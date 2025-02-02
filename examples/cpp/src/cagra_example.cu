/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>

#include <chrono>
#include <cstdint>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuvs/neighbors/common.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

// based on cuvs::neighbors::vpq_dataset, but with only a single global centroid instead of a vq codebook
template <typename MathT, typename IdxT>
struct jpq_dataset : cuvs::neighbors::dataset<IdxT> {
    /** Global centroid */
    raft::device_vector<MathT, uint32_t> vq_center;
    /** Product Quantization codebook */
    raft::device_matrix<MathT, uint32_t, raft::row_major> pq_codebook;
    /** Compressed dataset (indexes into codebook) */
    raft::device_matrix<uint8_t, IdxT, raft::row_major> codepoints;
    /** Dimensionality of a subspace */
    uint32_t pq_len;

    jpq_dataset(raft::device_vector<MathT, uint32_t>&& vq_center,
                raft::device_matrix<MathT, uint32_t, raft::row_major>&& pq_codebook,
                raft::device_matrix<uint8_t, IdxT, raft::row_major>&& codepoints,
                uint32_t pq_len)
            : vq_center{std::move(vq_center)},
              pq_codebook{std::move(pq_codebook)},
              codepoints{std::move(codepoints)},
              pq_len{pq_len}
    {
    }

    [[nodiscard]] auto n_rows() const noexcept -> IdxT final { return codepoints.extent(0); }
    [[nodiscard]] auto dim() const noexcept -> uint32_t final { return vq_center.size(); }
    [[nodiscard]] auto is_owning() const noexcept -> bool final { return true; }

    /** Row length of the encoded data in bytes. */
    [[nodiscard]] constexpr inline auto encoded_row_length() const noexcept -> uint32_t
    {
        return codepoints.extent(1);
    }
    /** The bit length of an encoded vector element after compression by PQ. */
    [[nodiscard]] constexpr inline auto pq_bits() const noexcept -> uint32_t
    {
        /*
        NOTE: pq_bits and the book size

        Normally, we'd store `pq_bits` as a part of the index.
        However, we know there's an invariant `pq_n_centers = 1 << pq_bits`, i.e. the codebook size is
        the same as the number of possible code values. Hence, we don't store the pq_bits and derive it
        from the array dimensions instead.
         */
        auto pq_width = pq_n_centers();
#ifdef __cpp_lib_bitops
        return std::countr_zero(pq_width);
#else
        uint32_t pq_bits = 0;
        while (pq_width > 1) {
            pq_bits++;
            pq_width >>= 1;
        }
        return pq_bits;
#endif
    }
    /** The dimensionality of an encoded vector after compression by PQ. */
    [[nodiscard]] constexpr inline auto pq_dim() const noexcept -> uint32_t
    {
        return raft::div_rounding_up_unsafe(dim(), pq_len);
    }
    /** The number of vectors in a PQ codebook (`1 << pq_bits`). */
    [[nodiscard]] constexpr inline auto pq_n_centers() const noexcept -> uint32_t
    {
        return pq_codebook.extent(0);
    }
};

__global__ void compute_adc_lut_kernel(
        const float* query,
        const float* vq_center,
        const float* pq_codebook,
        float* adc_lut,
        int64_t pq_dim,
        int64_t pq_len,
        int dim
) {
    int centroid_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int threads_per_block = blockDim.x;

    // Each thread processes multiple subspaces if needed
    for (int subspace_idx = thread_idx; subspace_idx < pq_dim; subspace_idx += threads_per_block) {
        float partial_distance = 0.0f;
        for (int i = 0; i < pq_len; i++) {
            int feature_idx = subspace_idx * pq_len + i;
            if (feature_idx >= dim) break;

            float q_val = query[feature_idx];
            float vq_val = vq_center[feature_idx];
            float pq_val = pq_codebook[subspace_idx * 256 * pq_len + centroid_idx * pq_len + i];

            partial_distance += q_val * (vq_val + pq_val);
        }

        adc_lut[centroid_idx * pq_dim + subspace_idx] = partial_distance;
    }
}

float* compute_dp_setup(
        raft::device_resources const& res,
        const raft::device_vector<float, int64_t>& d_query,
        const jpq_dataset<float, int64_t>& jpq_data)
{
    cudaStream_t stream = res.get_stream();
    int64_t pq_dim = jpq_data.pq_dim();
    int64_t pq_len = jpq_data.pq_len;
    int64_t dim = jpq_data.dim();

    // Allocate device memory for ADC lookup table
    float* d_adc_lut;
    RAFT_CUDA_TRY(cudaMalloc(&d_adc_lut, pq_dim * 256 * sizeof(float)));

    // Launch kernel to compute ADC lookup table
    int block_size = std::min(256, static_cast<int>(pq_dim));
    dim3 block_dim(block_size);
    dim3 grid_dim(256);  // always 256 subspaces
    compute_adc_lut_kernel<<<grid_dim, block_dim, 0, stream>>>(
            d_query.data_handle(),
            jpq_data.vq_center.data_handle(),
            jpq_data.pq_codebook.data_handle(),
            d_adc_lut,
            pq_dim,
            pq_len,
            dim
    );

    // Synchronize to ensure computation is complete
    res.sync_stream();

    return d_adc_lut;
}

__global__ void compute_dp_similarities_kernel(
        const float* adc_lut,
        const uint8_t* codepoints,
        const int32_t* node_ids,
        float* similarities,
        int64_t pq_dim,
        int n_nodes
) {
    int node_idx = blockIdx.x;
    int tid = threadIdx.x;
    int threads_per_block = blockDim.x;

    __shared__ float shared_distance[256];

    float thread_distance = 0.0f;
    const uint8_t* pq_codes = codepoints + node_ids[node_idx] * pq_dim;

    // Each thread processes multiple subspaces if needed
    for (int subspace_idx = tid; subspace_idx < pq_dim; subspace_idx += threads_per_block) {
        uint8_t pq_code = pq_codes[subspace_idx];
        thread_distance += adc_lut[pq_code * pq_dim + subspace_idx];
    }
    shared_distance[tid] = thread_distance;
    __syncthreads();

    // Reduce within block
    for (int s = threads_per_block / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_distance[tid] += shared_distance[tid + s];
        }
        __syncthreads();
    }

    // Compute final similarity
    if (tid == 0) {
        float total_distance = shared_distance[0];
        similarities[node_idx] = (1 + total_distance) / 2;
    }
}

void compute_dp_similarities(
        raft::device_resources const& res,
        float* adc_lut,
        const jpq_dataset<float, int64_t>& jpq_data,
        const int32_t* host_node_ids,
        float* host_similarities,
        int64_t n_nodes)
{
    cudaStream_t stream = res.get_stream();

    // Copy node ids to device
    auto d_node_ids = raft::make_device_vector<int32_t, int64_t>(res, n_nodes);
    raft::copy(d_node_ids.data_handle(), host_node_ids, n_nodes, stream);

    // Allocate device memory for similarities
    auto d_similarities = raft::make_device_vector<float, int64_t>(res, n_nodes);

    // Launch kernel to compute similarities
    int64_t pq_dim = jpq_data.pq_dim();
    int block_size = std::min(256, static_cast<int>(pq_dim));
    dim3 block_dim(block_size);
    dim3 grid_size(n_nodes);
    compute_dp_similarities_kernel<<<grid_size, block_size, 0, stream>>>(
            adc_lut,
            jpq_data.codepoints.data_handle(),
            d_node_ids.data_handle(),
            d_similarities.data_handle(),
            pq_dim,
            n_nodes
    );

    // Copy results back to host
    raft::copy(host_similarities, d_similarities.data_handle(), n_nodes, stream);
    res.sync_stream();
}

int32_t readIntBE(std::ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(int32_t));
    if (file.gcount() != sizeof(int32_t)) {
        throw std::runtime_error("Failed to read 4 bytes for int32");
    }
    return static_cast<int32_t>(__builtin_bswap32(value));  // For GCC/Clang
}

float readFloatBE(std::ifstream& file) {
    uint32_t intValue;
    file.read(reinterpret_cast<char*>(&intValue), sizeof(float));
    if (file.gcount() != sizeof(float)) {
        throw std::runtime_error("Failed to read 4 bytes for float");
    }
    intValue = __builtin_bswap32(intValue);  // For GCC/Clang
    return *reinterpret_cast<float*>(&intValue);
}

template <typename MathT, typename IdxT>
jpq_dataset<MathT, IdxT> load_pq_vectors(raft::device_resources const &res, const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Read and check magic number
    int32_t magic = readIntBE(file);
    if (magic != 0x75EC4012) {
        throw std::runtime_error("Invalid magic number in file");
    }

    // Read and check version
    int32_t version = readIntBE(file);
    if (version != 3) {
        throw std::runtime_error("Unsupported file version: " + std::to_string(version));
    }

    // Read global centroid
    int32_t global_centroid_length = readIntBE(file);
    std::vector<MathT> global_centroid(global_centroid_length, 0);
    for (int i = 0; i < global_centroid_length; ++i) {
        global_centroid[i] = static_cast<MathT>(readFloatBE(file));
    }

    // Read M (number of subspaces)
    int32_t M = readIntBE(file);
    if (M <= 0) {
        throw std::runtime_error("Invalid number of subspaces: " + std::to_string(M));
    }

    // Read subvector sizes
    int32_t subspace_size = readIntBE(file);
    for (int i = 1; i < M; ++i) {
        int32_t ss = readIntBE(file);
        if (ss != subspace_size) {
            throw std::runtime_error("CUVS only supports the case where all subvectors have the same size");
        }
    }

    // Read anisotropic threshold (ignored)
    float anisotropic_threshold = readFloatBE(file);

    // Read cluster count
    int32_t cluster_count = readIntBE(file);
    if (cluster_count != 256) {
        // CUVS can support other configurations, but the rest of our code here assumes 8 bit codes = 256 clusters
        throw std::runtime_error("Unsupported cluster count: " + std::to_string(cluster_count));
    }

    // Read PQ codebooks
    uint32_t total_dim = subspace_size * M;
    if (global_centroid_length != 0 && global_centroid_length != total_dim) {
        throw std::runtime_error("Global centroid length mismatch: " + std::to_string(global_centroid_length) +
                                 ", expected " + std::to_string(total_dim));
    }
    std::vector<MathT> host_pq_codebook(cluster_count * total_dim);
    for (size_t i = 0; i < host_pq_codebook.size(); ++i) {
        host_pq_codebook[i] = static_cast<MathT>(readFloatBE(file));
    }

    // Read compressed vectors
    int32_t vector_count = readIntBE(file);
    int32_t compressed_dimension = readIntBE(file);
    if (compressed_dimension != M) {
        throw std::runtime_error("Invalid compressed dimension: " + std::to_string(compressed_dimension));
    }

    // Debug: Print vector count and compressed dimension
    std::cout << "Loading " << vector_count << " vectors with compressed dimension " << compressed_dimension << std::endl;

    // Prepare device arrays
    auto vq_center = raft::make_device_vector<MathT, uint32_t>(res, total_dim);
    auto pq_codebook = raft::make_device_matrix<MathT, uint32_t>(res, cluster_count, total_dim);
    auto compressed_data = raft::make_device_matrix<uint8_t, IdxT>(res, vector_count, compressed_dimension);

    // Copy data to device
    raft::copy(vq_center.data_handle(), global_centroid.data(), global_centroid.size(), raft::resource::get_cuda_stream(res));
    raft::copy(pq_codebook.data_handle(), host_pq_codebook.data(), host_pq_codebook.size(), raft::resource::get_cuda_stream(res));

    // Read the pq code points
    std::vector<uint8_t> host_compressed_data(vector_count * compressed_dimension);
    for (int i = 0; i < vector_count; ++i) {
        std::streamsize bytes_read = file.read(
                reinterpret_cast<char*>(host_compressed_data.data() + i * compressed_dimension),
                compressed_dimension
        ).gcount();

        if (bytes_read != compressed_dimension) {
            throw std::runtime_error("Failed to read compressed vector " + std::to_string(i));
        }
    }
    // copy to device memory
    raft::copy(compressed_data.data_handle(), host_compressed_data.data(), host_compressed_data.size(), raft::resource::get_cuda_stream(res));
    res.sync_stream();

    // instantiate the jpq_dataset
    auto jpq_data = jpq_dataset<MathT, IdxT>{std::move(vq_center), std::move(pq_codebook), std::move(compressed_data), static_cast<uint32_t>(subspace_size)};

    // Validate
    if (jpq_data.n_rows() != vector_count) {
        throw std::runtime_error("Row count mismatch: jpq_data.n_rows() = " + std::to_string(jpq_data.n_rows()) +
                                 ", expected " + std::to_string(vector_count));
    }
    if (jpq_data.dim() != total_dim) {
        throw std::runtime_error("Dimension mismatch: jpq_data.dim() = " + std::to_string(jpq_data.dim()) +
                                 ", expected " + std::to_string(total_dim));
    }
    if (jpq_data.encoded_row_length() != compressed_dimension) {
        throw std::runtime_error("Encoded row length mismatch: jpq_data.encoded_row_length() = " + std::to_string(jpq_data.encoded_row_length()) +
                                 ", expected " + std::to_string(compressed_dimension));
    }
    if (jpq_data.pq_bits() != 8) {
        throw std::runtime_error("PQ bits mismatch: jpq_data.pq_bits() = " + std::to_string(jpq_data.pq_bits()) +
                                 ", expected 8");
    }
    if (jpq_data.pq_dim() != M) {
        throw std::runtime_error("PQ dimension mismatch: jpq_data.pq_dim() = " + std::to_string(jpq_data.pq_dim()) +
                                 ", expected " + std::to_string(M));
    }
    if (jpq_data.pq_len != subspace_size) {
        throw std::runtime_error("PQ length mismatch: jpq_data.pq_len = " + std::to_string(jpq_data.pq_len) +
                                 ", expected " + std::to_string(subspace_size));
    }
    if (jpq_data.pq_n_centers() != cluster_count) {
        throw std::runtime_error("PQ centers count mismatch: jpq_data.pq_n_centers() = " + std::to_string(jpq_data.pq_n_centers()) +
                                 ", expected " + std::to_string(cluster_count));
    }

    return jpq_data;
}

void jpq_test_cohere(raft::device_resources const &res) {
    auto jpq_data = load_pq_vectors<float, int64_t>(res, "cohere.pqv");
    const int max_nodes = 200;
    std::array<int32_t, max_nodes> node_ids{};
    std::array<float, max_nodes> similarities{};
    std::array<float, 1024> host_q;
    auto d_q = raft::make_device_vector<float, int64_t>(res, 1024);

    // fixed vector of ones
    int dim = jpq_data.dim();
    std::vector<float> host_ones(dim, 1.0f);
    auto d_ones = raft::make_device_vector<float, int64_t>(res, dim);
    raft::copy(d_ones.data_handle(), host_ones.data(), dim, res.get_stream());
    res.sync_stream();

    // compare ones with the first 10 vectors in the dataset
    int n_nodes = 10;
    std::iota(node_ids.begin(), node_ids.begin() + n_nodes, 0);
    float* adc_lut_ones = compute_dp_setup(res, d_ones, jpq_data);
    compute_dp_similarities(res, adc_lut_ones, jpq_data, node_ids.data(), similarities.data(), n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        std::cout << "Similarity with ones: " << similarities[i] << std::endl;
    }
    RAFT_CUDA_TRY(cudaFree(adc_lut_ones));

    // Benchmark random queries
    std::random_device rd;
    std::mt19937 gen(rd());
    // Query vector elements from -1 .. 1
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    // Node IDs from 0 .. 99999
    std::uniform_int_distribution<> node_dis(0, 99999);

    std::chrono::duration<double> elapsed = std::chrono::duration<double>::zero();
    for (int i = 0; i < 10000; ++i) {
        // query vector
        std::generate(host_q.begin(), host_q.end(), [&]() { return dis(gen); });
        raft::copy(d_q.data_handle(), host_q.data(), 1024, res.get_stream());
        res.sync_stream();

        auto start = std::chrono::high_resolution_clock::now();

        // Compute ADC lookup table for the query
        float* adc_lut = compute_dp_setup(res, d_q, jpq_data);

        // node IDs
        std::generate(node_ids.begin(), node_ids.end(), [&]() { return node_dis(gen); });
        // compute similarities
        compute_dp_similarities(res, adc_lut, jpq_data, node_ids.data(), similarities.data(), node_ids.size());

        RAFT_CUDA_TRY(cudaFree(adc_lut));

        elapsed += std::chrono::high_resolution_clock::now() - start;
    }

    std::cout << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;
}

int main()
{
  raft::device_resources res;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used within RAFT
  // algorithms). In that case only the internal arrays would use the pool, any other allocation
  // uses the default RMM memory resource. Here is how to change the workspace memory resource to
  // a pool with 2 GiB upper limit.
  // raft::resource::set_workspace_to_pool_resource(res, 2 * 1024 * 1024 * 1024ull);

  jpq_test_cohere(res);
}
