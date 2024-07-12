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

#include <chrono>
#include <cstdint>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>

#include "../../../cpp/src/neighbors/vpq_dataset.cuh" // fuck me
#include <cuvs/neighbors/common.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "common.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void compute_l2_similarities_kernel(
        const float* query,
        const float* vq_code_book,
        const float* pq_code_book,
        const uint8_t* encoded_data,
        const int32_t* node_ids,
        float* similarities,
        int64_t pq_dim,
        int64_t pq_len,
        int n_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_nodes) return;

    int32_t node_idx = node_ids[idx];
    const uint8_t* node_data = encoded_data + node_idx * (1 + pq_dim);
    uint8_t vq_code = *node_data;
    if (vq_code != 0) {
        printf("Bad vq code %d!\n", vq_code); // VSTODO add error code
    }
    const uint8_t* pq_codes = node_data + 1;

    float squared_distance = 0.0f;
    for (int i = 0; i < pq_dim; ++i) {
        uint8_t pq_code = pq_codes[i];
        const float* vq_subvector = vq_code_book + i * pq_len;
        const float* pq_subvector = pq_code_book + pq_code * pq_len;
        const float* query_subvector = query + i * pq_len;

        for (int j = 0; j < pq_len; ++j) {
            float diff = query_subvector[j] - (vq_subvector[j] + pq_subvector[j]);
            squared_distance += diff * diff;
        }
    }

    similarities[idx] = 1 / (1 + squared_distance);
}

// Wrapper function for the compute kernel
void compute_l2_similarities(
        raft::device_resources const& dev_resources,
        const float* host_query,
        const cuvs::neighbors::vpq_dataset<float, int64_t>& vpq_data,
        const int32_t* host_node_ids,
        float* host_similarities,
        int64_t n_nodes)
{
    cudaStream_t stream = dev_resources.get_stream();

    // Extract necessary fields from vpq_data
    const float* vq_code_book = vpq_data.vq_code_book.data_handle();
    const float* pq_code_book = vpq_data.pq_code_book.data_handle();
    const uint8_t* encoded_data = vpq_data.data.data_handle();
    int64_t dim = vpq_data.dim();
    int64_t pq_dim = vpq_data.pq_dim();
    int64_t pq_len = vpq_data.pq_len();
    int64_t vq_n_centers = vpq_data.vq_n_centers();
    if (vq_n_centers != 1) {
        throw std::runtime_error("VQ centers count must be 1");
    }

    // Allocate device memory for query, node IDs and similarities
    auto d_query = raft::make_device_vector<float, int64_t>(dev_resources, dim);
    auto d_node_ids = raft::make_device_vector<int32_t, int64_t>(dev_resources, n_nodes);
    auto d_similarities = raft::make_device_vector<float, int64_t>(dev_resources, n_nodes);

    // Copy query and node IDs to device
    raft::copy(d_query.data_handle(), host_query, dim, stream);
    raft::copy(d_node_ids.data_handle(), host_node_ids, n_nodes, stream);

    // Launch kernel
    int block_size = 256;
    int grid_size = (n_nodes + block_size - 1) / block_size;

    compute_l2_similarities_kernel<<<grid_size, block_size, 0, stream>>>(
            d_query.data_handle(),
            vq_code_book,
            pq_code_book,
            encoded_data,
            d_node_ids.data_handle(),
            d_similarities.data_handle(),
            pq_dim,
            pq_len,
            n_nodes
    );

    // Copy results back to host
    raft::copy(host_similarities, d_similarities.data_handle(), n_nodes, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

int32_t readIntBE(std::ifstream& file) {
    uint8_t buffer[4];
    file.read(reinterpret_cast<char*>(buffer), 4);
    if (file.gcount() != 4) {
        throw std::runtime_error("Failed to read 4 bytes for int32");
    }
    return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
}

float readFloatBE(std::ifstream& file) {
    uint8_t buffer[4];
    file.read(reinterpret_cast<char*>(buffer), 4);
    if (file.gcount() != 4) {
        throw std::runtime_error("Failed to read 4 bytes for float");
    }
    uint32_t intValue = (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
    float floatValue;
    std::memcpy(&floatValue, &intValue, sizeof(float));
    return floatValue;
}

template <typename MathT, typename IdxT>
cuvs::neighbors::vpq_dataset<MathT, IdxT> load_pq_vectors(raft::device_resources const &res, const std::string& filename)
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
    uint32_t codes_rowlen = 1 + compressed_dimension;
    auto vq_code_book = raft::make_device_matrix<MathT, uint32_t>(res, 1, total_dim);
    auto pq_code_book = raft::make_device_matrix<MathT, uint32_t>(res, cluster_count, total_dim);
    auto compressed_data = raft::make_device_matrix<uint8_t, IdxT>(res, vector_count, codes_rowlen);

    // Copy data to device
    raft::copy(vq_code_book.data_handle(), global_centroid.data(), global_centroid.size(), raft::resource::get_cuda_stream(res));
    raft::copy(pq_code_book.data_handle(), host_pq_codebook.data(), host_pq_codebook.size(), raft::resource::get_cuda_stream(res));

    // initizalize the first label (the vq code) to 0
    std::vector<uint8_t> host_compressed_data(vector_count * codes_rowlen, 0);
    // Read the pq code points
    for (int i = 0; i < vector_count; ++i) {
        std::streamsize bytes_read = file.read(
                reinterpret_cast<char*>(host_compressed_data.data() + i * codes_rowlen + 1),
                compressed_dimension
        ).gcount();

        if (bytes_read != compressed_dimension) {
            throw std::runtime_error("Failed to read compressed vector " + std::to_string(i));
        }
    }
    // copy to device memory
    raft::copy(compressed_data.data_handle(), host_compressed_data.data(), host_compressed_data.size(), raft::resource::get_cuda_stream(res));
    RAFT_CUDA_TRY(cudaStreamSynchronize(raft::resource::get_cuda_stream(res)));

    // instantiate the vpq_dataset
    auto vpq_data = cuvs::neighbors::vpq_dataset<MathT, IdxT>{std::move(vq_code_book), std::move(pq_code_book), std::move(compressed_data)};

    // Validate
    if (vpq_data.n_rows() != vector_count) {
        throw std::runtime_error("Row count mismatch: vpq_data.n_rows() = " + std::to_string(vpq_data.n_rows()) +
                                 ", expected " + std::to_string(vector_count));
    }
    if (vpq_data.dim() != total_dim) {
        throw std::runtime_error("Dimension mismatch: vpq_data.dim() = " + std::to_string(vpq_data.dim()) +
                                 ", expected " + std::to_string(total_dim));
    }
    if (vpq_data.encoded_row_length() != codes_rowlen) {
        throw std::runtime_error("Encoded row length mismatch: vpq_data.encoded_row_length() = " + std::to_string(vpq_data.encoded_row_length()) +
                                 ", expected " + std::to_string(codes_rowlen));
    }
    if (vpq_data.vq_n_centers() != 1) {
        throw std::runtime_error("VQ centers count mismatch: vpq_data.vq_n_centers() = " + std::to_string(vpq_data.vq_n_centers()) +
                                 ", expected 1");
    }
    if (vpq_data.pq_bits() != 8) {
        throw std::runtime_error("PQ bits mismatch: vpq_data.pq_bits() = " + std::to_string(vpq_data.pq_bits()) +
                                 ", expected 8");
    }
    if (vpq_data.pq_dim() != M) {
        throw std::runtime_error("PQ dimension mismatch: vpq_data.pq_dim() = " + std::to_string(vpq_data.pq_dim()) +
                                 ", expected " + std::to_string(M));
    }
    if (vpq_data.pq_len() != subspace_size) {
        throw std::runtime_error("PQ length mismatch: vpq_data.pq_len() = " + std::to_string(vpq_data.pq_len()) +
                                 ", expected " + std::to_string(subspace_size));
    }
    if (vpq_data.pq_n_centers() != cluster_count) {
        throw std::runtime_error("PQ centers count mismatch: vpq_data.pq_n_centers() = " + std::to_string(vpq_data.pq_n_centers()) +
                                 ", expected " + std::to_string(cluster_count));
    }

    return vpq_data;
}

void vpq_test_java(raft::device_resources const &dev_resources)
{
    auto vpq_data = load_pq_vectors<float, int64_t>(dev_resources, "test.pqv");

    int dim = vpq_data.dim();
    float* zeros = new float[dim]();
    std::fill(zeros, zeros + dim, 0.0f);
    float* ones = new float[dim]();
    std::fill(ones, ones + dim, 1.0f);

    int64_t n_nodes = 10;
    int32_t* node_ids = new int32_t[n_nodes];
    for (int32_t i = 0; i < n_nodes; ++i) {
        node_ids[i] = i;
    }
    float* similarities = new float[n_nodes];

    // compare zeros with the first 10 vectors in the dataset
    compute_l2_similarities(dev_resources, zeros, vpq_data, node_ids, similarities, n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        std::cout << "Similarity with zero: " << similarities[i] << std::endl;
    }

    // compare ones with the first 10 vectors in the dataset
    compute_l2_similarities(dev_resources, ones, vpq_data, node_ids, similarities, n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        std::cout << "Similarity with ones: " << similarities[i] << std::endl;
    }

    free(node_ids);
    free(similarities);
}

void vpq_test_random(raft::device_resources const &dev_resources)
{
    using namespace cuvs::neighbors;
    uint32_t PQ_BITS = 8;
    uint32_t PQ_LEN = 16;

    // Create input arrays.
    int64_t n_samples = 10000;
    int64_t n_dim     = 1024;
    int64_t n_queries = 10;
    auto dataset      = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
    auto queries      = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);
    generate_dataset(dev_resources, dataset.view(), queries.view());

    // Create vpq_params
    vpq_params params;
    params.vq_n_centers = 1;
    const uint32_t dim = dataset.extent(1);
    assert(dim % PQ_LEN == 0);
    params.pq_dim = dim / PQ_LEN;
    params.pq_bits = PQ_BITS;

    // Build VPQ dataset
    auto vpq_data = cuvs::neighbors::vpq_build<decltype(dataset), float, int64_t>(dev_resources, params, dataset);
    if (vpq_data.vq_n_centers() != 1) {
        throw std::runtime_error("VQ centers count mismatch: vpq_data.vq_n_centers() = " + std::to_string(vpq_data.vq_n_centers()) +
                                 ", expected 1");
    }
    if (vpq_data.pq_bits() != 8) {
        throw std::runtime_error("PQ bits mismatch: vpq_data.pq_bits() = " + std::to_string(vpq_data.pq_bits()) +
                                 ", expected 8");
    }

    // Prepare host memory for random nodes and similarities
    const int n_random_nodes = 32;
    size_t alignment = 32;
    int32_t* node_ids = static_cast<int32_t*>(aligned_alloc(alignment, n_random_nodes * sizeof(int32_t)));
    float* similarities = static_cast<float*>(aligned_alloc(alignment, n_random_nodes * sizeof(float)));

    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dist(0, dataset.extent(0) - 1);

    auto start = std::chrono::high_resolution_clock::now();
    // For each query
    for (int64_t query_idx = 0; query_idx < queries.extent(0); ++query_idx) {
        // Generate random nodes
        for (int i = 0; i < n_random_nodes; ++i) {
            node_ids[i] = dist(gen);
        }

        // Compute similarities
        compute_l2_similarities(
                dev_resources,
                queries.data_handle() + query_idx * dim,
                vpq_data,
                node_ids,
                similarities,
                n_random_nodes
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Total time to evaluate all queries: " << duration.count() << " seconds" << std::endl;

    // Free allocated memory
    free(node_ids);
    free(similarities);
}

int main()
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used within RAFT
  // algorithms). In that case only the internal arrays would use the pool, any other allocation
  // uses the default RMM memory resource. Here is how to change the workspace memory resource to
  // a pool with 2 GiB upper limit.
  // raft::resource::set_workspace_to_pool_resource(dev_resources, 2 * 1024 * 1024 * 1024ull);

  vpq_test_random(dev_resources);
}
