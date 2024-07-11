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

#include <cstdint>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>

#include "../../../cpp/src/neighbors/vpq_dataset.cuh" // fuck me
#include <cuvs/neighbors/common.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "common.cuh"

#include <raft/linalg/map.cuh>
#include <raft/stats/mean.cuh>

template <typename T, typename IdxT>
__global__ void decode_vpq_data_kernel(
        const uint8_t* encoded_data,
        const T* vq_codebook,
        const T* pq_codebook,
        T* decoded_data,
        IdxT n_rows,
        uint32_t dim,
        uint32_t pq_dim,
        uint32_t pq_bits,
        uint32_t vq_n_centers)
{
    const IdxT row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const uint32_t pq_len = dim / pq_dim;
    const uint32_t encoded_row_len = sizeof(uint32_t) + raft::div_rounding_up_safe<uint32_t>(pq_dim * pq_bits, 8);
    const uint8_t* row_data = encoded_data + row * encoded_row_len;

    // Get VQ center index
    uint32_t vq_index = *reinterpret_cast<const uint32_t*>(row_data);
    row_data += sizeof(uint32_t);

    for (uint32_t i = 0; i < pq_dim; i++) {
        uint32_t pq_code = 0;
        uint32_t bit_offset = i * pq_bits;
        uint32_t byte_offset = bit_offset / 8;
        uint32_t bit_shift = bit_offset % 8;

        // Extract PQ code
        for (uint32_t b = 0; b < pq_bits; b++) {
            pq_code |= ((row_data[byte_offset] >> (bit_shift + b)) & 1) << b;
        }

        // Decode PQ
        for (uint32_t j = 0; j < pq_len; j++) {
            uint32_t col = i * pq_len + j;
            decoded_data[row * dim + col] = pq_codebook[pq_code * pq_len + j];
        }
    }

    // Add VQ center
    for (uint32_t col = 0; col < dim; col++) {
        decoded_data[row * dim + col] += vq_codebook[vq_index * dim + col];
    }
}

__global__ void calculate_squared_diff(
        const float* dataset,
        const float* decoded_data,
        float* squared_diff,
        int64_t n_rows,
        int64_t n_cols
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_rows * n_cols) {
        float diff = dataset[idx] - decoded_data[idx];
        squared_diff[idx] = diff * diff;
    }
}

__global__ void compute_mse_kernel(const float* squared_diff, float* mse, int64_t n_elements)
{
    extern __shared__ float shared_sum[];

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;

    float sum = 0.0f;
    for (int64_t i = tid; i < n_elements; i += stride) {
        sum += squared_diff[i];
    }

    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (threadIdx.x == 0) {
        atomicAdd(mse, shared_sum[0]);
    }
}

void compute_mse(const raft::resources& dev_resources,
                 const float* squared_diff,
                 float* mse,
                 int64_t n_rows,
                 int64_t n_cols)
{
    int64_t n_elements = n_rows * n_cols;
    int block_size = 256;
    int grid_size = (n_elements + block_size - 1) / block_size;

    auto stream = raft::resource::get_cuda_stream(dev_resources);

    // Set initial value of mse to 0
    RAFT_CUDA_TRY(cudaMemsetAsync(mse, 0, sizeof(float), stream));

    // Launch kernel
    compute_mse_kernel<<<grid_size, block_size, block_size * sizeof(float), stream>>>(
            squared_diff, mse, n_elements);

    // Divide by total number of elements to get mean
    float inv_n_elements = 1.0f / static_cast<float>(n_elements);
    raft::linalg::scalarMultiply(mse, mse, inv_n_elements, 1, stream);
}

void vpq_test(raft::device_resources const &dev_resources,
              raft::device_matrix_view<const float, int64_t> dataset,
              raft::device_matrix_view<const float, int64_t> queries)
{
    using namespace cuvs::neighbors;

    uint32_t PQ_BITS = 8;
    uint32_t PQ_LEN = 16;

    // Create vpq_params
    vpq_params params;
    params.vq_n_centers = 256;
    const uint32_t dim = dataset.extent(1);
    assert(dim % PQ_LEN == 0);
    params.pq_dim = dim / PQ_LEN;
    params.pq_bits = PQ_BITS;

    // Build VPQ dataset
    auto vpq_data = cuvs::neighbors::vpq_build<decltype(dataset), float, int64_t>(dev_resources, params, dataset);

    // Calculate the quantization error

    // 1. Decode the VPQ-encoded data
    auto decoded_data = raft::make_device_matrix<float, int64_t>(dev_resources, dataset.extent(0), dataset.extent(1));

    // Launch kernel to decode data
    const int block_size = 256;
    const int grid_size = (dataset.extent(0) + block_size - 1) / block_size;

    decode_vpq_data_kernel<<<grid_size, block_size, 0, dev_resources.get_stream()>>>(
            vpq_data.data.data_handle(),
            vpq_data.vq_code_book.data_handle(),
            vpq_data.pq_code_book.data_handle(),
            decoded_data.data_handle(),
            dataset.extent(0),
            dim,
            params.pq_dim,
            params.pq_bits,
            params.vq_n_centers
    );

    // Calculate squared differences
    auto squared_diff = raft::make_device_matrix<float, int64_t>(dev_resources, dataset.extent(0), dataset.extent(1));

    calculate_squared_diff<<<grid_size, block_size, 0, dev_resources.get_stream()>>>(
            dataset.data_handle(),
            decoded_data.data_handle(),
            squared_diff.data_handle(),
            dataset.extent(0),
            dataset.extent(1)
    );

    // Compute mean of squared differences
    auto mse = raft::make_device_scalar<float>(dev_resources, 0.0f);
    compute_mse(dev_resources,
                squared_diff.data_handle(),
                mse.data_handle(),
                dataset.extent(0),
                dataset.extent(1));

    // Copy result to host and print
    float host_mse;
    raft::copy(&host_mse, mse.data_handle(), 1, dev_resources.get_stream());
    dev_resources.sync_stream();
    std::cout << "Mean Squared Error (Quantization Error): " << host_mse << std::endl;
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

  // Create input arrays.
  int64_t n_samples = 10000;
  int64_t n_dim     = 1024;
  int64_t n_queries = 10;
  auto dataset      = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries      = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);
  generate_dataset(dev_resources, dataset.view(), queries.view());

  vpq_test(dev_resources,
           raft::make_const_mdspan(dataset.view()),
           raft::make_const_mdspan(queries.view()));
}
