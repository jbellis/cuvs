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

template <uint32_t DATASET_BLOCK_DIM, uint32_t TEAM_SIZE, cuvs::distance::DistanceType METRIC, typename DataT, typename MathT, typename IdxT>
__device__ auto compute_similarity_coarse(
        const MathT* const query_ptr,
        raft::device_matrix_view<const DataT, IdxT, raft::row_major> dataset,
        raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers,
        const IdxT node_id,
        const bool valid) -> float
{
    if (!valid) {
        return 0;
    }

    float norm = 0;
    const uint32_t vq_code = dataset(node_id, 0);
    const uint32_t dim = vq_centers.extent(1);

    for (uint32_t elem_offset = 0; elem_offset < dim; elem_offset++) {
        MathT diff = query_ptr[elem_offset] - vq_centers(vq_code, elem_offset);
        norm += diff * diff;
    }

    if (node_id < 50) {
        printf("Node %ld: norm = %f\n", node_id, norm);
    }

    return norm;
}

// Define a kernel to test compute_similarity_coarse
template <uint32_t DATASET_BLOCK_DIM, uint32_t TEAM_SIZE, cuvs::distance::DistanceType METRIC, typename DataT, typename MathT, typename IdxT>
__global__ void test_compute_similarity_coarse_kernel(
        const MathT* query_ptr,
        raft::device_matrix_view<const DataT, IdxT, raft::row_major> dataset,
        raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers,
        MathT* similarities,
        IdxT n_rows)
{
    IdxT idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_rows) {
        similarities[idx] = compute_similarity_coarse<DATASET_BLOCK_DIM, TEAM_SIZE, METRIC>(
                query_ptr, dataset, vq_centers, idx, true);
    }
}

void vpq_coarse_test(raft::device_resources const& dev_resources,
                     raft::device_matrix_view<const float, int64_t> dataset,
                     raft::device_matrix_view<const float, int64_t> queries)
{
    using namespace cuvs::neighbors;

    // Create vpq_params
    vpq_params params;
    params.vq_n_centers = 256;  // Example value, adjust as needed

    // Build VPQ coarse index
    auto vpq_coarse = cuvs::neighbors::vpq_build_coarse<decltype(dataset), float, int64_t>(dev_resources, params, dataset);

    std::cout << "VPQ Coarse index built with " << vpq_coarse.n_rows() << " vectors" << std::endl;
    std::cout << "VQ codebook size: [" << vpq_coarse.vq_code_book.extent(0) << ", " << vpq_coarse.vq_code_book.extent(1) << "]" << std::endl;
    std::cout << "VQ codebook (first few elements of first few centroids):" << std::endl;
    std::vector<float> h_vq_codebook(vpq_coarse.vq_code_book.extent(0) * 10);
    raft::copy(h_vq_codebook.data(), vpq_coarse.vq_code_book.data_handle(), vpq_coarse.vq_code_book.extent(0) * 10, dev_resources.get_stream());
    raft::resource::sync_stream(dev_resources);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 10; ++j) {
            std::cout << h_vq_codebook[i * vpq_coarse.vq_code_book.extent(1) + j] << " ";
        }
        std::cout << std::endl;
    }

    // Use the first query vector directly from the queries parameter
    const float* query_ptr = queries.data_handle();

    // Allocate memory for similarities
    auto similarities = raft::make_device_vector<float>(dev_resources, vpq_coarse.n_rows());

    // Launch kernel to compute similarities
    constexpr uint32_t DATASET_BLOCK_DIM = 32;
    constexpr uint32_t TEAM_SIZE = 32;
    constexpr auto METRIC = cuvs::distance::DistanceType::L2Expanded;
    const dim3 block(256);
    const dim3 grid((vpq_coarse.n_rows() + block.x - 1) / block.x);

    // Debug: Check the contents of vpq_coarse.data
    std::vector<uint32_t> h_data(vpq_coarse.n_rows());
    raft::copy(h_data.data(), vpq_coarse.data.data_handle(), vpq_coarse.n_rows(), dev_resources.get_stream());
    raft::resource::sync_stream(dev_resources);

    std::cout << "First few VQ codes:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    test_compute_similarity_coarse_kernel<DATASET_BLOCK_DIM, 1, METRIC, uint32_t, float, int64_t><<<grid, block>>>(
            query_ptr,
            vpq_coarse.data.view(),
            vpq_coarse.vq_code_book.view(),
            similarities.data_handle(),
            vpq_coarse.n_rows());
    
    // Synchronize and check for errors
    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    // Print the first few similarities
    std::cout << "First few similarities:" << std::endl;
    std::vector<float> h_similarities(10);
    raft::copy(h_similarities.data(), similarities.data_handle(), 10, dev_resources.get_stream());
    raft::resource::sync_stream(dev_resources);

    for (int i = 0; i < 10; ++i) {
        std::cout << h_similarities[i] << " ";
    }
    std::cout << std::endl;
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
  int64_t n_dim     = 90;
  int64_t n_queries = 10;
  auto dataset      = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries      = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);
  generate_dataset(dev_resources, dataset.view(), queries.view());

  vpq_coarse_test(dev_resources,
                          raft::make_const_mdspan(dataset.view()),
                          raft::make_const_mdspan(queries.view()));
}
