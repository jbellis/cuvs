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

#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void compute_l2_similarities_kernel(
        const float* query,
        const float* vq_code_book,
        const float* pq_code_book,
        const uint8_t* encoded_data,
        const int64_t* node_ids,
        float* similarities,
        int64_t dim,
        int64_t pq_dim,
        int64_t pq_len,
        int64_t vq_n_centers,
        int n_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_nodes) return;

    int64_t node_idx = node_ids[idx];
    const uint8_t* node_data = encoded_data + node_idx * (sizeof(uint32_t) + pq_dim);
    uint32_t vq_code = *reinterpret_cast<const uint32_t*>(node_data);
    const uint8_t* pq_codes = node_data + sizeof(uint32_t);

    float squared_distance = 0.0f;

    for (int i = 0; i < pq_dim; ++i) {
        uint8_t pq_code = pq_codes[i];
        const float* vq_subvector = vq_code_book + vq_code * dim + i * pq_len;
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
        const float* query,
        const cuvs::neighbors::vpq_dataset<float, int64_t>& vpq_data,
        const std::vector<int64_t>& host_node_ids,
        std::vector<float>& host_similarities)
{
    cudaStream_t stream = dev_resources.get_stream();
    int n_nodes = host_node_ids.size();

    // Extract necessary fields from vpq_data
    const float* vq_code_book = vpq_data.vq_code_book.data_handle();
    const float* pq_code_book = vpq_data.pq_code_book.data_handle();
    const uint8_t* encoded_data = vpq_data.data.data_handle();
    int64_t dim = vpq_data.dim();
    int64_t pq_dim = vpq_data.pq_dim();
    int64_t pq_len = vpq_data.pq_len();
    int64_t vq_n_centers = vpq_data.vq_n_centers();

    // Allocate device memory for node IDs and similarities
    auto d_node_ids = raft::make_device_vector<int64_t, int64_t>(dev_resources, n_nodes);
    auto d_similarities = raft::make_device_vector<float, int64_t>(dev_resources, n_nodes);

    // Copy node IDs to device
    raft::copy(d_node_ids.data_handle(), host_node_ids.data(), n_nodes, stream);

    // Launch kernel
    int block_size = 256;
    int grid_size = (n_nodes + block_size - 1) / block_size;

    compute_l2_similarities_kernel<<<grid_size, block_size, 0, stream>>>(
            query,
            vq_code_book,
            pq_code_book,
            encoded_data,
            d_node_ids.data_handle(),
            d_similarities.data_handle(),
            dim,
            pq_dim,
            pq_len,
            vq_n_centers,
            n_nodes
    );

    // Copy results back to host
    raft::copy(host_similarities.data(), d_similarities.data_handle(), n_nodes, stream);

    // Synchronize to ensure copy is complete
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

void vpq_test(raft::device_resources const& dev_resources,
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

    // Prepare host vectors for random nodes and similarities
    const int n_random_nodes = 32;
    std::vector<int64_t> h_random_nodes(n_random_nodes);
    std::vector<float> h_similarities(n_random_nodes);

    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(0, dataset.extent(0) - 1);

    // For each query
    for (int64_t query_idx = 0; query_idx < queries.extent(0); ++query_idx) {
        // Generate random nodes on host
        std::generate(h_random_nodes.begin(), h_random_nodes.end(), [&]() { return dist(gen); });

        // Compute similarities
        compute_l2_similarities(
                dev_resources,
                queries.data_handle() + query_idx * dim,
                vpq_data,
                h_random_nodes,
                h_similarities
        );

        // Print results
        std::cout << "Query " << query_idx << " similarities:" << std::endl;
        for (int i = 0; i < n_random_nodes; ++i) {
            std::cout << "Node " << h_random_nodes[i] << ": " << h_similarities[i] << std::endl;
        }
        std::cout << std::endl;
    }
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
