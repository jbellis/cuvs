/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

/*
 * NOTE: this file is generated by generate_ivf_pq.py
 *
 * Make changes there and run in this directory:
 *
 * > python generate_ivf_pq.py
 *
 */

#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft_runtime/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::ivf_pq {

#define CUVS_INST_IVF_PQ_SEARCH(T, IdxT)                                        \
  void search(raft::resources const& handle,                                    \
              const cuvs::neighbors::ivf_pq::search_params& params,             \
              cuvs::neighbors::ivf_pq::index<IdxT>& index,                      \
              raft::device_matrix_view<const T, IdxT, raft::row_major> queries, \
              raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,  \
              raft::device_matrix_view<float, IdxT, raft::row_major> distances) \
  {                                                                             \
    raft::runtime::neighbors::ivf_pq::search(                                   \
      handle, params, *index.get_raft_index(), queries, neighbors, distances);  \
  }
CUVS_INST_IVF_PQ_SEARCH(int8_t, int64_t);

#undef CUVS_INST_IVF_PQ_SEARCH

}  // namespace cuvs::neighbors::ivf_pq