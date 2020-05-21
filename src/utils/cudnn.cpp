////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/gpu/dnn_primitives.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/number_theory.hpp"

#include <El.hpp>

#include <iostream>
#include <map>
#include <unordered_map>
#include <tuple>

#ifdef LBANN_HAS_CUDNN

namespace lbann {
namespace cudnn {

////////////////////////////////////////////////////////////
// Global cuDNN objects
////////////////////////////////////////////////////////////

namespace {

/** Wrapper for cuDNN handle. */
struct cudnnHandleWrapper {
  cudnnHandle_t handle;
  cudnnHandleWrapper() : handle(nullptr) {
    if (handle == nullptr) { LBANN_CHECK_CUDNN(cudnnCreate(&handle)); }
    if (handle == nullptr) { LBANN_ERROR("failed to create cuDNN handle"); }
    LBANN_CHECK_CUDNN(cudnnSetStream(handle, hydrogen::cuda::GetDefaultStream()));
  }
  cudnnHandleWrapper(const cudnnHandleWrapper&) = delete;
  cudnnHandleWrapper& operator=(const cudnnHandleWrapper&) = delete;
  ~cudnnHandleWrapper() {
    if (handle != nullptr) { cudnnDestroy(handle); }
  }
};

struct cudnnStreamManager
{
  cudnnStreamManager(cudnnHandle_t handle, cudaStream_t stream)
    : handle_(handle)
  {
    LBANN_CHECK_CUDNN(cudnnGetStream(handle_, old_stream_));
    LBANN_CHECK_CUDNN(cudnnSetStream(handle_, stream));
  }

  ~cudnnStreamManager()
  {
    try {
      LBANN_CHECK_CUDNN(cudnnSetStream(handle_, old_stream_));
    }
    catch (std::exception const& e) {
      std::cerr << "Caught error in ~cudnnStreamManager().\n\n  e.what(): "
                << e.what() << "\n\nCalling std::terminate()."
                << std::endl;
      std::terminate();
    }
    catch (...) {
      std::cerr << "Caught unknown error in ~cudnnStreamManager().\n\n"
                << "Calling std::terminate()."
                << std::endl;
      std::terminate();
    }
  }
  cudnnHandle_t handle_;
  cudaStream_t old_stream_;
};// struct cudnnStreamManager

/** Global instance of cuDNN handle. */
std::unique_ptr<cudnnHandleWrapper> handle_instance;

} // namespace

void initialize() {
  handle_instance.reset(new cudnnHandleWrapper());
}

void destroy() {
  handle_instance.reset();
}

cudnnHandle_t& get_handle() {
  if (!handle_instance) { initialize(); }
  LBANN_CHECK_CUDNN(
    cudnnSetStream(handle_instance->handle,
                   hydrogen::cuda::GetDefaultStream()));
  return handle_instance->handle;
}

////////////////////////////////////////////////////////////
// Helper functions for cuDNN types
////////////////////////////////////////////////////////////

template <typename TensorDataType>
void set_tensor_desc(cudnnTensorDescriptor_t& desc,
                     std::vector<int> dims,
                     std::vector<int> strides) {
  if (dims.empty()) {
    LBANN_ERROR("attempted to set cuDNN tensor descriptor with no dimensions");
  }

  // Assume data is contiguous if no strides are provided
  if (strides.empty()) {
    strides.resize(dims.size());
    strides.back() = 1;
    for(int i = strides.size() - 1; i > 0; --i) {
        strides[i-1] = strides[i] * dims[i];
    }
  }

#ifdef LBANN_DEBUG
  std::stringstream err;

  // Check that dimensions and strides are valid
  if (strides.size() != dims.size()) {
    err << "attempted to set cuDNN tensor descriptor "
        << "with invalid strides (";
    for (size_t i = 0; i < strides.size(); ++i) {
      err << (i == 0 ? "" : ",") << strides[i];
    }
    err << ") for dimensions (";
    for (size_t i = 0; i < dims.size(); ++i) {
      err << (i == 0 ? "" : "x") << dims[i];
    }
    err << ")";
    LBANN_ERROR(err.str());
  }
  for (size_t j = 0; j < dims.size(); ++j) {
    if (dims[j] <= 0) {
      err << "attempted to set cuDNN tensor descriptor "
          << "with invalid dimensions (";
      for (size_t i = 0; i < dims.size(); ++i) {
        err << (i == 0 ? "" : "x") << dims[i];
      }
      err << ")";
      LBANN_ERROR(err.str());
    }
    if (j > 0 && strides[j-1] < dims[j] * strides[j]) {
      err << "attempted to set cuDNN tensor descriptor "
          << "with invalid strides (";
      for (size_t i = 0; i < strides.size(); ++i) {
        err << (i == 0 ? "" : ",") << strides[i];
      }
      err << ") for dimensions (";
      for (size_t i = 0; i < dims.size(); ++i) {
        err << (i == 0 ? "" : "x") << dims[i];
      }
      err << ")";
      LBANN_ERROR(err.str());
    }
  }
#endif // LBANN_DEBUG

  // Set cuDNN tensor descriptor
  // Note: cuDNN tensors should have at least 4 dimensions
  /// @todo Think about 1D convolution
  while (dims.size() < 4) {
    dims.push_back(1);
    strides.push_back(1);
  }
  if (desc == nullptr) {
    LBANN_CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
  }
  LBANN_CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc,
                                         get_data_type<TensorDataType>(),
                                         dims.size(),
                                         dims.data(),
                                         strides.data()));

}

void copy_tensor_desc(const cudnnTensorDescriptor_t& src,
                      cudnnTensorDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
        LBANN_CHECK_CUDNN(cudnnCreateTensorDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
        LBANN_CHECK_CUDNN(cudnnDestroyTensorDescriptor(dst));
        dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
        cudnnDataType_t data_type;
        int num_dims;
        LBANN_CHECK_CUDNN(cudnnGetTensorNdDescriptor(src,
                                               0,
                                               &data_type,
                                               &num_dims,
                                               nullptr,
                                               nullptr));
        std::vector<int> dims(num_dims), strides(num_dims);
        LBANN_CHECK_CUDNN(cudnnGetTensorNdDescriptor(src,
                                               num_dims,
                                               &data_type,
                                               &num_dims,
                                               dims.data(),
                                               strides.data()));
        LBANN_CHECK_CUDNN(cudnnSetTensorNdDescriptor(dst,
                                               data_type,
                                               num_dims,
                                               dims.data(),
                                               strides.data()));
    }

}

void copy_activation_desc(const cudnnActivationDescriptor_t& src,
                          cudnnActivationDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
        LBANN_CHECK_CUDNN(cudnnCreateActivationDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
        LBANN_CHECK_CUDNN(cudnnDestroyActivationDescriptor(dst));
        dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
        cudnnActivationMode_t mode;
        cudnnNanPropagation_t nan_propagation;
        double relu_ceiling;
        LBANN_CHECK_CUDNN(cudnnGetActivationDescriptor(src,
                                                 &mode,
                                                 &nan_propagation,
                                                 &relu_ceiling));
        LBANN_CHECK_CUDNN(cudnnSetActivationDescriptor(dst,
                                                 mode,
                                                 nan_propagation,
                                                 relu_ceiling));
    }

}

////////////////////////////////////////////////////////////
// Base cuDNN tensor manager
////////////////////////////////////////////////////////////

template <typename TensorDataType>
layer_tensor_manager<TensorDataType>::layer_tensor_manager(const data_type_layer<TensorDataType>* l)
  : m_layer(nullptr) {
  set_layer(l);
}

template <typename TensorDataType>
layer_tensor_manager<TensorDataType>::layer_tensor_manager(const layer_tensor_manager<TensorDataType>& other)
  : m_layer(other.m_layer),
    m_prev_activations(other.m_prev_activations.size(), nullptr),
    m_activations(other.m_activations.size(), nullptr),
    m_prev_error_signals(other.m_prev_error_signals.size(), nullptr),
    m_error_signals(other.m_error_signals.size(), nullptr) {
  for (size_t i = 0; i < m_prev_activations.size(); ++i) {
    copy_tensor_desc(other.m_prev_activations[i], m_prev_activations[i]);
  }
  for (size_t i = 0; i < m_activations.size(); ++i) {
    copy_tensor_desc(other.m_activations[i], m_activations[i]);
  }
  for (size_t i = 0; i < m_prev_error_signals.size(); ++i) {
    copy_tensor_desc(other.m_prev_error_signals[i], m_prev_error_signals[i]);
  }
  for (size_t i = 0; i < m_error_signals.size(); ++i) {
    copy_tensor_desc(other.m_error_signals[i], m_error_signals[i]);
  }
}

template <typename TensorDataType>
layer_tensor_manager<TensorDataType>& layer_tensor_manager<TensorDataType>::operator=(const layer_tensor_manager<TensorDataType>& other) {

  // Set layer being managed
  m_layer = other.m_layer;

  // Destroy tensor descriptors
  set_num_parents(0);
  set_num_children(0);

  // Create copies of tensor descriptors
  m_prev_activations.resize(other.m_prev_activations.size(), nullptr);
  m_activations.resize(other.m_activations.size(), nullptr);
  m_prev_error_signals.resize(other.m_prev_error_signals.size(), nullptr);
  m_error_signals.resize(other.m_error_signals.size(), nullptr);
  for (size_t i = 0; i < m_prev_activations.size(); ++i) {
    copy_tensor_desc(other.m_prev_activations[i], m_prev_activations[i]);
  }
  for (size_t i = 0; i < m_activations.size(); ++i) {
    copy_tensor_desc(other.m_activations[i], m_activations[i]);
  }
  for (size_t i = 0; i < m_prev_error_signals.size(); ++i) {
    copy_tensor_desc(other.m_prev_error_signals[i], m_prev_error_signals[i]);
  }
  for (size_t i = 0; i < m_error_signals.size(); ++i) {
    copy_tensor_desc(other.m_error_signals[i], m_error_signals[i]);
  }

  return *this;
}

template <typename TensorDataType>
layer_tensor_manager<TensorDataType>::~layer_tensor_manager() {
  for (auto&& desc : m_prev_activations) {
    if (desc != nullptr) { cudnnDestroyTensorDescriptor(desc); }
  }
  for (auto&& desc : m_activations) {
    if (desc != nullptr) { cudnnDestroyTensorDescriptor(desc); }
  }
  for (auto&& desc : m_prev_error_signals) {
    if (desc != nullptr) { cudnnDestroyTensorDescriptor(desc); }
  }
  for (auto&& desc : m_error_signals) {
    if (desc != nullptr) { cudnnDestroyTensorDescriptor(desc); }
  }
}

template <typename TensorDataType>
void layer_tensor_manager<TensorDataType>::set_layer(const data_type_layer<TensorDataType>* new_layer) {
  m_layer = new_layer;
  set_num_parents(this->m_layer == nullptr ? 0 : m_layer->get_num_parents());
  set_num_children(this->m_layer == nullptr ? 0 : m_layer->get_num_children());
}

template <typename TensorDataType>
void layer_tensor_manager<TensorDataType>::set_num_parents(int num_parents) {
#ifdef LBANN_DEBUG
  if (num_parents < 0) { LBANN_ERROR("negative number of parents"); }
#endif // LBANN_DEBUG
  for (size_t i = num_parents; i < m_prev_activations.size(); ++i) {
    if (m_prev_activations[i] != nullptr) {
      LBANN_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_prev_activations[i]));
      m_prev_activations[i] = nullptr;
    }
  }
  for (size_t i = num_parents; i < m_error_signals.size(); ++i) {
    if (m_error_signals[i] != nullptr) {
      LBANN_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_error_signals[i]));
      m_error_signals[i] = nullptr;
    }
  }
  m_prev_activations.resize(num_parents, nullptr);
  m_error_signals.resize(num_parents, nullptr);
}

template <typename TensorDataType>
void layer_tensor_manager<TensorDataType>::set_num_children(int num_children) {
#ifdef LBANN_DEBUG
  if (num_children < 0) { LBANN_ERROR("negative number of children"); }
#endif // LBANN_DEBUG
  for (size_t i = num_children; i < m_activations.size(); ++i) {
    if (m_activations[i] != nullptr) {
      LBANN_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_activations[i]));
      m_activations[i] = nullptr;
    }
  }
  for (size_t i = num_children; i < m_prev_error_signals.size(); ++i) {
    if (m_prev_error_signals[i] != nullptr) {
      LBANN_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_prev_error_signals[i]));
      m_prev_error_signals[i] = nullptr;
    }
  }
  m_activations.resize(num_children, nullptr);
  m_prev_error_signals.resize(num_children, nullptr);
}

////////////////////////////////////////////////////////////
// Data-parallel cuDNN tensor manager
////////////////////////////////////////////////////////////

template <typename TensorDataType>
data_parallel_layer_tensor_manager<TensorDataType>
::data_parallel_layer_tensor_manager(const data_type_layer<TensorDataType>* l)
  : layer_tensor_manager<TensorDataType>(l) {}

namespace {

/** Set a cuDNN tensor descriptor for a data-parallel data layout.
 */
template <typename TensorDataType>
void set_data_parallel_tensor_desc(cudnnTensorDescriptor_t& desc,
                                   std::vector<int> dims,
                                   const El::AbstractMatrix<TensorDataType>& local_data) {
#ifdef LBANN_DEBUG
  if (local_data.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("attempted to setup cuDNN tensor with non-GPU data");
  }
#endif // LBANN_DEBUG
  if (local_data.Height() > 0 && local_data.Width() > 0) {
    std::vector<int> strides(dims.size(), 1);
    for(int i = strides.size() - 1; i > 0; --i) {
      strides[i-1] = strides[i] * dims[i];
    }
    dims.insert(dims.begin(), local_data.Width());
    strides.insert(strides.begin(), local_data.LDim());
    set_tensor_desc<TensorDataType>(desc, dims, strides);
  }
}

} // namespace

template <typename TensorDataType>
cudnnTensorDescriptor_t& data_parallel_layer_tensor_manager<TensorDataType>::get_prev_activations(int parent_index) {
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = this->m_layer->get_local_prev_activations(parent_index);
  const auto& dims = this->m_layer->get_input_dims(parent_index);
  this->set_num_parents(this->m_layer->get_num_parents());
  auto& desc = this->m_prev_activations[parent_index];
  set_data_parallel_tensor_desc<TensorDataType>(desc, dims, local_data);
  return desc;
}

template <typename TensorDataType>
cudnnTensorDescriptor_t& data_parallel_layer_tensor_manager<TensorDataType>::get_activations(int child_index) {
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = this->m_layer->get_local_activations(child_index);
  const auto& dims = this->m_layer->get_output_dims(child_index);
  this->set_num_children(this->m_layer->get_num_children());
  auto& desc = this->m_activations[child_index];
  set_data_parallel_tensor_desc<TensorDataType>(desc, dims, local_data);
  return desc;
}

template <typename TensorDataType>
cudnnTensorDescriptor_t& data_parallel_layer_tensor_manager<TensorDataType>::get_prev_error_signals(int child_index) {
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = this->m_layer->get_local_prev_error_signals(child_index);
  const auto& dims = this->m_layer->get_output_dims(child_index);
  this->set_num_children(this->m_layer->get_num_children());
  auto& desc = this->m_prev_error_signals[child_index];
  set_data_parallel_tensor_desc<TensorDataType>(desc, dims, local_data);
  return desc;
}

template <typename TensorDataType>
cudnnTensorDescriptor_t& data_parallel_layer_tensor_manager<TensorDataType>::get_error_signals(int parent_index) {
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = this->m_layer->get_local_error_signals(parent_index);
  const auto& dims = this->m_layer->get_input_dims(parent_index);
  this->set_num_parents(this->m_layer->get_num_parents());
  auto& desc = this->m_error_signals[parent_index];
  set_data_parallel_tensor_desc<TensorDataType>(desc, dims, local_data);
  return desc;
}

////////////////////////////////////////////////////////////
// Entry-wise cuDNN tensor manager
////////////////////////////////////////////////////////////

template <typename TensorDataType>
entrywise_layer_tensor_manager<TensorDataType>
::entrywise_layer_tensor_manager(const data_type_layer<TensorDataType>* l)
  : layer_tensor_manager<TensorDataType>(l) {}

namespace {

/** Set a cuDNN tensor descriptor for an entrywise tensor operation.
 *  Given local data in a (height x width) matrix, the tensor is
 *  initialized with dimensions (width, a, b, c), where
 *  a*b*c=height. This is because cuDNN is optimized for 4D tensors
 *  and gets poor performance with 1D tensors and 2D tensors.
 */
template <typename TensorDataType>
void set_entrywise_tensor_desc(cudnnTensorDescriptor_t& desc,
                               const El::AbstractMatrix<TensorDataType>& local_data) {
#ifdef LBANN_DEBUG
  if (local_data.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("attempted to setup cuDNN tensor with non-GPU data");
  }
#endif // LBANN_DEBUG
  const int height = local_data.Height();
  const int width = local_data.Width();
  const int ldim = local_data.LDim();
  if (height > 0 && width > 0) {

    // Factorize height into three factors
    // Note: factorization is memoized
    static std::unordered_map<int,std::vector<int>> cache;
    auto& factors = cache[height];
    if (factors.empty()) {
      factors = number_theory::balanced_factors(height, 3);
    }

    // Set cuDNN tensor descriptor with 4D tensor
    set_tensor_desc<TensorDataType>(desc,
                    {width, factors[2], factors[1], factors[0]},
                    {ldim, factors[1]*factors[0], factors[0], 1});

  }
}

} // namespace

template <typename TensorDataType>
cudnnTensorDescriptor_t& entrywise_layer_tensor_manager<TensorDataType>::get_prev_activations(int parent_index) {
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = this->m_layer->get_local_prev_activations(parent_index);
  this->set_num_parents(this->m_layer->get_num_parents());
  auto& desc = this->m_prev_activations[parent_index];
  set_entrywise_tensor_desc<TensorDataType>(desc, local_data);
  return desc;
}

template <typename TensorDataType>
cudnnTensorDescriptor_t& entrywise_layer_tensor_manager<TensorDataType>::get_activations(int child_index) {
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = this->m_layer->get_local_activations(child_index);
  this->set_num_children(this->m_layer->get_num_children());
  auto& desc = this->m_activations[child_index];
  set_entrywise_tensor_desc<TensorDataType>(desc, local_data);
  return desc;
}

template <typename TensorDataType>
cudnnTensorDescriptor_t& entrywise_layer_tensor_manager<TensorDataType>::get_prev_error_signals(int child_index) {
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = this->m_layer->get_local_prev_error_signals(child_index);
  this->set_num_children(this->m_layer->get_num_children());
  auto& desc = this->m_prev_error_signals[child_index];
  set_entrywise_tensor_desc<TensorDataType>(desc, local_data);
  return desc;
}

template <typename TensorDataType>
cudnnTensorDescriptor_t& entrywise_layer_tensor_manager<TensorDataType>::get_error_signals(int parent_index) {
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = this->m_layer->get_local_error_signals(parent_index);
  this->set_num_parents(this->m_layer->get_num_parents());
  auto& desc = this->m_error_signals[parent_index];
  set_entrywise_tensor_desc<TensorDataType>(desc, local_data);
  return desc;
}

////////////////////////////////////////////////////////////
// cuDNN algorithm selection
////////////////////////////////////////////////////////////

namespace {

// Non-deterministic algorithms.
std::vector<cudnnConvolutionFwdAlgo_t> nondet_fwd_algos = {};
std::vector<cudnnConvolutionBwdDataAlgo_t> nondet_bwd_data_algos = {
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
};
std::vector<cudnnConvolutionBwdFilterAlgo_t> nondet_bwd_filter_algos = {
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3
};

template <typename AlgoType, typename PerfType>
AlgoType find_best_heuristic_algorithm(
  const std::vector<PerfType>& perf_results,
  const std::vector<AlgoType>& nondeterministic_algos,
  bool deterministic,
  size_t max_ws_size) {
  std::vector<AlgoType> algos;
  for (const auto& p : perf_results) {
    if (p.status != CUDNN_STATUS_SUCCESS) {
      continue;
    }
    if (deterministic &&
        std::find(nondeterministic_algos.begin(), nondeterministic_algos.end(),
                  p.algo) != nondeterministic_algos.end()) {
      continue;
    }
    if (p.memory > max_ws_size) {
      continue;
    }
    algos.push_back(p.algo);
  }
  if (algos.empty()) {
    LBANN_ERROR("No valid convolution algorithms.");
  }
  return algos[0];
}

template <typename AlgoType, typename PerfType>
AlgoType find_best_algorithm(
  const std::vector<PerfType>& perf_results,
  const std::vector<AlgoType>& nondeterministic_algos,
  bool deterministic,
  size_t max_ws_size) {
  std::map<AlgoType, float> time_map;
  for (const auto& p : perf_results) {
    if (p.status != CUDNN_STATUS_SUCCESS) {
      // If an algorithm fails, we still add it in case the failure is
      // nondeterministic.
      time_map[p.algo] = std::numeric_limits<float>::max();
      continue;
    }
    if (deterministic &&
        std::find(nondeterministic_algos.begin(), nondeterministic_algos.end(),
                  p.algo) != nondeterministic_algos.end()) {
      continue;
    }
    if (p.memory > max_ws_size) {
      continue;
    }
    if (time_map.count(p.algo) == 0) {
      time_map[p.algo] = p.time;
    } else {
      time_map[p.algo] += p.time;
    }
  }
  if (time_map.empty()) {
    LBANN_ERROR("No valid convolution algorithms.");
  }
  AlgoType best_algo = time_map.begin()->first;
  float min_time = std::numeric_limits<float>::max();
  for (const auto& x : time_map) {
    AlgoType algo = x.first;
    float time = x.second;
    if (time < min_time) {
      min_time = time;
      best_algo = algo;
    }
  }
  if (min_time == std::numeric_limits<float>::max()) {
    LBANN_ERROR("No valid convolution algorithms.");
  }
  return best_algo;
}

cudnnConvolutionFwdAlgo_t get_fwd_algo_heuristic(
  bool deterministic,
  const cudnnTensorDescriptor_t& input_desc,
  const cudnnFilterDescriptor_t& kernel_desc,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnTensorDescriptor_t& output_desc,
  size_t ws_size) {
  int num_algos;
  LBANN_CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  LBANN_CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
                get_handle(), input_desc, kernel_desc, conv_desc, output_desc,
                num_algos, &num_tested_algos, perf_results.data()));
  return find_best_heuristic_algorithm(perf_results, nondet_fwd_algos,
                                       deterministic, ws_size);
}

cudnnConvolutionBwdDataAlgo_t get_bwd_data_algo_heuristic(
  bool deterministic,
  const cudnnFilterDescriptor_t& kernel_desc,
  const cudnnTensorDescriptor_t& prev_error_signal_desc,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnTensorDescriptor_t& error_signal_desc,
  size_t ws_size) {
  int num_algos;
  LBANN_CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  LBANN_CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                get_handle(), kernel_desc, prev_error_signal_desc, conv_desc,
                error_signal_desc, num_algos, &num_tested_algos,
                perf_results.data()));
  return find_best_heuristic_algorithm(perf_results, nondet_bwd_data_algos,
                                       deterministic, ws_size);
}

cudnnConvolutionBwdFilterAlgo_t get_bwd_filter_algo_heuristic(
  bool deterministic,
  const cudnnTensorDescriptor_t& input_desc,
  const cudnnTensorDescriptor_t& prev_error_signal_desc,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnFilterDescriptor_t& kernel_gradient_desc,
  size_t ws_size) {
  int num_algos;
  LBANN_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  LBANN_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                get_handle(), input_desc, prev_error_signal_desc, conv_desc,
                kernel_gradient_desc, num_algos, &num_tested_algos,
                perf_results.data()));
  return find_best_heuristic_algorithm(perf_results, nondet_bwd_filter_algos,
                                       deterministic, ws_size);
}

cudnnConvolutionFwdAlgo_t get_fwd_algo_autotune(
  bool deterministic,
  const cudnnTensorDescriptor_t& input_desc,
  const void* input,
  const cudnnFilterDescriptor_t& kernel_desc,
  const void* kernel,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnTensorDescriptor_t& output_desc,
  void* output,
  size_t ws_size,
  void* ws) {
  constexpr int num_trials = 3;
  constexpr int num_skip = 1;
  int num_algos;
  LBANN_CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results_all;
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(num_algos);
  for (int trial = 0; trial < num_trials + num_skip; ++trial) {
    int num_tested_algos;
    LBANN_CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithmEx(
                  get_handle(), input_desc, input, kernel_desc, kernel,
                  conv_desc, output_desc, output, num_algos, &num_tested_algos,
                  perf_results.data(), ws, ws_size));
    if (trial > num_skip) {
      for (const auto& p : perf_results) {
        perf_results_all.push_back(p);
      }
    }
  }
  return find_best_algorithm(perf_results_all, nondet_fwd_algos,
                             deterministic, ws_size);
}

cudnnConvolutionBwdDataAlgo_t get_bwd_data_algo_autotune(
  bool deterministic,
  const cudnnFilterDescriptor_t& kernel_desc,
  const void* kernel,
  const cudnnTensorDescriptor_t& prev_error_signal_desc,
  const void* prev_error_signal,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnTensorDescriptor_t& error_signal_desc,
  void* error_signal,
  size_t ws_size,
  void* ws) {
  constexpr int num_trials = 3;
  constexpr int num_skip = 1;
  int num_algos;
  LBANN_CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results_all;
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(num_algos);
  for (int trial = 0; trial < num_trials + num_skip; ++trial) {
    int num_tested_algos;
    LBANN_CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithmEx(
                  get_handle(), kernel_desc, kernel,
                  prev_error_signal_desc, prev_error_signal,
                  conv_desc, error_signal_desc, error_signal, num_algos,
                  &num_tested_algos, perf_results.data(), ws, ws_size));
    if (trial > num_skip) {
      for (const auto& p : perf_results) {
        perf_results_all.push_back(p);
      }
    }
  }
  return find_best_algorithm(perf_results_all, nondet_bwd_data_algos,
                             deterministic, ws_size);
}

cudnnConvolutionBwdFilterAlgo_t get_bwd_filter_algo_autotune(
  bool deterministic,
  const cudnnTensorDescriptor_t& input_desc,
  const void* input,
  const cudnnTensorDescriptor_t& prev_error_signal_desc,
  const void* prev_error_signal,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnFilterDescriptor_t& kernel_gradient_desc,
  void* kernel_gradient,
  size_t ws_size,
  void* ws) {
  constexpr int num_trials = 3;
  constexpr int num_skip = 1;
  int num_algos;
  LBANN_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results_all;
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(num_algos);
  for (int trial = 0; trial < num_trials + num_skip; ++trial) {
    int num_tested_algos;
    LBANN_CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithmEx(
                  get_handle(), input_desc, input,
                  prev_error_signal_desc, prev_error_signal,
                  conv_desc, kernel_gradient_desc, kernel_gradient, num_algos,
                  &num_tested_algos, perf_results.data(), ws, ws_size));
    if (trial > num_skip) {
      for (const auto& p : perf_results) {
        perf_results_all.push_back(p);
      }
    }
  }
  return find_best_algorithm(perf_results_all, nondet_bwd_filter_algos,
                             deterministic, ws_size);
}

}  // namespace

cudnnConvolutionFwdAlgo_t get_fwd_algorithm(
  bool autotune,
  bool deterministic,
  const cudnnTensorDescriptor_t& input_desc,
  const void* input,
  const cudnnFilterDescriptor_t& kernel_desc,
  const void* kernel,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnTensorDescriptor_t& output_desc,
  void* output,
  size_t ws_size,
  void* ws) {
  if (autotune) {
    return get_fwd_algo_autotune(deterministic,
                                 input_desc, input,
                                 kernel_desc, kernel,
                                 conv_desc,
                                 output_desc, output,
                                 ws_size, ws);
  } else {
    return get_fwd_algo_heuristic(deterministic, input_desc, kernel_desc,
                                  conv_desc, output_desc, ws_size);
  }
}

cudnnConvolutionBwdDataAlgo_t get_bwd_data_algorithm(
  bool autotune,
  bool deterministic,
  const cudnnFilterDescriptor_t& kernel_desc,
  const void* kernel,
  const cudnnTensorDescriptor_t& prev_error_signal_desc,
  const void* prev_error_signal,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnTensorDescriptor_t& error_signal_desc,
  void* error_signal,
  size_t ws_size,
  void* ws) {
  if (autotune) {
    return get_bwd_data_algo_autotune(deterministic,
                                      kernel_desc, kernel,
                                      prev_error_signal_desc, prev_error_signal,
                                      conv_desc,
                                      error_signal_desc, error_signal,
                                      ws_size, ws);
  } else {
    return get_bwd_data_algo_heuristic(deterministic, kernel_desc,
                                       prev_error_signal_desc, conv_desc,
                                       error_signal_desc, ws_size);
  }
}

cudnnConvolutionBwdFilterAlgo_t get_bwd_filter_algorithm(
  bool autotune,
  bool deterministic,
  const cudnnTensorDescriptor_t& input_desc,
  const void* input,
  const cudnnTensorDescriptor_t& prev_error_signal_desc,
  const void* prev_error_signal,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnFilterDescriptor_t& kernel_gradient_desc,
  void* kernel_gradient,
  size_t ws_size,
  void* ws) {
  if (autotune) {
    return get_bwd_filter_algo_autotune(deterministic,
                                        input_desc, input,
                                        prev_error_signal_desc, prev_error_signal,
                                        conv_desc,
                                        kernel_gradient_desc, kernel_gradient,
                                        ws_size, ws);
  } else {
    return get_bwd_filter_algo_heuristic(deterministic, input_desc,
                                         prev_error_signal_desc, conv_desc,
                                         kernel_gradient_desc, ws_size);
  }
}

namespace {
cudnnMathType_t default_tensor_ops_mode = CUDNN_DEFAULT_MATH;
}

void default_to_tensor_ops() noexcept
{
  default_tensor_ops_mode = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
}

cudnnMathType_t get_default_convolution_math_type() noexcept
{
  return default_tensor_ops_mode;
}

#define PROTO(T)                                       \
  template cudnnDataType_t get_data_type<T>();                   \
  template void set_tensor_desc<T>(cudnnTensorDescriptor_t&, std::vector<int>, std::vector<int>); \
  template class layer_tensor_manager<T>;               \
  template class data_parallel_layer_tensor_manager<T>; \
  template class entrywise_layer_tensor_manager<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace cudnn

namespace dnn_primitive {

PoolingMode_t get_pooling_mode(pooling_mode pm)
{
  switch (pm)
  {
  case pooling_mode::MAX:
#ifdef LBANN_DETERMINISTIC
    return CUDNN_POOLING_MAX;
#else
    return CUDNN_POOLING_MAX_DETERMINISTIC;
#endif // LBANN_DETERMINISTIC
  case pooling_mode::AVERAGE:
    return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  case pooling_mode::AVERAGE_NO_PAD:
    return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  default:
    LBANN_ERROR("Unknown pooling mode type.");
  }
  return CUDNN_POOLING_MAX; // Silence sub-par compiler warnings.
}

LRNDescriptor_t create_lrn_descriptor()
{
  LRNDescriptor_t lrnd;
  LBANN_CHECK_CUDNN(cudnnCreateLRNDescriptor(&lrnd));
  return lrnd;
}

void destroy_lrn_descriptor(LRNDescriptor_t& lrnd)
{
  if (lrnd) {
    LBANN_CHECK_CUDNN(cudnnDestroyLRNDescriptor(&lrnd));
    lrnd = nullptr;
  }
}

void set_lrn_descriptor(LRNDescriptor_t& lrnd,
                        unsigned const lrnN,
                        double const lrnAlpha,
                        double const lrnBeta,
                        double const lrnK)
{
  LBANN_CHECK_CUDNN(
    cudnnSetLRNDescriptor(lrnd, lrnN, lrnAlpha, lrnBeta, lrnK));
}

void copy_lrn_descriptor(LRNDescriptor_t const& src, LRNDescriptor_t& dst)
{
  unsigned n;
  double alpha, beta, k;
  CHECK_CUDNN(cudnnGetLRNDescriptor(src, &n, &alpha, &beta, &k));
  set_lrn_descriptor(dst, n, alpha, beta, k);
}

template <typename T>
void lrn_cross_channel_forward(
  LRNDescriptor_t normDesc,
  T const alpha,
  TensorDescriptor_t const xDesc,
  T const* x,
  T const beta,
  TensorDescriptor_t const yDesc,
  T* y,
  El::SyncInfo<El::Device::GPU> const& si)
{
  auto handle = cudnn::get_handle();
  cudnnStreamManager mgr(handle, si.Stream());
  auto alpha_scaling_type = El::To<ScalingParamType<T>>(alpha);
  auto beta_scaling_type = El::To<ScalingParamType<T>>(beta);
  LBANN_CHECK_CUDNN(
    cudnnLRNCrossChannelForward(
      handle, nromDesc,
      CUDNN_LRN_CROSS_CHANNEL_DIM1,
      &alpha_scaling_type, xDesc, x,
      &beta_scaling_type, yDesc, y));
}

template <typename T>
void lrn_cross_channel_backward(
  LRNDescriptor_t normDesc,
  T const alpha,
  TensorDescriptor_t const yDesc,
  T const* y,
  TensorDescriptor_t const dyDesc,
  T const* dy,
  TensorDescriptor_t const xDesc,
  T const* x,
  T const beta,
  TensorDescriptor_t const dxDesc,
  T* dx,
  El::SyncInfo<El::Device::GPU> const& si)
{
  auto handle = cudnn::get_handle();
  cudnnStreamManager mgr(handle, si.Stream());

  auto alpha_scaling_type = El::To<ScalingParamType<T>>(alpha);
  auto beta_scaling_type = El::To<ScalingParamType<T>>(beta);

  LBANN_CHECK_CUDNN(
    cudnnLRNCrossChannelBackward(handle,
                                 normDesc,
                                 CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                 &alpha_scaling_type,
                                 yDesc, y, dyDesc, dy, xDesc,
                                 &beta_scaling_type, dxDesc, dx));
}

PoolingDescriptor_t create_pooling_descriptor()
{
  PoolingDescriptor_t desc;
  LBANN_CHECK_CUDNN(cudnnCreatePoolingDescriptor(&desc));
  return desc;
}

void destroy_pooling_descriptor(PoolingDescriptor_t& pd)
{
  if (pd) {
    LBANN_CHECK_CUDNN(cudnnDestroyPoolingDescriptor(&pd));
    pd = nullptr;
  }
}

void set_pooling_descriptor(PoolingDescriptor_t pd,
                            PoolingMode mode,
                            int dims,
                            int const windowDimsA[],
                            int const padA[],
                            int const stridesA[])
{
  LBANN_CHECK_CUDNN(
    cudnnSetPoolingNdDescriptor(
      pd, get_native_pooling_mode(mode), CUDNN_PROPAGATE_NAN,
      dims, windowDimsA, padA, stridesA));
}

/** @brief Copy pooling cuDNN descriptor. */
void copy_pooling_descriptor(const cudnn::PoolingDescriptor_t& src,
                             cudnn::PoolingDescriptor_t& dst) {

  // Create or destroy descriptor if needed
  if(src != nullptr && dst == nullptr) {
    LBANN_CHECK_CUDNN(cudnnCreatePoolingDescriptor(&dst));
  }
  else if(src == nullptr && dst != nullptr) {
    LBANN_CHECK_CUDNN(cudnnDestroyPoolingDescriptor(dst));
    dst = nullptr;
  }

  // Copy descriptor data if needed
  if(src != nullptr) {
    cudnnPoolingMode_t mode;
    cudnnNanPropagation_t nan_propagation;
    int num_dims;
    LBANN_CHECK_CUDNN(
      cudnnGetPoolingNdDescriptor(src,
                                  0,
                                  &mode,
                                  &nan_propagation,
                                  &num_dims,
                                  nullptr,
                                  nullptr,
                                  nullptr));
    std::vector<int> dims(num_dims), pads(num_dims), strides(num_dims);
    LBANN_CHECK_CUDNN(
      cudnnGetPoolingNdDescriptor(src,
                                  num_dims,
                                  &mode,
                                  &nan_propagation,
                                  &num_dims,
                                  dims.data(),
                                  pads.data(),
                                  strides.data()));
    LBANN_CHECK_CUDNN(
      cudnnSetPoolingNdDescriptor(dst,
                                  mode,
                                  nan_propagation,
                                  num_dims,
                                  dims.data(),
                                  pads.data(),
                                  strides.data()));
  }
}

template <typename T>
void pooling_forward(
  PoolingDescriptor_t const poolingDesc,
  T const alpha,
  TensorDescriptor_t const xDesc,
  T const* x,
  T const beta,
  TensorDescriptor_t const yDesc,
  T* y,
  El::SyncInfo<El::Device::GPU> const& si)
{
  auto handle = cudnn::get_handle();
  cudnnStreamManager mgr(handle, si.Stream());
  auto alpha_scaling_type = El::To<ScalingParamType<T>>(alpha);
  auto beta_scaling_type = El::To<ScalingParamType<T>>(beta);
  LBANN_CHECK_CUDNN(
    cudnnPoolingForward(handle,
                        poolingDesc,
                        &alpha_scaling_type, xDesc, x,
                        &beta_scaling_type, yDesc, y));
}

template <typename T>
void pooling_backward(
    PoolingDescriptor_t const poolingDesc,
    T const alpha,
    TensorDescriptor_t const yDesc,
    T const* y,
    TensorDescriptor_t const dyDesc,
    T const* dy,
    TensorDescriptor_t const xDesc,
    T const* xData,
    T const beta,
    TensorDescriptor_t const dxDesc,
    T* dx,
    El::SyncInfo<El::Device::GPU> const& si)
{
  auto handle = cudnn::get_handle();
  cudnnStreamManager mgr(handle, si.Stream());
  auto alpha_scaling_type = El::To<ScalingParamType<T>>(alpha);
  auto beta_scaling_type = El::To<ScalingParamType<T>>(beta);
  LBANN_CHECK_CUDNN(
    cudnnPoolingBackward(handle,
                         poolingDesc,
                         &alpha, yDesc, y, dyDesc, dy, xDesc, xData,
                         &beta, dxDesc, dx));
}

}// namespace dnn_primitive
} // namespace lbann

#endif // LBANN_HAS_CUDNN
