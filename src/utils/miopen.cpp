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

#include "lbann/utils/miopen.hpp"

#include "lbann/layers/layer.hpp"
#include "lbann/utils/gpu/dnn_primitives.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/number_theory.hpp"

#include <El.hpp>

#include <iostream>
#include <map>
#include <numeric>
#include <tuple>
#include <unordered_map>

// Error handling macro
#define LBANN_CHECK_MIOPEN(cmd)                                         \
    do                                                                  \
    {                                                                   \
        H_SYNC_HIP();                                                   \
        auto h_check_miopen_error_code__ = cmd;                         \
        H_ASSERT(h_check_miopen_error_code__ == miopenStatusSuccess,    \
                 ::lbann::MIOpenError,                                  \
                 (hipDeviceReset(),                                     \
                  ::lbann::dnn_primitive::BuildMIOpenErrorMessage(      \
                    #cmd, h_check_miopen_error_code__)));               \
        H_SYNC_HIP();                                                   \
    } while (false)

namespace lbann {

// Or is this a "HIPError"?
H_ADD_BASIC_EXCEPTION_CLASS(MIOpenError, ::hydrogen::GPUError);

namespace dnn_primitive {

std::string BuildMIOpenErrorMessage(
  std::string const& cmd, miopenStatus_t status);

namespace {

/** @brief Get the default GPU stream. */
hipStream_t DefaultStream() noexcept
{
  return hydrogen::rocm::GetDefaultStream();
}

/** @brief RAII wrapper for an MIOpen handle. */
struct MIOpenHandleWrapper
{
  MIOpenHandleWrapper() : handle_(nullptr)
  {
    LBANN_CHECK_MIOPEN(
      miopenCreateWithStream(&handle_, DefaultStream()));
  }

  MIOpenHandleWrapper(MIOpenHandleWrapper const&) = delete;
  MIOpenHandleWrapper& operator=(MIOpenHandleWrapper const&) = delete;

  MIOpenHandleWrapper(MIOpenHandleWrapper&& other)
    : handle_(other.handle_)
  {
    other.handle_ = nullptr;
  }

  MIOpenHandleWrapper& operator=(MIOpenHandleWrapper&& other) noexcept
  {
    MIOpenHandleWrapper tmp(std::move(other));
    this->swap(tmp);
    return *this;
  }

  ~MIOpenHandleWrapper()
  {
    try
    {
      if (handle_ != nullptr)
        LBANN_CHECK_MIOPEN(miopenDestroy(handle_));
    }
    catch (std::exception const& e)
    {
      std::cerr << "Detected error in ~MIOpenHandleWrapper() ("
                << __FILE__ << ":" << __LINE__ << ")\n\n  e.what(): "
                << e.what() << "\n\nAbout to call std::terminate."
                << std::endl;
      std::terminate();
    }
    catch (...)
    {
      std::cerr << "Detected unknown error in ~MIOpenHandleWrapper() ("
                << __FILE__ << ":" << __LINE__ << ")\n\n"
                << "About to call std::terminate."
                << std::endl;
      std::terminate();
    }
  }

  void swap(MIOpenHandleWrapper& other) noexcept
  {
    auto handle = other.handle_;
    other.handle_ = handle_;
    handle_ = handle;
  }

  Handle_t handle_;

};// struct MIOpenHandleWrapper

/** Global instance of MIOpen handle. */
std::unique_ptr<MIOpenHandleWrapper> miopen_instance_;

std::string get_miopen_status_name(miopenStatus_t status)
{
#define MIOPEN_STATUS_NAME_CASE(status) \
  case status: return #status

  switch (status)
  {
    MIOPEN_STATUS_NAME_CASE(miopenStatusSuccess);
    MIOPEN_STATUS_NAME_CASE(miopenStatusNotInitialized);
    MIOPEN_STATUS_NAME_CASE(miopenStatusInvalidValue);
    MIOPEN_STATUS_NAME_CASE(miopenStatusBadParm);
    MIOPEN_STATUS_NAME_CASE(miopenStatusAllocFailed);
    MIOPEN_STATUS_NAME_CASE(miopenStatusInternalError);
    MIOPEN_STATUS_NAME_CASE(miopenStatusNotImplemented);
    MIOPEN_STATUS_NAME_CASE(miopenStatusUnknownError);
    MIOPEN_STATUS_NAME_CASE(miopenStatusUnsupportedOp);
  default:
    return "Unknown miopenStatus_t value.";
  }
#undef MIOPEN_STATUS_NAME_CASE
}

std::string get_miopen_status_string(miopenStatus_t status)
{

  switch (status)
  {
  case miopenStatusSuccess: return "No errors";
  case miopenStatusNotInitialized: return "Data not initialized.";
  case miopenStatusInvalidValue: return "Incorrect variable value.";
  case miopenStatusBadParm: return "Incorrect parameter detected.";
  case miopenStatusAllocFailed: return "Memory allocation error.";
  case miopenStatusInternalError: return "MIOpen failure.";
  case miopenStatusNotImplemented: return "Use of unimplemented feature.";
  case miopenStatusUnknownError: return "Unknown error occurred.";
  case miopenStatusUnsupportedOp: return "Unsupported operator for fusion.";
  default:
    return "Unknown miopenStatus_t value.";
  }
}

} // namespace

std::string BuildMIOpenErrorMessage(
  std::string const& cmd, miopenStatus_t status)
{
  std::ostringstream oss;
  oss << "ROCm error detected in command: \"" << cmd << "\"\n\n"
      << "    Error Code: " << status << "\n"
      << "    Error Name: " << get_miopen_status_name(status) << "\n"
      << "    Error Mesg: " << get_miopen_status_string(status);
  return oss.str();
}

void initialize()
{
  miopen_instance_ = make_unique<MIOpenHandleWrapper>();
}

void destroy()
{
  miopen_instance_.reset();
}

Handle_t& get_handle()
{
  if (!miopen_instance_)
    initialize();
  return miopen_instance_->handle_;
}

namespace {

std::vector<int> build_strides_from_dims(std::vector<int> dims)
{
  std::vector<int> strides(dims.size());
  if (dims.size())
  {
    strides.back() = 1;
    auto out_it = strides.rbegin()+1;
    auto dims_start = dims.rbegin();
    auto dims_end = dims.rend() - 1;
    std::partial_sum(dims_start, dims_end, out_it, std::multiplies<int>());
  }
  return strides;
}

}// namepace <anon>

template <typename TensorDataType>
void set_tensor_desc(TensorDescriptor_t& desc,
                     std::vector<int> dims,
                     std::vector<int> strides)
{
  if (dims.empty())
    LBANN_ERROR("Attempted to set MIOpen descriptor with no dimensions");

  // Assume data is contiguous if no strides are provided
  if (strides.empty())
    strides = build_strides_from_dims(dims);

#ifdef LBANN_DEBUG
  std::ostringstream err;
  // Check that dimensions and strides are valid
  if (strides.size() != dims.size()) {
    err << "attempted to set MIOpen tensor descriptor "
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
      err << "attempted to set MIOpen tensor descriptor "
          << "with invalid dimensions (";
      for (size_t i = 0; i < dims.size(); ++i) {
        err << (i == 0 ? "" : "x") << dims[i];
      }
      err << ")";
      LBANN_ERROR(err.str());
    }
    if (j > 0 && strides[j-1] < dims[j] * strides[j]) {
      err << "attempted to set MIOpen tensor descriptor "
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

  // Set MIOpen tensor descriptor
  //
  // FIXME (trb 04/23/2020): There's a note in the cuDNN impl about
  // tensors needing to be at least 4th order. I'm carrying that
  // assumption to MIOpen. We can investigate later.
  if (dims.size() < 4)
  {
    dims.resize(4, 1);
    strides.resize(4, 1);
  }

  if (!desc)
    LBANN_CHECK_MIOPEN(miopenCreateTensorDescriptor(&desc));

  LBANN_CHECK_MIOPEN(
    miopenSetTensorDescriptor(desc,
                              get_data_type<TensorDataType>(),
                              dims.size(),
                              dims.data(),
                              strides.data()));
}

void copy_tensor_desc(TensorDescriptor_t const& src, TensorDescriptor_t& dst)
{
  if (src)
  {
    if (!dst)
      LBANN_CHECK_MIOPEN(miopenCreateTensorDescriptor(&dst));

    DataType_t data_type;
    int num_dims;
    std::vector<int> dims, strides;

    // There's a source code example in
    // $MIOPEN_SRC_ROOT/src/batch_norm_api.cpp that suggests that
    // this function returns the number of dimensions of the tensor
    // -- not the number of "elements", as the documentation
    // suggests.
    LBANN_CHECK_MIOPEN(
      miopenGetTensorDescriptorSize(src, &num_dims));

    dims.resize(num_dims);
    strides.resize(num_dims);
    LBANN_CHECK_MIOPEN(
      miopenGetTensorDescriptor(src,
                                &data_type,
                                dims.data(),
                                strides.data()));

    LBANN_CHECK_MIOPEN(
      miopenSetTensorDescriptor(dst,
                                data_type,
                                num_dims,
                                dims.data(),
                                strides.data()));
  }
  else if (dst)
  {
    LBANN_CHECK_MIOPEN(miopenDestroyTensorDescriptor(dst));
    dst = nullptr;
  }
}

void copy_activation_desc(ActivationDescriptor_t const& src,
                          ActivationDescriptor_t& dst)
{
  if (src)
  {
    if (!dst)
      LBANN_CHECK_MIOPEN(miopenCreateActivationDescriptor(&dst));

    ActivationMode_t mode;
    double alpha, beta, gamma;

    LBANN_CHECK_MIOPEN(
      miopenGetActivationDescriptor(src, &mode, &alpha, &beta, &gamma));
    LBANN_CHECK_MIOPEN(
      miopenSetActivationDescriptor(dst, mode, alpha, beta, gamma));
  }
  else if (dst)
  {
    LBANN_CHECK_MIOPEN(miopenDestroyActivationDescriptor(dst));
    dst = nullptr;
  }
}

template <typename TensorDataType>
LayerTensorManager<TensorDataType>::LayerTensorManager(
  const data_type_layer<TensorDataType>* l)
  : m_layer(nullptr)
{
  set_layer(l);
}

template <typename TensorDataType>
LayerTensorManager<TensorDataType>::LayerTensorManager(
  const LayerTensorManager<TensorDataType>& other)
  : m_layer(other.m_layer),
    m_prev_activations(other.m_prev_activations.size()),
    m_activations(other.m_activations.size()),
    m_prev_error_signals(other.m_prev_error_signals.size()),
    m_error_signals(other.m_error_signals.size())
{
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
LayerTensorManager<TensorDataType>
::SafeTensorDescriptionManager
::~SafeTensorDescriptionManager()
{
  if (tensor_desc_)
  {
    try
    {
      LBANN_CHECK_MIOPEN(miopenDestroyTensorDescriptor(tensor_desc_));
      tensor_desc_ = nullptr;
    }
    catch (std::exception const& e)
    {
      std::cerr << "Detected error in ~SafeTensorDescriptionManager() ("
                << __FILE__ << ":" << __LINE__ << ")\n\n  e.what(): "
                << e.what() << "\n\nAbout to call std::terminate."
                << std::endl;
      std::terminate();
    }
    catch (...)
    {
      std::cerr << "Detected unknown error in "
                << "~SafeTensorDescriptionManager() ("
                << __FILE__ << ":" << __LINE__ << ")\n\n"
                << "About to call std::terminate."
                << std::endl;
      std::terminate();
    }
  }
}

template <typename TensorDataType>
LayerTensorManager<TensorDataType>&
LayerTensorManager<TensorDataType>::operator=(
  const LayerTensorManager<TensorDataType>& other)
{
  // Set layer being managed
  m_layer = other.m_layer;

  // Destroy tensor descriptors
  set_num_parents(0);
  set_num_children(0);

  // Create copies of tensor descriptors
  m_prev_activations.resize(other.m_prev_activations.size());
  m_activations.resize(other.m_activations.size());
  m_prev_error_signals.resize(other.m_prev_error_signals.size());
  m_error_signals.resize(other.m_error_signals.size());
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
LayerTensorManager<TensorDataType>::~LayerTensorManager()
{
  try
  {
    clear_data_();
  }
  catch (std::exception const& e)
  {
    std::cerr << "Detected error in ~LayerTensorManager() ("
              << __FILE__ << ":" << __LINE__ << ")\n\n  e.what(): "
              << e.what() << "\n\nAbout to call std::terminate."
              << std::endl;
    std::terminate();
  }
  catch (...)
  {
    std::cerr << "Detected unknown error in ~LayerTensorManager() ("
              << __FILE__ << ":" << __LINE__ << ")\n\n"
              << "About to call std::terminate."
              << std::endl;
    std::terminate();
  }
}

template <typename TensorDataType>
void LayerTensorManager<TensorDataType>::set_layer(
  const data_type_layer<TensorDataType>* new_layer)
{
  if (new_layer)
  {
    m_layer = new_layer;
    set_num_parents(m_layer->get_num_parents());
    set_num_children(m_layer->get_num_children());
  }
  else
  {
    clear_data_();
  }
}

namespace
{
template <typename VectorT>
void force_deallocate(VectorT& v)
{
  VectorT().swap(v);
}
}// namespace anon

template <typename TensorDataType>
void LayerTensorManager<TensorDataType>::clear_data_()
{
  force_deallocate(m_prev_activations);
  force_deallocate(m_activations);
  force_deallocate(m_prev_error_signals);
  force_deallocate(m_error_signals);
}

template <typename TensorDataType>
void LayerTensorManager<TensorDataType>::set_num_parents(int num_parents)
{
#ifdef LBANN_DEBUG
  if (num_parents < 0) { LBANN_ERROR("negative number of parents"); }
#endif // LBANN_DEBUG
  m_prev_activations.resize(num_parents);
  m_error_signals.resize(num_parents);
}

template <typename TensorDataType>
void LayerTensorManager<TensorDataType>::set_num_children(int num_children)
{
#ifdef LBANN_DEBUG
  if (num_children < 0) { LBANN_ERROR("negative number of children"); }
#endif // LBANN_DEBUG
  m_activations.resize(num_children);
  m_prev_error_signals.resize(num_children);
}

template <typename TensorDataType>
DataParallelLayerTensorManager<TensorDataType>
::DataParallelLayerTensorManager(const data_type_layer<TensorDataType>* l)
  : LayerTensorManager<TensorDataType>(l)
{}

namespace {

/** Set an MIOpen tensor descriptor for a data-parallel data layout.
 */
template <typename TensorDataType>
void set_data_parallel_tensor_desc(
  TensorDescriptor_t& desc,
  std::vector<int> dims,
  El::AbstractMatrix<TensorDataType> const& local_data)
{
#ifdef LBANN_DEBUG
  if (local_data.GetDevice() != El::Device::GPU)
    LBANN_ERROR("Attempted to setup MIOpen tensor with non-GPU data");
#endif // LBANN_DEBUG
  if (local_data.Height() > 0 && local_data.Width() > 0)
  {
    // Account for the size of the minibatch
    dims.insert(dims.begin(), local_data.Width());

    // Build it as if tensor were contiguous
    std::vector<int> strides = build_strides_from_dims(dims);

    // Correct the first entry.
    strides.front() = local_data.LDim();

    // Setup the descriptor
    set_tensor_desc<TensorDataType>(desc, dims, strides);
  }
}

} // namespace <anon>

template <typename TensorDataType>
auto
DataParallelLayerTensorManager<TensorDataType>::get_prev_activations(
  int parent_index) -> tensor_desc_type&
{
  if (this->m_layer == nullptr)
    LBANN_ERROR("LayerTensorManager is not managing a layer!");
  const auto& local_data =
    this->m_layer->get_local_prev_activations(parent_index);
  const auto& dims = this->m_layer->get_input_dims(parent_index);
  this->set_num_parents(this->m_layer->get_num_parents());
  auto& desc = this->m_prev_activations[parent_index];
  set_data_parallel_tensor_desc(desc, dims, local_data);
  return desc;
}

template <typename TensorDataType>
auto DataParallelLayerTensorManager<TensorDataType>::get_activations(
  int child_index) -> tensor_desc_type&
{
  if (this->m_layer == nullptr)
    LBANN_ERROR("LayerTensorManager is not managing a layer!");
  const auto& local_data = this->m_layer->get_local_activations(child_index);
  const auto& dims = this->m_layer->get_output_dims(child_index);
  this->set_num_children(this->m_layer->get_num_children());
  auto& desc = this->m_activations[child_index];
  set_data_parallel_tensor_desc(desc, dims, local_data);
  return desc;
}

template <typename TensorDataType>
auto
DataParallelLayerTensorManager<TensorDataType>::get_prev_error_signals(
  int child_index) -> tensor_desc_type&
{
  if (this->m_layer == nullptr)
    LBANN_ERROR("LayerTensorManager is not managing a layer!");
  const auto& local_data =
    this->m_layer->get_local_prev_error_signals(child_index);
  const auto& dims = this->m_layer->get_output_dims(child_index);
  this->set_num_children(this->m_layer->get_num_children());
  auto& desc = this->m_prev_error_signals[child_index];
  set_data_parallel_tensor_desc(desc, dims, local_data);
  return desc;
}

template <typename TensorDataType>
auto
DataParallelLayerTensorManager<TensorDataType>::get_error_signals(
  int parent_index) -> tensor_desc_type&
{
  if (this->m_layer == nullptr)
    LBANN_ERROR("LayerTensorManager is not managing a layer!");
  const auto& local_data =
    this->m_layer->get_local_error_signals(parent_index);
  const auto& dims = this->m_layer->get_input_dims(parent_index);
  this->set_num_parents(this->m_layer->get_num_parents());
  auto& desc = this->m_error_signals[parent_index];
  set_data_parallel_tensor_desc(desc, dims, local_data);
  return desc;
}

////////////////////////////////////////////////////////////
// Entry-wise cuDNN tensor manager
////////////////////////////////////////////////////////////

template <typename TensorDataType>
EntrywiseLayerTensorManager<TensorDataType>
::EntrywiseLayerTensorManager(const data_type_layer<TensorDataType>* l)
  : LayerTensorManager<TensorDataType>(l) {}

namespace {

/** @brief Set a cuDNN tensor descriptor for an entrywise tensor
 *         operation.
 *
 *  Given local data in a (height x width) matrix, the tensor is
 *  initialized with dimensions (width, a, b, c), where
 *  a*b*c=height. This is because cuDNN is optimized for 4D tensors
 *  and gets poor performance with 1D tensors and 2D tensors.
 */
template <typename TensorDataType>
void set_entrywise_tensor_desc(
  TensorDescriptor_t& desc,
  El::AbstractMatrix<TensorDataType> const& local_data)
{
#ifdef LBANN_DEBUG
  if (local_data.GetDevice() != El::Device::GPU)
    LBANN_ERROR("Attempted to setup MIOpen tensor with non-GPU data");
#endif // LBANN_DEBUG
  const int height = local_data.Height();
  const int width = local_data.Width();
  const int ldim = local_data.LDim();
  if (height > 0 && width > 0)
  {
    // Factorize height into three factors
    // Note: factorization is memoized
    static std::unordered_map<int,std::vector<int>> cache;
    auto& factors = cache[height];
    if (factors.empty())
      factors = number_theory::balanced_factors(height, 3);

    // Set cuDNN tensor descriptor with 4D tensor
    set_tensor_desc<TensorDataType>(
      desc,
      {width, factors[2], factors[1], factors[0]},
      {ldim, factors[1]*factors[0], factors[0], 1});
  }
}

} // namespace <anon>

template <typename TensorDataType>
auto
EntrywiseLayerTensorManager<TensorDataType>::get_prev_activations(
  int parent_index) -> tensor_desc_type&
{
  if (this->m_layer == nullptr)
    LBANN_ERROR("tensor manager is not managing a layer");
  const auto& local_data =
    this->m_layer->get_local_prev_activations(parent_index);
  this->set_num_parents(this->m_layer->get_num_parents());
  auto& desc = this->m_prev_activations[parent_index];
  set_entrywise_tensor_desc(desc, local_data);
  return desc;
}

template <typename TensorDataType>
auto
EntrywiseLayerTensorManager<TensorDataType>::get_activations(
  int child_index) -> tensor_desc_type&
{
  if (this->m_layer == nullptr)
    LBANN_ERROR("tensor manager is not managing a layer");
  const auto& local_data = this->m_layer->get_local_activations(child_index);
  this->set_num_children(this->m_layer->get_num_children());
  auto& desc = this->m_activations[child_index];
  set_entrywise_tensor_desc(desc, local_data);
  return desc;
}

template <typename TensorDataType>
auto
EntrywiseLayerTensorManager<TensorDataType>::get_prev_error_signals(
  int child_index) -> tensor_desc_type&
{
  if (this->m_layer == nullptr)
    LBANN_ERROR("tensor manager is not managing a layer");
  const auto& local_data =
    this->m_layer->get_local_prev_error_signals(child_index);
  this->set_num_children(this->m_layer->get_num_children());
  auto& desc = this->m_prev_error_signals[child_index];
  set_entrywise_tensor_desc(desc, local_data);
  return desc;
}

template <typename TensorDataType>
auto
EntrywiseLayerTensorManager<TensorDataType>::get_error_signals(
  int parent_index) -> tensor_desc_type&
{
  if (this->m_layer == nullptr)
    LBANN_ERROR("tensor manager is not managing a layer");
  const auto& local_data = this->m_layer->get_local_error_signals(parent_index);
  this->set_num_parents(this->m_layer->get_num_parents());
  auto& desc = this->m_error_signals[parent_index];
  set_entrywise_tensor_desc(desc, local_data);
  return desc;
}

#ifdef TOM_LIKES_LLAMAS
namespace {

// Non-deterministic algorithms.
std::vector<ConvolutionFwdAlgo_t> nondet_fwd_algos = {};
std::vector<ConvolutionBwdDataAlgo_t> nondet_bwd_data_algos = {
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
};
std::vector<ConvolutionBwdFilterAlgo_t> nondet_bwd_filter_algos = {
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
  LBANN_CHECK_MIOPEN(cudnnGetConvolutionForwardAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  LBANN_CHECK_MIOPEN(cudnnGetConvolutionForwardAlgorithm_v7(
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
  LBANN_CHECK_MIOPEN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  LBANN_CHECK_MIOPEN(cudnnGetConvolutionBackwardDataAlgorithm_v7(
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
  LBANN_CHECK_MIOPEN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  LBANN_CHECK_MIOPEN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
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
  LBANN_CHECK_MIOPEN(cudnnGetConvolutionForwardAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results_all;
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(num_algos);
  for (int trial = 0; trial < num_trials + num_skip; ++trial) {
    int num_tested_algos;
    LBANN_CHECK_MIOPEN(cudnnFindConvolutionForwardAlgorithmEx(
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
  LBANN_CHECK_MIOPEN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results_all;
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(num_algos);
  for (int trial = 0; trial < num_trials + num_skip; ++trial) {
    int num_tested_algos;
    LBANN_CHECK_MIOPEN(cudnnFindConvolutionBackwardDataAlgorithmEx(
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
  LBANN_CHECK_MIOPEN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results_all;
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(num_algos);
  for (int trial = 0; trial < num_trials + num_skip; ++trial) {
    int num_tested_algos;
    LBANN_CHECK_MIOPEN(cudnnFindConvolutionBackwardFilterAlgorithmEx(
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
#endif // TOM_LIKES_LLAMAS

#define PROTO(T)                                                        \
  template void set_tensor_desc<T>(TensorDescriptor_t&,                 \
                                   std::vector<int>, std::vector<int>); \
  template class LayerTensorManager<T>;                                 \
  template class DataParallelLayerTensorManager<T>;                     \
  template class EntrywiseLayerTensorManager<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace dnn_primitive
} // namespace lbann

static_assert(lbann::dnn_primitive::get_data_type<float>() == miopenFloat,
              "MIOpen: Bad type-enum mapping.");
