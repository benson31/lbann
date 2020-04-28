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

#ifndef LBANN_UTILS_DNN_PRIMITIVES_HPP_INCLUDED
#define LBANN_UTILS_DNN_PRIMITIVES_HPP_INCLUDED

#include "lbann_config.hpp"

#if defined LBANN_HAS_CUDNN
#include "cudnn.hpp"
#elif defined LBANN_HAS_MIOPEN
#include "miopen.hpp"
#endif

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/ml_enums.hpp"

#include <vector>

/** @file
 *
 *  This file provides high-level APIs for interfacing with a DNN
 *  primitive library, such as cuDNN or MIOpen.
 *
 *  @note These are strongly GPU-centric; packages like MKL-DNN might
 *  have compatibility issues. This will be addressed once there's a
 *  use-case and a Github issue to track it.
 */
namespace lbann {

/** @namespace dnn_primitive
 *  @brief An abstract API for DNN primitives.
 *
 *  Functions and classes in this namespace provide a common,
 *  consistent API across (primarily GPU-centric) DNN primitive
 *  libraries.
 */
namespace dnn_primitive {

#if defined LBANN_HAS_CUDNN
namespace impl = ::lbann::cudnn;
#elif defined LBANN_HAS_MIOPEN
namespace impl = ::lbann::miopen;
#endif

/** @name Common typedefs. */
///@{

using DataType_t = impl::DataType_t;
using Handle_t = impl::Handle_t;

using ActivationDescriptor_t = impl::ActivationDescriptor_t;
using ActivationMode_t = impl::ActivationMode_t;
using FilterDescriptor_t = impl::FilterDescriptor_t;
using TensorDescriptor_t = impl::TensorDescriptor_t;

///@}
/** @brief Type conversion and metaprogramming. */
///@{

/** @brief Metafunction to get the DNN scaling parameter type from a
 *         primary data type.
 */
template <typename T>
struct ScalingParameterT
{
  using type = T;
};

template <typename T>
using ScalingParamType = typename ScalingParameterT<T>::type;

#ifdef LBANN_HAS_GPU_FP16
template <>
struct ScalingParameterT<fp16>
{
  using type = float;
};
#endif // LBANN_USE_GPU_FP16

/** @brief Get DNN primitive data type associated with type. */
template <typename T>
constexpr DataType_t get_data_type()
{
    return impl::details::ConvertDataTypeV<T>::value;
}

///@}
/** @name Control of global DNN objects. */
///@{

/** @brief Initialize global DNN objects. */
void initialize();
/** @brief Destroy global DNN objects. */
void destroy();
/** Get DNN handle.
 *  This resets the active HIP device and stream to the Hydrogen
 *  defaults. The DNN handle is initialized if needed.
 */
Handle_t& get_handle();

///@}
/** @brief Type helpers */
///@{


/** @brief Set DNN tensor descriptor.
 *
 *  @param desc The tensor descriptor object; will be created if
 *              necessary.
 *  @param dims List of tensor dimensions.
 *  @param strides A list of tensor strides. If not given, the tensor
 *                 is assumed to be packed.
 */
template <typename TensorDataType>
void set_tensor_desc(TensorDescriptor_t& desc,
                     std::vector<int> dims,
                     std::vector<int> strides = {});

/** @brief Copy DNN tensor descriptor.
 *  @param src The original tensor descriptor.
 *  @param dst The target tensor descriptor; created or destroyed if
 *             needed.
 */
void copy_tensor_desc(const TensorDescriptor_t& src,
                      TensorDescriptor_t& dst);

/** @brief Copy DNN activation descriptor.
 *  @param src The original activation descriptor.
 *  @param dst The target descriptor; created or destroyed if needed.
 */
void copy_activation_desc(const ActivationDescriptor_t& src,
                          ActivationDescriptor_t& dst);

///@}
/** @brief Tensor managers */
///@{

/** @brief Manager for a layer's DNN tensor descriptors. */
template <typename TensorDataType>
class LayerTensorManager
{
public:
  using tensor_desc_type = TensorDescriptor_t;
  using layer_type = data_type_layer<TensorDataType>;
public:
  LayerTensorManager(const layer_type* l = nullptr);
  virtual ~LayerTensorManager();

  /** @brief Get the layer being managed. */
  const layer_type& get_layer() const noexcept { return *m_layer; }
  /** @brief Set the layer being managed. */
  void set_layer(const layer_type* l);

  /** @brief Get DNN tensor descriptor for layer input. */
  virtual tensor_desc_type& get_prev_activations(int parent_index = 0) = 0;
  /** @brief Get DNN tensor descriptor for layer output. */
  virtual tensor_desc_type& get_activations(int child_index = 0) = 0;
  /** @brief Get DNN tensor descriptor for gradient w.r.t. layer
   *         output.
   */
  virtual tensor_desc_type& get_prev_error_signals(int child_index = 0) = 0;
  /** @brief Get DNN tensor descriptor for gradient w.r.t. layer
   *         input.
   */
  virtual tensor_desc_type& get_error_signals(int parent_index = 0) = 0;

protected:

  /** @name Protected type information. */
  ///@{

  /** @brief A struct to ensure tensor descriptions are deleted properly.
   *
   *  This is essentially half an RAII wrapper -- only destruction is
   *  managed. The purpose is to create a vector of these things and
   *  be able to safely shrink/expand it.
   */
  struct SafeTensorDescriptionManager
  {
    SafeTensorDescriptionManager() = default;
    ~SafeTensorDescriptionManager();
    operator tensor_desc_type& () { return tensor_desc_; }
    operator tensor_desc_type const& () const { return tensor_desc_; }
    tensor_desc_type tensor_desc_ = nullptr;
  };// struct SafeTensorDescriptionManager

  using tensor_desc_storage_type = SafeTensorDescriptionManager;

  ///@}
  /** @name Non-public copy/move construct and assign. */
  ///@{
  LayerTensorManager(LayerTensorManager const& other);
  LayerTensorManager(LayerTensorManager&& other);
  LayerTensorManager& operator=(const LayerTensorManager& other);
  ///@}

  /** @brief Clear the internal data */
  void clear_data_();

  /** @brief Set number of tensor descriptors corresponding to layer
   *         inputs.
   */
  void set_num_parents(int num_parents);

  /** @brief Set number of tensor descriptors corresponding to layer
   *         outputs.
   */
  void set_num_children(int num_children);

  /** @brief Layer being managed. */
  const layer_type* m_layer;
  /** @brief DNN tensor descriptors for layer inputs. */
  std::vector<tensor_desc_storage_type> m_prev_activations;
  /** @brief DNN tensor descriptors for layer outputs. */
  std::vector<tensor_desc_storage_type> m_activations;
  /** @brief DNN tensor descriptors for gradients w.r.t. layer
   *         outputs.
   */
  std::vector<tensor_desc_storage_type> m_prev_error_signals;
  /** @brief DNN tensor descriptors for gradients w.r.t. layer
   *         inputs.
   */
  std::vector<tensor_desc_storage_type> m_error_signals;

};

/** @brief Manager for a data-parallel layer's DNN tensor
 *         descriptors.
 */
template <typename TensorDataType>
class DataParallelLayerTensorManager final
  : public LayerTensorManager<TensorDataType>
{
public:
  using tensor_desc_type = TensorDescriptor_t;
  using layer_type = data_type_layer<TensorDataType>;
public:
  DataParallelLayerTensorManager(const layer_type* l = nullptr);
  DataParallelLayerTensorManager(
    const DataParallelLayerTensorManager& other) = default;
  DataParallelLayerTensorManager&
    operator=(const DataParallelLayerTensorManager& other) = default;
  ~DataParallelLayerTensorManager() = default;
  tensor_desc_type& get_prev_activations(int parent_index = 0) override;
  tensor_desc_type& get_activations(int child_index = 0) override;
  tensor_desc_type& get_prev_error_signals(int child_index = 0) override;
  tensor_desc_type& get_error_signals(int parent_index = 0) override;
};// class DataParallelLayerTensorManager

/** Manager for an entry-wise layer's DNN tensor descriptors. */
template <typename TensorDataType>
class EntrywiseLayerTensorManager final
  : public LayerTensorManager<TensorDataType>
{
public:
  using tensor_desc_type = TensorDescriptor_t;
  using layer_type = data_type_layer<TensorDataType>;
public:
  EntrywiseLayerTensorManager(const layer_type* l = nullptr);
  EntrywiseLayerTensorManager(
    const EntrywiseLayerTensorManager& other) = default;
  EntrywiseLayerTensorManager&
    operator=(const EntrywiseLayerTensorManager& other) = default;
  ~EntrywiseLayerTensorManager() = default;
  tensor_desc_type& get_prev_activations(int parent_index = 0) override;
  tensor_desc_type& get_activations(int child_index = 0) override;
  tensor_desc_type& get_prev_error_signals(int child_index = 0) override;
  tensor_desc_type& get_error_signals(int parent_index = 0) override;
}; // class EntrywiseLayerTensorManager

/** @name Backward-compatible type aliases. */
///@{

template <typename TensorDataType>
using layer_tensor_manager = LayerTensorManager<TensorDataType>;

template <typename TensorDataType>
using data_parallel_layer_tensor_manager =
  DataParallelLayerTensorManager<TensorDataType>;

template <typename TensorDataType>
using entrywise_layer_tensor_manager =
  EntrywiseLayerTensorManager<TensorDataType>;

///@}
///@}

/** @name Specific DNN operations */
///@{

/** @name Batch normalization operations */
///@{

///@}
/** @name Convolution operations */
///@{

using ConvolutionDescriptor_t = impl::ConvolutionDescriptor_t;
using ConvolutionFwdAlgo_t = impl::ConvolutionFwdAlgo_t;
using ConvolutionBwdDataAlgo_t = impl::ConvolutionBwdDataAlgo_t;
using ConvolutionBwdFilterAlgo_t = impl::ConvolutionBwdFilterAlgo_t;

/** @name Algorithm selection */
///@{

/** @brief Select a forward convolution algorithm.
 *
 *  If autotuning, memory for DNN algorithm runs is needed and
 *  should be provided via the pointer arguments.
 *
 *  @param autotune True to attempt all DNN algorithms and select
 *                  the fastest.
 *  @param deterministic True to require deterministic algorithms.
 */
ConvolutionFwdAlgo_t get_fwd_algorithm(
  bool autotune,
  bool deterministic,
  const TensorDescriptor_t& input_desc,
  const void* input,
  const FilterDescriptor_t& kernel_desc,
  const void* kernel,
  const ConvolutionDescriptor_t& conv_desc,
  const TensorDescriptor_t& output_desc,
  void* output,
  size_t ws_size,
  void* ws);

/** @brief Select a backward data convolution algorithm.
 *
 *  If autotuning, memory for DNN algorithm runs is needed and
 *  should be provided via the pointer arguments.
 *
 *  @param autotune True to attempt all DNN algorithms and select
 *                  the fastest.
 *  @param deterministic True to require deterministic algorithms.
 */
ConvolutionBwdDataAlgo_t get_bwd_data_algorithm(
  bool autotune,
  bool deterministic,
  const FilterDescriptor_t& kernel_desc,
  const void* kernel,
  const TensorDescriptor_t& prev_error_signal_desc,
  const void* prev_error_signal,
  const ConvolutionDescriptor_t& conv_desc,
  const TensorDescriptor_t& error_signal_desc,
  void* error_signal,
  size_t ws_size,
  void* ws);

/** @brief Select a backward filter convolution algorithm.
 *
 *  If autotuning, memory for DNN algorithm runs is needed and
 *  should be provided via the pointer arguments.
 *
 *  @param autotune True to attempt all DNN algorithms and select
 *                  the fastest.
 *  @param deterministic True to require deterministic algorithms.
 */
ConvolutionBwdFilterAlgo_t get_bwd_filter_algorithm(
  bool autotune,
  bool deterministic,
  const TensorDescriptor_t& input_desc,
  const void* input,
  const TensorDescriptor_t& prev_error_signal_desc,
  const void* prev_error_signal,
  const ConvolutionDescriptor_t& conv_desc,
  const FilterDescriptor_t& kernel_gradient_desc,
  void* kernel_gradient,
  size_t ws_size,
  void* ws);

///@}
/** @name Forward and backward operations. */
///@{

void ConvolutionForward(...);
void ConvolutionBackward(...);

///@}
///@}
/** @name Dropout operations */
///@{

///@}
/** @name Pooling operations */
///@

using PoolingDescriptor_t = impl::PoolingDescriptor_t;
using PoolingMode_t = impl::PoolingMode_t;

PoolingDescriptor_t CreatePoolingDescriptor();
// Destroy?
// SetPoolingDescriptor?
// Copy?

/** @brief The DNN primitive execution of the forward pooling operation. */
void PoolingForward(...);

/** @brief The DNN primitive execution of the backward pooling operation. */
void PoolingBackward(...);

///@}
/** @name Softmax operations */
///@{

using SoftmaxMode_t = impl::SoftmaxMode_t;
using SoftmaxAlgorithm_t = impl::SoftmaxAlgorithm_t;

/** @brief Convert the LBANN softmax mode to the DNN primitive token. */
SoftmaxMode_t get_softmax_mode(lbann::softmax_mode);

/** @brief The DNN primitive execution of the forward softmax operation. */
template <typename T>
void softmax_forward(lbann::softmax_algo, lbann::softmax_mode,
                     T alpha, TensorDescriptor_t xdesc, T const* x,
                     T beta, TensorDescriptor_t ydesc, T* y);

/** @brief The DNN primitive execution of the backward softmax operation. */
template <typename T>
void softmax_backward(lbann::softmax_algo, lbann::softmax_mode,
                      T alpha,
                      TensorDescriptor_t ydesc, T const* y,
                      TensorDescriptor_t dydesc, T const* dy,
                      T beta,
                      TensorDescriptor_t dxdesc, T* dx);

///@}
///@}
} // namespace dnn_primitive
} // namespace lbann
#endif // LBANN_UTILS_DNN_PRIMITIVES_HPP_INCLUDED
