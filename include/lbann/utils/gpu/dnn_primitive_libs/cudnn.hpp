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

#ifndef LBANN_UTILS_CUDNN_HPP
#define LBANN_UTILS_CUDNN_HPP

#ifdef LBANN_HAS_CUDNN

#include "lbann/base.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/gpu/cuda.hpp"
#include <vector>

#include <cudnn.h>

// Error utility macros
#define LBANN_CHECK_CUDNN_NODEBUG(cudnn_call)                           \
  do {                                                                  \
    const cudnnStatus_t status_LBANN_CHECK_CUDNN = (cudnn_call);        \
    if (status_LBANN_CHECK_CUDNN != CUDNN_STATUS_SUCCESS) {             \
      cudaDeviceReset();                                                \
      LBANN_ERROR(std::string("cuDNN error (")                          \
                  + cudnnGetErrorString(status_LBANN_CHECK_CUDNN)       \
                  + std::string(")"));                                  \
    }                                                                   \
  } while (0)
#define LBANN_CHECK_CUDNN_DEBUG(cudnn_call)     \
  do {                                          \
    LBANN_CUDA_CHECK_LAST_ERROR(true);          \
    LBANN_CHECK_CUDNN_NODEBUG(cudnn_call);      \
  } while (0)
#ifdef LBANN_DEBUG
#define LBANN_CHECK_CUDNN(cudnn_call) LBANN_CHECK_CUDNN_DEBUG(cudnn_call)
#else
#define LBANN_CHECK_CUDNN(cudnn_call) LBANN_CHECK_CUDNN_NODEBUG(cudnn_call)
#endif // #ifdef LBANN_DEBUG

#define LBANN_CHECK_CUDNN_DTOR(cudnn_call)                              \
  try {                                                                 \
    LBANN_CHECK_CUDNN(cudnn_call);                                      \
  }                                                                     \
  catch (std::exception const& e) {                                     \
    std::cerr << "Caught exception:\n\n    what(): "                    \
              << e.what() << "\n\nCalling std::terminate() now."        \
              <<  std::endl;                                            \
    std::terminate();                                                   \
  }                                                                     \
  catch (...) {                                                         \
    std::cerr << "Caught something that isn't an std::exception.\n\n"   \
              << "Calling std::terminate() now." << std::endl;          \
    std::terminate();                                                   \
  }


namespace lbann {
namespace cudnn {

/** @name Common typedefs */
///@{

using DataType_t = cudnnDataType_t;
using Handle_t = cudnnHandle_t;

using ActivationDescriptor_t = cudnnActivationDescriptor_t;
using ActivationMode_t = cudnnActivationMode_t;
using ConvolutionDescriptor_t = cudnnConvolutionDescriptor_t;
using FilterDescriptor_t = cudnnFilterDescriptor_t;
using TensorDescriptor_t = cudnnTensorDescriptor_t;

using ConvolutionFwdAlgo_t = cudnnConvolutionFwdAlgo_t;
using ConvolutionBwdDataAlgo_t = cudnnConvolutionBwdDataAlgo_t;
using ConvolutionBwdFilterAlgo_t = cudnnConvolutionBwdFilterAlgo_t;

using LRNDescriptor_t = cudnnLRNDescriptor_t;

using PoolingDescriptor_t = cudnnPoolingDescriptor_t;
using PoolingMode_t = cudnnPoolingMode_t;

using SoftmaxAlgorithm_t = cudnnSoftmaxAlgorithm_t;
using SoftmaxMode_t = cudnnSoftmaxMode_t;

///@}

namespace details {

template <typename T>
struct ConvertDataTypeV;

#define ADD_CUDNN_DATATYPE(TYPE,ENUMVAL)        \
  template <>                                   \
  struct ConvertDataTypeV<TYPE>                 \
  {                                             \
    static constexpr auto value = ENUMVAL;      \
  }

#ifdef LBANN_HAS_GPU_FP16
ADD_CUDNN_DATATYPE(fp16, CUDNN_DATA_HALF);
#endif // LBANN_HAS_GPU_FP16
ADD_CUDNN_DATATYPE(float, CUDNN_DATA_FLOAT);
ADD_CUDNN_DATATYPE(double, CUDNN_DATA_DOUBLE);

}// namespace details

/** @brief Set the default to use tensor core operations, allowing
 *         FP32->FP16 conversions.
 */
void default_to_tensor_ops() noexcept;

/** @brief Get the default math type.
 *
 *  Will query the command-line args.
 */
cudnnMathType_t get_default_convolution_math_type() noexcept;

} // namespace cudnn
} // namespace lbann
#endif // LBANN_HAS_CUDNN
#endif // LBANN_UTILS_CUDNN_HPP
