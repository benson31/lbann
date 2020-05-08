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

#ifndef LBANN_UTILS_MIOPEN_HPP_INCLUDED
#define LBANN_UTILS_MIOPEN_HPP_INCLUDED

#include <miopen/miopen.h>

namespace lbann {
namespace miopen {

/** @name Common typedefs */
///@{

using DataType_t = miopenDataType_t;
using Handle_t = miopenHandle_t;

using ActivationDescriptor_t = miopenActivationDescriptor_t;
using ActivationMode_t = miopenActivationMode_t;
using ConvolutionDescriptor_t = miopenConvolutionDescriptor_t;
using FilterDescriptor_t = miopenTensorDescriptor_t;
using TensorDescriptor_t = miopenTensorDescriptor_t;

using ConvolutionFwdAlgo_t = miopenConvFwdAlgorithm_t;
using ConvolutionBwdDataAlgo_t = miopenConvBwdDataAlgorithm_t;
using ConvolutionBwdFilterAlgo_t = miopenConvBwdWeightsAlgorithm_t;

using PoolingDescriptor_t = miopenPoolingDescriptor_t;
using PoolingMode_t = miopenPoolingMode_t;

using SoftmaxMode_t = miopenSoftmaxMode_t;
using SoftmaxAlgorithm_t = miopenSoftmaxAlgorithm_t;

///@}

namespace details
{

// This is absolutely terrible, and I'm sorry, world.
static constexpr int notActuallyMIOpenDouble = INT_MAX;

template <typename T>
struct ConvertDataTypeV;

#define ADD_MIOPEN_DATATYPE(TYPE, ENUMVAL)                              \
  template <>                                                           \
  struct ConvertDataTypeV<TYPE>                                         \
  {                                                                     \
    static constexpr auto value = static_cast<DataType_t>(ENUMVAL);     \
  }

#ifdef LBANN_HAS_GPU_FP16
ADD_MIOPEN_DATATYPE(fp16, miopenHalf);
#endif // LBANN_HAS_GPU_FP16
ADD_MIOPEN_DATATYPE(float, miopenFloat);
ADD_MIOPEN_DATATYPE(double, notActuallyMIOpenDouble);

}// namespace details
} // namespace miopen
} // namespace lbann
#endif // LBANN_UTILS_MIOPEN_HPP_INCLUDED
