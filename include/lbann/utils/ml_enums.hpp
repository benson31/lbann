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
#ifndef LBANN_UTILS_ML_ENUMS_HPP_INCLUDED
#define LBANN_UTILS_ML_ENUMS_HPP_INCLUDED

namespace lbann
{

/** @name Pooling layer */
///@{

enum class pooling_mode
{
  MAX,
  AVERAGE,
  AVERAGE_NO_PAD,
};// enum class PoolingMode

///@}
/** @name Softmax layer */
///@{

/** @brief Which tensor dimensions to apply softmax over. */
enum class softmax_mode
{
  /** @brief Flag that no softmax mode has been chosen. */
  INVALID,
  /** @brief Sample-wise softmax.
   *
   *  Slice tensor along the sample dimension (assuming data in NCHW
   *  format) and apply softmax independently to each slice (once per
   *  sample).
   */
  INSTANCE,
  /** @brief Position-wise softmax.
   *
   *  Split tensor along all but the channel dimension (assuming data
   *  in NCHW format) and apply softmax independently to each piece
   *  (once per spatial position per sample).
   *
   *  This is not to be confused with @c channelwise_softmax, which
   *  slices along the sample and channel dimensions.
   */
  CHANNEL,
};// enum class softmax_mode

/** @brief Which algorithm to use to apply softmax. */
enum class softmax_algo
{
  /** @brief Straightforward Softmax implementation. */
  FAST,
  /** @brief Avoid potential floating-point overflows. */
  ACCURATE,
  /** @brief Log softmax operation. */
  LOG,
};// enum class softmax_algo

///@}

}// namespace lbann
#endif // LBANN_UTILS_ML_ENUMS_HPP_INCLUDED
