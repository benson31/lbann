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

#ifndef LBANN_UTILS_GPU_EVENT_WRAPPER_HPP_INCLUDED
#define LBANN_UTILS_GPU_EVENT_WRAPPER_HPP_INCLUDED

#include "lbann_config.hpp"

#if defined LBANN_HAS_CUDA
#include <cuda_runtime.h>
#elif defined LBANN_HAS_ROCM
#include <hip/hip_runtime.h>
#endif

namespace lbann
{
namespace gpu
{

/** @brief Wrapper class for a GPU event.
 *
 *  On CUDA-based platforms, this wraps a @c cudaEvent_t. On
 *  ROCm-based platforms, this wraps a @c hipEvent_t.
 */
class event_wrapper {
#ifdef LBANN_HAS_CUDA
    using event_type = cudaEvent_t;
    using stream_type = cudaStream_t;
#elif defined LBANN_HAS_ROCM
    using event_type = hipEvent_t;
    using stream_type = hipStream_t;
#endif

public:
  event_wrapper();
  event_wrapper(const event_wrapper& other);
  event_wrapper& operator=(const event_wrapper& other);
  ~event_wrapper();
  /** @brief Enqueue GPU event on a GPU stream. */
  void record(stream_type stream);
  /** @brief Check whether GPU event has completed. */
  bool query() const;
  /** @brief Wait until GPU event has completed. */
  void synchronize();
  /** @brief Get GPU event object. */
  event_type& get_event();
private:
  /** @brief GPU event object.
   *  @details The event object lifetime is managed internally.
   */
  event_type m_event;
  /** @brief GPU stream object.
   *  @details The stream object lifetime is assumed to be managed
   *           externally.
   */
  stream_type m_stream;
};// class event_wrapper

}// namespace gpu
}// namespace lbann
#endif // LBANN_UTILS_GPU_EVENT_WRAPPER_HPP_INCLUDED
