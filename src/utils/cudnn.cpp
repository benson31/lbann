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
// Helper functions for cuDNN types
////////////////////////////////////////////////////////////



}// namespace dnn_primitive
} // namespace lbann

#endif // LBANN_HAS_CUDNN
