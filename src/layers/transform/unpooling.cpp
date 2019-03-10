////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include "lbann/layers/transform/unpooling.hpp"
#include "lbann/proto/factory_helpers.hpp"
#include "lbann/utils/memory.hpp"

// Source tree includes
#include "lbann.pb.h"

namespace lbann {
namespace { // anon
template <data_layout Layout, El::Device Device>
struct unpooling_layer_builder {
  static std::unique_ptr<unpooling_layer<Layout, Device>>
  build(lbann_comm*, google::protobuf::Message const&) {
    LBANN_ERROR("unpooling layer is only supported with "
                "a data-parallel layout and on CPU");
    return nullptr;
  }
};// struct unpooling_layer_builder

template <>
struct unpooling_layer_builder<data_layout::DATA_PARALLEL, El::Device::CPU> {
  static constexpr data_layout Layout = data_layout::DATA_PARALLEL;
  static constexpr El::Device Device = El::Device::CPU;
  static std::unique_ptr<unpooling_layer<Layout, Device>>
  build(lbann_comm* comm, google::protobuf::Message const&) {
    return lbann::make_unique<unpooling_layer<Layout, Device>>(comm);
  }
};// struct unpooling_layer_builder<DATA_PARALLEL, CPU>
}// namespace <anon>

template <data_layout Layout, El::Device Device>
std::unique_ptr<unpooling_layer<Layout, Device>>
build_unpooling_layer_from_protobuf(
  lbann_comm* comm, google::protobuf::Message const& in_msg) {
  return unpooling_layer_builder<Layout, Device>::build(comm, in_msg);
}

INSTANTIATE_LAYER_BUILD(unpooling);

}// namespace lbann
