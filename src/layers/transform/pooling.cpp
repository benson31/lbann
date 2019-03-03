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

#include "lbann/layers/transform/pooling.hpp"

#include "lbann/proto/factories.hpp"

// Source tree includes
#include "factory_helpers.hpp"
#include "lbann.pb.h"

namespace lbann {
namespace { // anon
template <data_layout Layout, El::Device Device>
struct pooling_layer_builder {
  static std::unique_ptr<pooling_layer<Layout, Device>>
  build(lbann_comm*, google::protobuf::Message const&) {
    LBANN_ERROR("pooling layer is only supported with "
                "a data-parallel layout");
    return nullptr;
  }
};// struct pooling_layer_builder

template <El::Device Device>
struct pooling_layer_builder<data_layout::DATA_PARALLEL, Device> {
  static std::unique_ptr<pooling_layer<data_layout::DATA_PARALLEL, Device>>
  build(lbann_comm* comm, google::protobuf::Message const& in_msg) {
    const auto& params = dynamic_cast<lbann_data::Pooling const&>(in_msg);
    const auto& mode_str = params.pool_mode();
    // FIXME (trb): This should use an enum, not a string. String typos
    // are caught at runtime; enum typos are caught at compile time.
    pool_mode mode = pool_mode::invalid;
    if (mode_str == "max" )            { mode = pool_mode::max; }
    if (mode_str == "average" )        { mode = pool_mode::average; }
    if (mode_str == "average_no_pad" ) { mode = pool_mode::average_no_pad; }
    if (params.has_vectors()) {
      const auto& dims = proto::parse_list<int>(params.pool_dims());
      const auto& pads = proto::parse_list<int>(params.pool_pads());
      const auto& strides = proto::parse_list<int>(params.pool_strides());
      return lbann::make_unique<
        pooling_layer<data_layout::DATA_PARALLEL, Device>>(
          comm, dims.size(), dims, pads, strides, mode);
    }
    else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.pool_dims_i();
      const auto& pad = params.pool_pads_i();
      const auto& stride = params.pool_strides_i();
      return lbann::make_unique<
        pooling_layer<data_layout::DATA_PARALLEL, Device>>(
          comm, num_dims, dim, pad, stride, mode);
    }
  }
};// struct pooling_layer_builder<DATA_PARALLEL, Device>
}// namespace <anon>

template <data_layout Layout, El::Device Device>
std::unique_ptr<pooling_layer<Layout, Device>>
build_pooling_layer_from_protobuf(
  lbann_comm* comm, google::protobuf::Message const& in_msg) {
  return pooling_layer_builder<Layout, Device>::build(comm, in_msg);
}

INSTANTIATE_LAYER_BUILD(pooling);

}// namespace lbann
