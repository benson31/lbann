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

#include "lbann/layers/learning/convolution.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/factories.hpp"
#include "lbann/proto/factory_helpers.hpp"
#include "lbann/utils/memory.hpp"

// Source tree includes
#include "lbann.pb.h"

namespace lbann {
namespace { // anon
template <data_layout Layout, El::Device Device>
struct convolution_layer_builder {
  static std::unique_ptr<convolution_layer<Layout, Device>>
  build(lbann_comm*, google::protobuf::Message const&) {
    LBANN_ERROR("convolution layer is only supported with "
                "a data-parallel layout");
    return nullptr;
  }
};// struct convolution_layer_builder

template <El::Device Device>
struct convolution_layer_builder<data_layout::DATA_PARALLEL, Device> {
  static constexpr data_layout Layout = data_layout::DATA_PARALLEL;
  static std::unique_ptr<convolution_layer<Layout, Device>>
  build(lbann_comm* comm, google::protobuf::Message const& in_msg) {
    using lbann_data::Convolution;
    const auto& params = dynamic_cast<Convolution const&>(in_msg);
    const auto& num_output_channels = params.num_output_channels();
    const auto& bias = params.has_bias();
    int num_groups = params.num_groups();
    if (num_groups == 0) {
      num_groups = 1;
    }
    if (params.has_vectors()) {
      const auto& dims = proto::parse_list<int>(params.conv_dims());
      const auto& pads = proto::parse_list<int>(params.conv_pads());
      const auto& strides = proto::parse_list<int>(params.conv_strides());
      std::vector<int> dilations
        = proto::parse_list<int>(params.conv_dilations());
      if (dilations.empty()) {
        dilations.resize(dims.size(), 1);
      }
      return lbann::make_unique<convolution_layer<Layout, Device>>(
               comm, dims.size(), num_output_channels,
               dims, pads, strides, dilations, num_groups, bias);
    }
    else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.conv_dims_i();
      const auto& pad = params.conv_pads_i();
      const auto& stride = params.conv_strides_i();
      int dilation = params.conv_dilations_i();
      if (dilation == 0) {
        dilation = 1;
      }
      return lbann::make_unique<convolution_layer<Layout, Device>>(
               comm, num_dims, num_output_channels,
               dim, pad, stride, dilation, num_groups, bias);
    }
  }
};// struct convolution_layer_builder<DATA_PARALLEL, CPU>
}// namespace <anon>

template <data_layout Layout, El::Device Device>
std::unique_ptr<convolution_layer<Layout, Device>>
build_convolution_layer_from_protobuf(
  lbann_comm* comm, google::protobuf::Message const& in_msg) {
  return convolution_layer_builder<Layout, Device>::build(comm, in_msg);
}

INSTANTIATE_LAYER_BUILD(convolution);

}// namespace lbann
