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

#include "lbann/proto/factories.hpp"
#include "lbann/utils/factory.hpp"
#include "lbann/utils/peek_map.hpp"

namespace lbann {
namespace proto {
template <typename OutT, typename... Args>
struct GenerateBuilderType_struct
{
  using type = std::function<std::unique_ptr<OutT>(Args...)>;
};

template <typename OutT, typename... Args>
using GenerateBuilderType =
  typename GenerateBuilderType_struct<OutT, Args...>::type;

using layer_factory = lbann::generic_factory<
  Layer,
  std::string,
  GenerateBuilderType<
    Layer, lbann_comm*, google::protobuf::Message const&>>;

namespace { // Anon
#define LAYER_DEFAULT_BUILDER(KEY, NAME)                                \
  factory.register_builder(                                             \
    #KEY, &default_layer_builder<NAME##_layer<Layout, Device>>)
#define LAYER_MESSAGE_BUILDER(KEY, NAME)                                \
  factory.register_builder(                                             \
    #KEY, &build_##NAME##_layer_from_protobuf<Layout, Device>)
#define LAYER_SPECIAL_BUILDER(...) (void) factory

template <typename LayerT>
std::unique_ptr<LayerT>
default_layer_builder(
  lbann_comm* comm, google::protobuf::Message const&)
{
  return make_unique<LayerT>(comm);
}

template <data_layout Layout, El::Device Device>
void setup_default_builders(layer_factory& factory)
{
  using namespace lbann_data;

  // MotifLayer, motif_layer?????

  // Input layers
  //LAYER_SPECIAL_BUILDER(Input, input);

  // Transform layers
  //LAYER_SPECIAL_BUILDER(Reshape, reshape);
  LAYER_MESSAGE_BUILDER(Pooling, pooling);
  //LAYER_MESSAGE_BUILDER(Concatenation, concatenation);
  //LAYER_SPECIAL_BUILDER(Slice, slice);
  LAYER_DEFAULT_BUILDER(Split, split);
  LAYER_DEFAULT_BUILDER(Sum, sum);
  //LAYER_MESSAGE_BUILDER(WeightedSum, weighted_sum);
  LAYER_MESSAGE_BUILDER(Unpooling, unpooling);
  LAYER_DEFAULT_BUILDER(Hadamard, hadamard);
  //LAYER_MESSAGE_BUILDER(Constant, constant);
  //LAYER_MESSAGE_BUILDER(Zero, zero);
  //LAYER_MESSAGE_BUILDER(Reduction, reduction);
  LAYER_DEFAULT_BUILDER(Evaluation, evaluation);
  //LAYER_MESSAGE_BUILDER(Gaussian, gaussian);
  //LAYER_MESSAGE_BUILDER(Bernoulli, bernoulli);
  //LAYER_MESSAGE_BUILDER(Uniform, uniform);
  //LAYER_MESSAGE_BUILDER(Crop, crop);
  //LAYER_MESSAGE_BUILDER(CategoricalRandom, categorical_random);
  //LAYER_MESSAGE_BUILDER(DiscreteRandom, discrete_random);
  LAYER_DEFAULT_BUILDER(Dummy, dummy);
  LAYER_DEFAULT_BUILDER(StopGradient, stop_gradient);
  //LAYER_MESSAGE_BUILDER(InTopK, in_top_k);
  //LAYER_MESSAGE_BUILDER(Sort, sort);
  //LAYER_MESSAGE_BUILDER(WeightsLayer, weights_layer);
  //LAYER_MESSAGE_BUILDER(Tessellate, tessellate);

  // Learning layers
  //LAYER_SPECIAL_BUILDER(FullyConnected, fully_connected);
  LAYER_MESSAGE_BUILDER(Convolution, convolution);
  //LAYER_SPECIAL_BUILDER(Deconvolution, deconvolution);

  // Loss layers
  LAYER_DEFAULT_BUILDER(CrossEntropy, cross_entropy);
  LAYER_DEFAULT_BUILDER(MeanSquaredError, mean_squared_error);
  LAYER_DEFAULT_BUILDER(MeanAbsoluteError, mean_absolute_error);
  LAYER_DEFAULT_BUILDER(CategoricalAccuracy, categorical_accuracy);
  //LAYER_MESSAGE_BUILDER(TopKCategoricalAccuracy, top_k_categorical_accuracy);
  LAYER_DEFAULT_BUILDER(L2Norm2, l2_norm2);
  LAYER_DEFAULT_BUILDER(L1Norm, l1_norm);
  LAYER_DEFAULT_BUILDER(BinaryCrossEntropy, binary_cross_entropy);
  LAYER_DEFAULT_BUILDER(SigmoidBinaryCrossEntropy, sigmoid_binary_cross_entropy);
  LAYER_DEFAULT_BUILDER(BooleanAccuracy, boolean_accuracy);
  LAYER_DEFAULT_BUILDER(BooleanFalseNegative, boolean_false_negative);
  LAYER_DEFAULT_BUILDER(BooleanFalsePositive, boolean_false_positive);

  // Math layers
  LAYER_DEFAULT_BUILDER(LogicalNot, logical_not);
  LAYER_DEFAULT_BUILDER(Abs, abs);
  LAYER_DEFAULT_BUILDER(Negative, negative);
  LAYER_DEFAULT_BUILDER(Sign, sign);
  LAYER_DEFAULT_BUILDER(Round, round);
  LAYER_DEFAULT_BUILDER(Ceil, ceil);
  LAYER_DEFAULT_BUILDER(Floor, floor);
  LAYER_DEFAULT_BUILDER(Reciprocal, reciprocal);
  LAYER_DEFAULT_BUILDER(Square, square);
  LAYER_DEFAULT_BUILDER(Sqrt, sqrt);
  LAYER_DEFAULT_BUILDER(Rsqrt, rsqrt);
  LAYER_DEFAULT_BUILDER(SafeReciprocal, safe_reciprocal);
  LAYER_DEFAULT_BUILDER(Exp, exp);
  LAYER_DEFAULT_BUILDER(Expm1, expm1);
  LAYER_DEFAULT_BUILDER(Log, log);
  LAYER_DEFAULT_BUILDER(Log1p, log1p);
  LAYER_DEFAULT_BUILDER(Cos, cos);
  LAYER_DEFAULT_BUILDER(Sin, sin);
  LAYER_DEFAULT_BUILDER(Tan, tan);
  LAYER_DEFAULT_BUILDER(Acos, acos);
  LAYER_DEFAULT_BUILDER(Asin, asin);
  LAYER_DEFAULT_BUILDER(Atan, atan);
  LAYER_DEFAULT_BUILDER(Cosh, cosh);
  LAYER_DEFAULT_BUILDER(Sinh, sinh);
  LAYER_DEFAULT_BUILDER(Tanh, tanh);
  LAYER_DEFAULT_BUILDER(Acosh, acosh);
  LAYER_DEFAULT_BUILDER(Asinh, asinh);
  LAYER_DEFAULT_BUILDER(Atanh, atanh);
  LAYER_DEFAULT_BUILDER(Add, add);
  LAYER_DEFAULT_BUILDER(Subtract, subtract);
  LAYER_DEFAULT_BUILDER(Multiply, multiply);
  LAYER_DEFAULT_BUILDER(Divide, divide);
  LAYER_DEFAULT_BUILDER(Mod, mod);
  LAYER_DEFAULT_BUILDER(Pow, pow);
  LAYER_DEFAULT_BUILDER(SafeDivide, safe_divide);
  LAYER_DEFAULT_BUILDER(SquaredDifference, squared_difference);
  LAYER_DEFAULT_BUILDER(Max, max);
  LAYER_DEFAULT_BUILDER(Min, min);
  LAYER_DEFAULT_BUILDER(Equal, equal);
  LAYER_DEFAULT_BUILDER(NotEqual, not_equal);
  LAYER_DEFAULT_BUILDER(Less, less);
  LAYER_DEFAULT_BUILDER(LessEqual, less_equal);
  LAYER_DEFAULT_BUILDER(Greater, greater);
  LAYER_DEFAULT_BUILDER(GreaterEqual, greater_equal);
  LAYER_DEFAULT_BUILDER(LogicalAnd, logical_and);
  LAYER_DEFAULT_BUILDER(LogicalOr, logical_or);
  LAYER_DEFAULT_BUILDER(LogicalXor, logical_xor);
  //LAYER_MESSAGE_BUILDER(Clamp, clamp);

  // Regularization layers
  //LAYER_MESSAGE_BUILDER(BatchNormalization, batch_normalization);
  //LAYER_MESSAGE_BUILDER(LocalResponseNormalization, local_response_normalization);
  //LAYER_MESSAGE_BUILDER(Dropout, dropout);
  //LAYER_MESSAGE_BUILDER(SeluDropout, selu_dropout);

  // Activation layers
  //LAYER_MESSAGE_BUILDER(Elu, elu);
  LAYER_DEFAULT_BUILDER(Identity, identity);
  //LAYER_MESSAGE_BUILDER(LeakyRelu, leaky_relu);
  LAYER_DEFAULT_BUILDER(LogSigmoid, log_sigmoid);
  LAYER_DEFAULT_BUILDER(LogSoftmax, log_softmax);
  LAYER_DEFAULT_BUILDER(Relu, relu);
  LAYER_DEFAULT_BUILDER(Selu, selu);
  LAYER_DEFAULT_BUILDER(Sigmoid, sigmoid);
  LAYER_DEFAULT_BUILDER(Softmax, softmax);
  LAYER_DEFAULT_BUILDER(Softplus, softplus);
  LAYER_DEFAULT_BUILDER(Softsign, softsign);

  // Image layers
  //LAYER_MESSAGE_BUILDER(BilinearResize, bilinear_resize);

  // Miscellaneous layers
  //LAYER_MESSAGE_BUILDER(Covariance, covariance);
  //LAYER_MESSAGE_BUILDER(Variance, variance);
  //LAYER_MESSAGE_BUILDER(ChannelwiseMean, channelwise_mean);
  LAYER_DEFAULT_BUILDER(MiniBatchIndex, mini_batch_index);
  LAYER_DEFAULT_BUILDER(MiniBatchSize, mini_batch_size);

}
#undef LAYER_SPECIAL_BUILDER
#undef LAYER_MESSAGE_BUILDER
#undef LAYER_DEFAULT_BUILDER

template <data_layout Layout, El::Device Device>
layer_factory const&
get_layer_factory() {
  static layer_factory factory;
  if (factory.size() == 0) {
    setup_default_builders<Layout, Device>(factory);
  }
  return factory;
}

template <data_layout Layout, El::Device Device>
std::unique_ptr<Layer>
build_layer(lbann_comm* comm, google::protobuf::Message const& in_msg) {
  auto&& msg = get_oneof_message(in_msg, "layer_type");
  auto&& factory = get_layer_factory<Layout, Device>();
  return factory.create_object(msg.GetDescriptor()->name(), comm, msg);
}
}// namespace <anon>

std::vector<El::Int> get_slice_points_from_reader(
  const generic_data_reader* dr, const std::string& var_category,
  bool& is_supported);

template <data_layout Layout, El::Device Device>
std::unique_ptr<Layer> construct_layer(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer) {

  auto&& layer_msg = get_oneof_message(proto_layer, "layer_type");
#if 0
  auto&& layer_pb_name = layer_msg.GetDescriptor()->name();
  if (layer_pb_name == "Input") {
    return build_input_layer_from_protobuf<Layout, Device>(
      comm, data_readers, num_parallel_readers, proto_layer);
  }
  else if (layer_pb_name == "FullyConnected") {
    return build_fully_connected_layer_from_protobuf<Layout, Device>(
      comm, data_readers, num_parallel_readers, proto_layer);
  }
  else if (layer_pb_name == "Deconvolution") {
    return build_deconvolution_layer_from_protobuf<Layout, Device>(
      comm, data_readers, num_parallel_readers, proto_layer);
  }
  else if (layer_pb_name == "Reshape") {
    return build_reshape_layer_from_protobuf<Layout, Device>(
      comm, data_readers, num_parallel_readers, proto_layer);
  }
  else if (layer_pb_name == "Slice") {
    return build_slice_layer_from_protobuf<Layout, Device>(
      comm, data_readers, num_parallel_readers, proto_layer);
  }
  else {
    return build_layer<Layout, Device>(comm, layer_msg);
  }
#endif
  return build_layer<Layout, Device>(comm, layer_msg);
}

// Template instantiation
template std::unique_ptr<Layer>
construct_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
template std::unique_ptr<Layer>
construct_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
#ifdef LBANN_HAS_GPU
template std::unique_ptr<Layer>
construct_layer<data_layout::DATA_PARALLEL, El::Device::GPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
template std::unique_ptr<Layer>
construct_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
#endif // LBANN_HAS_GPU

/// Obtain the slice points from the data reader
std::vector<El::Int> get_slice_points_from_reader(const generic_data_reader* dr_generic,
                                                  const std::string& var_category,
                                                  bool& is_supported) {
  std::vector<El::Int> slice_points;
  is_supported = false;
#if defined(LBANN_HAS_CONDUIT)
  // TODO: remove the dynamic cast when this feature gets merged into the base class
  const auto dr = dynamic_cast<const data_reader_jag_conduit*>(dr_generic);

  if (dr != nullptr) {
    is_supported = true;
    if (var_category == "independent") {
      slice_points = dr->get_slice_points_independent();
    } else if (var_category == "dependent") {
      slice_points = dr->get_slice_points_independent();
    } else {
      LBANN_ERROR("Unknown variable category \"" + var_category \
                  + "\". Must be either \"independent\" or \"dependent\".");
    }
  }
#endif
  return slice_points;
}

} // namespace proto
} // namespace lbann
