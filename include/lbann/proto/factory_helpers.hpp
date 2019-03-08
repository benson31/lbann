#pragma once
#ifndef _LBANN_SRC_LAYERS_FACTORY_HELPERS__INCLUDED
#define _LBANN_SRC_LAYERS_FACTORY_HELPERS__INCLUDED

#define LAYER_BUILD_SIGNATURE(NAME, LAYOUT, DEVICE, COMM, MSG)          \
  std::unique_ptr<NAME##_layer<LAYOUT, DEVICE>>                         \
  build_##NAME##_layer_from_protobuf(                                   \
    lbann_comm* COMM, google::protobuf::Message const& MSG)

#define DECLARE_LAYER_BUILDER(NAME)                                     \
  template <data_layout Layout, El::Device Device>                      \
  LAYER_BUILD_SIGNATURE(NAME, Layout, Device,,)

#define INSTANTIATE_LAYER_BUILD_FULL(NAME, LAYOUT, DEVICE)              \
  template LAYER_BUILD_SIGNATURE(NAME, LAYOUT, DEVICE,,)

#define INSTANTIATE_LAYER_BUILD_DEV(NAME, DEVICE)                       \
  INSTANTIATE_LAYER_BUILD_FULL(                                         \
    NAME, data_layout::DATA_PARALLEL, DEVICE);                          \
  INSTANTIATE_LAYER_BUILD_FULL(                                         \
    NAME, data_layout::MODEL_PARALLEL, DEVICE)

#ifndef LBANN_HAS_GPU
#define INSTANTIATE_LAYER_BUILD(NAME)                   \
  INSTANTIATE_LAYER_BUILD_DEV(NAME, El::Device::CPU)
#else
#define INSTANTIATE_LAYER_BUILD(NAME)                   \
  INSTANTIATE_LAYER_BUILD_DEV(NAME, El::Device::CPU);   \
  INSTANTIATE_LAYER_BUILD_DEV(NAME, El::Device::GPU)
#endif

// Macros for dealing with the default builder -- important for
// compilation speed.
#define DEFAULT_LAYER_BUILDER_SIG(TYPE)                 \
  std::unique_ptr<TYPE>                                 \
  default_layer_builder(                                \
    lbann_comm*, google::protobuf::Message const&)

#define INSTANTIATE_LAYER_DEFAULT_BUILDER(NAME)           \
  template DEFAULT_LAYER_BUILDER_SIG(NAME##_layer)

#define DECLARE_LAYER_DEFAULT_BUILDER(NAME)                     \
  extern template DEFAULT_LAYER_BUILDER_SIG(NAME##_layer)

#endif // _LBANN_SRC_LAYERS_FACTORY_HELPERS__INCLUDED
