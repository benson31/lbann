#pragma once
#ifndef _LBANN_SRC_LAYERS_FACTORY_HELPERS__INCLUDED
#define _LBANN_SRC_LAYERS_FACTORY_HELPERS__INCLUDED

#define INSTANTIATE_LAYER_BUILD_FULL(NAME, LAYOUT, DEVICE)              \
  template <>                                                           \
  std::unique_ptr<NAME##_layer<LAYOUT, DEVICE>>                         \
  build_##NAME##_layer_from_protobuf<LAYOUT, DEVICE>(                   \
    lbann_comm*, google::protobuf::Message const&)

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

#endif // _LBANN_SRC_LAYERS_FACTORY_HELPERS__INCLUDED
