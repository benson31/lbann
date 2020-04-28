# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  batch_normalization.cpp
  dropout.cpp
  entrywise_batch_normalization.cpp
  instance_norm.cpp
  layer_norm.cpp
  local_response_normalization.cpp
  selu_dropout.cpp
  )

if (LBANN_HAS_GPU)
  # Add the GPU source files for this directory
  target_sources(lbann_gpu_srcs INTERFACE
    batch_normalization.cu
    entrywise_batch_normalization.cu
    instance_norm.cu
    layer_norm.cu
    )
endif ()
