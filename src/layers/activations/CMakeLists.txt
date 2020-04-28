# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  activations.cpp
  elu.cpp
  identity.cpp
  relu.cpp
  leaky_relu.cpp
  log_softmax.cpp
  softmax.cpp
  softmax_builder.cpp
  )

if (LBANN_HAS_GPU)
  # Add the GPU source files for this directory
  target_sources(lbann_gpu_srcs INTERFACE
    activations.cu
    elu.cu
    relu.cu
    leaky_relu.cu
    log_softmax.cu
    softmax.cu
    )
endif ()
