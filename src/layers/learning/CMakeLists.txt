# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  channelwise_fully_connected.cpp
  channelwise_scale_bias.cpp
  channelwise_scale_bias_builder.cpp
  convolution.cpp
  deconvolution.cpp
  entrywise_scale_bias.cpp
  embedding.cpp
  embedding_builder.cpp
  fully_connected.cpp
  )

if (LBANN_HAS_GPU)
  # Add the GPU source files for this directory
  target_sources(lbann_gpu_srcs INTERFACE
    channelwise_scale_bias.cu
    embedding.cu
    entrywise_scale_bias.cu
    )
endif ()
