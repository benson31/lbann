# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  argmax.cpp
  argmin.cpp
  channelwise_mean.cpp
  channelwise_softmax.cpp
  covariance.cpp
  mini_batch_index.cpp
  mini_batch_size.cpp
  one_hot.cpp
  variance.cpp
  )

if (LBANN_HAS_GPU)
  # Add the GPU source files for this directory
  target_sources(lbann_gpu_srcs INTERFACE
    covariance.cu
    variance.cu
    channelwise_mean.cu
    channelwise_softmax.cu
    one_hot.cu
    )
endif ()
