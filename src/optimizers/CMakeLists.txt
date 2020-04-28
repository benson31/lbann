# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  adagrad.cpp
  adam.cpp
  data_type_optimizer.cpp
  hypergradient_adam.cpp
  optimizer.cpp
  rmsprop.cpp
  sgd.cpp
  )

if (LBANN_HAS_GPU)
  # Add the GPU source files for this directory
  target_sources(lbann_gpu_srcs INTERFACE
    adagrad.cu
    adam.cu
    rmsprop.cu
    sgd.cu
    )
endif ()
