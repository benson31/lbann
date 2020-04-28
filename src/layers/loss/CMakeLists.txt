# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  categorical_accuracy.cpp
  cross_entropy.cpp
  entrywise.cpp
  l1_norm.cpp
  l2_norm2.cpp
  mean_absolute_error.cpp
  mean_squared_error.cpp
  top_k_categorical_accuracy.cpp
  )

if (LBANN_HAS_GPU)
  # Add the GPU source files for this directory
  target_sources(lbann_gpu_srcs INTERFACE
    categorical_accuracy.cu
    cross_entropy.cu
    entrywise.cu
    l1_norm.cu
    l2_norm2.cu
    mean_absolute_error.cu
    mean_squared_error.cu
    top_k_categorical_accuracy.cu
    )
endif ()
