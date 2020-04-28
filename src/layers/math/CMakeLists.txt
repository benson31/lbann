# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  unary.cpp
  binary.cpp
  clamp.cpp
  matmul.cpp
  )

if (LBANN_HAS_GPU)
  # Add the GPU source files for this directory
  target_sources(lbann_gpu_srcs INTERFACE
    unary.cu
    binary.cu
    clamp.cu
    )
endif ()
