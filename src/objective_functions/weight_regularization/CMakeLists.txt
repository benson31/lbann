# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  l2.cpp
  )

if (TARGET lbann_gpu_srcs)
  # Add the GPU source files for this directory
  target_sources(lbann_gpu_srcs INTERFACE
    l2.cu
    )
endif ()
