# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  bilinear_resize.cpp
  )

if (LBANN_HAS_GPU)
  # Add the GPU source files for this directory
  target_sources(lbann_gpu_srcs INTERFACE
    bilinear_resize.cu
    )
endif ()
