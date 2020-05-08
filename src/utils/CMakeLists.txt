# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  argument_parser.cpp
  cnpy_utils.cpp
  commify.cpp
  description.cpp
  environment_variable.cpp
  exception.cpp
  file_utils.cpp
  graph.cpp
  im2col.cpp
  image.cpp
  jag_common.cpp
  lbann_library.cpp
  number_theory.cpp
  omp_diagnostics.cpp
  options.cpp
  profiling.cpp
  protobuf_utils.cpp
  python.cpp
  random.cpp
  stack_profiler.cpp
  stack_trace.cpp
  statistics.cpp
  summary.cpp
  system_info.cpp
  trainer_file_utils.cpp
)

if (LBANN_HAS_CUDNN)
  target_sources(lbann_cxx PRIVATE
    cudnn.cpp
    )
endif ()

if (LBANN_HAS_MIOPEN)
  target_sources(lbann_cxx PRIVATE
    miopen.cpp
    )
endif ()

if (LBANN_HAS_HALF)
  target_sources(lbann_cxx PRIVATE
    serialization.cpp
    )
endif (LBANN_HAS_HALF)

if (LBANN_HAS_CUDA)
  # Add the CUDA source files for this directory
  target_sources(lbann_cxx PRIVATE
    cuda.cpp
    )
  target_sources(lbann_gpu_srcs INTERFACE
    nvshmem.cu
    )
endif ()

if (LBANN_HAS_DISTCONV)
  target_sources(lbann_cxx PRIVATE
    distconv.cpp
    )
endif ()

add_subdirectory(threads)
