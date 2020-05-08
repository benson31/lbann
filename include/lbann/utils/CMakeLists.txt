# Add the headers for this directory
target_sources(lbann_cxx PRIVATE
  any.hpp
  argument_parser.hpp
  compiler_control.hpp
  dataset.hpp
  description.hpp
  entrywise_operator.hpp
  enum_iterator.hpp
  environment_variable.hpp
  exception.hpp
  factory.hpp
  factory_error_policies.hpp
  file_utils.hpp
  glob.hpp
  im2col.hpp
  image.hpp
  jag_utils.hpp
  lbann_library.hpp
  mild_exception.hpp
  number_theory.hpp
  nvshmem.hpp
  omp_diagnostics.hpp
  opencv.hpp
  options.hpp
  profiling.hpp
  python.hpp
  random.hpp
  serialization.hpp
  statistics.hpp
  summary.hpp
  summary_impl.hpp
  timer.hpp
  trainer_file_utils.hpp
  type_erased_matrix.hpp
  typename.hpp
  )

if (LBANN_HAS_HALF)
target_sources(lbann_cxx PRIVATE
  serialization.hpp
  )
endif (LBANN_HAS_HALF)

if (LBANN_HAS_DISTCONV)
  target_sources(lbann_cxx PRIVATE
    distconv.hpp)
endif ()

# Add the subdirectories
add_subdirectory(threads)
add_subdirectory(impl)
