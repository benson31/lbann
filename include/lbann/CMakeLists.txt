# Add the headers for this directory
target_sources(lbann_cxx PRIVATE
  Elemental_extensions.hpp
  base.hpp
  comm.hpp
  lbann.hpp
  )

# Add the subdirectories
add_subdirectory(callbacks)
add_subdirectory(data_coordinator)
add_subdirectory(data_readers)
add_subdirectory(data_store)
add_subdirectory(execution_contexts)
add_subdirectory(io)
add_subdirectory(layers)
add_subdirectory(macros)
add_subdirectory(metrics)
add_subdirectory(models)
add_subdirectory(objective_functions)
add_subdirectory(optimizers)
add_subdirectory(proto)
add_subdirectory(trainers)
add_subdirectory(training_algorithms)
add_subdirectory(transforms)
add_subdirectory(utils)
add_subdirectory(weights)
