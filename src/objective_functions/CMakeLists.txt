# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  objective_function.cpp
  objective_function_term.cpp
  layer_term.cpp
  )

# Add the subdirectories
add_subdirectory(weight_regularization)
