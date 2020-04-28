# Add the headers for this directory
target_sources(lbann_cxx PRIVATE
  objective_function.hpp
  objective_function_term.hpp
  layer_term.hpp
  )

# Add the subdirectories
add_subdirectory(weight_regularization)
