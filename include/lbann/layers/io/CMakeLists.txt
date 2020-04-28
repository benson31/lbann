# Add the headers for this directory
target_sources(lbann_cxx PRIVATE
  io_layer.hpp
  )

# Add the subdirectories
add_subdirectory(input)
