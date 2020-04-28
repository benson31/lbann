# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  data_type_layer.cpp
  layer.cpp
  )

if (LBANN_HAS_DISTCONV)
  target_sources(lbann_cxx PRIVATE
    data_type_distconv_adapter.cpp
    distconv_adapter.cpp
    )
endif ()

# Add the subdirectories
add_subdirectory(activations)
add_subdirectory(image)
add_subdirectory(io/input)
add_subdirectory(learning)
add_subdirectory(loss)
add_subdirectory(math)
add_subdirectory(misc)
add_subdirectory(regularizers)
add_subdirectory(transform)
