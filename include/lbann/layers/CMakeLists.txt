# Add the headers for this directory
target_sources(lbann_cxx PRIVATE
  data_type_layer.hpp
  layer.hpp
  )

if (LBANN_HAS_DISTCONV)
  target_sources(lbann_cxx PRIVATE
    data_type_distconv_adapter.hpp
    distconv_adapter.hpp
    )
endif ()

# Add the subdirectories
add_subdirectory(activations)
add_subdirectory(image)
add_subdirectory(io)
add_subdirectory(learning)
add_subdirectory(loss)
add_subdirectory(math)
add_subdirectory(misc)
add_subdirectory(regularizers)
add_subdirectory(transform)
