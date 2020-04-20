# Add the headers for this directory
set_source_path(THIS_DIR_HEADERS
  instantiate.hpp
  instantiate_device.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
