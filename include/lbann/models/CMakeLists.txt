# Add the headers for this directory
set_source_path(THIS_DIR_HEADERS
  directed_acyclic_graph.hpp
  model.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
