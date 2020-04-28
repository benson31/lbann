add_subdirectory(data_buffers)

# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  file_io.cpp
  persist.cpp
  )
