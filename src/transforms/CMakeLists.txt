# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  normalize.cpp
  repack_HWC_to_CHW_layout.cpp
  sample_normalize.cpp
  scale.cpp
  scale_and_translate.cpp
  transform_pipeline.cpp
  )

add_subdirectory(vision)
