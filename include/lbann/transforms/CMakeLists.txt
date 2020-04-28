# Add the headers for this directory
target_sources(lbann_cxx PRIVATE
  normalize.hpp
  repack_HWC_to_CHW_layout.hpp
  sample_normalize.hpp
  scale.hpp
  scale_and_translate.hpp
  transform.hpp
  transform_pipeline.hpp
  )

add_subdirectory(vision)
