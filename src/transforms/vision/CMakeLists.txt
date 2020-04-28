# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  adjust_brightness.cpp
  adjust_contrast.cpp
  adjust_saturation.cpp
  center_crop.cpp
  colorize.cpp
  color_jitter.cpp
  cutout.cpp
  grayscale.cpp
  horizontal_flip.cpp
  normalize_to_lbann_layout.cpp
  random_affine.cpp
  random_crop.cpp
  random_resized_crop.cpp
  random_resized_crop_with_fixed_aspect_ratio.cpp
  resize.cpp
  resized_center_crop.cpp
  to_lbann_layout.cpp
  vertical_flip.cpp
  )
