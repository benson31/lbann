# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  bernoulli.cpp
  categorical_random.cpp
  concatenate.cpp
  concatenate_builder.cpp
  constant.cpp
  crop.cpp
  crop_builder.cpp
  discrete_random.cpp
  dummy.cpp
  evaluation.cpp
  gaussian.cpp
  hadamard.cpp
  in_top_k.cpp
  pooling.cpp
  reduction.cpp
  reshape.cpp
  slice.cpp
  sort.cpp
  split.cpp
  stop_gradient.cpp
  sum.cpp
  tessellate.cpp
  uniform.cpp
  unpooling.cpp
  weighted_sum.cpp
  weights.cpp
  )

if (LBANN_HAS_GPU)
  # Add the GPU source files for this directory
  target_sources(lbann_gpu_srcs INTERFACE
    concatenate.cu
    crop.cu
    in_top_k.cu
    sort.cu
    slice.cu
    tessellate.cu
    split.cu
    sum.cu
    )
endif ()
