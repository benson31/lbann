# Add the headers for this directory
target_sources(lbann_cxx PRIVATE
  batch_normalization.hpp
  dropout.hpp
  entrywise_batch_normalization.hpp
  layer_norm.hpp
  local_response_normalization.hpp
  regularizer.hpp
  selu_dropout.hpp
  )
