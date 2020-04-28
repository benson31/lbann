# Add the source files for this directory
target_sources(lbann_cxx PRIVATE
  callback_factory.cpp
  layer_factory.cpp
  layer_graph_factory.cpp
  model_factory.cpp
  objective_function_factory.cpp
  optimizer_factory.cpp
  trainer_factory.cpp
  transform_factory.cpp
  weights_factory.cpp
)
