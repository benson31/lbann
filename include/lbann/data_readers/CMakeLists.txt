# Add the headers for this directory
target_sources(lbann_cxx PRIVATE
  compound_data_reader.hpp
  data_reader.hpp
  data_reader_cifar10.hpp
  data_reader_csv.hpp
  data_reader_image.hpp
  data_reader_imagenet.hpp
  data_reader_merge_features.hpp
  data_reader_merge_samples.hpp
  data_reader_mnist.hpp
  data_reader_nci.hpp
  data_reader_numpy.hpp
  data_reader_numpy_npz.hpp
  data_reader_numpy_npz_conduit.hpp
  data_reader_pilot2_molecular.hpp
  data_reader_python.hpp
  data_reader_synthetic.hpp
  data_reader_smiles.hpp
  )
