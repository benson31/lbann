////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/learning/fully_connected.hpp"

namespace lbann {

template <>
void fully_connected_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
  ::setup_matrices(const El::Grid& grid) {
  learning_layer::setup_matrices(grid);
  deallocate_matrices();
  m_linearity_gradient = new MCMRMat<El::Device::CPU>(grid);
  m_bias_gradient = new MCStarMat<El::Device::CPU>(grid);
}

template <>
void fully_connected_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
  ::setup_matrices(const El::Grid& grid) {
  learning_layer::setup_matrices(grid);
  deallocate_matrices();
  m_linearity_gradient = new StarMat<El::Device::CPU>(grid);
  m_bias_gradient = new StarMat<El::Device::CPU>(grid);
}

template <>
void fully_connected_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
  ::setup_matrices(const El::Grid& grid) {
  learning_layer::setup_matrices(grid);
  deallocate_matrices();
  m_linearity_gradient = new MCMRMat<El::Device::GPU>(grid);
  m_bias_gradient = new MCStarMat<El::Device::GPU>(grid);
}

template <>
void fully_connected_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
  ::setup_matrices(const El::Grid& grid) {
  learning_layer::setup_matrices(grid);
  deallocate_matrices();
  m_linearity_gradient = new StarMat<El::Device::GPU>(grid);
  m_bias_gradient = new StarMat<El::Device::GPU>(grid);
}

/** CPU implementation of forward prop computation. */
template <>
void fully_connected_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::fp_compute() {

  // Matrices
  const auto& input = get_prev_activations();
  auto& output = get_activations();

  // Apply linearity
  // Note: Perform GEMMs independently if possible
  const auto& linearity = m_weights[0]->get_values();
  if (linearity.DistSize() == 1) {
    El::Gemm(El::NORMAL, El::NORMAL,
             DataType(1), linearity.LockedMatrix(), input.LockedMatrix(),
             DataType(0), output.Matrix());
  } else {
    El::Gemm(El::NORMAL, El::NORMAL,
             DataType(1), linearity, input,
             DataType(0), output);
  }

  // Apply bias if needed
  if(m_bias_scaling_factor != DataType(0)) {
    const auto& local_bias = m_weights[1]->get_values().LockedMatrix();
    auto& local_output = output.Matrix();
    El::IndexDependentMap(local_output,
                          (std::function<DataType(El::Int,El::Int,const DataType&)>)
                          ([this,&local_bias](El::Int r, El::Int c,const DataType& z)
                           ->DataType {
                            return z + m_bias_scaling_factor * local_bias(r, 0);
                          }));
  }

}

/** CPU implementation of backward prop computation. */
template <>
void fully_connected_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::bp_compute() {

  // Effective mini-batch size
  const int mini_batch_size = this->m_model->get_effective_mini_batch_size();

  // Matrices
  const auto& linearity = m_weights[0]->get_values();
  const auto& input = get_prev_activations();
  const auto& gradient_wrt_output = get_prev_error_signals();
  auto& gradient_wrt_input = get_error_signals();
  const auto& local_linearity = linearity.LockedMatrix();
  const auto& local_input = input.LockedMatrix();
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();

  // Compute gradient w.r.t. bias if needed
  optimizer* bias_optimizer = this->m_weights[1]->get_optimizer();
  if (m_bias_scaling_factor != DataType(0)
      && bias_optimizer != nullptr) {
    El::RowSum(local_gradient_wrt_output,
               m_bias_gradient->Matrix());
    bias_optimizer->add_to_gradient_staging(
      *m_bias_gradient,
      m_bias_scaling_factor / mini_batch_size);
  }

  // Compute gradient w.r.t. linearity if needed
  // Note: Perform GEMMs independently if possible
  optimizer* linearity_optimizer = this->m_weights[0]->get_optimizer();
  if (linearity_optimizer != nullptr) {
    if (linearity.DistSize() == 1) {
      El::Gemm(El::NORMAL, El::TRANSPOSE,
               DataType(1), local_gradient_wrt_output, local_input,
               DataType(0), m_linearity_gradient->Matrix());
      linearity_optimizer->add_to_gradient_staging(
        *m_linearity_gradient,
        DataType(1) / mini_batch_size);
    } else {
      El::Gemm(El::NORMAL, El::TRANSPOSE,
               DataType(1), gradient_wrt_output, input,
               DataType(0), *m_linearity_gradient);
      linearity_optimizer->add_to_gradient(
        *m_linearity_gradient,
        DataType(1) / mini_batch_size);
    }
  }

  // Compute gradient w.r.t. input
  // Note: Perform GEMMs independently if possible
  if (linearity.DistSize() == 1) {
    El::Gemm(El::TRANSPOSE, El::NORMAL,
             DataType(1), local_linearity, local_gradient_wrt_output,
             DataType(1), local_gradient_wrt_input);
  } else {
    El::Gemm(El::TRANSPOSE, El::NORMAL,
             DataType(1), linearity, gradient_wrt_output,
             DataType(1), gradient_wrt_input);
  }

}

/** CPU implementation of forward prop computation. */
template <>
void fully_connected_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {

  // Matrices
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();

  // Apply linearity
  const auto& local_linearity = m_weights[0]->get_values().LockedMatrix();
  El::Gemm(El::NORMAL, El::NORMAL,
           DataType(1), local_linearity, local_input,
           DataType(0), local_output);

  // Apply bias if needed
  if(m_bias_scaling_factor != DataType(0)) {
    const auto& local_bias = m_weights[1]->get_values().LockedMatrix();
    El::IndexDependentMap(local_output,
                          (std::function<DataType(El::Int,El::Int,const DataType&)>)
                          ([this,&local_bias](El::Int r, El::Int c,const DataType& z)
                           ->DataType {
                            return z + m_bias_scaling_factor * local_bias(r, 0);
                          }));
  }

}

/** CPU implementation of backward prop computation. */
template <>
void fully_connected_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::bp_compute() {

  // Effective mini-batch size
  const int mini_batch_size = this->m_model->get_effective_mini_batch_size();

  // Matrices
  const auto& local_linearity = m_weights[0]->get_values().LockedMatrix();
  const auto& local_input = get_local_prev_activations();
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();

  // Compute gradient w.r.t. bias if needed
  optimizer* bias_optimizer = this->m_weights[1]->get_optimizer();
  if (m_bias_scaling_factor != DataType(0)
      && bias_optimizer != nullptr) {
    El::RowSum(local_gradient_wrt_output,
               m_bias_gradient->Matrix());
    bias_optimizer->add_to_gradient_staging(
      *m_bias_gradient,
      m_bias_scaling_factor / mini_batch_size);
  }

  // Compute gradient w.r.t. linearity if needed
  optimizer* linearity_optimizer = this->m_weights[0]->get_optimizer();
  if (linearity_optimizer != nullptr) {
    El::Gemm(El::NORMAL, El::TRANSPOSE,
             DataType(1), local_gradient_wrt_output, local_input,
             DataType(0), m_linearity_gradient->Matrix());
    linearity_optimizer->add_to_gradient_staging(
      *m_linearity_gradient,
      DataType(1) / mini_batch_size);
  }

  // Compute gradient w.r.t. input
  El::Gemm(El::TRANSPOSE, El::NORMAL,
           DataType(1), local_linearity, local_gradient_wrt_output,
           DataType(1), local_gradient_wrt_input);

}

/** GPU implementation of forward prop computation. */
template <>
void fully_connected_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
#ifndef LBANN_HAS_CUDNN
  throw lbann_exception("fully_connected: CUDA not detected");
#else

  // GPU matrices
  const auto& linearity_d = m_weights[0]->get_values_gpu();
  const auto& input_d = this->m_prev_activations_d[0];
  auto& output_d = this->m_activations_d[0];

  // Matrix parameters
  const int input_size = get_num_prev_neurons();
  const int output_size = get_num_neurons();
  const int mini_batch_size = m_mini_batch_size_per_gpu;
  const int num_gpus = this->m_cudnn->get_num_gpus();
  const int input_ldim = input_d.get_leading_dim();
  const int output_ldim = output_d.get_leading_dim();

  // Stop early if possible
  if (mini_batch_size == 0) { return; }

  // Apply linearity
  for (int i=0; i<num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
    cublas::gemm(this->m_cudnn->get_cublas_handle(i),
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 output_size, mini_batch_size, input_size,
                 DataType(1),
                 linearity_d[i], output_size,
                 input_d.get_locked_data(i), input_ldim,
                 DataType(0),
                 output_d.get_data(i), output_ldim);
  }

  // Apply bias if needed
  if(m_bias_scaling_factor != DataType(0)) {
    const auto& bias_d = m_weights[1]->get_values_gpu();

    // Initialize work space with ones
    cudnn::matrix ones_d(this->m_cudnn);
    ones_d.attach_to_work_spaces(mini_batch_size);
    m_cudnn->set_on_gpus(ones_d.get_data(), DataType(1), mini_batch_size);

    // Apply bias with outer product
    for (int i = 0; i < num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      cublas::gemm(this->m_cudnn->get_cublas_handle(i),
                   CUBLAS_OP_N, CUBLAS_OP_T,
                   output_size, mini_batch_size, 1,
                   DataType(1),
                   bias_d[i], output_size,
                   ones_d.get_data(i), mini_batch_size,
                   DataType(1),
                   output_d.get_data(i), output_ldim);
    }

  }
#endif // LBANN_HAS_CUDNN
}

/** GPU implementation of backward prop computation. */
template <>
void fully_connected_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
#ifndef LBANN_HAS_CUDNN
  throw lbann_exception("fully_connected: CUDA not detected");
#else

  // GPU matrices
  const auto& linearity_d = m_weights[0]->get_values_gpu();
  const auto& input_d = this->m_prev_activations_d[0];
  const auto& gradient_wrt_output_d = this->m_prev_error_signals_d[0];
  auto& gradient_wrt_input_d = this->m_error_signals_d[0];

  // Matrix parameters
  const int input_size = get_num_prev_neurons();
  const int output_size = get_num_neurons();
  const int mini_batch_size = m_mini_batch_size_per_gpu;
  const int num_gpus = this->m_cudnn->get_num_gpus();
  const int input_ldim = input_d.get_leading_dim();
  const int gradient_wrt_output_ldim = gradient_wrt_output_d.get_leading_dim();
  const int gradient_wrt_input_ldim = gradient_wrt_input_d.get_leading_dim();

  // Compute gradient w.r.t. bias if needed
  optimizer* bias_optimizer = this->m_weights[1]->get_optimizer();
  if (m_bias_scaling_factor != DataType(0)
      && bias_optimizer != nullptr) {

    // Initialize work space with ones
    cudnn::matrix ones_d(this->m_cudnn);
    ones_d.attach_to_work_spaces(mini_batch_size);
    m_cudnn->set_on_gpus(ones_d.get_data(), DataType(1), mini_batch_size);

    // Obtain gradient with a sum over rows
    for (int i = 0; i < num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      cublas::gemv(this->m_cudnn->get_cublas_handle(i),
                   CUBLAS_OP_N,
                   output_size, mini_batch_size,
                   DataType(1),
                   gradient_wrt_output_d.get_locked_data(i), gradient_wrt_output_ldim,
                   ones_d.get_data(i), 1,
                   DataType(0),
                   m_bias_gradient_d.get_data(i), 1);
    }
    bias_optimizer->add_to_gradient_staging(
                                            m_bias_gradient_d,
                                            m_bias_scaling_factor / this->m_model->get_effective_mini_batch_size());
  }

  // Compute gradient w.r.t. linearity if needed
  optimizer* linearity_optimizer = this->m_weights[0]->get_optimizer();
  if (linearity_optimizer != nullptr) {
    for (int i = 0; i < num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      cublas::gemm(this->m_cudnn->get_cublas_handle(i),
                   CUBLAS_OP_N, CUBLAS_OP_T,
                   output_size, input_size, mini_batch_size,
                   DataType(1),
                   gradient_wrt_output_d.get_locked_data(i), gradient_wrt_output_ldim,
                   input_d.get_locked_data(i), input_ldim,
                   DataType(0),
                   m_linearity_gradient_d.get_data(i), output_size);
    }
    linearity_optimizer->add_to_gradient_staging(
                                                 m_linearity_gradient_d,
                                                 DataType(1) / this->m_model->get_effective_mini_batch_size());
  }

  // Compute gradient w.r.t. input
  if (mini_batch_size != 0) {
    for (int i = 0; i < num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      cublas::gemm(this->m_cudnn->get_cublas_handle(i),
                   CUBLAS_OP_T, CUBLAS_OP_N,
                   input_size, mini_batch_size, output_size,
                   DataType(1),
                   linearity_d[i], output_size,
                   gradient_wrt_output_d.get_locked_data(i), gradient_wrt_output_ldim,
                   DataType(1),
                   gradient_wrt_input_d.get_data(i), gradient_wrt_input_ldim);
    }
  }

#endif // LBANN_HAS_CUDNN
}

template <>
void fully_connected_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::fp_compute() {
#ifndef LBANN_HAS_CUDNN
  throw lbann_exception("fully_connected: CUDA not detected");
#else
#endif // LBANN_HAS_CUDNN
}

template <>
void fully_connected_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {
#ifndef LBANN_HAS_CUDNN
  throw lbann_exception("fully_connected: CUDA not detected");
#else
#endif // LBANN_HAS_CUDNN
}

} // namespace lbann
