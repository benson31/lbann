////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_UTILS_HIP_HPP_INCLUDED
#define LBANN_UTILS_HIP_HPP_INCLUDED

#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>

namespace lbann {
namespace hip {

// Atomic add
template <typename T> __device__ __forceinline__
T atomic_add(T* address, T val);

/** @brief Sum over threads in HIP block
 *
 *  Every thread in a HIP block must enter this function. The sum is
 *  returned on thread 0.
 *
 *  @tparam bdimx   x-dimension of HIP block
 *  @tparam bdimy   y-dimension of HIP block
 *  @tparam bdimz   z-dimension of HIP block
 *  @tparam T       Data type
 *  @param  val     Contribution from thread
 *  @returns On thread 0, the sum. Not meaningful on other threads.
 */
template <size_t bdimx, size_t bdimy, size_t bdimz, class T>
__device__ __forceinline__
T block_reduce(T val);

/** @brief Reduction over threads in HIP block
 *
 *  Every thread in a HIP block must enter this function. The reduced
 *  value is returned on thread 0.
 *
 *  @tparam bdimx   x-dimension of HIP block
 *  @tparam bdimy   y-dimension of HIP block
 *  @tparam bdimz   z-dimension of HIP block
 *  @tparam T       Data type
 *  @tparam Op      Functor for reduction operation
 *  @param  val     Contribution from each thread
 *  @returns On thread 0, the reduced value. Not meaningful on other
 *  threads.
 */
template <size_t bdimx, size_t bdimy, size_t bdimz, class T, class Op>
__device__ __forceinline__
T block_reduce(T val);

// Unary predicates
template <typename T> __forceinline__ __device__ bool isfinite(T const& x);
template <typename T> __forceinline__ __device__ bool isnan(T const& x);

// Nullary operators
template <typename T> __forceinline__ __device__ T infinity() { return T(1); }
template <typename T> constexpr __forceinline__ __device__ T epsilon() { return T(1); }
template <typename T> constexpr __forceinline__ __device__ T max() { return T(1); }
template <typename T> constexpr __forceinline__ __device__ T min() { return T(1); }

// Unary operators
template <typename T> __forceinline__ __device__ T abs(T const& x);
template <typename T> __forceinline__ __device__ T acos(T const& x);
template <typename T> __forceinline__ __device__ T acosh(T const& x);
template <typename T> __forceinline__ __device__ T asin(T const& x);
template <typename T> __forceinline__ __device__ T asinh(T const& x);
template <typename T> __forceinline__ __device__ T atan(T const& x);
template <typename T> __forceinline__ __device__ T atanh(T const& x);
template <typename T> __forceinline__ __device__ T ceil(T const& x);
template <typename T> __forceinline__ __device__ T cos(T const& x);
template <typename T> __forceinline__ __device__ T cosh(T const& x);
template <typename T> __forceinline__ __device__ T exp(T const& x);
template <typename T> __forceinline__ __device__ T expm1(T const& x);
template <typename T> __forceinline__ __device__ T floor(T const& x);
template <typename T> __forceinline__ __device__ T log(T const& x);
template <typename T> __forceinline__ __device__ T log1p(T const& x);
template <typename T> __forceinline__ __device__ T round(T const& x);
template <typename T> __forceinline__ __device__ T rsqrt(T const& x);
template <typename T> __forceinline__ __device__ T sin(T const& x);
template <typename T> __forceinline__ __device__ T sinh(T const& x);
template <typename T> __forceinline__ __device__ T sqrt(T const& x);
template <typename T> __forceinline__ __device__ T tan(T const& x);
template <typename T> __forceinline__ __device__ T tanh(T const& x);

// Binary operators
template <typename T> __forceinline__ __device__ T max(T const& x, T const& y);
template <typename T> __forceinline__ __device__ T min(T const& x, T const& y);
template <typename T>
__forceinline__ __device__ T mod(T const& x, T const& y); // x % y
template <typename T>
__forceinline__ __device__ T pow(T const& x, T const& y); // x ^ y


/*
template <typename T> __forceinline__ __device__ T (T const& x);
template <typename T> __forceinline__ __device__ T (T const& x);
template <typename T> __forceinline__ __device__ T (T const& x);
template <typename T> __forceinline__ __device__ T (T const& x);
template <typename T> __forceinline__ __device__ T (T const& x);
template <typename T> __forceinline__ __device__ T (T const& x);
*/

/** Apply an entry-wise unary operator to GPU data.
 *  The input and output data must be on GPU and must have the same
 *  dimensions.
 */
template <template <typename> class UnaryOperator, typename TensorDataType>
void apply_entrywise_unary_operator(
  const El::AbstractMatrix<TensorDataType>& input,
  El::AbstractMatrix<TensorDataType>& output);

/** Apply an entry-wise binary operator to GPU data.
 *  The input and output data must be on GPU and must have the same
 *  dimensions.
 */
template <template <typename> class BinaryOperator, typename TensorDataType>
void apply_entrywise_binary_operator(
  const El::AbstractMatrix<TensorDataType>& input1,
  const El::AbstractMatrix<TensorDataType>& input2,
  El::AbstractMatrix<TensorDataType>& output);


/** Apply an entry-wise unary operator to GPU data.
 *  The input and output data must be on GPU, have the same
 *  dimensions, and be aligned.
 */
template <template <typename> class UnaryOperator, typename TensorDataType>
void apply_entrywise_unary_operator(
  const El::AbstractDistMatrix<TensorDataType>& input,
  El::AbstractDistMatrix<TensorDataType>& output);

/** Apply an entry-wise binary operator to GPU data.
 *  The input and output data must be on GPU, have the same
 *  dimensions, and be aligned.
 */
template <template <typename> class BinaryOperator, typename TensorDataType>
void apply_entrywise_binary_operator(
  const El::AbstractDistMatrix<TensorDataType>& input1,
  const El::AbstractDistMatrix<TensorDataType>& input2,
  El::AbstractDistMatrix<TensorDataType>& output);

/** @brief Array with fixed type and size. */
template <typename T, size_t N>
struct array
{
  T vals[N];
  __host__ __device__ __forceinline__ size_t size() const noexcept
  {
    return N;
  }
  __host__ __device__ __forceinline__ T& operator[](size_t i) noexcept
  {
    return vals[i];
  }
  __host__ __device__ __forceinline__ const T& operator[](size_t i) const noexcept
  {
    return vals[i];
  }
};

namespace thrust {

/** @brief Thrust execution policy. */
using execute_on_stream = ::thrust::hip_rocprim::execute_on_stream;

/** GPU memory allocator that can interact with Thrust.
 *  Operations are performed on a provided HIP stream. Uses
 *  Hydrogen's CUB memory pool if available.
 */
template <typename T = El::byte>
class allocator
  : public ::thrust::detail::tagged_allocator<
               T, execute_on_stream,
               ::thrust::pointer<T, execute_on_stream>> {
public:
  // Convenient typedefs
  typedef ::thrust::detail::tagged_allocator<
              T, execute_on_stream,
              ::thrust::pointer<T, execute_on_stream>> parent_class;
  typedef typename parent_class::value_type  value_type;
  typedef typename parent_class::pointer     pointer;
  typedef typename parent_class::size_type   size_type;
  typedef typename parent_class::system_type system_type;

  /** Default constructor. */
  allocator(hipStream_t stream = El::rocm::GetDefaultStream());
  /** Allocate GPU buffer. */
  pointer allocate(size_type size);
  /** Deallocate GPU buffer.
   *  'size' is unused and maintained for compatibility with Thrust.
   */
  void deallocate(pointer buffer, size_type size = 0);
  /** Get Thrust execution policy. */
  system_type& system();

private:
  /** Active HIP stream. */
  hipStream_t m_stream;
  /** Thrust execution policy. */
  system_type m_system;

};

/** Thrust device vector. */
template <typename T>
using vector = ::thrust::device_vector<T, allocator<T>>;

} // namespace thrust
}// namespace hip
}// namespace lbann
#endif // LBANN_UTILS_HIP_HPP_INCLUDED
