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
//
// lbann_comm .hpp .cpp - LBANN communication utilities
////////////////////////////////////////////////////////////////////////////////

#define LBANN_COMM_INSTANTIATE
#include "lbann/comm_impl.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/gpu/helpers.hpp"
#include "lbann/utils/timer.hpp"
#include "mpi.h"
#include "omp.h"
#include <memory>
#include <sstream>
#include <thread>

namespace lbann {

// Error utility macro
#ifdef LBANN_DEBUG
#define checkMPI(mpi_call)                                                     \
  {                                                                            \
    const int status = mpi_call;                                               \
    if (status != MPI_SUCCESS) {                                               \
      char error_string[MPI_MAX_ERROR_STRING];                                 \
      int error_string_len;                                                    \
      MPI_Error_string(status, error_string, &error_string_len);               \
      std::cerr << "MPI error: "                                               \
                << std::string(error_string, error_string_len) << "\n"         \
                << "Error at " << __FILE__ << ":" << __LINE__ << "\n";         \
      throw lbann_exception("MPI error");                                      \
    }                                                                          \
  }
#else
#define checkMPI(status) status
#endif // #ifdef LBANN_DEBUG

namespace Al {
request::request() : impl_{std::make_unique<request_impl>()} {}
request::~request() noexcept {}
request::request(request const&) : request{} {}
} // namespace Al

lbann_comm::lbann_comm(int ppm, El::mpi::Comm world)
  : m_world_comm(std::move(world)), m_grid(nullptr), m_procs_per_trainer(ppm),
    m_num_trainer_barriers(0), m_num_intertrainer_barriers(0),
    m_num_global_barriers(0), m_bytes_sent(0), m_bytes_received(0)
{
#ifdef LBANN_HAS_ALUMINUM
  // Don't have argc/argv here, but MPI should already be init'd.
  int argc_dummy = 0;
  char** argv_dummy = nullptr;
  ::Al::Initialize(argc_dummy, argv_dummy);
#endif
  // Set up the initial trainer split
  split_trainers(m_procs_per_trainer);

  // Initialize node communicators
  setup_node_comm();
  m_procs_per_node = El::mpi::Size(m_node_comm);
  m_rank_in_node = El::mpi::Rank(m_node_comm);

  // Setup threads
  setup_threads();
}

lbann_comm::~lbann_comm()
{
  delete m_grid;
  El::mpi::Free(m_trainer_comm);
  El::mpi::Free(m_intertrainer_comm);
  El::mpi::Free(m_node_comm);
#ifdef LBANN_HAS_ALUMINUM
  ::Al::Finalize();
#endif
}

void lbann_comm::split_trainers(const int ppm)
{
  int world_size = El::mpi::Size(get_world_comm());
  m_procs_per_trainer = ppm;
  if (ppm == 0) {
    m_procs_per_trainer = world_size;
  }
  // Check if parameters are valid
  if (m_procs_per_trainer > world_size) {
    LBANN_ERROR(
      "Not enough processes to create one trainer; procs_per_trainer: ",
      m_procs_per_trainer,
      " is larger than world_size: ",
      world_size);
  }
  if (world_size % m_procs_per_trainer != 0) {
    LBANN_ERROR("Procs per trainer does not divide total number of procs; "
                "procs_per_trainer: ",
                m_procs_per_trainer,
                " total number of procs (world size): ",
                world_size);
  }

  m_num_trainers = world_size / m_procs_per_trainer;
  m_trainer_rank = El::mpi::Rank(get_world_comm()) / m_procs_per_trainer;
  m_rank_in_trainer = El::mpi::Rank(get_world_comm()) % m_procs_per_trainer;

  // Initialize trainer and intertrainer communicators
  El::mpi::Split(get_world_comm(),
                 m_trainer_rank,
                 m_rank_in_trainer,
                 m_trainer_comm);
  El::mpi::Split(get_world_comm(),
                 m_rank_in_trainer,
                 m_trainer_rank,
                 m_intertrainer_comm);

  // Initialize Elemental grid
  if (m_grid != nullptr) {
    delete m_grid;
  }
  m_grid = new Grid(m_trainer_comm.GetMPIComm());
}

void lbann_comm::intertrainer_sum_matrix(AbsMat& mat) const
{
  m_bytes_sent += sizeof(DataType) * mat.Height() * mat.Width();
  El::AllReduce(mat, m_intertrainer_comm, El::mpi::SUM);
  m_bytes_received += sizeof(DataType) * mat.Height() * mat.Width();
}

void lbann_comm::intertrainer_sum_matrix(AbsDistMat& mat) const
{
  allreduce(mat, m_intertrainer_comm, El::mpi::SUM);
}

namespace {

template <typename BackendT> struct BackendTag
{
};

#if defined(LBANN_HAS_GPU) && defined(LBANN_HAS_ALUMINUM)
// This will set the variant to use the given request type.
template <typename BackendT>
auto GetRequest(Al::request& r, BackendTag<BackendT>) ->
  typename BackendT::req_type&
{
  using RequestT = typename BackendT::req_type;
  return r().m_req.emplace<RequestT>();
}

void UpdateRequest(typename ::Al::MPIBackend::req_type&,
                   El::SyncInfo<El::Device::CPU> const&) noexcept
{}

#ifdef AL_HAS_NCCL
void UpdateRequest(typename ::Al::NCCLBackend::req_type& req,
                   El::SyncInfo<El::Device::GPU> const& si) noexcept
{
  if (req)
    req->orig_stream = si.Stream();
}
#endif // AL_HAS_NCCL

#ifdef AL_HAS_MPI_CUDA
void UpdateRequest(typename ::Al::MPICUDABackend::req_type& req,
                   El::SyncInfo<El::Device::GPU> const& si) noexcept
{
  if (req)
    req->orig_stream = si.Stream();
}
#endif // AL_HAS_MPI_CUDA
#ifdef AL_HAS_HOST_TRANSFER
void UpdateRequest(typename ::Al::HostTransferBackend::req_type& req,
                   El::SyncInfo<El::Device::GPU> const& si) noexcept
{
  if (req)
    req->orig_stream = si.Stream();
}
#endif // AL_HAS_MPI_CUDA
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_HAS_ALUMINUM)

// The best we can do on CPU is exactly the Elemental implementation:
// If the buffer is contiguous, call the El::mpi interface, which will
// dispatch to Aluminum if possible for the type; otherwise,
// pack-allreduce-unpack.
//
// Likewise, if we don't have Aluminum, this is the best we can do on GPU.
//
// If we DO have Aluminum, the compiler should select that overload
// for GPUs as it is "more specialized" than this template. If that's
// not what's happening, there's a compiler bug.
template <typename T, El::Device D>
void allreduce_impl(El::Matrix<T, D>& m,
                    const El::mpi::Comm& c,
                    El::mpi::Op const& op)
{
  return El::AllReduce(m, c, op);
}

template <typename T>
void nb_allreduce_impl(El::Matrix<T, El::Device::CPU>& m,
                       const El::mpi::Comm& c,
                       Al::request& req_wrapper,
                       El::mpi::Op const& op)
{
  if (m.Height() == m.LDim() || m.Width() == 1) {
    auto& req = req_wrapper().m_req.emplace<MPI_Request>(MPI_REQUEST_NULL);
    auto const count = m.Height() * m.Width();
    MPI_Iallreduce(MPI_IN_PLACE,
                   m.Buffer(),
                   count,
                   El::mpi::TypeMap<T>(),
                   op.op,
                   c.GetMPIComm(),
                   &(req));
  }
  else {
    return El::AllReduce(m, c, op);
  }
}

#if defined(LBANN_HAS_GPU) && defined(LBANN_HAS_ALUMINUM)

template <typename T,
          typename BackendT,
          El::EnableWhen<
            El::AluminumSupportsBackendAndCollective<T,
                                                     El::Collective::ALLREDUCE,
                                                     BackendT>,
            int> = 0>
void allreduce_aluminum(El::Matrix<T, El::Device::GPU>& m,
                        const El::mpi::Comm& c,
                        El::mpi::Op const& op,
                        BackendTag<BackendT>,
                        typename BackendT::allreduce_algo_type algo =
                          BackendT::allreduce_algo_type::automatic)
{
  const auto local_size = m.Height() * m.Width();
  ::Al::Allreduce<BackendT>(
    m.Buffer(),
    local_size,
    mpi_op_to_al_op(op),
    c.template GetComm<BackendT>(El::SyncInfoFromMatrix(m)),
    algo);
}

template <typename T,
          typename BackendT,
          El::EnableWhen<
            El::AluminumSupportsBackendAndCollective<T,
                                                     El::Collective::ALLREDUCE,
                                                     BackendT>,
            int> = 0>
void nb_allreduce_aluminum(El::Matrix<T, El::Device::GPU>& m,
                           const El::mpi::Comm& c,
                           Al::request& req,
                           El::mpi::Op const& op,
                           BackendTag<BackendT> const& tag,
                           typename BackendT::allreduce_algo_type algo =
                             BackendT::allreduce_algo_type::automatic)
{
  const auto local_size = m.Height() * m.Width();
  const auto& syncinfo = El::SyncInfoFromMatrix(m);
  auto& request = GetRequest(req, tag);
  ::Al::NonblockingAllreduce<BackendT>(m.Buffer(),
                                       local_size,
                                       mpi_op_to_al_op(op),
                                       c.template GetComm<BackendT>(syncinfo),
                                       request,
                                       algo);
  UpdateRequest(request, syncinfo);
}

template <typename T,
          typename BackendT,
          El::EnableUnless<
            El::AluminumSupportsBackendAndCollective<T,
                                                     El::Collective::ALLREDUCE,
                                                     BackendT>,
            int> = 0>
void nb_allreduce_aluminum(El::Matrix<T, El::Device::GPU>& m,
                           const El::mpi::Comm& c,
                           Al::request& req,
                           El::mpi::Op const& op,
                           BackendTag<BackendT> const& tag,
                           typename BackendT::allreduce_algo_type algo =
                             BackendT::allreduce_algo_type::automatic)
{
  El::AllReduce(m, c, op);
}

template <typename T,
          typename BackendT,
          El::EnableUnless<
            El::AluminumSupportsBackendAndCollective<T,
                                                     El::Collective::ALLREDUCE,
                                                     BackendT>,
            int> = 0>
void allreduce_aluminum(El::Matrix<T, El::Device::GPU>& m,
                        const El::mpi::Comm& c,
                        El::mpi::Op const& op,
                        BackendTag<BackendT>,
                        typename BackendT::allreduce_algo_type =
                          BackendT::allreduce_algo_type::automatic)
{
  // We cannot dispatch with this backend directly to Aluminum. Let
  // Elemental handle it.
  El::AllReduce(m, c, op);
}

template <typename T>
void allreduce_impl(El::Matrix<T, El::Device::GPU>& m,
                    El::mpi::Comm const& c,
                    El::mpi::Op const& op)
{
  return El::AllReduce(m, c, op);
}

template <typename T>
void nb_allreduce_impl(El::Matrix<T, El::Device::GPU>& m,
                       El::mpi::Comm const& c,
                       Al::request& req,
                       El::mpi::Op const& op)
{
  if (m.Width() > 1 && m.Height() != m.LDim()) {
    // Aluminum doesn't do allreduces on strided matrices
    return El::AllReduce(m, c, op);
  }

#if defined(AL_HAS_NCCL)
  return nb_allreduce_aluminum(m, c, req, op, BackendTag<::Al::NCCLBackend>{});
#elif defined(AL_HAS_MPI_CUDA)
  return nb_allreduce_aluminum(
    m,
    c,
    req,
    op,
    BackendTag<::Al::MPICUDABackend>{},
    ::Al::MPICUDABackend::allreduce_algo_type::host_transfer);
#else
  // At this point just call Elemental again
  return El::AllReduce(m, c, op);
#endif
}

#endif // defined(LBANN_HAS_GPU) && defined(LBANN_HAS_ALUMINUM)
} // namespace

template <typename TensorDataType>
void lbann_comm::allreduce(El::AbstractMatrix<TensorDataType>& m,
                           const El::mpi::Comm& c,
                           El::mpi::Op op) const
{
  if (El::mpi::Size(c) == 1 || m.Height() < 1 || m.Width() < 1) {
    return;
  }

  const int local_size = m.Height() * m.Width();
  m_bytes_sent += sizeof(DataType) * local_size;
  m_bytes_received += sizeof(DataType) * local_size * (El::mpi::Size(c) - 1);

  switch (m.GetDevice()) {
  case El::Device::CPU:
    return allreduce_impl(
      static_cast<El::Matrix<TensorDataType, El::Device::CPU>&>(m),
      c,
      op);
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return allreduce_impl(
      static_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(m),
      c,
      op);
#endif // LBANN_HAS_GPU
  }
}

template <typename TensorDataType>
void lbann_comm::allreduce(El::AbstractDistMatrix<TensorDataType>& m,
                           const El::mpi::Comm& c,
                           El::mpi::Op op) const
{
  allreduce(m.Matrix(), c, op);
}

template <typename TensorDataType>
void lbann_comm::nb_allreduce(El::AbstractMatrix<TensorDataType>& m,
                              const El::mpi::Comm& c,
                              Al::request& req,
                              El::mpi::Op op) const
{
  if (El::mpi::Size(c) == 1 || m.Height() < 1 || m.Width() < 1) {
    return;
  }

  const int local_size = m.Height() * m.Width();
  m_bytes_sent += sizeof(DataType) * local_size;
  m_bytes_received += sizeof(DataType) * local_size * (El::mpi::Size(c) - 1);

  switch (m.GetDevice()) {
  case El::Device::CPU:
    return nb_allreduce_impl(
      static_cast<El::Matrix<TensorDataType, El::Device::CPU>&>(m),
      c,
      req,
      op);
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return nb_allreduce_impl(
      static_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(m),
      c,
      req,
      op);
#endif // LBANN_HAS_GPU
  }
}

template <typename TensorDataType>
void lbann_comm::nb_allreduce(El::AbstractDistMatrix<TensorDataType>& m,
                              const El::mpi::Comm& c,
                              Al::request& req,
                              El::mpi::Op op) const
{
  nb_allreduce(m.Matrix(), c, req, op);
}

void lbann_comm::wait(Al::request& req) const
{
#ifdef LBANN_HAS_ALUMINUM
  using AlMPIRequestT = typename ::Al::MPIBackend::req_type;
  if (auto* mpi_backend_req = std::get_if<AlMPIRequestT>(&(req().m_req))) {
    ::Al::Wait<::Al::MPIBackend>(*mpi_backend_req);
    return;
  }
#ifdef AL_HAS_NCCL
  using AlNCCLRequestT = typename ::Al::NCCLBackend::req_type;
  if (auto* nccl_backend_req = std::get_if<AlNCCLRequestT>(&(req().m_req))) {
    // Note this does not block the host.
    ::Al::Wait<::Al::NCCLBackend>(*nccl_backend_req);
    return;
  }
#endif // AL_HAS_NCCL
#ifdef AL_HAS_MPI_CUDA
  using AlMPICUDARequestT = typename ::Al::MPICUDABackend::req_type;
  if (auto* mpicuda_backend_req =
        std::get_if<AlMPICUDARequestT>(&(req().m_req))) {
    // Note this does not block the host.
    ::Al::Wait<::Al::MPICUDABackend>(*mpicuda_backend_req);
    return;
  }
#endif // AL_HAS_MPI_CUDA
#ifdef AL_HAS_HOST_TRANSFER
  using AlHostXferRequestT = typename ::Al::HostTransferBackend::req_type;
  if (auto* host_xfer_backend_req =
        std::get_if<AlHostXferRequestT>(&(req().m_req))) {
    // Note this does not block the host.
    ::Al::Wait<::Al::HostTransferBackend>(*host_xfer_backend_req);
    return;
  }
#endif // AL_HAS_NCCL
#endif // LBANN_HAS_ALUMINUM
  if (auto* mpi_req = std::get_if<MPI_Request>(&(req().m_req))) {
    if (*mpi_req == MPI_REQUEST_NULL)
      LBANN_ERROR("Request is NULL");
    MPI_Wait(mpi_req, MPI_STATUS_IGNORE);
    return; // not needed, but symmetry and all that...
  }
}

bool lbann_comm::test(Al::request& req) const
{
#ifdef LBANN_HAS_ALUMINUM
  using AlMPIRequestT = typename ::Al::MPIBackend::req_type;
  if (auto* mpi_backend_req = std::get_if<AlMPIRequestT>(&(req().m_req))) {
    return ::Al::Test<::Al::MPIBackend>(*mpi_backend_req);
  }
#ifdef AL_HAS_NCCL
  using AlNCCLRequestT = typename ::Al::NCCLBackend::req_type;
  if (auto* nccl_backend_req = std::get_if<AlNCCLRequestT>(&(req().m_req))) {
    // Note this does not block the host.
    return ::Al::Test<::Al::NCCLBackend>(*nccl_backend_req);
  }
#endif // AL_HAS_NCCL
#ifdef AL_HAS_MPI_CUDA
  using AlMPICUDARequestT = typename ::Al::MPICUDABackend::req_type;
  if (auto* mpicuda_backend_req =
        std::get_if<AlMPICUDARequestT>(&(req().m_req))) {
    // Note this does not block the host.
    return ::Al::Test<::Al::MPICUDABackend>(*mpicuda_backend_req);
  }
#endif // AL_HAS_MPI_CUDA
#ifdef AL_HAS_HOST_TRANSFER
  using AlHostXferRequestT = typename ::Al::HostTransferBackend::req_type;
  if (auto* host_xfer_backend_req =
        std::get_if<AlHostXferRequestT>(&(req().m_req))) {
    // Note this does not block the host.
    return ::Al::Test<::Al::HostTransferBackend>(*host_xfer_backend_req);
  }
#endif // AL_HAS_NCCL
#endif // LBANN_HAS_ALUMINUM
  if (auto* mpi_req = std::get_if<MPI_Request>(&(req().m_req))) {
    int flag = -1;
    MPI_Test(mpi_req, &flag, MPI_STATUS_IGNORE);
    return (flag == 1 ? true : false);
  }
  return false;
}

void lbann_comm::intertrainer_broadcast_matrix(AbsMat& mat, int root) const
{
  El::Broadcast(mat, m_intertrainer_comm, root);
}

void lbann_comm::intertrainer_broadcast_matrix(AbsDistMat& mat, int root) const
{
  El::Broadcast(mat, m_intertrainer_comm, root);
}

template <>
void lbann_comm::broadcast<std::string>(const int root,
                                        std::string& str,
                                        const El::mpi::Comm& c) const
{
  std::vector<char> data(str.begin(), str.end());
  broadcast(root, data, c);
  str.assign(data.begin(), data.end());
}

void lbann_comm::intertrainer_barrier() const
{
  ++m_num_intertrainer_barriers;
  barrier(m_intertrainer_comm);
}

void lbann_comm::trainer_barrier() const
{
  ++m_num_trainer_barriers;
  barrier(m_trainer_comm);
}

void lbann_comm::global_barrier() const
{
  ++m_num_global_barriers;
  barrier(get_world_comm());
}

void lbann_comm::barrier(const El::mpi::Comm& c) const { El::mpi::Barrier(c); }

void lbann_comm::send(const AbsMat& mat,
                      const int trainer,
                      const int rank) const
{
  El::Send(mat, get_world_comm(), get_world_rank(trainer, rank));
}

void lbann_comm::send(const DistMat& mat,
                      const int trainer,
                      const int rank) const
{
  send(mat.LockedMatrix(), trainer, rank);
}

void lbann_comm::nb_send(const AbsMat& mat,
                         const int trainer,
                         const int rank,
                         El::mpi::Request<DataType>& req) const
{
  nb_send(mat.LockedBuffer(), mat.Height() * mat.Width(), trainer, rank, req);
}

void lbann_comm::nb_send(const DistMat& mat,
                         const int trainer,
                         const int rank,
                         El::mpi::Request<DataType>& req) const
{
  nb_send(mat.LockedBuffer(),
          mat.LocalHeight() * mat.LocalWidth(),
          trainer,
          rank,
          req);
}

void lbann_comm::recv(AbsMat& mat, const int trainer, const int rank) const
{
  El::Recv(mat, get_world_comm(), get_world_rank(trainer, rank));
}

void lbann_comm::recv(DistMat& mat, const int trainer, const int rank) const
{
  recv(mat.Matrix(), trainer, rank);
}

void lbann_comm::recv(AbsMat& mat) const
{
  El::Recv(mat, get_world_comm(), El::mpi::ANY_SOURCE);
}

void lbann_comm::recv(DistMat& mat) const { recv(mat.Matrix()); }

void lbann_comm::nb_recv(AbsMat& mat,
                         const int trainer,
                         const int rank,
                         El::mpi::Request<DataType>& req) const
{
  nb_recv(mat.Buffer(), mat.Height() * mat.Width(), trainer, rank, req);
}

void lbann_comm::nb_recv(DistMat& mat,
                         const int trainer,
                         const int rank,
                         El::mpi::Request<DataType>& req) const
{
  nb_recv(mat.Buffer(),
          mat.LocalHeight() * mat.LocalWidth(),
          trainer,
          rank,
          req);
}

void lbann_comm::nb_recv(AbsMat& mat, El::mpi::Request<DataType>& req) const
{
  nb_recv(mat.Buffer(), mat.Height() * mat.Width(), req);
}

void lbann_comm::nb_recv(DistMat& mat, El::mpi::Request<DataType>& req) const
{
  nb_recv(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth(), req);
}

void lbann_comm::setup_node_comm()
{

  // Get string specifying compute node
  char node_name[MPI_MAX_PROCESSOR_NAME];
  int node_name_len;
  checkMPI(MPI_Get_processor_name(node_name, &node_name_len));
  const std::string node_string(node_name);

  // Hash node names and split MPI processes
  int hash = std::hash<std::string>()(node_string);
  hash = hash >= 0 ? hash : -hash; // Make sure hash is non-negative
  El::mpi::Comm hash_comm;
  El::mpi::Split(get_world_comm(),
                 hash,
                 El::mpi::Rank(get_world_comm()),
                 hash_comm);
  const int hash_comm_size = El::mpi::Size(hash_comm);

  // Compare node names and split MPI processes
  int node_num = El::mpi::Rank(hash_comm);
  {
    std::vector<char> node_name_list(hash_comm_size * MPI_MAX_PROCESSOR_NAME);
    checkMPI(MPI_Allgather(node_name,
                           MPI_MAX_PROCESSOR_NAME,
                           MPI_CHAR,
                           node_name_list.data(),
                           MPI_MAX_PROCESSOR_NAME,
                           MPI_CHAR,
                           hash_comm.GetMPIComm()));
    for (int i = 0; i < hash_comm_size; ++i) {
      const std::string other_node_string(node_name_list.data() +
                                          i * MPI_MAX_PROCESSOR_NAME);
      if (node_string == other_node_string) {
        node_num = i;
        break;
      }
    }
  }
  El::mpi::Split(hash_comm,
                 node_num,
                 El::mpi::Rank(get_world_comm()),
                 m_node_comm);
  El::mpi::Free(hash_comm);

  // Set up list of ranks that are local.
  int node_comm_size = El::mpi::Size(m_node_comm);
  for (int i = 0; i < node_comm_size; ++i) {
    m_world_ranks_on_node.push_back(
      El::mpi::Translate(m_node_comm, i, get_world_comm()));
  }
}

void lbann_comm::setup_threads()
{
  const char* env_num_threads = getenv("OMP_NUM_THREADS");
  if (env_num_threads != nullptr) {
    m_threads_per_proc = std::atoi(env_num_threads);
  }
  else {
    m_threads_per_proc = std::thread::hardware_concurrency() / m_procs_per_node;
  }
  reset_threads();
}

void lbann_comm::reset_threads() const noexcept
{
  if (m_threads_per_proc != omp_get_max_threads()) {
    omp_set_num_threads(m_threads_per_proc);
  }
}

const El::mpi::Comm& lbann_comm::get_packed_group_comm(int num_per_group) const
{
  if (m_group_communicators.count(num_per_group) == 0) {
    // Ensure we can get an even number of groups.
    if (get_procs_in_world() % num_per_group != 0) {
      std::ostringstream err;
      err << "Cannot create a packed group comm with group size "
          << num_per_group << " out of " << get_procs_in_world()
          << " processes";
      LBANN_ERROR(err.str());
    }
    MPI_Comm comm;
    MPI_Comm_split(get_world_comm().GetMPIComm(),
                   get_rank_in_world() / (get_procs_in_world() / num_per_group),
                   0,
                   &comm);
    m_group_communicators.emplace(num_per_group, comm);
    MPI_Comm_free(&comm); // El::mpi::Comm duplicates internally.
  }
  return m_group_communicators[num_per_group];
}

void lbann_comm::lbann_comm_abort(std::string msg) const
{
  throw lbann_exception(msg);
}

#ifdef LBANN_HAS_ALUMINUM
::Al::ReductionOperator mpi_op_to_al_op(El::mpi::Op op)
{
  if (op == El::mpi::SUM) {
    return ::Al::ReductionOperator::sum;
  }
  else if (op == El::mpi::PROD) {
    return ::Al::ReductionOperator::prod;
  }
  else if (op == El::mpi::MIN) {
    return ::Al::ReductionOperator::min;
  }
  else if (op == El::mpi::MAX) {
    return ::Al::ReductionOperator::max;
  }
  else {
    throw lbann_exception("Reduction operator not supported in Aluminum");
  }
}
#endif

int get_rank_in_world()
{
  int initialized = 0, finalized = 1, rank = -1;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (initialized && !finalized) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
  return rank;
}

#define PROTO(T)                                                               \
  template void lbann_comm::allreduce(El::AbstractMatrix<T>& m,                \
                                      const El::mpi::Comm& c,                  \
                                      El::mpi::Op op) const;                   \
  template void lbann_comm::allreduce(El::AbstractDistMatrix<T>& m,            \
                                      const El::mpi::Comm& c,                  \
                                      El::mpi::Op op) const;                   \
  template void lbann_comm::nb_allreduce(El::AbstractMatrix<T>& m,             \
                                         const El::mpi::Comm& c,               \
                                         Al::request& req,                     \
                                         El::mpi::Op op) const;                \
  template void lbann_comm::nb_allreduce(El::AbstractDistMatrix<T>& m,         \
                                         const El::mpi::Comm& c,               \
                                         Al::request& req,                     \
                                         El::mpi::Op op) const

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
