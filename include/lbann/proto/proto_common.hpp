#ifndef LBANN_PROTO_PROTO_COMMON_HPP_INCLUDED
#define LBANN_PROTO_PROTO_COMMON_HPP_INCLUDED

#include "lbann/lbann.hpp"
#include <lbann.pb.h>
#include "lbann/proto/factories.hpp"

namespace lbann {

/// Returns true if the Model contains at least one MotifLayer
bool has_motifs(const lbann_comm& comm, const lbann_data::LbannPB& p);

void expand_motifs(const lbann_comm& comm, lbann_data::LbannPB& pb);

/** Customize the name of the index list with the following options:
 *   - trainer ID
 *   - model name
 * The format for the naming convention if the provided name is <index list> is:
 *   <index list> == <basename>.<extension>
 *   <model name>_t<ID>_<basename>.<extension>
 */
void customize_data_readers_index_list(const lbann_comm& comm, lbann_data::LbannPB& p);

/// instantiates one or more generic_data_readers and inserts them in &data_readers
void init_data_readers(
  lbann_comm *comm,
  const lbann_data::LbannPB& p,
  std::map<execution_mode, generic_data_reader *>& data_readers,
  bool is_shareable_training_data_reader, bool is_shareable_testing_data_reader,
  bool is_shareable_validation_data_reader = false);

/// adjusts the number of parallel data readers
void set_num_parallel_readers(const lbann_comm& comm, lbann_data::LbannPB& p);

/// adjusts the values in p by querying the options db
void get_cmdline_overrides(const lbann_comm& comm, lbann_data::LbannPB& p);

/// print various params (learn_rate, etc) to cout
void print_parameters(const lbann_comm& comm, lbann_data::LbannPB& p);

/// prints usage information
void print_help(const lbann_comm& comm);

/// prints usage information
void print_help(std::ostream& os);

/// prints prototext file, cmd line, etc to file
void save_session(const lbann_comm& comm, const int argc, char * const* argv, lbann_data::LbannPB& p);

///
void read_prototext_file(
  const std::string& fn,
  google::protobuf::Message& pb,
  const bool master);

///
bool write_prototext_file(
  const std::string& fn,
  lbann_data::LbannPB& pb);

} // namespace lbann

#endif // LBANN_PROTO_PROTO_COMMON_HPP_INCLUDED
