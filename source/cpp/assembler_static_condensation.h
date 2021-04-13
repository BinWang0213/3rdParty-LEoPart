// Author: Nate sime
// Contact: nsime _at_ carnegiescience.edu
// Copyright: (c) 2020
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifndef AssemblerStaticCondensation_H
#define AssemblerStaticCondensation_H

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>

namespace dolfin
{
  // Forward declarations
  class Form;
  class Mesh;
  class DirichletBC;
  class Function;


  class AssemblerStaticCondensation
{
public:
  // Constructors with assumed symmetry
  AssemblerStaticCondensation(std::shared_ptr<const Form> A, std::shared_ptr<const Form> G,
                           std::shared_ptr<const Form> B, std::shared_ptr<const Form> Q, std::shared_ptr<const Form> S);
  AssemblerStaticCondensation(std::shared_ptr<const Form> A, std::shared_ptr<const Form> G,
                           std::shared_ptr<const Form> B, std::shared_ptr<const Form> Q, std::shared_ptr<const Form> S,
                           std::vector<std::shared_ptr<const DirichletBC>> bcs);
  // Constructors assuming full [2x2] block specification
  AssemblerStaticCondensation(std::shared_ptr<const Form> A, std::shared_ptr<const Form> G,
                           std::shared_ptr<const Form> GT, std::shared_ptr<const Form> B, std::shared_ptr<const Form> Q,
                           std::shared_ptr<const Form> S);

  AssemblerStaticCondensation(std::shared_ptr<const Form> A, std::shared_ptr<const Form> G,
                           std::shared_ptr<const Form> GT, std::shared_ptr<const Form> B, std::shared_ptr<const Form> Q,
                           std::shared_ptr<const Form> S,
                           std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Destructor
  ~AssemblerStaticCondensation();

  // Public Methods
  void assemble_global(GenericMatrix& A_g, GenericVector& f_g);
  void assemble_global_lhs(GenericMatrix& A_g);
  void assemble_global_rhs(GenericVector& f_g);
  void assemble_global_system(GenericMatrix& A_g, GenericVector& b_g, bool assemble_lhs = true);

  void backsubstitute(const Function& Uglobal, Function& Ulocal);

private:
  // Private Methods
  void test_rank(const Form& a, const std::size_t rank);

  // Private Attributes
  std::shared_ptr<const Mesh> mesh;
  std::shared_ptr<const Form> A;
  std::shared_ptr<const Form> B;
  std::shared_ptr<const Form> G;
  std::shared_ptr<const Form> Q;
  std::shared_ptr<const Form> S;
  std::shared_ptr<const Form> GT;

  bool assume_symmetric;

  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      invAe_list;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      Ge_list;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      Be_list;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> Qe_list;

  // Facilitate non-symmetric case
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      GTe_list;

  void init_tensors(GenericMatrix* A_g, GenericVector* f_g);

  const MPI_Comm mpi_comm;
  std::vector<std::shared_ptr<const DirichletBC>> bcs;
};
} // namespace dolfin

#endif // AssemblerStaticCondensation_H
