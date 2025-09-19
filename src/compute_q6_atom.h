/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(q6/atom,ComputeQ6Atom);
// clang-format on
#else

#ifndef LMP_COMPUTE_Q6_ATOM_H
#define LMP_COMPUTE_Q6_ATOM_H

#include "compute.h"
#include "compute_diff_atom.h"

namespace LAMMPS_NS {

class ComputeQ6Atom : public ComputeDiffAtom {
 public:
  ComputeQ6Atom(class LAMMPS *, int, char **);
  void compute_all() override;
  virtual int pack_reverse_comm(int n, int first, double *buf) override;
  virtual void unpack_reverse_comm(int n, int *list, double *buf) override;

 private:
  std::array<double,104> calculate_Y6m(const std::array<double,3>& dist);


  double diff_Q6[3];
  int chosen_type;
  double cutoff;
  static constexpr int nbnum_col = 4;
};

}    // namespace LAMMPS_NS

#endif
#endif
