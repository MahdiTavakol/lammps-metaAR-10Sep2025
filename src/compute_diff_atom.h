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

#ifndef LMP_COMPUTE_DIFF_ATOM_H
#define LMP_COMPUTE_DIFF_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeDiffAtom : public Compute {
 public:
  ComputeDiffAtom(class LAMMPS *, int, char **);
  ~ComputeDiffAtom() override;
  virtual void init() override;
  void init_list(int, class NeighList *) override;
  double compute_scalar() override;
  void compute_peratom() override;


  virtual void compute_all() = 0;
  static constexpr int val_col = 0;
  static constexpr int diff_x_col = 1;
  static constexpr int diff_y_col = 2;
  static constexpr int diff_z_col = 3;

 protected:
  int nmax;
  class NeighList *list;
  class NeighRequest* request;

  bigint last_compute;
};

}    // namespace LAMMPS_NS

#endif

