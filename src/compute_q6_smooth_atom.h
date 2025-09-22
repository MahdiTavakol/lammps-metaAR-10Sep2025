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
ComputeStyle(q6-smooth/atom,ComputeQ6SmoothAtom);
// clang-format on
#else

#ifndef LMP_COMPUTE_Q6_SMOOTH_ATOM_H
#define LMP_COMPUTE_Q6_SMOOTH_ATOM_H

#include "compute.h"
#include "compute_diff_atom.h"

#include "random_park.h"

namespace LAMMPS_NS {

class ComputeQ6SmoothAtom : public ComputeDiffAtom {
 public:
  ComputeQ6SmoothAtom(class LAMMPS *, int, char **);
  ~ComputeQ6SmoothAtom();
  void init() override;
  void compute_all() override;
  virtual int pack_forward_comm(int , int*, double *, int, int*) override;
  virtual int pack_reverse_comm(int n, int first, double *buf) override;
  virtual void unpack_forward_comm(int, int , double *) override;
  virtual void unpack_reverse_comm(int n, int *list, double *buf) override;

 private:
  int mode;
  int chosen_type;
  double cutoff;

  // the real part of q6ms for each atom
  // nmax X 13
  double** q6ms_real;
  // the imaginary part of q6ms for each atom
  // nmax X 13
  double** q6ms_imag;
  // natoms (nlocal + nghosts) X 13 X 3
  double ***diff_q6ms_real, ***diff_q6ms_imag;
  // nmax
  double* inv_q6_norm_i;
  // nmax
  double* inv_nbnum_i;
  // npairs (npairs) X 13 X 3 
  double ****diff_q6ms_real_pair, ****diff_q6ms_imag_pair;
  // nmax X 3
  double **diff_q6_norm_pair;


  int npair;
  void update_npair();
  void allocate_diff_pairs();
  void deallocate_diff_pairs();
  
  // nmax;
  double* Ni;
  // nmax
  double* ds2i;
  // nmax X 3
  double** diff_Ni;
  // npairs X 3
  double*** diff_Ni_pair;
  // nmax
  double* gi;
  // nmax 
  double* Gi;
  // nmax X 3
  double  **hj;
  // the flag for the comm_forward
  int forward_mode;

  // nmax 
  double *s0j, *s1j, *ds0j, *ds1j;


  // constant parameters 
  static constexpr double Q6_coeff = 4.0*M_PI/13.0;
  static constexpr int nbnum_min = 4; 
  static constexpr int nbnum_col = 4;
  static constexpr int second_val_col = 5;

  // helper functions
  double beta1 = 25.0; //8.0;
  double x01 = 0.65; //0.5;
  double beta2 = 7.0;
  double x02 = 2.5;
  static void orient(const double& input, const double& beta, const double& x0, double& output, double& diff);
  static constexpr int dist_deg = 6;
  static constexpr double dist_coeffs[6] = {1.0,0.0,0.0,-10.0,15.0,-6.0}; 
  static void dist(const double& input, const double& cutoff, double& output, double& diff);
  static std::array<double,104> calculate_Y6m(const std::array<double,3>& dist);
  static std::array<double,3> calculate_dq6i_dxj(const std::array<double,3>& dist, const double& inv_nbnum, const double& Q6_norm);


};

}    // namespace LAMMPS_NS

#endif
#endif
