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

#include <array>
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
  virtual int pack_forward_comm(int, int *, double *, int, int *) override;
  virtual int pack_reverse_comm(int n, int first, double *buf) override;
  virtual void unpack_forward_comm(int, int, double *) override;
  virtual void unpack_reverse_comm(int n, int *list, double *buf) override;

 private:
  int mode;
  int chosen_type;
  double cutoff;
  int switch_flag;

  // the real part of q6ms for each atom
  // nmax X 13
  double **q6ms_real;
  // the imaginary part of q6ms for each atom
  // nmax X 13
  double **q6ms_imag;
  // nmax X 13 X 3
  double ***diff_q6ms_real, ***diff_q6ms_imag;
  // nmax
  double *inv_q6_norm_i;
  // nmax
  double *inv_nbnum_i;

  // nmax;
  double *Ni;
  // nmax
  double *ds2i;
  // nmax X 3
  double **diff_Ni;
  // nmax X 3;
  double **diff_Ntotal;
  // nmax X 13
  double **gi_real, **gi_imag;
  // nmax
  double *Gi;
  // nmax
  double *Cjj;
  // nmax X 3
  double **hj;
  // nmax X 3
  double **hj2;
  // The differential of the scaling factor with ri
  double **diff_Z_all;
  double **diff_W_all;
  // the flag for the comm_forward
  int forward_mode;
  // nmax
  double *s0j, *s1j, *ds0j, *ds1j;

  // Q6_ARRAY_SIZE * N_DIM
  double **dqi_drj_real, **dqi_drj_imag;

  // Random number generation object
  std::unique_ptr<RanPark> rng;

  // constant parameters
  static constexpr int Q6_ARRAY_SIZE = 13;
  static constexpr int N_DIM = 3;
  static constexpr double Q6_coeff = 4.0 * M_PI / 13.0;
  static constexpr int nbnum_min = 4;
  static constexpr int nbnum_col = 4;
  static constexpr int second_val_col = 5;

  // helper functions
  double beta1 = 25.0;    //8.0;
  double x01 = 0.65;      //0.5;
  double beta2 = 7.0;
  double x02 = 2.5;
  static void orient(const double &input, const double &beta, const double &x0, double &output,
                     double &diff);

  static constexpr int dist_deg = 6;
  static constexpr double dist_coeffs[6] = {1.0, 0.0, 0.0, -10.0, 15.0, -6.0};
  static void dist(const double &input, const double &cutoff, double &output, double &diff);

  // Just for debugging.
  static  inline void unit(const double &input,  double &output, double &diff) {
    output = 1.0;
    diff = 0.0;
  }

  static std::array<double, 104> calculate_Y6m(const std::array<double, 3> &dist);
  static void calculate_dq6i_drj(const std::array<double, N_DIM> &dist, const double *q6m_real_i,
                                 const double *q6m_imag_i, const double &inv_nbnum,
                                 const double &inv_q6_norm_i, double **dqi_drj_real,
                                 double **dqi_drj_imag);

  inline static double smooth_floor(double a, double floor, double k=16.0) noexcept
  {
    // returns ~max(a, floor) but C^1
    return floor + std::log1p(std::exp(k*(a - floor)))/k;
  }

  inline static double pow_fun(double s, int n) noexcept
  {
    double out = 1.0;
    switch (n) {
      case 6:
        out *= s;
        [[fallthrough]];
      case 5:
        out *= s;
        [[fallthrough]];
      case 4:
        out *= s;
        [[fallthrough]];
      case 3:
        out *= s;
        [[fallthrough]];
      case 2:
        out *= s;
        [[fallthrough]];
      case 1:
        out *= s;
        [[fallthrough]];
      case 0:
        return out;
      default:;    // fall back below if n>6
    }
    // Fallback for bigger n: exponentiation by squaring (O(log n))
    double r = out, b = s;
    unsigned k = static_cast<unsigned>(n);
    r = 1.0;
    while (k) {
      if (k & 1) r *= b;
      b *= b;
      k >>= 1;
    }
    return r;
  }
};

}    // namespace LAMMPS_NS

#endif
#endif
