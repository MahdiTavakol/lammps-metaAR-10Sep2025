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
/* ---v7.6.03---- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(metaAR, FixMetaAR);
// clang-format on
#else

#ifndef LMP_FIX_METAAR_H
#define LMP_FIX_METAAR_H

#include <fstream>
#include <memory>
#include <array>
#include "compute_diff_atom.h"

#include "fix.h"

namespace LAMMPS_NS {

class FixMetaAR : public Fix {
 public:
  FixMetaAR(class LAMMPS *, int, char **);
  ~FixMetaAR() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  double compute_scalar() override;
  double compute_vector(int) override;
  double memory_usage() override;
  void init_list(int, class NeighList *) override;
  void write_restart(FILE* ) override;
  void restart(char *) override;

 private:
  // Input variables for constant values
  double omega, deltaT;
  double sigmaCV1, sigmaCV2;
  double deltaCV1, deltaCV2;
  double DCV1, DCV2;
  double minCV1, maxCV1, minCV2, maxCV2;
  int binsCV1, binsCV2;
  bool setMinMax = true;

  // Input variables for variable values
  std::string omegastr, deltaTstr;
  std::string sigmaCV1str, sigmaCV2str;
  std::string binsCV1str, binsCV2str;
  
  int omegavar, deltaTvar;
  int sigmaCV1var, sigmaCV2var;
  int binsCV1var, binsCV2var;

  int omegastyle, deltaTstyle;
  int sigmaCV1style, sigmaCV2style;
  int binsCV1style, binsCV2style;


  // Neighborlist is required for the compute entropy
  class NeighList *list;

  // Optional region input
  std::string idregion;
  class Region *region;

  // Metadynamics variables
  double CV1t, CV2t;
  double vCV1, vCV2;
  double CV1tInstant, CV2tInstant;
  int numInstant = 10;
  
  std::vector<double>  CV1, CV2;
  std::vector<double> CV1Instant, CV2Instant;
  std::vector<double> locCV1, locCV2;
  // double* entropy_bounds, * enthalpy_bounds; // leave it for now!!!
  // S and H in previous step to calculate Vh and Vs
  double CV1prev, CV2prev;
  // Forces and their transformation
  double fCV1, fCV2;


  // biast in previous step to calculate Vbiast
  double biastprev;
  // how often should the bias distribution be printed
  int biasevery = 0;
  // Welltempered flag
  bool welltempered_flag = false;


  double foriginal[4], foriginal_all[5];
  int force_flag;
  int ilevel_respa;
  double **bias;
  double biast;
  double Vbiast;

  // previous biast values
  std::vector<double>  biasHistory;

  double pv;

  // Maximum atoms in each processor
  int nmax;

  // Variables for printing the output
  int printfreq = 0;
  std::ofstream fp;

  // debugging functions and variables
  std::fstream fp3;
  void print_debug_1d_header();
  void print_debug_1d(const int& );

  // Metadynamics functions
  void init_loc();
  void step0();
  void modify_force();
  void calculate_bias(const int &);

  // Metadynamics Computes
  std::string CV1CId, CV2CId;
  class ComputeDiffAtom* CV1Compute;
  class ComputeDiffAtom* CV2Compute;


  // Print functions
  void writeheader();
  void writebiasdist(std::fstream &fp2, const int &);
  void writefp2(const int &);
  void print_meta();


  void print_debug_1d_unittest(const int &);


  // the function to pack the restart_data
  int pack_restart_data(double *);
};

}    // namespace LAMMPS_NS

#endif
#endif

