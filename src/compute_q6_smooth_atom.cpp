/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_q6_smooth_atom.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <numeric>

#include <iostream>

using namespace LAMMPS_NS;

// Transfer mode flag
enum { Q6_TRANSFER = 1 << 0, N_TRANSFER = 1 << 1, G_TRANSFER = 1 << 2, DS2_TRANSFER = 1 << 3 };

// Execution flags
enum { N_MODE = 1 << 0, PHI_MODE = 1 << 1, SIMPLE_PHI_MODE = 1 << 2, SIMPLE_N_MODE = 1 << 3, NO_DIFF = 1 << 4, ALL_TYPES = 1 << 5 };

// Transfer strides
enum { Q6_FOR_STRIDE = 26, N_FOR_STRIDE = 1, G_FOR_STRIDE = 1 , DS2_FOR_STRIDE = 1, N_REV_STRIDE = 3, S_REV_STRIDE=6};

// Function flags for s0/s1/s2/s3
enum { NONE = 0, S0_SW=1<<0, S1_SW=1<<1, S2_SW=1<<2, S3_SW=1<<3};

static constexpr double eps = 1e-8;


/* ---------------------------------------------------------------------- */

ComputeQ6SmoothAtom::ComputeQ6SmoothAtom(LAMMPS *lmp, int narg, char **arg) :
    ComputeDiffAtom{lmp, narg, arg}, mode{N_MODE}, chosen_type{-1}, 
    switch_flag{0}, cutoff{3.2}, min_slope{0.02},
    q6ms_real{nullptr}, q6ms_imag{nullptr}, diff_q6ms_real{nullptr}, diff_q6ms_imag{nullptr},
    inv_q6_norm_i{nullptr}, inv_nbnum_i{nullptr},
    Ni{nullptr}, ds2i{nullptr}, diff_Ni{nullptr}, diff_Ntotal{nullptr}, gi_real{nullptr},
    gi_imag{nullptr}, Gi{nullptr}, Cjj{nullptr}, hj{nullptr}, hj2{nullptr}, 
    diff_Z_all{nullptr}, diff_W_all{nullptr}, forward_mode{Q6_TRANSFER},
    s0j{nullptr}, s1j{nullptr}, ds0j{nullptr}, ds1j{nullptr}, dqi_drj_real{nullptr}, dqi_drj_imag{
                                                                                         nullptr}
{
  chosen_type = utils::numeric(FLERR, arg[3], false, lmp);
  cutoff = utils::numeric(FLERR, arg[4], false, lmp);

  if (chosen_type == -1)
    mode |= ALL_TYPES;

  // before calling the comm->forward this parameter is set since it has two different values.
  comm_forward = Q6_FOR_STRIDE;
  comm_reverse = N_REV_STRIDE;
  

  // Every switch is on by default.
  switch_flag |= S0_SW;
  switch_flag |= S1_SW;
  switch_flag |= S2_SW;
  switch_flag |= S3_SW;

  int iarg = 5;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "no_diff") == 0) {
      if (comm->me == 0) error->warning(FLERR,"Turning off the diffs, this compute should not be used with any fixes that requests for diffs!");
      mode |= NO_DIFF;
      iarg++;
    } else if (strcmp(arg[iarg], "phi") == 0) {
      mode = (mode & ~N_MODE) | PHI_MODE;
      iarg++;
    } else if (strcmp(arg[iarg], "S0_off") == 0) {
      switch_flag &= ~S0_SW;
      iarg++;
    } else if (strcmp(arg[iarg], "S1_off") == 0) {
      switch_flag &= ~S1_SW;
      iarg++;
    } else if (strcmp(arg[iarg], "S2_off") == 0) {
      switch_flag &= ~S2_SW;
      iarg++;
    } else if (strcmp(arg[iarg], "S3_off") == 0) {
      switch_flag &= ~S3_SW;
      iarg++;
    } else if (strcmp(arg[iarg], "S0") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing parameters after S0");
      threshold0 = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg+=2;
    } else if (strcmp(arg[iarg], "S1") == 0) {
      if (iarg + 2 >= narg) error->all(FLERR, "Missing parameters after S1");
      beta1 = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      x01 = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      iarg+=3;
    } else if (strcmp(arg[iarg], "S2") == 0) {
      if (iarg + 2 >= narg) error->all(FLERR, "Missing parameters after S2");
      beta2 = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      x02 = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      iarg+=3;
    } else if (strcmp(arg[iarg], "S3") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing parameters after S3");
      threshold3 = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg+=2;
    } else if (strcmp(arg[iarg], "min_slope") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing parameters after S3");
      min_slope = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg+=2;
    } else
      error->all(FLERR, "Illegal compute q6-smooth/atom command");
  }

  // A warning message
  if ((mode & N_MODE) && !(switch_flag & S3_SW))
    error->warning(FLERR,"There is no S3 function in the N_MODE mode, ignoring the S3_off flag!");
}

/* ----------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::init()
{
  ComputeDiffAtom::init();
  memory->create(q6ms_real, nmax, Q6_ARRAY_SIZE, "compute_q6_smooth_atom:q6ms_real");
  memory->create(q6ms_imag, nmax, Q6_ARRAY_SIZE, "compute_q6_smooth_atom:q6ms_imag");
  memory->create(diff_q6ms_real, nmax, Q6_ARRAY_SIZE, N_DIM,
                 "compute_q6_smooth_atom:diff_q6ms_real");
  memory->create(diff_q6ms_imag, nmax, Q6_ARRAY_SIZE, N_DIM,
                 "compute_q6_smooth_atom:diff_q6ms_imag");
  memory->create(inv_q6_norm_i, nmax, "compute_q6_smooth_atom:q6_norm_i");
  memory->create(inv_nbnum_i, nmax, "compute_q6_smooth_atom:inv_nbnum_i");
  memory->create(Ni, nmax, "compute_q6_smooth_atom:Ni");
  memory->create(ds2i, nmax, "compute_q6_smooth:ds2i");
  memory->create(diff_Ni, nmax, N_DIM, "compute_q6_smooth_atom:diff_Ni");
  memory->create(diff_Ntotal, nmax, N_DIM, "compute_q6_smooth_atom:diff_Ntotal");
  memory->create(gi_real, nmax, Q6_ARRAY_SIZE, "compute_q6_smooth_atom:gi_real");
  memory->create(gi_imag, nmax, Q6_ARRAY_SIZE, "compute_q6_smooth_atom:gi_imag");
  memory->create(Gi, nmax, "compute_q6_smooth_atom:Gi");
  memory->create(Cjj, nmax, "compute_q6_smooth_atom:Cjj");
  memory->create(hj, nmax, N_DIM, "compute_q6_smooth_atom:hj");
  memory->create(hj2, nmax, N_DIM, "compute_q6_smooth_atom:hj2");
  memory->create(diff_Z_all,nmax,N_DIM,"compute_q6_smooth:diff_Z_all");
  memory->create(diff_W_all,nmax,N_DIM,"compute_q6_smooth:diff_W_all");
  memory->create(s0j, nmax, "compute_q6_smooth_atom:s0j");
  memory->create(s1j, nmax, "compute_q6_smooth_atom:s1j");
  memory->create(ds0j, nmax, "compute_q6_smooth_atom:ds0j");
  memory->create(ds1j, nmax, "compute_q6_smooth_atom:ds1j");
  memory->create(dqi_drj_real, Q6_ARRAY_SIZE, N_DIM, "compute_q6_smooth_atom:dqi_drj_real");
  memory->create(dqi_drj_imag, Q6_ARRAY_SIZE, N_DIM, "compute_q6_smooth_atom:dqi_drj_imag");

  // Setting the neighbor list cutoff
  request->cutoff = cutoff;
  // Initializing the random number generation object
  rng = std::make_unique<RanPark>(lmp, 11111);

  // All the switches are off
  if (!(switch_flag & S0_SW) && 
      !(switch_flag & S1_SW) &&
      !(switch_flag & S2_SW) &&
      !(switch_flag & S3_SW)) {
    if (mode & PHI_MODE) {
      if (comm->me == 0) {
        error->warning(FLERR,"Running the faster version of the phi mode!");
      }
      mode = (mode & NO_DIFF) | SIMPLE_PHI_MODE;
    } else if (mode & N_MODE) {
      if (comm->me == 0) {
        error->warning(FLERR,"Running the faster version of the N mode!");
      }
      mode = (mode & NO_DIFF) | SIMPLE_N_MODE;     
    }
  }

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  extvector = 0;

  memory->create(vector,size_vector,"compute_q6_smooth_atom.cpp:vector");

}

/* ----------------------------------------------------------------------- */

ComputeQ6SmoothAtom::~ComputeQ6SmoothAtom()
{
  if (vector) memory->destroy(vector);
  if (q6ms_real) memory->destroy(q6ms_real);
  if (q6ms_imag) memory->destroy(q6ms_imag);
  if (diff_q6ms_real) memory->destroy(diff_q6ms_real);
  if (diff_q6ms_imag) memory->destroy(diff_q6ms_imag);
  if (inv_q6_norm_i) memory->destroy(inv_q6_norm_i);
  if (inv_nbnum_i) memory->destroy(inv_nbnum_i);
  if (Ni) memory->destroy(Ni);
  if (ds2i) memory->destroy(ds2i);
  if (diff_Ni) memory->destroy(diff_Ni);
  if (diff_Ntotal) memory->destroy(diff_Ntotal);
  if (gi_real) memory->destroy(gi_real);
  if (gi_imag) memory->destroy(gi_imag);
  if (Gi) memory->destroy(Gi);
  if (Cjj) memory->destroy(Cjj);
  if (hj) memory->destroy(hj);
  if (hj2) memory->destroy(hj2);
  if (diff_Z_all) memory->destroy(diff_Z_all);
  if (diff_W_all) memory->destroy(diff_W_all);
  if (s0j) memory->destroy(s0j);
  if (s1j) memory->destroy(s1j);
  if (ds0j) memory->destroy(ds0j);
  if (ds1j) memory->destroy(ds1j);
  if (dqi_drj_real) memory->destroy(dqi_drj_real);
  if (dqi_drj_imag) memory->destroy(dqi_drj_imag);

  q6ms_real = nullptr;
  q6ms_imag = nullptr;
  diff_q6ms_real = nullptr;
  diff_q6ms_imag = nullptr;
  inv_q6_norm_i = nullptr;
  inv_nbnum_i = nullptr;
  Ni = nullptr;
  ds2i = nullptr;
  diff_Ni = nullptr;
  diff_Ntotal = nullptr;
  gi_real = nullptr;
  gi_imag = nullptr;
  Gi = nullptr;
  Cjj = nullptr;
  hj = nullptr;
  hj2 = nullptr;
  diff_Z_all = nullptr;
  diff_W_all = nullptr;
  s0j = nullptr;
  s1j = nullptr;
  ds0j = nullptr;
  ds1j = nullptr;
  dqi_drj_real = nullptr;
  dqi_drj_imag = nullptr;
}

/* ---------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::compute_all()
{
  //if (last_compute == update->ntimestep) return;
  last_compute = update->ntimestep;
  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;

  neighbor->build_one(list);

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    memory->grow(array_atom, nmax, size_peratom_cols, "compute_diff:array_atom");
    memory->grow(q6ms_real, nmax, Q6_ARRAY_SIZE, "compute_q6_smooth_atom:q6ms_real");
    memory->grow(q6ms_imag, nmax, Q6_ARRAY_SIZE, "compute_q6_smooth_atom:q6ms_imag");
    memory->grow(diff_q6ms_real, nmax, Q6_ARRAY_SIZE, N_DIM,
                 "compute_q6_smooth_atom:diff_q6ms_real");
    memory->grow(diff_q6ms_imag, nmax, Q6_ARRAY_SIZE, N_DIM,
                 "compute_q6_smooth_atom:diff_q6ms_imag");
    memory->grow(inv_q6_norm_i, nmax, "compute_q6_smooth_atom:inv_q6_norm_i");
    memory->grow(inv_nbnum_i, nmax, "compute_q6_smooth_atom:inv_nbnum_i");
    memory->grow(Ni, nmax, "compute_q6_smooth_atom:Ni");
    memory->grow(ds2i, nmax, "compute_q6_smooth_atom:ds2i");
    memory->grow(diff_Ni, nmax, N_DIM, "compute_q6_smooth_atom:diff_Ni");
    memory->grow(diff_Ntotal, nmax, N_DIM, "compute_q6_smooth_atom:diff_Ntotal");
    memory->grow(gi_real, nmax, Q6_ARRAY_SIZE, "compute_q6_smooth_atom:gi_real");
    memory->grow(gi_imag, nmax, Q6_ARRAY_SIZE, "compute_q6_smooth_atom:gi_imag");
    memory->grow(Gi, nmax, "compute_q6_smooth_atom:Gi");
    memory->grow(Cjj, nmax, "compute_q6_smooth_atom:Cjj");
    memory->grow(hj, nmax, N_DIM, "compute_q6_smooth_atom:hj");
    memory->grow(hj2, nmax, N_DIM, "compute_q6_smooth_atom:hj2");
    memory->grow(diff_Z_all, nmax, N_DIM, "compute_q6_smooth_atom:diff_Z_all");
    memory->grow(diff_W_all, nmax, N_DIM, "compute_q6_smooth_atom:diff_W_all");
    memory->grow(s0j, nmax, "compute_q6_smooth_atom:s0j");
    memory->grow(s1j, nmax, "compute_q6_smooth_atom:s1j");
    memory->grow(ds0j, nmax, "compute_q6_smooth_atom:ds0j");
    memory->grow(ds1j, nmax, "compute_q6_smooth_atom:ds1j");
  }

  int i, j, ii, jj, inum, jnum;
  int *ilist, *jlist, *numneigh, **firstneigh;

  //it is the same as atom->nlocal and list->gnum is zero since we have not requested for ghost neighbors in the list.
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // zeroing all the variables
  double Q6_sum = 0.0;
  double Q6_sum_all = 0.0;
  double phi_sum = 0.0;
  double phi_sum_all = 0.0;
  int num_selected = 0;
  int num_selected_all = 0;
  // Si = sigma(s1(bij)*s0(rij))
  // Zi = s0(rij)
  // Z = sigma(s0(rij))
  // Z is the normalization factor for N
  double Z = 0.0;
  double Z_all = 0.0;
  // Phi = sigma(Ni*Nj*s3(rij))
  // W = sigma(s3(rij))
  // Zavg^2*W is the normalization factor for phi
  double W = 0.0;
  double W_all = 0.0;

  // filling all the arrays with zeros
  std::fill_n(&array_atom[0][0], nmax * size_peratom_cols, 0.0);
  std::fill_n(&q6ms_real[0][0], nmax * Q6_ARRAY_SIZE, 0.0);
  std::fill_n(&q6ms_imag[0][0], nmax * Q6_ARRAY_SIZE, 0.0);
  std::fill_n(&diff_q6ms_real[0][0][0], nmax * Q6_ARRAY_SIZE * N_DIM, 0.0);
  std::fill_n(&diff_q6ms_imag[0][0][0], nmax * Q6_ARRAY_SIZE * N_DIM, 0.0);
  std::fill_n(&gi_real[0][0], nmax * Q6_ARRAY_SIZE, 0.0);
  std::fill_n(&gi_imag[0][0], nmax * Q6_ARRAY_SIZE, 0.0);
  std::fill_n(&diff_Ni[0][0], nmax * N_DIM, 0.0);
  std::fill_n(&diff_Ntotal[0][0], nmax * N_DIM, 0.0);
  std::fill_n(Gi, nmax, 0.0);
  std::fill_n(&hj[0][0], nmax * N_DIM, 0.0);
  std::fill_n(&hj2[0][0], nmax * N_DIM, 0.0);
  std::fill_n(&diff_Z_all[0][0],nmax*N_DIM,0.0);
  std::fill_n(&diff_W_all[0][0],nmax*N_DIM,0.0);

  /*
   These variables either do not accumulate values
   or they for the first time used with = sign
   so zeroing them is not needed.
   As their size is pretty large zeroing them might 
   have an overhead.
  std::fill_n(inv_q6_norm_i, nmax, 0.0);
  std::fill_n(inv_nbnum_i, nmax, 0.0);
  std::fill_n(Ni, nmax, 0.0);
  std::fill_n(ds2i, nmax, 0.0);
  std::fill_n(Cjj, nmax, 0.0);
  std::fill_n(s0j, nmax, 0.0);
  std::fill_n(s1j, nmax, 0.0);
  std::fill_n(ds0j, nmax, 0.0);
  std::fill_n(ds1j, nmax, 0.0);
  */

  // lambda functions
  auto s0 = [&](const double &input, double &output, double &diff) {
    if (switch_flag & S0_SW)
      dist(input,cutoff,output,diff,min_slope,threshold0);
    else
      unit(input,output,diff);
  };
  auto s1 = [&](const double &input, double &output, double &diff) {
    if (switch_flag & S1_SW)
      orient(input, beta1, x01, output, diff,min_slope);
    else
      unit(input,output,diff);
  };
  auto s2 = [&](const double &input, double &output, double &diff) {
    if (switch_flag & S2_SW)
      orient(input, beta2, x02, output, diff,min_slope);
    else
      unit(input,output,diff);
  };
  auto s3 = [&](const double &input, double &output, double &diff) {
    if (switch_flag & S3_SW)
      dist(input, cutoff, output, diff,min_slope,threshold3);
    else
      unit(input,output,diff);
  };
  /*
  auto s1 = [&](const double& input, double& output, double& diff) {
    output = input;
    diff = 1.0;
  };
  auto s2 = [&](const double& input, double& output, double& diff) {
    output = input;
    diff = 1.0;
  };*/

  /*
   * This part calculates the q6ms and their self differential (diffqi/diffri).
   * The diffqi/diffrj are calculated on the fly since the 
   * array needed to store all the pairs would be so large 
   * and its communication is impossible.
   * For instance for 30000 atoms 12GB is required!!!
   */
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jnum = numneigh[i];
    jlist = firstneigh[i];

    if (!(mask[i] & groupbit)) continue;
    if (!(mode & ALL_TYPES) && type[i] != chosen_type) continue;
    num_selected++;

    int nbnum = 0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      double distx = x[i][0] - x[j][0];
      double disty = x[i][1] - x[j][1];
      double distz = x[i][2] - x[j][2];
      double r = sqrt(distx * distx + disty * disty + distz * distz);
      if (r < 1e-8 || r >= cutoff) continue;
      nbnum++;
      std::array<double, N_DIM> distance = {distx, disty, distz};
      std::array<double, 104> Y6m = calculate_Y6m(distance);
      for (int deg = -6; deg <= 6; deg++) {
        int offset = (deg + 6) * 8;
        int indx = deg + 6;
        q6ms_real[i][indx] += Y6m[offset + 0];
        q6ms_imag[i][indx] += Y6m[offset + 1];
        if (!(mode & NO_DIFF)) {
          diff_q6ms_real[i][indx][0] += Y6m[offset + 2];
          diff_q6ms_imag[i][indx][0] += Y6m[offset + 3];
          diff_q6ms_real[i][indx][1] += Y6m[offset + 4];
          diff_q6ms_imag[i][indx][1] += Y6m[offset + 5];
          diff_q6ms_real[i][indx][2] += Y6m[offset + 6];
          diff_q6ms_imag[i][indx][2] += Y6m[offset + 7];
        }
      }
    }
    array_atom[i][nbnum_col] = nbnum;

    const double inv_nbnum = nbnum >= nbnum_min ? 1.0 / nbnum : 0.0;

    inv_nbnum_i[i] = inv_nbnum;

    for (int deg = -6; deg <= 6; deg++) {
      int indx = deg + 6;
      q6ms_real[i][indx] *= inv_nbnum;
      q6ms_imag[i][indx] *= inv_nbnum;

      for (int k = 0; k < N_DIM; k++) {
        diff_q6ms_real[i][indx][k] *= inv_nbnum;
        diff_q6ms_imag[i][indx][k] *= inv_nbnum;
      }
    }

    double q6_norm = 0.0;

    for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
      q6_norm += q6ms_real[i][indx] * q6ms_real[i][indx] + q6ms_imag[i][indx] * q6ms_imag[i][indx];
    }

    q6_norm = std::sqrt(q6_norm);
    const double inv_q6_norm = q6_norm >= 1e-20 ? 1.0 / q6_norm : 0.0;

    inv_q6_norm_i[i] = inv_q6_norm;

    for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
      q6ms_real[i][indx] *= inv_q6_norm;
      q6ms_imag[i][indx] *= inv_q6_norm;
    }


    if (!(mode & NO_DIFF)) {
      
      double diff_q6_norm[N_DIM] = {0.0, 0.0, 0.0};

      for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
        for (int dim = 0; dim < N_DIM; dim++)
          diff_q6_norm[dim] += q6ms_real[i][indx] * diff_q6ms_real[i][indx][dim] +
            q6ms_imag[i][indx] * diff_q6ms_imag[i][indx][dim];
      }



      diff_q6_norm[0] *= inv_q6_norm;
      diff_q6_norm[1] *= inv_q6_norm;
      diff_q6_norm[2] *= inv_q6_norm;

      for (int dim = 0; dim < N_DIM; dim++) {
        for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
          diff_q6ms_real[i][indx][dim] =
            (inv_q6_norm) * (diff_q6ms_real[i][indx][dim] - q6ms_real[i][indx] * diff_q6_norm[dim]);
          diff_q6ms_imag[i][indx][dim] =
            (inv_q6_norm) * (diff_q6ms_imag[i][indx][dim] - q6ms_imag[i][indx] * diff_q6_norm[dim]);
        }
      }
    }
  }

  // forward_comm the q6ms_real, q6ms_imag, diff_q6ms_real and diff_q6ms_imag to ghost atoms
  forward_mode = Q6_TRANSFER;
  comm_forward = Q6_FOR_STRIDE;
  comm->forward_comm(this);

  /*
   * This part calculates the contribution of the atom i to the the differential of N_total [Ni]
   *  with respect to ri and also diff of N_total [Nj] with respect to rj (self diffs)
   * (1)  N_MODE: N_total = sigma(Ni) 
   *      Ni = s2(sigma(s1(cij)*s0(rij)))
   *      In this case despite the SIMPLE_N_MODE there is no symmetery since the ds2 coefficient for
   *      atom i is not the same as the ds2[j].
   *      dNi/dri = ds2[i]*sigma(ds0(rij)*s1(cij)*drij/dri+s0(rij)*ds1(cij)*dcij/dri)
   *      (A) So the distance dependent contribution of Ni to the dN/dri is
   *          ds2[i]*s1(i,j)*ds0(i,j)*drij/dri = wpair2*drij/dri
   *          the distance depedent contribition of Ni to the dN/drj is
   *          -ds2[i]*s1(i,j)*ds0(i,j)*drij/drj = -wpair2*drij/drj
   *      (B) The cij depdent term is dcij/drk = sigma(qj)*dqi/drk + sigma(dqj/drk)*qi
   *          The first term is ds2*ds1*s0*sigma(qj)*dqi/drk = gi*dqi/drk
   *          Each atom i contribution to its own diff  = gi*dqi/dri
   *          Each atom i contribution to the j diff = gi*dqi/drj
   *      (C) The third term is ds2[i]*ds1(i,j)*s0(i,j)*dqj/drk which is contributed only from neighbor atoms
   *          The contribution of atom i to the neighbor j is 
   *          sigma(ds2[j]*ds1(i,j)*s0(i,j)*qj)*dqi/drj = g_primei*dqi/drj
   *          gprimei is different than gi since wpair in the gprime is ds2[i]*... while in gprime is ds2[j] which breaks the symmetery..
   * 
   * (2)  PHI_MODE: Ni = qi*(wij sigmaqj) = qi*gi
   *      dNi/dri = dqi/dri * gi + qi* (sigma dqj/dri)
   *      dNj/dri = dqj/dri * gj + qj* (sigma dqk/dri)
   *      Thus Ni also contribu: I value focused work and tes to j and for the same reason we need hj.. but hj = wij*qj*dqi/drj
   *      diffN1/diffrk = diffq1 *(q2+q3+q4) + q1*(diffq2+diffq3+diffq4)
   *
   * (3)  SIMPLE_N_MODE: N_total = simga(Ni)
   *      Ni = sigma(wij*qi*qj)
   *      gi = sigma(wij*qj) --> N_total = sigma(qi*gi)
   *      dN_total/drk = sigma(wij*qj*dqi/drk +wij*qi*dqj/drk) = sigma(gi*dqi/drk + gj*dqj/drk)
   *                   = 2.0*sigma(gi*dqi/drk)
   *      Thanks to the forward comm we have the qj for the ghost atoms in this rank so 
   *      we can easily calculate the gi = sigma(wij*qj) over j.
   *      However, we do not have the gj and dgj/drk and it does not make sense to transfer the diff of qj with
   *      respect to every possible r since the array size would be massively large!
   *      So, here we define hj = sigma (gi*dqi/rj over i), each rank adds its own contribution to the hj.
   *      Since the hj can be a ghost atom it has to be reverse communicated and added to the hj on the current rank.
   *      Then we go through every owned atom and add hj[i] to every diff.
   */

  /*
   * (1)  We first calculate gi values.
   * (2)  For the contribution of qi to the diff with respect to rj   
   *      (2A)  if (mode & SIMPLE_N_MODE) N_total = N1 + N2 + N3 + ... = q1*g1+q2*g2 + ...
   *            Thus, the contribution of the qi to the rj comes from the gi * diffqi/diffrj
   *            Also we have both the ij and ji pairs. That is the reason why the result is multiplied by 2.
   *      (2B)  if (mode & PHI_MODE) Ni = q1*g2
   *            diff Ni/diff rk = diff qi/diff rk * gi + qi*(sigma diffqj/diffrk where j is i neighbor)
   *            Thus, the contribution of atom i to the rj comes from qj * diff qi/ diff rj
   *            Despite the (2A) the differential of each neighbor multiplied by j is its contribution.
   *      (2C)  An example:
   *            N_MODE: Q = q1*(q2+q3+q4) + q2*(q1+q3) + q3*(q1+q2+q4) + q4*(q1+q3)
   *            dQ/dri = dq1/dr1*(q2+q3+q4) + dq2/dr1*(q1+q3) + dq3/dr1*(q1+q2+q4) + dq4/dr1*(q1+q3) +
   *            q1*(dq2/dr1+dq3/dr1+dq4/dr1) + ......
   *            2 contribution to diffQ/dr1 = 2*dq2/dr1*(q1+q3) = 2*dq2/dr1*g2
   *            PHI_MODE: N1 = q1*(q2+q3+q4)
   *            dN1/dr1 = dq1/dr1*(q2+q3+q4) + q1*(dq2/dr1+dq3/dr1+dq4/dr1)
   *            q2 contribution comes from dq2/dr1*q1 not dq2/dr1*g1
   * (3)  As atom i can contribute to atom j which might be a ghost atom we fill the 
   *      its contribution to atom j in the hj array which we be reverse communicate
   *      to fill the ghost atoms of all the rankk.
   * (4)  An examples for the N_MODE:
   *      Ai = ds2[i]
   *      N = A1*q1*(q2+q3+q4)+A2*q2*(q1+q3)+A3*q3*(q1+q2+q4)+A4*q4*(q1+q3)
   *      N = A1*q1*g1+A2*q2*g2+A3*q3*g3+A4*q4*g4
   *      dN1/dr1 = A1*dq1/dr1*(q2+q3+q4) = A1*dq1/dr1*g1 (first term of N1)
   *      dN2/dr1 = A2*dq2/dr1*(q1+q3) = A2*dq2/dr1*g2 (first term of N2)
   *      dN3/dr1 = A3*dq3/dr1*(q1+q2+q4) = A3*dq3/dr1*g3
   *      dN4/dr1 = A4*dq4/dr1*(q1+q3) = A4*dq4/dr1*g4
   *      dN/dr1 = dq1/dr1*(A2*q2+A3*q3+A4*q4)=dq1/dr1*(gprimi1)
   *      dN/dr1 = dq2/dr1*(A1*q1+A3*q3)=dq2/dr1*(gprimi2)
   *      dN/dr1 = dq3/dr1*(A1*q1+A3*q3+A4*q4)=dq3/dr1*(gprimi3)
   *      dN/dr1 = dq4/dr1*(A1*q1+A3*q3)=dq4/dr1*(gprimi4)
   * 
   */

  /*
   * This loop fills the gi array, add the self diff contribution to diffs (diffqi/diffri)
   * and fills the hj components.
   * For the N_MODE we fill the diff_array_atom
   * For the PHI_MODE the diff_Ni is filled.
   * We cannot fill the gprime yet since it needs the ds2[j]
   */

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jnum = numneigh[i];
    jlist = firstneigh[i];
    if (!(mask[i] & groupbit)) continue;
    if (!(mode & ALL_TYPES) && type[i] != chosen_type) continue;

    double Si = 0.0;
    //resetting neighbor variables
    /* Since we are assigning initial values to these instead of 
     * accumulating it is safe to not zero them here but 
     * they will have values from the previous atom.
     * std::fill_n(s0j,jnum,0.0);
     * std::fill_n(s1j,jnum,0.0);
     * std::fill_n(ds0j,jnum,0.0);
     * std::fill_n(ds1j,jnum,0.0); 
     */

    /* Some tests -->>*/
    double ci_norm = 0.0;
    for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
      ci_norm += q6ms_real[i][indx] * q6ms_real[i][indx] + q6ms_imag[i][indx] * q6ms_imag[i][indx];
    }
    if (std::abs(ci_norm - 1.0) > eps && std::abs(ci_norm) > eps)
      error->warning(FLERR, "ci normalisation for the atom {} is wrong", i);
    /*<<--Some tests*/

    /*
     * bij = sigma(q_real_i_indx*q_real_j_indx+q_imag_i_indx+q_imag_j_indx) over index values
     * cij = s1(bij)*s0(rij)
     * Si = sigma(cij) over j
     * Ni = s2(Si)
     * N_MODE: N_total = sigma(Ni*inv_nbnum_i)
     * PHI_MODE: PHI = sigma(s3(rij)*Ni*Nj)
     * dNi/drk = ds2(Si)*dSi/drk 
     * dNi/drk = ds2(Si)*(sigma(dcij/drk))
     * dcij/drk = dbij/drk*ds1(bij)*s0(rij)+s1(bij)*ds0(rij)*drij/drk
     */

    /*
     * I need the S2 to calculated the diff /drk and also gi
     * The large number of neighbors is prohibiting to 
     * cache the cij values in the loop over j. 
     * Instead of caching them we recalculate their sum (gi) in the next loop.
     *
     * A solution is to cache the s0[j], ds0[j],
     * s1[j] and ds1[j] and multiply them by the ds2 or s2 
     * at the second loop over jj to obtain the wpair and wpair2
     * values.
     */

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      double bij = 0.0;
      double distx = x[i][0] - x[j][0];
      double disty = x[i][1] - x[j][1];
      double distz = x[i][2] - x[j][2];
      double r = sqrt(distx * distx + disty * disty + distz * distz);
      if (r < 1e-8 || r >= cutoff) continue;

      for (int indx = 0; indx < Q6_ARRAY_SIZE; ++indx) {
        bij += q6ms_real[i][indx] * q6ms_real[j][indx] + q6ms_imag[i][indx] * q6ms_imag[j][indx];
      }

      /* Some tests -->>*/
      if (std::abs(bij)> 1.0 + eps ) error->warning(FLERR, "This aint good {},{},{}", i, j,std::abs(bij));
      /*<<--Some tests*/
  
      double s0val, ds0val;
      double s1val, ds1val;
      s1(bij, s1val, ds1val);
      s0(r, s0val, ds0val);
      Si += s1val * s0val;

      if (switch_flag & S0_SW) {
        Z += s0val;
        if (!(mode & NO_DIFF)) {
          diff_Z_all[i][0] += 2.0*ds0val*distx/r;
          diff_Z_all[i][1] += 2.0*ds0val*disty/r;
          diff_Z_all[i][2] += 2.0*ds0val*distz/r;
        }
      } else Z += 1.0;

      s0j[j] = s0val;
      s1j[j] = s1val;
      ds0j[j] = ds0val;
      ds1j[j] = ds1val;
    }

    double ds2;
    double s2val;
    s2(Si, s2val, ds2);
    if (mode & N_MODE || mode & SIMPLE_N_MODE) {
      array_atom[i][val_col] = s2val;
    } else if (mode & PHI_MODE || mode & SIMPLE_PHI_MODE) {
      array_atom[i][second_val_col] = s2val;
    }
    Ni[i] = s2val;
    ds2i[i] = ds2;
    Q6_sum += s2val;

    /*
     * We have cached the s0, s1, ds0 and ds1 values.
     * In the current loop we fill the gi and 
     * also calculate the  distance dependent terms (ds0) of the
     * diffNi/diffri or diffN_total/diffri
     */
    
    // Check if we need diffs or not!
    if (!(mode & NO_DIFF)) {
      //Adding the distance dependent sentence
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        const double distx = x[i][0] - x[j][0];
        const double disty = x[i][1] - x[j][1];
        const double distz = x[i][2] - x[j][2];
        const double r = sqrt(distx * distx + disty * disty + distz * distz);
        if (r < 1e-8 || r >= cutoff) continue;

        const double wpair  = ds2 * s0j[j] * ds1j[j];
        const double wpair2 = ds2 * s1j[j] * ds0j[j];

        for (int indx = 0; indx < Q6_ARRAY_SIZE; ++indx) { 
          gi_real[i][indx] += wpair * q6ms_real[j][indx];
          gi_imag[i][indx] += wpair * q6ms_imag[j][indx];
        }

        const double diff_x = wpair2 * distx / r;
        const double diff_y = wpair2 * disty / r;
        const double diff_z = wpair2 * distz / r;
        if (mode & N_MODE) {
          // Since wpair2[i][j] = ds2(i)*s1(ij)*ds0(ij) this term is not symmetric
          // Ni contributions to the dN/dri  and dN/drj
          array_atom[i][diff_x_col] += diff_x;
          array_atom[i][diff_y_col] += diff_y;
          array_atom[i][diff_z_col] += diff_z;
          hj[j][0] -= diff_x;
          hj[j][1] -= diff_y;
          hj[j][2] -= diff_z;
        } else if (mode & PHI_MODE) {
          diff_Ni[i][0] += diff_x;
          diff_Ni[i][1] += diff_y;
          diff_Ni[i][2] += diff_z;
        } else if (mode & SIMPLE_PHI_MODE) {
          // No S2, so wpair2 is symmeteric
          // Contribution of Nj to dNtotal/dri is the same as Ni -->2.0*
          diff_Ni[i][0] += diff_x;
          diff_Ni[i][1] += diff_y;
          diff_Ni[i][2] += diff_z;
          diff_Ntotal[i][0] += 2.0 * diff_x;
          diff_Ntotal[i][1] += 2.0 * diff_y;
          diff_Ntotal[i][2] += 2.0 * diff_z;
        } else if (mode & SIMPLE_N_MODE) {
          // No S2, so wpair2 is symmeteric
          array_atom[i][diff_x_col] += 2.0*diff_x;
          array_atom[i][diff_y_col] += 2.0*diff_y;
          array_atom[i][diff_z_col] += 2.0*diff_z;
        }
      }
    }

    /*
     * gi = ds2*sigma(wij*qj) 
     * N_MODE : CV = sigma(qi*gi) dCV/dri = sigma(gi*dqi/dri (PART I) + qi*(wij*sigmadqj/dri) (PART 2))
     * PART 2  = gi*dqi/drj
     * PHI_MODE: Ni = gi*qi dNi/dri = gi*dqi/dri (PART I) + qi*(wij*sigmadqj/dri) (PART 2)
     * PART 2 = qi*dqi/drj
     */
    // This part calculates the qj*dqi/dri dependent component of diffNi/diffri or diffN_total/diffri
    // (the  gi*dqi/dri (qi*sigma(dqj/dri)) is calculated in the next part!)
    // cij = sigma (qi*qj)
    // dcij/dri = dqi/dri * gi + qi*dgi/dri (in the next section)
    // N_MODE : CV = sigma (gi*qi) --> dCV/dri = 2.0*gi*dqi/dri+...
    // PHI_MODE : Ni = gi*qi --> dNi/dri = gi*dqi/dri+...

    if (!(mode & NO_DIFF)) {
      for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
        const double diff_x = diff_q6ms_real[i][indx][0] * gi_real[i][indx] +
          diff_q6ms_imag[i][indx][0] * gi_imag[i][indx];
        const double diff_y = diff_q6ms_real[i][indx][1] * gi_real[i][indx] +
          diff_q6ms_imag[i][indx][1] * gi_imag[i][indx];
        const double diff_z = diff_q6ms_real[i][indx][2] * gi_real[i][indx] +
          diff_q6ms_imag[i][indx][2] * gi_imag[i][indx];
        if (mode & N_MODE) {
          array_atom[i][diff_x_col] += diff_x;
          array_atom[i][diff_y_col] += diff_y;
          array_atom[i][diff_z_col] += diff_z;
        } else if (mode & PHI_MODE) {
          diff_Ni[i][0] += diff_x;
          diff_Ni[i][1] += diff_y;
          diff_Ni[i][2] += diff_z;
        } else if (mode & SIMPLE_PHI_MODE) {
          diff_Ni[i][0] += diff_x;
          diff_Ni[i][1] += diff_y;
          diff_Ni[i][2] += diff_z;
          diff_Ntotal[i][0] += 2.0 * diff_x;
          diff_Ntotal[i][1] += 2.0 * diff_y;
          diff_Ntotal[i][2] += 2.0 * diff_z;
        } else if (mode & SIMPLE_N_MODE) {
          array_atom[i][diff_x_col] += 2.0*diff_x;
          array_atom[i][diff_y_col] += 2.0*diff_y;
          array_atom[i][diff_z_col] += 2.0*diff_z;
        }
      }
    }

    /* contribution of the qi to the dNi/drj [diffN_total/diffrj] 
     * which is equal to the qj*dqi/drj [gi*dqi/drj]
     * rj can be any neighbors of the i including ghost atoms and that is the reason why 
     * there is a need for a reverse communication of the hj
     * dqi/drj is calculated on the fly
     */

    if (!(mode & NO_DIFF)) { 
      for (int jj = 0; jj < jnum; jj++) {
        int j = jlist[jj];
        j &= NEIGHMASK;
        double distx = x[j][0] - x[i][0];
        double disty = x[j][1] - x[i][1];
        double distz = x[j][2] - x[i][2];
        double r = sqrt(distx * distx + disty * disty + distz * distz);
        if (r < 1e-8 || r >= cutoff) continue;
        std::array<double, N_DIM> distance = {distx, disty, distz};

        calculate_dq6i_drj(distance, q6ms_real[i], q6ms_imag[i], inv_nbnum_i[i], inv_q6_norm_i[i],
                         dqi_drj_real, dqi_drj_imag);

        // adding hj components
        if (mode & N_MODE) {
          for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
            for (int dim = 0; dim < N_DIM; dim++) {
              hj[j][dim] += dqi_drj_real[indx][dim] * gi_real[i][indx];
              hj[j][dim] += dqi_drj_imag[indx][dim] * gi_imag[i][indx];
            }
          }
        } else if (mode & PHI_MODE) {
          // Since the Nj = s2j*sigma(wjk*qj*qk) and on this node
          // we still do not have the s2 for all the neighbors we do not add anything to hj yet!
        } else if (mode & SIMPLE_PHI_MODE) {
          for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
            for (int dim = 0; dim < N_DIM; dim++) {
              hj[j][dim] += dqi_drj_real[indx][dim] * q6ms_real[i][indx];
              hj[j][dim] += dqi_drj_imag[indx][dim] * q6ms_imag[i][indx];
              hj2[j][dim] += 2.0 * dqi_drj_real[indx][dim] * gi_real[i][indx];
              hj2[j][dim] += 2.0 * dqi_drj_imag[indx][dim] * gi_imag[i][indx];
            }
          }
        } else if (mode & SIMPLE_N_MODE) {
          for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
            for (int dim = 0; dim < N_DIM; dim++) {
              hj[j][dim] += 2.0*dqi_drj_real[indx][dim]*gi_real[i][indx];
              hj[j][dim] += 2.0*dqi_drj_imag[indx][dim]*gi_imag[i][indx];
            }
          }

        }


      }
    }



  }


  // The S2_SW breakes the symmetry
  if (!(mode & NO_DIFF) && (switch_flag & S2_SW)) {
    comm_forward = DS2_FOR_STRIDE;
    forward_mode = DS2_TRANSFER;
    comm->forward_comm(this);

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jnum = numneigh[i];
      jlist = firstneigh[i];
      if (!(mask[i] & groupbit)) continue;
      if (!(mode & ALL_TYPES) && type[i] != chosen_type) continue;

      double gi_prime_real[Q6_ARRAY_SIZE];
      double gi_prime_imag[Q6_ARRAY_SIZE];

      std::fill_n(gi_prime_real,Q6_ARRAY_SIZE,0.0);
      std::fill_n(gi_prime_imag,Q6_ARRAY_SIZE,0.0);

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        double bij = 0.0;
        double distx = x[i][0] - x[j][0];
        double disty = x[i][1] - x[j][1];
        double distz = x[i][2] - x[j][2];
        double r = sqrt(distx * distx + disty * disty + distz * distz);
        if (r < 1e-8 || r >= cutoff) continue;
        for (int indx = 0; indx < Q6_ARRAY_SIZE; ++indx) {
          bij += q6ms_real[i][indx] * q6ms_real[j][indx] + q6ms_imag[i][indx] * q6ms_imag[j][indx];
        }
        double s0val, ds0val;
        double s1val, ds1val;
        s0(r, s0val, ds0val);
        s1(bij, s1val, ds1val);
        s0j[j] = s0val;
        ds1j[j] = ds1val;
        double wpair = ds2i[j]*ds1val*s0val;

        if (mode & N_MODE) {
          for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
            gi_prime_real[indx] += wpair*q6ms_real[j][indx];
            gi_prime_imag[indx] += wpair*q6ms_imag[j][indx];
          }
        } 


      }


      if (mode & N_MODE) {
        for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
          array_atom[i][diff_x_col] += gi_prime_real[indx]*diff_q6ms_real[i][indx][0] + gi_prime_imag[indx]*diff_q6ms_imag[i][indx][0];
          array_atom[i][diff_y_col] += gi_prime_real[indx]*diff_q6ms_real[i][indx][1] + gi_prime_imag[indx]*diff_q6ms_imag[i][indx][1]; 
          array_atom[i][diff_z_col] += gi_prime_real[indx]*diff_q6ms_real[i][indx][2] + gi_prime_imag[indx]*diff_q6ms_imag[i][indx][2];  
        }
      } 


      // We need to have the gi_prime_* array first
      for (int jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        double bij = 0.0;
        double distx = x[j][0] - x[i][0];
        double disty = x[j][1] - x[i][1];
        double distz = x[j][2] - x[i][2];
        double r = sqrt(distx * distx + disty * disty + distz * distz);
        if (r < 1e-8 || r >= cutoff) continue;
        std::array<double, N_DIM> distance = {distx, disty, distz};

        calculate_dq6i_drj(distance, q6ms_real[i], q6ms_imag[i], inv_nbnum_i[i], inv_q6_norm_i[i],
                         dqi_drj_real, dqi_drj_imag);

        if (mode & N_MODE) {
          for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
            for (int dim = 0; dim < 3; dim++) {
              hj[j][dim] += gi_prime_real[indx]*dqi_drj_real[indx][dim] + gi_prime_imag[indx]*dqi_drj_imag[indx][dim];
            }
          }
        } else if (mode & PHI_MODE) {
          double wpair = ds2i[j]*ds1j[j]*s0j[j];
          for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
            for (int dim = 0; dim < 3; dim++) {
              hj[j][dim] += wpair*(q6ms_real[j][indx]*dqi_drj_real[indx][dim] + q6ms_imag[j][indx]*dqi_drj_imag[indx][dim]);
            }
          }
        }
      }



    }
  }
  

  if (!(mode & NO_DIFF)) {
    // Transferring the hj from the ghost atoms
    if (mode & SIMPLE_PHI_MODE) {
      comm_reverse = S_REV_STRIDE;
    } else {
      comm_reverse = N_REV_STRIDE;
    }
    comm->reverse_comm(this);

    // Adding the contribution of atom j to the diffN_total/dri or diffNi/diffri
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      if (!(mask[i] & groupbit)) continue;
      if (!(mode & ALL_TYPES) && type[i] != chosen_type) continue;

      const double diff_x = hj[i][0];
      const double diff_y = hj[i][1];
      const double diff_z = hj[i][2];
      if (mode & N_MODE ) {
        array_atom[i][diff_x_col] += diff_x;
        array_atom[i][diff_y_col] += diff_y;
        array_atom[i][diff_z_col] += diff_z;
      } else if (mode & PHI_MODE) {
        diff_Ni[i][0] += diff_x;
        diff_Ni[i][1] += diff_y;
        diff_Ni[i][2] += diff_z;
      } else if (mode & SIMPLE_PHI_MODE) {
        diff_Ni[i][0] += diff_x;
        diff_Ni[i][1] += diff_y;
        diff_Ni[i][2] += diff_z;
        diff_Ntotal[i][0] += hj2[i][0];
        diff_Ntotal[i][1] += hj2[i][1];
        diff_Ntotal[i][2] += hj2[i][2];
      } else if (mode & SIMPLE_N_MODE) {
        array_atom[i][diff_x_col] += diff_x;
        array_atom[i][diff_y_col] += diff_y;
        array_atom[i][diff_z_col] += diff_z;
      }
    }
  }

  // Now we have dCV/dri (N_MODE) or dNi/dri (PHI_MODE)


  /*
   * phi = sigma(K(rij)*Ni*Nj)
   * Since the Nj values which are not neighbor of i do not
   * contribute to the dphi/dri we consider phi as 2*Ni(K(rij)*sigmaNj) where j is neighbor of i
   * and Gi = K(rij)*sigmaNj
   * So dphi/dri = 2.0*(diffNi/diffri*Gi + sigma(dK(rij)/dri*Ni*Nj) + Gj*dNj/dri))
   * the third term is calculated in the neighbor ranks.
   */

  /* 
    * So the following steps are used in the calculation of diffphi/diffri
    * step 0: forward transfer the N[j] (needed for the calculation of G[i])
    * step 1: Gi is calculated.
    * step 2: Gi is forward communicated to ghost atoms
    * step 3: term 1 (diffNi/diffri*Gi)  is calculated - We have diffNi/diffri on this rank.
    * step 4: term 2 components for each j  (dK(rij)/dri*Ni*Nj) is added.
    * step 5: term 3 (Gj*dNj/dri)) components (for atom i (step 5A dNj/dri) and atom j (steps 5B-5D - dNi/drj)) are calculated on the fly
    * step 5A: contribution of dqi/dri through various Nk terms in the Gi*sigma(K(ik)*dNk/drj) (dphi/dri)
    * step 5B: Contribution of the dqi/drj to the Gj*sigma(K(rjk)*dNk/drj) through Ni the term (dphi/drj)
    * step 5C: Contribution of the dqi/drj through Nk term of other neighbors of i expect for j to the dphi/drj (j has been added in Step 4A)
    * step 5D: The distance dependent term for each Ni in the Gj*dNi/drj is added.
    * This term is added since Ni = sigma(s0(rij)*s1(cij)).. This term is related to the diff(s0(rij))/drj
    * for instance the dq2/dr1 can also affect the dN2/dr1 = K(13)*(dq3/dr1*(q1+q2+q4)+q3*(dq1/dr1+dq2/dr1+dq4/dr1))
    * step 6: reverse comm of dphi/drj
    * step 7: adding the contribution from the atom j (hj) to the diff. 
    */

  /*
   * This is an example
   * G2*dN2/dr1 = G2*(dq2*(q1+q3)+q2*(dq1+dq3))
   * G3*dN3/dr1 = G3*N1*(dq3*(q1+q2+q4)+q3*(dq1+dq2+dq4))
   * G4*dN4/dr1 = G4)*N1(dq4*(q1+q3)+q4*(dq1+dq3))
   * 5A:  G2*dN2/dr1 += G2*dq1*q2 i == 1
   *      G3*dN3/dr1 += G3*dq1*q3 i == 1
   *      G4*dN4/dr1 += G4*dq1*q4 i == 1
   * 5B:  G2*dN2/dr1 += G2*dq2*(q1+q3) i == 2 j == 1
   *      G3*dN3/dr1 += G3*dq3*(q1+q2+q4) i == 3 j == 1
   *      G4*dN4/dr1 += G4*dq4*(q1+q3) i == 4 j == 1
   * 5C:  G2*dN2/dr1 += G2*q2*(dq3) i == 2
   *      G3*dN3/dr1 += G3*q3*(dq2+dq4) i == 3
   *      G4*dN4/dr1 += G4*q4*(dq3) i == 4
   * 
   */

  /*
   * A special case is the case where all the switches are off.
   * In this case diffphi/diffri = Gi*diffNi/diffri + (diffN_total/diffri - diffNi/diffri)
   * This implementation is more efficient than the general case.
   * As we have diffN_total/diffri we just need the Gi.
   */ 

  if (mode & PHI_MODE  || mode & SIMPLE_PHI_MODE) {
    /*
     * step 0
     */
    forward_mode = N_TRANSFER;
    comm_forward = N_FOR_STRIDE;
    comm->forward_comm(this);

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jnum = numneigh[i];
      jlist = firstneigh[i];
      if (!(mask[i] & groupbit)) continue;
      if (!(mode & ALL_TYPES) && type[i] != chosen_type) continue;

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        double distx = x[i][0] - x[j][0];
        double disty = x[i][1] - x[j][1];
        double distz = x[i][2] - x[j][2];
        double r = sqrt(distx * distx + disty * disty + distz * distz);
        if (r < 1e-8 || r >= cutoff) continue;

        double s3val, ds3;
        s3(r,s3val,ds3);
        phi_sum += Ni[i] * Ni[j] * s3val;

        /*
         * step 1
         */
        Gi[i] += s3val * Ni[j];

        W += s3val;
        
        if (!(mode & NO_DIFF)) {
          diff_W_all[i][0] += 2.0 * ds3 * distx / r; // factor 2.0 for ij+ji convention
          diff_W_all[i][1] += 2.0 * ds3 * disty / r;
          diff_W_all[i][2] += 2.0 * ds3 * distz / r;
        }


      }
    }

    /* 
     * step 2: 
     * forward comm Gi
     */
    forward_mode = G_TRANSFER;
    comm_forward = G_FOR_STRIDE;
    comm->forward_comm(this);
  }

  if (mode & PHI_MODE) {    
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jnum = numneigh[i];
      jlist = firstneigh[i];
      if (!(mask[i] & groupbit)) continue;
      if (!(mode & ALL_TYPES) && type[i] != chosen_type) continue;

      // phi[i] = Ni[i] * sigma(Kij*Ni[j]) = Ni[i]*Gi[i]
      array_atom[i][val_col] += Ni[i] * Gi[i];

      // The rest of the loop is related to the diff
      if (mode & NO_DIFF) continue; 

      /*
       * step 3 - Gi *diffNi/diffri
       */
      array_atom[i][diff_x_col] += 2.0 * diff_Ni[i][0] * Gi[i];
      array_atom[i][diff_y_col] += 2.0 * diff_Ni[i][1] * Gi[i];
      array_atom[i][diff_z_col] += 2.0 * diff_Ni[i][2] * Gi[i];




      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        double distx = x[j][0] - x[i][0];
        double disty = x[j][1] - x[i][1];
        double distz = x[j][2] - x[i][2];
        double r = sqrt(distx * distx + disty * disty + distz * distz);
        std::array<double, N_DIM> distance = {distx, disty, distz};
        if (r < 1e-8 || r >= cutoff) continue;
        /*
         * wpair=s3*ds2*s0*ds1
         * wpair2=s3*ds2*s1*ds0
         * wpair3=ds3
         */
        double wpair, wpair2, wpair3;
        double s0val, s1val, s3val;
        double ds0, ds1, ds3;

        /*
         * step 4
         * sigma(dK(rij)/dri*Ni*Nj)
         */
        s3(r,s3val,ds3);
        wpair3 = -ds3;

        array_atom[i][diff_x_col] += 2 * wpair3 * Ni[i] * Ni[j] * distx / r;
        array_atom[i][diff_y_col] += 2 * wpair3 * Ni[i] * Ni[j] * disty / r;
        array_atom[i][diff_z_col] += 2 * wpair3 * Ni[i] * Ni[j] * distz / r;

        /*
         * step 5:
         *
         */

        /*
         * step 5A
         * contribution of the dqi/dri to the simga(Gk*diffNk/diffri)
         * it contributes through the Nj terms in the 
         * Gj*diffNj/diffri
         */

        double cij = 0.0;
        for (int indx = 0; indx < Q6_ARRAY_SIZE; ++indx) {
          cij += q6ms_real[i][indx] * q6ms_real[j][indx] + q6ms_imag[i][indx] * q6ms_imag[j][indx];
        }
        // cache for step 5D
        Cjj[j] = cij;

        s0(r, s0val, ds0);
        s1(cij, s1val, ds1);

    
        // it contributes to the Nj that is the reason why ds2i[j]
        // Gi[j] contains the s3val
        // Gj*Nj
        wpair = ds2i[j] * ds1 * s0val * Gi[j];

        double qjdqi_dri[N_DIM];

        for (int dim = 0; dim < N_DIM; dim++) {
          qjdqi_dri[dim] = 0.0;
          for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
            qjdqi_dri[dim] += diff_q6ms_real[i][indx][dim] * q6ms_real[j][indx] +
                diff_q6ms_imag[i][indx][dim] * q6ms_imag[j][indx];
          }
        }

        array_atom[i][diff_x_col] += 2.0 * wpair * qjdqi_dri[0];
        array_atom[i][diff_y_col] += 2.0 * wpair * qjdqi_dri[1];
        array_atom[i][diff_z_col] += 2.0 * wpair * qjdqi_dri[2];

        /*
         * step 5B
         * Contribution of the dqi/drj to the sigma(Gk*dNk/drj)
         * and dphi/drj
         * It contributes to the Ni term in the sigma
         * Gi*dNi/drj
         */

        /*
         * It is possible that all of the neighbors of the atom j
         * do not belong to this rank.
         * If it is a ghost atom
         * we cannot have its nbnum and 
         * qnorm.. Accordingly these values have been forward communicated
         * for ghost atoms.
         */

        calculate_dq6i_drj(distance, q6ms_real[i], q6ms_imag[i], inv_nbnum_i[i], inv_q6_norm_i[i],
                           dqi_drj_real, dqi_drj_imag);

        // Ni contribution to the Nj*sigma(K(ij)*dN/dri)
        // gi_real and _imag have ds2i[i]*ds1*s0val
        // Gi[i] contains the s3val
        wpair = Gi[i];

        for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
          for (int dim = 0; dim < 3; dim++) {
            hj[j][dim] += wpair *
                (dqi_drj_real[indx][dim] * gi_real[i][indx] +
                 dqi_drj_imag[indx][dim] * gi_imag[i][indx]);
          }
        }
      }

      // Wanted to cache Cjj to be used in the loop over k 
      // and that is the reason why I needed another loop over jj
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        double distx = x[j][0] - x[i][0];
        double disty = x[j][1] - x[i][1];
        double distz = x[j][2] - x[i][2];
        double r = sqrt(distx * distx + disty * disty + distz * distz);
        std::array<double, N_DIM> distance = {distx, disty, distz};
        if (r < 1e-8 || r >= cutoff) continue;
        /*
         * step 5C
         * It is possible that other neighbors of i except for j contribute to the Nj*sigma(diffNl/drj)
         * This contribution is through Nk in which k!=j and k is neighbor of i
         * Those terms contribute through qi*qk
         */

        // k is a neighbor of i not j so we have its q6m thanks to the forward_comm
        // diffNk / diffrj -> diffqi/diffrj*qk
        for (int kk = 0; kk < jnum; kk++) {
          int k = jlist[kk];
          k &= NEIGHMASK;
          if (k == j) continue;
          double distxjk = x[j][0] - x[k][0];
          double distyjk = x[j][1] - x[k][1];
          double distzjk = x[j][2] - x[k][2];
          double rjk = sqrt(distxjk * distxjk + distyjk * distyjk + distzjk * distzjk);
          if (rjk < 1e-8 || rjk >= cutoff) continue;
          double distxik = x[i][0] - x[k][0];
          double distyik = x[i][1] - x[k][1];
          double distzik = x[i][2] - x[k][2];
          double rik = sqrt(distxik * distxik + distyik * distyik + distzik * distzik);
          if (rik < 1e-8 || rik >= cutoff) continue;

          /* It is the Gk*diffNk/diffrj contribution.
           * But s2 must be based on rik since its based on the qi*qk 
           * and ds1 must be based on the cik.
           * Since k is a neighbor of i, we have all of its q values thanks to the forward comm. 
           */

          double cik = Cjj[k];
          double s1ikval, ds1ik;
          double s0ikval, ds0ik;
          s1(cik, s1ikval, ds1ik);
          s0(rik, s0ikval, ds0ik);
          double wpair = ds2i[k] * ds1ik * s0ikval * Gi[k];
          for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++)
            for (int dim = 0; dim < N_DIM; dim++)
              hj[j][dim] += wpair *
                  (dqi_drj_real[indx][dim] * q6ms_real[k][indx] +
                   dqi_drj_imag[indx][dim] * q6ms_imag[k][indx]);
        }

        /*
         * step 5D: distance dependent term to the Gi*diffNi/drj
         */
        double s0val, ds0;
        double s1val, ds1;
        s1(Cjj[j], s1val, ds1);
        s0(r, s0val, ds0);
        double wpair = Gi[i] * ds2i[i] * s1val * ds0;
        for (int dim = 0; dim < 3; dim++)
          hj[j][dim] += wpair*distance[dim]/r;
      }
    }

    if (!(mode & NO_DIFF)) {
      /*
       * step 5
       * Transferring the hj from the ghost atoms
       */
      comm_reverse = N_REV_STRIDE;
      comm->reverse_comm(this);
      for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        if (!(mask[i] & groupbit)) continue;
        if (!(mode & ALL_TYPES) && type[i] != chosen_type) continue;

        /*
         * step 6
         */

        const double diff_x = hj[i][0];
        const double diff_y = hj[i][1];
        const double diff_z = hj[i][2];
        array_atom[i][diff_x_col] += 2.0 * diff_x;
        array_atom[i][diff_y_col] += 2.0 * diff_y;
        array_atom[i][diff_z_col] += 2.0 * diff_z;
      }
    }
  }

  if (mode & SIMPLE_PHI_MODE) {
    for (int ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      if (!(mask[i] & groupbit)) continue;
      if (!(mode & ALL_TYPES) && type[i] != chosen_type) continue;
      
      array_atom[i][val_col] += Ni[i]*Gi[i];

      if (!(mode & NO_DIFF)) {
        array_atom[i][diff_x_col] += 2.0*(diff_Ni[i][0]*Gi[i] + Ni[i]*(diff_Ntotal[i][0]-diff_Ni[i][0]));
        array_atom[i][diff_y_col] += 2.0*(diff_Ni[i][1]*Gi[i] + Ni[i]*(diff_Ntotal[i][1]-diff_Ni[i][1]));
        array_atom[i][diff_z_col] += 2.0*(diff_Ni[i][2]*Gi[i] + Ni[i]*(diff_Ntotal[i][2]-diff_Ni[i][2]));
      }
    }
  }

  MPI_Allreduce(&Q6_sum, &Q6_sum_all, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&phi_sum, &phi_sum_all, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&num_selected, &num_selected_all, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&Z,&Z_all,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&W,&W_all,1,MPI_DOUBLE,MPI_SUM,world);


  double num_double = static_cast<double>(num_selected_all);
  double Z_avg = Z_all / num_double;
  double W_avg = W_all / num_double;
  double scaling = 0.0;

  if (mode & (N_MODE | SIMPLE_N_MODE))
    scaling = (Z_all >= eps ? 1.0/Z_all : 0.0);
  else if (mode & (PHI_MODE | SIMPLE_PHI_MODE))
    scaling = (W_avg >= eps ? 1.0/W_avg : 0.0);
  phi_sum_all *= scaling;
  Q6_sum_all *= scaling;
  
  if (mode & (N_MODE | SIMPLE_N_MODE)) {
    vector[0] = Q6_sum_all;
    vector[1] = 0.0;
    scalar = Q6_sum_all;
  } else if (mode & (PHI_MODE | SIMPLE_PHI_MODE)) {
    vector[0] = Q6_sum_all;
    vector[1] = phi_sum_all;
    scalar = phi_sum_all;
  }

  if (mode & NO_DIFF)
    vector[2] = -1.0;


  double slopeModType = 0.0;
  double slopeModTypeAll = 0.0;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jnum = numneigh[i];
    jlist = firstneigh[i];
    
    if (!(mask[i] & groupbit)) continue;
    if (!(mode & ALL_TYPES) && type[i] != chosen_type) continue;

    if ( (mode & (PHI_MODE | SIMPLE_PHI_MODE)) && !(mode & NO_DIFF) ) {
      // diff of Z_avg 
      diff_Z_all[i][0] /= num_double;
      diff_Z_all[i][1] /= num_double;
      diff_Z_all[i][2] /= num_double;
      // calculating the diff of (Zavg^2*W_all)
      // 2*Zavg*diff_Zavg*W_all + Zavg^2*diff_W_all
      diff_Z_all[i][0] = 2.0*Z_avg*diff_Z_all[i][0]*W_all+Z_avg*Z_avg*diff_W_all[i][0];
      diff_Z_all[i][1] = 2.0*Z_avg*diff_Z_all[i][1]*W_all+Z_avg*Z_avg*diff_W_all[i][1];
      diff_Z_all[i][2] = 2.0*Z_avg*diff_Z_all[i][2]*W_all+Z_avg*Z_avg*diff_W_all[i][2];
    }

    array_atom[i][val_col] *= scaling;
    if (mode & (PHI_MODE | SIMPLE_PHI_MODE)) {
      array_atom[i][second_val_col] *= (1.0/Z_avg);
    }



    if (!(mode & NO_DIFF)) {
      // diff(x*s) = s*diffx + diffs*x;
      // here we are adding the x*diffs
      // s = 1/Z
      // x*diffs = -x*diffZ/Z^2
      // x*diffs = -xscaled * diffZ/Z
      // x*diffs = -xscaled * scaling*diffZ (scaling = 1/Z)
      array_atom[i][diff_x_col] *= scaling;
      array_atom[i][diff_y_col] *= scaling;
      array_atom[i][diff_z_col] *= scaling;
      array_atom[i][diff_x_col] += -scalar*scaling*diff_Z_all[i][0];
      array_atom[i][diff_y_col] += -scalar*scaling*diff_Z_all[i][1];
      array_atom[i][diff_z_col] += -scalar*scaling*diff_Z_all[i][2];

      double x_comp = array_atom[i][diff_x_col];
      double y_comp = array_atom[i][diff_y_col];
      double z_comp = array_atom[i][diff_z_col];

      double slope = std::sqrt(x_comp * x_comp + y_comp * y_comp + z_comp * z_comp);
      if (slope < min_slope) {
        slopeModType += 0.5;
        //const double target = std::abs(rng->gaussian()) * min_slope; // >= 0
        const double target = min_slope;
        // slope is under radical so it will never be negative..
        if (slope <= 0.0) {
          slopeModType += 0.5;
          /*if (comm->me == 0)
           error->warning(FLERR,"Dead gradient of zero in all the direction! \
                                 Setting a random value in the x-direction!");
          */
          //int dirs[3] = {diff_x_col,diff_y_col,diff_z_col};
          //int idx = (static_cast<in      iarg += 2;t>(std::abs(rng->uniform()*100.0))) %3;
          //array_atom[i][dirs[idx]] = target;
          array_atom[i][diff_x_col] = target;
        } else {
          const double s = target / slope;
          array_atom[i][diff_x_col] *= s;
          array_atom[i][diff_y_col] *= s;
          array_atom[i][diff_z_col] *= s;
        }
      }
    }

  }

  MPI_Allreduce(&slopeModType,&slopeModTypeAll,1,MPI_DOUBLE,MPI_SUM,world);
  vector[2] = slopeModTypeAll / num_double;

  //scalar = num_selected_all ? Q6_sum_all / static_cast<double>(num_selected_all) :0.0;
}

/* --------------------------------------------------------------------- */

int ComputeQ6SmoothAtom::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
                                           int * /*pbc*/)
{
  int m, j;
  m = 0;

  if (forward_mode & Q6_TRANSFER) {
    if (comm_forward != Q6_FOR_STRIDE)
      error->one(FLERR, "Wrong value in the comm_forward {}", comm_forward);
    for (int i = 0; i < n; i++) {
      j = list[i];
      for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
        buf[m++] = q6ms_real[j][indx];
        buf[m++] = q6ms_imag[j][indx];
      }
    }
  } else if (forward_mode & N_TRANSFER) {
    if (comm_forward != N_FOR_STRIDE)
      error->one(FLERR, "Wrong value in the comm_forward {}", comm_forward);
    for (int i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = Ni[j];
    }
  } else if (forward_mode & G_TRANSFER) {
    if (comm_forward != G_FOR_STRIDE)
      error->one(FLERR, "Wrong value in the comm_forward {}", comm_forward);
    for (int i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = Gi[j];
    }
  } else if (forward_mode & DS2_TRANSFER) {
    if (comm_forward != DS2_FOR_STRIDE)
      error->one(FLERR, "Wrong value in the comm_forward {}", comm_forward);
    for (int i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ds2i[j];
    }
  } else
    error->one(FLERR, "Wrong forward_comm flag");

  return m;
}

/* ------------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::unpack_forward_comm(int n, int first, double *buf)
{
  int j, m, last;

  m = 0;
  last = first + n;

  if (forward_mode & Q6_TRANSFER) {
    if (comm_forward != Q6_FOR_STRIDE)
      error->one(FLERR, "Wrong value in the comm_forward {}", comm_forward);
    for (j = first; j < last; j++) {
      for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
        q6ms_real[j][indx] = buf[m++];
        q6ms_imag[j][indx] = buf[m++];
      }
    }
  } else if (forward_mode & N_TRANSFER) {
    if (comm_forward != N_FOR_STRIDE)
      error->one(FLERR, "Wrong value in the comm_forward {}", comm_forward);
    for (j = first; j < last; j++) { Ni[j] = buf[m++]; }
  } else if (forward_mode & G_TRANSFER) {
    if (comm_forward != G_FOR_STRIDE)
      error->one(FLERR, "Wrong value in the comm_forward {}", comm_forward);
    for (j = first; j < last; j++) { Gi[j] = buf[m++]; }
  } else if (forward_mode & DS2_TRANSFER) {
    if (comm_forward != DS2_FOR_STRIDE)
      error->one(FLERR,"Wrong value in the comm_forward {}", comm_forward);
    for (j = first; j < last; j++) {ds2i[j] = buf[m++];}
  } else 
    error->one(FLERR, "Wrong forward_comm flag");
}

/* ----------------------------------------------------------------------
   Q6sum = sqrt(sigma(average(Y6mreal)^2+average(Y6mimage)^2)overm)_atomi + 
   sqrt(sigma(average(Y6mreal)^2+average(Y6mimage)^2)overm)_atomj + ...
   even though the dY6m/dx(rij) = -dY6m/dx(ji) since they are under radical
   and powered by two we need to have ghost atom exchange

*/

int ComputeQ6SmoothAtom::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m, last;
  m = 0;
  last = first + n;
  if ( ((mode & SIMPLE_PHI_MODE) && (comm_reverse != S_REV_STRIDE)) ||
       (!(mode & SIMPLE_PHI_MODE) && (comm_reverse != N_REV_STRIDE)) )
       error->one(FLERR, "Wrong value in the comm_reverse {}", comm_reverse);

  if (mode & SIMPLE_PHI_MODE) {
    for (i = first; i < last; i++) {
      for (int dim = 0; dim < N_DIM; dim++) buf[m++] = hj[i][dim];
      for (int dim = 0; dim < N_DIM; dim++) buf[m++] = hj2[i][dim];
    }
  } else {
    for (i = first; i < last; i++) {
      for (int dim = 0; dim < N_DIM; dim++) buf[m++] = hj[i][dim];
    }
  }
  return m;
}

/* ----------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i, j, m;

  m = 0;
  if (mode & SIMPLE_PHI_MODE) {
    for (i = 0; i < n; i++) {
      j = list[i];
      for (int dim = 0; dim < N_DIM; dim++) hj[j][dim] += buf[m++];
      for (int dim = 0; dim < N_DIM; dim++) hj2[j][dim] += buf[m++];
    }
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      for (int dim = 0; dim < N_DIM; dim++) hj[j][dim] += buf[m++];
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::orient(const double &input, const double &beta, const double &x0,
                                 double &output, double &diff,const double& min_slope)
{
  output = 1.0 / (1.0 + std::exp(-beta * (input - x0)));
  diff = beta * output * (1.0 - output);

  // avoiding dead gradient
  if (std::abs(diff) <= min_slope) diff = smooth_floor(diff,min_slope);
}

/* --------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::dist(const double &input, const double &cutoff, double &output,
                               double &diff, const double& min_slope, const double& threshold)
{
  diff = output = 0.0;
  const double *const coeff = dist_coeffs;
  double r0 = threshold * cutoff;
  double x = (input - r0) / (cutoff - r0);
  if (x <= 0.0) {
    output = 1.0;
  } else if (x >= 1.0) {
    output = 0.0;
  } else {
    output += coeff[0];
    for (int i = 1; i < dist_deg; i++) {
      output += coeff[i] * pow_fun(x, i);
      diff += i * coeff[i] * pow_fun(x, i - 1);
    }
  }
  // The chain rule
  diff *= 1.0 / (cutoff - r0);

  // avoiding dead gradient
  if (std::abs(diff) <= min_slope)

  if (x > 0.0 && x < 1.0)
     diff = -smooth_floor(-diff,min_slope);
}


/* ---------------------------------------------------------------------
   since the size of q6mi_dxj become very large we need to calculate the
    diff on the fly with this function 
   --------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::calculate_dq6i_drj(const std::array<double, N_DIM> &dist,
                                             const double *q6m_real_i, const double *q6m_imag_i,
                                             const double &inv_nbnum, const double &inv_q6_norm,
                                             double **dqi_drj_real, double **dqi_drj_imag)
{
  std::array<double, 104> Y6m = calculate_Y6m(dist);

  /*
   * It is possible that all of the neighbors of the atom k
   * do not belong to this rank, if it is a ghost atom.
   * That is the reason why we cannot have its nbnum and 
   * qnorm.. Accordingly these values have been forward communicated
   * for ghost atoms.
   */
  for (int deg = -6; deg <= 6; deg++) {
    int offset = (deg + 6) * 8;
    int indx = deg + 6;
    dqi_drj_real[indx][0] = Y6m[offset + 2] * inv_nbnum;
    dqi_drj_imag[indx][0] = Y6m[offset + 3] * inv_nbnum;
    dqi_drj_real[indx][1] = Y6m[offset + 4] * inv_nbnum;
    dqi_drj_imag[indx][1] = Y6m[offset + 5] * inv_nbnum;
    dqi_drj_real[indx][2] = Y6m[offset + 6] * inv_nbnum;
    dqi_drj_imag[indx][2] = Y6m[offset + 7] * inv_nbnum;
  }

  double dq_norm_drj[N_DIM] = {0.0, 0.0, 0.0};
  for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
    for (int dim = 0; dim < N_DIM; dim++)
      dq_norm_drj[dim] +=
          dqi_drj_real[indx][dim] * q6m_real_i[indx] + dqi_drj_imag[indx][dim] * q6m_imag_i[indx];
  }

  for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
    for (int dim = 0; dim < N_DIM; dim++) {
      dqi_drj_real[indx][dim] *= inv_q6_norm;
      dqi_drj_imag[indx][dim] *= inv_q6_norm;
      dqi_drj_real[indx][dim] -= q6m_real_i[indx] * dq_norm_drj[dim] * inv_q6_norm;
      dqi_drj_imag[indx][dim] -= q6m_imag_i[indx] * dq_norm_drj[dim] * inv_q6_norm;
    }
  }
}

/* ----------------------------------------------------------------------
  computing the Y6m values
------------------------------------------------------------------------- */

std::array<double, 104> ComputeQ6SmoothAtom::calculate_Y6m(const std::array<double, 3> &dist)
{
  double x = dist[0];
  double y = dist[1];
  double z = dist[2];

  double r = std::sqrt(x * x + y * y + z * z);
  double rxy = std::sqrt(x * x + y * y);
  constexpr double eps = 1e-6;

  std::array<double, 104> Y6m;
  std::fill_n(Y6m.begin(), Y6m.size(), 0.0);
  if (r < eps) return Y6m;


  // Handle pole: rxy == 0 → x=y=0, on z-axis (φ undefined)
  if (rxy < eps) {
    const double ct = z / r;           // cos(theta) = ±1
    double ct_pow[7]; ct_pow[0] = 1.0;
    for (int k = 1; k <= 6; ++k) ct_pow[k] = ct_pow[k-1] * ct;

    // Coefficient for m=0 (matches your C0 above)
    const double C0 = (1.0/32.0) * std::sqrt(13.0/M_PI);

    // Associated Legendre for l=6,m=0 as polynomial in cos(theta)
    const double second = 231.0*ct_pow[6] - 315.0*ct_pow[4] + 105.0*ct_pow[2] - 5.0;

    // Fill only m=0 (deg = 0). Re part is nonzero, Im part is zero.
    const int off = (0 + 6) * 8;
    Y6m[off + 0] = C0 * /*first*/ 1.0 * second; // Re(Y_6^0)
    Y6m[off + 1] = 0.0;                         // Im(Y_6^0) = 0

    // Derivatives at the pole: set to zero (stable convention)
    Y6m[off + 2] = Y6m[off + 3] = 0.0; // d/dx of Re/Im
    Y6m[off + 4] = Y6m[off + 5] = 0.0; // d/dy of Re/Im
    Y6m[off + 6] = Y6m[off + 7] = 0.0; // d/dz of Re/Im

    // m≠0 entries already zeroed by the std::fill_n earlier
    return Y6m;
  }

  const double ct = z / r;                // cos(theta)
  const double st = std::sqrt(std::max(0.0, 1.0 - ct*ct)); // sin(theta), robust to FP
  const double c1 = x / rxy;              // cos(phi)
  const double s1 = y / rxy;              // sin(phi)

  double dtheta_dx = (x * z) / (r * r * rxy);
  double dtheta_dy = (y * z) / (r * r * rxy);
  double dtheta_dz = -rxy / (r * r);

  double dphi_dx = -y / (rxy * rxy);
  double dphi_dy = x / (rxy * rxy);
  double dphi_dz = 0.0;


  // Precompute sin(m*phi), cos(m*phi) for m=0..6 via recurrence
  double c[7], s[7];
  c[0]=1.0; s[0]=0.0;
  c[1]=c1;  s[1]=s1;
  for (int m=2; m<=6; ++m) {
    c[m] = c1*c[m-1] - s1*s[m-1];
    s[m] = s1*c[m-1] + c1*s[m-1];
  }
  
  // Precompute sin^m(theta) and cos^m(theta)
  double st_pow[7]; st_pow[0]=1.0;
  for (int m=1; m<=6; ++m) st_pow[m] = st_pow[m-1]*st;
  double ct_pow[7]; ct_pow[0]=1.0;
  for (int k=1; k<=6; ++k) ct_pow[k] = ct_pow[k-1]*ct;
  
  // Coefficient magnitudes for m=0..6 (Condon–Shortley phase handled below)
  const double C0 = (1.0/32.0)*std::sqrt(13.0/M_PI);
  const double C1 = (1.0/16.0)*std::sqrt(273.0/(2.0*M_PI));
  const double C2 = (1.0/64.0)*std::sqrt(1365.0/M_PI);
  const double C3 = (1.0/32.0)*std::sqrt(1365.0/M_PI);
  const double C4 = (3.0/32.0)*std::sqrt(91.0/(2.0*M_PI));
  const double C5 = (3.0/32.0)*std::sqrt(1001.0/M_PI);
  const double C6 = (1.0/64.0)*std::sqrt(3003.0/M_PI);
  
  // Helper to write a single deg = ±m slot
  auto emit = [&](int deg, double coeff, double first, double dfirst,
                    double second, double dsecond, double cos_mphi, double sin_mphi)
  {
    const int   off = (deg + 6) * 8;
    const double Re = coeff * cos_mphi * first * second;
    const double Im = coeff * sin_mphi * first * second;
  
    const double dRe_dtheta = coeff * cos_mphi * (first*dsecond + second*dfirst);
    const double dIm_dtheta = coeff * sin_mphi * (first*dsecond + second*dfirst);
    const double dRe_dphi   = -coeff * deg * sin_mphi * first * second;
    const double dIm_dphi   =  coeff * deg * cos_mphi * first * second;
  
    Y6m[off + 0] = Re;
    Y6m[off + 1] = Im;
    Y6m[off + 2] = dRe_dtheta * dtheta_dx + dRe_dphi * dphi_dx;
    Y6m[off + 3] = dIm_dtheta * dtheta_dx + dIm_dphi * dphi_dx;
    Y6m[off + 4] = dRe_dtheta * dtheta_dy + dRe_dphi * dphi_dy;
    Y6m[off + 5] = dIm_dtheta * dtheta_dy + dIm_dphi * dphi_dy;
    Y6m[off + 6] = dRe_dtheta * dtheta_dz + dRe_dphi * dphi_dz;
    Y6m[off + 7] = dIm_dtheta * dtheta_dz + dIm_dphi * dphi_dz;
  };
  
  // Handy lambda to push both +m and -m, applying the odd-m sign flip (Condon–Shortley).
  auto emit_pair = [&](int m, double C,
                         double second, double dsecond)
  {
    // first_theta and d/dθ
    const double first  = st_pow[m];
    const double dfirst = (m==0) ? 0.0 : m * st_pow[m-1] * ct;
  
    // +m: coeff = base * (odd ? -1 : +1)
    double coeff_p = ((m & 1) ? -C : +C);
    emit(+m, coeff_p, first, dfirst, second, dsecond, c[m], +s[m]);
  
    // -m: coeff = base * (odd ? +1 : +1)  (i.e., flip for odd m)
    double coeff_m = ((m & 1) ? +C : +C);
    emit(-m, coeff_m, first, dfirst, second, dsecond, c[m], -s[m]);
  };
  
  // m = 0
  {
    const int   m = 0;
    const double first  = 1.0;
    const double dfirst = 0.0;
    const double second =
        231.0*ct_pow[6] - 315.0*ct_pow[4] + 105.0*ct_pow[2] - 5.0;
    const double dsecond =
        -st * (231.0*6.0*ct_pow[5] - 315.0*4.0*ct_pow[3] + 105.0*2.0*ct_pow[1]);
  
    emit(0, C0, first, dfirst, second, dsecond, 1.0, 0.0); // cos(0φ)=1, sin(0φ)=0
  }
  
  // m = 1
  {
    const double second   = 33.0*ct_pow[5] - 30.0*ct_pow[3] + 5.0*ct;
    const double dsecond  = -st*(33.0*5.0*ct_pow[4] - 30.0*3.0*ct_pow[2] + 5.0);
    emit_pair(1, C1, second, dsecond);
  }
  
  // m = 2
  {
    const double second   = 33.0*ct_pow[4] - 18.0*ct_pow[2] + 1.0;
    const double dsecond  = -st*(4.0*33.0*ct_pow[3] - 2.0*18.0*ct);
    emit_pair(2, C2, second, dsecond);
  }
  
  // m = 3
  {
    const double second   = 11.0*ct_pow[3] - 3.0*ct;
    const double dsecond  = -st*(11.0*3.0*ct_pow[2] - 3.0);
    emit_pair(3, C3, second, dsecond);
  }
  
  // m = 4
  {
    const double second   = 11.0*ct_pow[2] - 1.0;
    const double dsecond  = -st*(2.0*11.0*ct);
    emit_pair(4, C4, second, dsecond);
  }
  
  // m = 5
  {
    const double second   = ct;
    const double dsecond  = -st;
    emit_pair(5, C5, second, dsecond);
  }
  
  // m = 6
  {
    const double second   = 1.0;
    const double dsecond  = 0.0;
    emit_pair(6, C6, second, dsecond);
  }

  return Y6m;
}
