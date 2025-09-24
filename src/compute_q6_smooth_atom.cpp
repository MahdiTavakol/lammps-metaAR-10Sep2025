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

using namespace LAMMPS_NS;

enum { Q6_TRANSFER = 1 << 1, N_TRANSFER = 1 << 2 };

enum { N_MODE = 0, PHI_MODE = 1 << 1 };

enum { Q6_STRIDE = 26, N_STRIDE = 1 };

// parameter to avoid dead gradient issue.
static double constexpr min_slope = 0.02;

/* ---------------------------------------------------------------------- */

ComputeQ6SmoothAtom::ComputeQ6SmoothAtom(LAMMPS *lmp, int narg, char **arg) :
    ComputeDiffAtom{lmp, narg, arg}, mode{N_MODE}, chosen_type{-1}, cutoff{3.2}, q6ms_real{nullptr},
    q6ms_imag{nullptr}, diff_q6ms_real{nullptr}, diff_q6ms_imag{nullptr}, inv_q6_norm_i{nullptr},
    inv_nbnum_i{nullptr}, Ni{nullptr}, ds2i{nullptr}, diff_Ni{nullptr}, gi_real{nullptr},
    gi_imag{nullptr}, Gi{nullptr}, Ci{nullptr}, hj{nullptr}, forward_mode{Q6_TRANSFER},
    s0j{nullptr}, s1j{nullptr}, ds0j{nullptr}, ds1j{nullptr}, dqi_drj_real{nullptr}, dqi_drj_imag{
                                                                                         nullptr}
{
  chosen_type = utils::numeric(FLERR, arg[3], false, lmp);
  cutoff = utils::numeric(FLERR, arg[4], false, lmp);

  // before calling the comm->forward this parameter is set since it has two different values.
  comm_forward = Q6_STRIDE;
  comm_reverse = 3;

  int iarg = 5;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "sigmoid") == 0) {
      if (narg < iarg + 5) error->all(FLERR, "Illegal q6-smooth/atom command");
      beta1 = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      x01 = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      beta2 = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      x02 = utils::numeric(FLERR, arg[iarg + 4], false, lmp);
      iarg += 5;
    } else if (strcmp(arg[iarg], "phi") == 0) {
      mode = PHI_MODE;
      iarg++;
    } else
      error->all(FLERR, "Illegal compute q6-smooth/atom command");
  }
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
  memory->create(gi_real, nmax, Q6_ARRAY_SIZE, "compute_q6_smooth_atom:gi_real");
  memory->create(gi_imag, nmax, Q6_ARRAY_SIZE, "compute_q6_smooth_atom:gi_imag");
  memory->create(Gi, nmax, "compute_q6_smooth_atom:Gi");
  memory->create(Ci, nmax, N_DIM, "compute_q6_smooth_atom:Ci");
  memory->create(hj, nmax, N_DIM, "compute_q6_smooth_atom:hj");
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
}

/* ----------------------------------------------------------------------- */

ComputeQ6SmoothAtom::~ComputeQ6SmoothAtom()
{
  if (q6ms_real) memory->destroy(q6ms_real);
  if (q6ms_imag) memory->destroy(q6ms_imag);
  if (diff_q6ms_real) memory->destroy(diff_q6ms_real);
  if (diff_q6ms_imag) memory->destroy(diff_q6ms_imag);
  if (inv_q6_norm_i) memory->destroy(inv_q6_norm_i);
  if (inv_nbnum_i) memory->destroy(inv_nbnum_i);
  if (Ni) memory->destroy(Ni);
  if (ds2i) memory->destroy(ds2i);
  if (diff_Ni) memory->destroy(diff_Ni);
  if (gi_real) memory->destroy(gi_real);
  if (gi_imag) memory->destroy(gi_imag);
  if (Gi) memory->destroy(Gi);
  if (Ci) memory->destroy(Ci);
  if (hj) memory->destroy(hj);
  if (s0j) memory->destroy(s0j);
  if (s1j) memory->destroy(s1j);
  if (ds0j) memory->destroy(ds0j);
  if (ds1j) memory->destroy(ds1j);
  if (dqi_drj_real) memory->destroy(dqi_drj_real);
  if (dqi_drj_imag) memory->destroy(dqi_drj_imag);
}

/* ---------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::compute_all()
{
  //if (last_compute == update->ntimestep) return;
  last_compute = update->ntimestep;
  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;

  //neighbor->build_one(list);

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
    memory->grow(gi_real, nmax, Q6_ARRAY_SIZE, "compute_q6_smooth_atom:gi_real");
    memory->grow(gi_imag, nmax, Q6_ARRAY_SIZE, "compute_q6_smooth_atom:gi_imag");
    memory->grow(Gi, nmax, "compute_q6_smooth_atom:Gi");
    memory->grow(Ci, nmax, N_DIM, "compute_q6_smooth_atom:Ci");
    memory->grow(hj, nmax, N_DIM, "compute_q6_smooth_atom:hj");
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

  // filling all the arrays with zeros
  std::fill_n(&array_atom[0][0], nmax * size_peratom_cols, 0.0);
  std::fill_n(&q6ms_real[0][0], nmax * Q6_ARRAY_SIZE, 0.0);
  std::fill_n(&q6ms_imag[0][0], nmax * Q6_ARRAY_SIZE, 0.0);
  std::fill_n(&diff_q6ms_real[0][0][0], nmax * Q6_ARRAY_SIZE * N_DIM, 0.0);
  std::fill_n(&diff_q6ms_imag[0][0][0], nmax * Q6_ARRAY_SIZE * N_DIM, 0.0);
  std::fill_n(inv_q6_norm_i, nmax, 0.0);
  std::fill_n(inv_nbnum_i, nmax, 0.0);
  std::fill_n(Ni, nmax, 0.0);
  std::fill_n(ds2i, nmax, 0.0);
  std::fill_n(&diff_Ni[0][0], nmax * N_DIM, 0.0);
  std::fill_n(&gi_real[0][0], nmax * Q6_ARRAY_SIZE, 0.0);
  std::fill_n(&gi_imag[0][0], nmax * Q6_ARRAY_SIZE, 0.0);
  std::fill_n(Gi, nmax, 0.0);
  std::fill_n(&Ci[0][0], nmax * N_DIM, 0.0);
  std::fill_n(&hj[0][0], nmax * N_DIM, 0.0);
  std::fill_n(s0j, nmax, 0.0);
  std::fill_n(s1j, nmax, 0.0);
  std::fill_n(ds0j, nmax, 0.0);
  std::fill_n(ds1j, nmax, 0.0);

  // lambda functions
  auto s1 = [&](const double &input, double &output, double &diff) {
    orient(input, beta1, x01, output, diff);
  };
  auto s2 = [&](const double &input, double &output, double &diff) {
    orient(input, beta2, x02, output, diff);
  };
  auto s3 = [&](const double &input, double &output, double &diff) {
    dist(input, cutoff, output, diff);
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

    if (type[i] != chosen_type || !(mask[i] & groupbit)) continue;
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

        diff_q6ms_real[i][indx][0] += Y6m[offset + 2];
        diff_q6ms_imag[i][indx][0] += Y6m[offset + 3];
        diff_q6ms_real[i][indx][1] += Y6m[offset + 4];
        diff_q6ms_imag[i][indx][1] += Y6m[offset + 5];
        diff_q6ms_real[i][indx][2] += Y6m[offset + 6];
        diff_q6ms_imag[i][indx][2] += Y6m[offset + 7];
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
    double diff_q6_norm[N_DIM] = {0.0, 0.0, 0.0};

    for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
      q6_norm += q6ms_real[i][indx] * q6ms_real[i][indx] + q6ms_imag[i][indx] * q6ms_imag[i][indx];
      for (int dim = 0; dim < N_DIM; dim++)
        diff_q6_norm[dim] += q6ms_real[i][indx] * diff_q6ms_real[i][indx][dim] +
            q6ms_imag[i][indx] * diff_q6ms_imag[i][indx][dim];
    }

    q6_norm = std::sqrt(q6_norm);
    const double inv_q6_norm = q6_norm >= 1e-20 ? 1.0 / q6_norm : 0.0;

    inv_q6_norm_i[i] = inv_q6_norm;

    for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
      q6ms_real[i][indx] *= inv_q6_norm;
      q6ms_imag[i][indx] *= inv_q6_norm;
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

  // forward_comm the q6ms_real, q6ms_imag, diff_q6ms_real and diff_q6ms_imag to ghost atoms
  forward_mode = Q6_TRANSFER;
  comm_forward = Q6_STRIDE;

  comm->forward_comm(this);

  /*
   * This part calculates the contribution of the atom i to the differential with all the rj values. 
   * (1)  PHI_MODE: N_total = sigma(wij*qi*qj)
   *      gi = sigma(wij*qj)
   *      dN_total/drk = sigma(wij*qj*dqi/drk +wij*qi*dqj/drk) = gi*dqi/drk + gj*dqj/drk
   *      Thanks to the forward comm we have the qj for the ghost atoms in this rank so 
   *      we can easily calculate the gi = sigma(wij*qj) over j.
   *      However, we do not have the gj and dgj/drk and it does not make sense to transfer the diff of qj with
   *      respect to every possible r since the array size would be massively large!
   *      So, here we define hj = sigma (gi*dqi/rj over i), each rank adds its own contribution to the hj.
   *      Since the hj can be a ghost atom it has to be reverse communicated and added to the hj on the current rank.
   *      Then we go through every owned atom and add hj[i] to every diff.
   * (2)  N_MODE: Ni = qi*(wij sigmaqj) = qi*gi
   *      the Ni also contributes to j and for the same reason we need hj.. but hj = wij*qj*dqi/drj
   *      diffN1/diffrk = diffq1 *(q2+q3+q4) + q1*(diffq2+diffq3+diffq4) 
   */

  /*
   * (1)  We first calculate gi values.
   * (2)  For the contribution of qi to the diff with respect to rj   
   *      (2A)  if (mode & N_MODE) N_total = N1 + N2 + N3 + ... = q1*g1+q2*g2 + ...
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
   * 
   */

  /*
   * This loop fills the gi array, add the self diff contribution to diffs (diffqi/diffri)
   * and fills the hj components.
   * For the N_MODE we fill the diff_array_atom
   * For the PHI_MODE the diff_Ni is filled.
   */

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jnum = numneigh[i];
    jlist = firstneigh[i];
    if (type[i] != chosen_type || !(mask[i] & groupbit)) continue;

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
    const double eps = 1e-8;
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
     * N_MODE: N_total = sigma(Ni)
     * PHI_MODE: PHI_i = sigma(s3(rij)*Ni*Nj)
     * dNi/drk = ds2(Si)*dSi/drk 
     * dNi/drk = ds2(Si)*(sigma(dcij/drk))
     * dcij/drk = dbij/drk*ds1(bij)*s0(rij)+s1(bij)*ds0(rij)*drij/drk
     */

    /*
     * I need the S2 to calculated the diff /drk
     * The large number of neighbors is prohibiting to 
     * cache the cij values in the loop over j.
     *
     * A solution is to cache the s0[j], ds0[j],
     * s1[j] and ds1[j] and multiply them by the ds2 or s2 
     * at the second loop over jj to obtain the wpair and wpair2
     * values.
     */

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      double cij = 0.0;

      double distx = x[i][0] - x[j][0];
      double disty = x[i][1] - x[j][1];
      double distz = x[i][2] - x[j][2];

      double r = sqrt(distx * distx + disty * disty + distz * distz);
      if (r < 1e-8 || r >= cutoff) continue;

      for (int indx = 0; indx < Q6_ARRAY_SIZE; ++indx) {
        cij += q6ms_real[i][indx] * q6ms_real[j][indx] + q6ms_imag[i][indx] * q6ms_imag[j][indx];
      }

      /* Some tests -->>*/
      if (std::abs(cij) + eps > 1.0) error->warning(FLERR, "This aint good {},{}", i, j);
      /*<<--Some tests*/

      double s0val, ds0val;
      double s1val, ds1val;
      s1(cij, s1val, ds1val);
      dist(r, cutoff, s0val, ds0val);
      Si += s1val * s0val;

      s0j[j] = s0val;
      s1j[j] = s1val;
      ds0j[j] = ds0val;
      ds1j[j] = ds1val;
    }

    double ds2;
    double s2val;
    s2(Si, s2val, ds2);
    if (mode & N_MODE) {
      array_atom[i][val_col] = s2val;
    } else if (mode & PHI_MODE) {
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
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      const double distx = x[i][0] - x[j][0];
      const double disty = x[i][1] - x[j][1];
      const double distz = x[i][2] - x[j][2];
      const double r = sqrt(distx * distx + disty * disty + distz * distz);
      if (r < 1e-8 || r >= cutoff) continue;

      const double wpair = ds2 * s0j[j] * ds1j[j];
      const double wpair2 = ds2 * s1j[j] * ds0j[j];

      for (int indx = 0; indx < Q6_ARRAY_SIZE; ++indx) {    // 2l+1 for l=6 → 13 harmonics
        // values I have Rej, Imj and diff_q6s_real/imag for j atoms thank to the forward_comm.

        const double Rej = q6ms_real[j][indx];
        const double Imj = q6ms_imag[j][indx];

        gi_real[i][indx] += wpair * Rej;
        gi_imag[i][indx] += wpair * Imj;
      }

      // the distance contribution to the dcij/dri
      // the rij distance does not contribute to the dcij/drk if k!=j
      const double diff_x = wpair2 * distx / r;
      const double diff_y = wpair2 * disty / r;
      const double diff_z = wpair2 * distz / r;
      if (mode & N_MODE) {
        array_atom[i][diff_x_col] += 2.0 * diff_x;
        array_atom[i][diff_y_col] += 2.0 * diff_y;
        array_atom[i][diff_z_col] += 2.0 * diff_z;
      } else if (mode & PHI_MODE) {
        diff_Ni[i][0] += diff_x;
        diff_Ni[i][1] += diff_y;
        diff_Ni[i][2] += diff_z;
      }
    }

    // This part calculates the cij dependent component of diffNi/diffri or diffN_total/diffri
    // cij = sigma (qi*qj)
    // dcij/dri = dqi/dri * gi
    // N_MODE : CV = sigma (gi*qi) --> dCV/dri = 2.0*gi*dqi/dri+...
    // PHI_MODE : CV = gi*qi --> dCV/dri = gi*dqi/dri+...
    for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
      const double diff_x = diff_q6ms_real[i][indx][0] * gi_real[i][indx] +
          diff_q6ms_imag[i][indx][0] * gi_imag[i][indx];
      const double diff_y = diff_q6ms_real[i][indx][1] * gi_real[i][indx] +
          diff_q6ms_imag[i][indx][1] * gi_imag[i][indx];
      const double diff_z = diff_q6ms_real[i][indx][2] * gi_real[i][indx] +
          diff_q6ms_imag[i][indx][2] * gi_imag[i][indx];
      if (mode & N_MODE) {
        array_atom[i][diff_x_col] += 2.0 * diff_x;
        array_atom[i][diff_y_col] += 2.0 * diff_y;
        array_atom[i][diff_z_col] += 2.0 * diff_z;
      } else if (mode & PHI_MODE) {
        diff_Ni[i][0] += diff_x;
        diff_Ni[i][1] += diff_y;
        diff_Ni[i][2] += diff_z;
      }
    }

    /* contribution of the qi to the dNi/drj [diffN_total/diffrj] 
     * which is equal to the qj*dqi/drj [gi*dqi/drj]
     * rj can be any neighbors of the i including ghost atoms and that is the reason why 
     * there is a need for a reverse communication of the hj
     * dqi/drj is calculated on the fly
     */
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
            hj[j][dim] += 2.0 * dqi_drj_real[indx][dim] * gi_real[i][indx];
            hj[j][dim] += 2.0 * dqi_drj_imag[indx][dim] * gi_imag[i][indx];
          }
        }
      } else if (mode & PHI_MODE) {
        for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
          for (int dim = 0; dim < N_DIM; dim++) {
            hj[j][dim] += dqi_drj_real[indx][dim] * q6ms_real[i][indx];
            hj[j][dim] += dqi_drj_imag[indx][dim] * q6ms_imag[i][indx];
          }
        }
      }
    }
  }

  // Transfering the hj from the ghost atoms
  comm->reverse_comm(this);

  // Adding the contribution of atom j to the diffN_total/dri or diffNi/diffri
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (type[i] != chosen_type || !(mask[i] & groupbit)) continue;

    const double diff_x = hj[i][0];
    const double diff_y = hj[i][1];
    const double diff_z = hj[i][2];
    if (mode & N_MODE) {
      array_atom[i][diff_x_col] += diff_x;
      array_atom[i][diff_y_col] += diff_y;
      array_atom[i][diff_z_col] += diff_z;
    } else if (mode & PHI_MODE) {
      diff_Ni[i][0] += diff_x;
      diff_Ni[i][1] += diff_y;
      diff_Ni[i][2] += diff_z;
    }
  }

  // Transfering the Ni, to the ghost atoms
  if (mode & PHI_MODE) {
    forward_mode = N_TRANSFER;
    comm_forward = N_STRIDE;
    comm->forward_comm(this);
  }

  /*
   * phi = sigma(K(rij)*Ni*Nj)
   * Since the Nj values which are not neighbor of i do not
   * contribute to the dphi/dri we consider phi as 2*Ni(K(rij)*sigmaNj)
   * and Gi = K(rij)*sigmaNj
   * So dphi/dri = diffNi/diffri*Gi + sigma(dK(rij)/dri*Ni*Nj) + Ni*(sigma K(rij)*dNj/dri)
   * the second term is calculated in the neighbor ranks.
   */

  /* 
    * So the following steps are used in the calculation of diffphi/diffri
    *
    * step 1: Gi is calculated.
    * step 2: term 1 (diffNi/diffri*Gi)  is calculated
    * step 3: term 2 components for each j  (dK(rij)/dri*Ni*Nj) is added.
    * step 4: term 3 (Ni*K(rij)*dNj/dri)) components are calculated on the fly
    * step 4A: contribution of dqi/dri through various Nk terms in the Ni*sigma(K(ik)*dNk/drj) (dphi/dri)
    * step 4B: Contribution of the dqi/drj to the Nj*sigma(K(rjk)*dNk/drj) through Ni the term (dphi/drj)
    * step 4C: Contribution of the dqi/drj through Nk term of other neighbors of i expect for j to the dphi/drj
    * step 4D: The distance dependent term for each Ni in the Nj*dNi/drj is added.
    * This term is added since Ni = sigma(s0(rij)*s1(cij)).. This term is related to the diff(s0(rij))/drj
    * for instance the dq2/dr1 can also affect the dN2/dr1 = K(13)*(dq3/dr1*(q1+q2+q4)+q3*(dq1/dr1+dq2/dr1+dq4/dr1))
    * step 5: reverse comm of dphi/drj
    * step 6: adding the contrivution from the atom j (hj) to the diff. 
    */

  /*
     * This is an example
     * K12*N1*dN2/dr1 = K(12)*N1*(dq2*(q1+q3)+q2*(dq1+dq3))
     * K13*N1*dN3/dr1 = K(13)*N1*(dq3*(q1+q2+q4)+q3*(dq1+dq2+dq4))
     * K14*N1*dN4/dr1 = K(14)*N1(dq4*(q1+q3)+q4*(dq1+dq3))
     * 4A:  K12*N1*dN2/dr1 += K(12)*N1*dq1*q2 i == 1
     *      K13*N1*dN3/dr1 += K(13)*N1*dq1*q3 i == 1
     *      K14*N1*dN4/dr1 += K(14)*N1*dq1*q4 i == 1
     * 4B:  K12*N1*dN2/dr1 += K(12)*N1*dq2*(q1+q3) i == 2 j == 1
     *      K13*N1*dN3/dr1 += K(13)*N1*dq3*(q1+q2+q4) i == 3 j == 1
     *      K14*N1*dN4/dr1 += K(14)*N1*dq4*(q1+q3) i == 4 j == 1
     * 4C:  K12*N1*dN2/dr1 += K(12)*N1*q2*(dq3) i == 2
     *      K13*N1*dN3/dr1 += K(13)*N1*q3*(dq2+dq4) i == 3
     *      K14*N1*dN4/dr1 += K(14)*N1*q4*(dq3) i == 4
     * 
     */

  if (mode & PHI_MODE) {
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jnum = numneigh[i];
      jlist = firstneigh[i];
      if (type[i] != chosen_type || !(mask[i] & groupbit)) continue;

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        double distx = x[i][0] - x[j][0];
        double disty = x[i][1] - x[j][1];
        double distz = x[i][2] - x[j][2];
        double r = sqrt(distx * distx + disty * disty + distz * distz);
        if (r < 1e-8 || r >= cutoff) continue;

        double s3val, ds3;
        dist(r, cutoff, s3val, ds3);
        phi_sum += Ni[i] * Ni[j] * s3val;

        /*
         * step 1
         */
        Gi[i] += s3val * Ni[j];
      }

      // phi[i] = Ni[i] * sigma(Kij*Ni[j])
      array_atom[i][val_col] += Ni[i] * Gi[i];

      /*
       * step 2 - diffNi/diffri*(sigma K(rij)*Nj)
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

        /* step 3
         * s3 dependent contribution of dNi/drj to the Nj*sigma(diffNk/diffrj)
         *
         *
         */
        dist(r, cutoff, s3val, ds3);
        wpair3 = -ds3;

        array_atom[i][diff_x_col] += 2 * wpair3 * Ni[i] * Ni[j] * distx / r;
        array_atom[i][diff_y_col] += 2 * wpair3 * Ni[i] * Ni[j] * disty / r;
        array_atom[i][diff_z_col] += 2 * wpair3 * Ni[i] * Ni[j] * distz / r;

        /*
         * step 4:
         *
         */

        /*
         * step 4A
         * contribution of the dqi/dri to the Ni*simga(diffNk/diffri)
         * it contributes to the Nj terms in the sigma(diffNk/diffri)
         *
         */

        double cij = 0.0;
        for (int indx = 0; indx < Q6_ARRAY_SIZE; ++indx) {
          cij += q6ms_real[i][indx] * q6ms_real[j][indx] + q6ms_imag[i][indx] * q6ms_imag[j][indx];
        }

        dist(r, cutoff, s0val, ds0);
        s1(cij, s1val, ds1);

        // it contributes to the Nj that is the reason why ds2i[j]
        wpair = s3val * ds2i[j] * ds1 * s0val * Ni[i];

        double dcij_dri[N_DIM];

        for (dim = 0; dim < N_DIM; dim++) {
          dcij_drj[dim] = 0.0;
          for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
            dcij_dri[dim] = diff_q6ms_real[i][indx][dim] * q6ms_real[j][indx] +
                diff_q6ms_imag[i][indx][dim] * q6ms_imag[j][indx];
          }
        }

        array_atom[i][diff_x_col] = 2.0 * wpair * dcij_dri[0];
        array_atom[i][diff_y_col] = 2.0 * wpair * dcij_dri[1];
        array_atom[i][diff_z_col] = 2.0 * wpair * dcij_dri[2];

        /*
         * step 4B
         * Contribution of the dqi/drj to the Nj*sigma(dNk/drj)
         * It contributes to the Ni term in the sigma
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

        //Ni contribution to the Nj*sigma(K(ij)*dN/dri)
        // gi_real and _imag have ds2i[i]*ds1*s0val
        wpair = s3val * Ni[j];

        for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
          for (int dim = 0; dim < 3; dim++) {
            hj[j][dim] += wpair *
                (dqi_drj_real[indx][dim] * gi_real[i][indx] +
                 dqi_drj_imag[indx][dim] * gi_imag[i][indx]);
          }
        }

        /*
         * step 4C
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

          /* It is the Nj*diffNk/diffrj contribution.
            * Thus the s3 must be based on rjk since phi = sigma (s3(rij)*Ni*Nj)
            * But s2 must be based on rik since its based on the qi*qk 
            * and ds1 must be based on the cik
            * Since k is neighbor of i we have all of its q values thanks to the forward comm. 
            */

          double cik = 0.0;
          for (int indx = 0; indx < 13; indx++)
            cik +=
                q6ms_real[i][indx] * q6ms_real[k][indx] + q6ms_imag[i][indx] * q6ms_imag[k][indx];

          double s3jkval, ds3jk;
          double s2ikval, ds2ik;
          double s1ikval, ds1ik;
          double s0ikval, ds0ik;
          dist(rjk, cutoff, s3jkval, ds3jk);
          s1(cik, s1ikval, ds1ik);
          dist(rik, cutoff, s0val, ds0);
          wpair = s3jk * ds2i[k] * ds1ik * s0ikval * Ni[j];
          for (int ind x = 0; indx < Q6_ARRAY_SIZE; indx++)
            for (int dim = 0; dim < N_DIM; dim++)
              hj[j][dim] += wpair *
                  (dqi_drj_real[indx][dim] * q6ms_real[k][indx] +
                   dqi_drj_imag[indx][dim] * q6ms_imag[k][indx]);
        }

        /*
          * step 4D: distance dependent term to the Nj*diffNi/drj
          */

        wpair = -Ni[j] * ds2[i] * s1val * ds0;
        for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++)
          for (int dim = 0; dim < N_DIM; dim++)
            hj[j][dim] += wpair * (q6ms_real[i] * q6ms_real[j] + q6ms_imag[i] * q6m_imag[j]) *
                distance[dim] / r;
      }
    }
  }

  /*
   * step 5
   */
  // Transfering the hj from the ghost atoms
  comm->reverse_comm(this);

  if (mode & PHI_MODE) {
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      if (type[i] != chosen_type || !(mask[i] & groupbit)) continue;

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

  MPI_Allreduce(&Q6_sum, &Q6_sum_all, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&phi_sum, &phi_sum_all, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&num_selected, &num_selected_all, 1, MPI_INT, MPI_SUM, world);

  double num_double = static_cast<double>(num_selected_all);
  double scaling = num_double >= 2 ? 2.0 / (num_double * (num_double - 1.0)) : 0.0;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    array_atom[i][val_col] *= scaling;
    array_atom[i][diff_x_col] *= scaling;
    array_atom[i][diff_y_col] *= scaling;
    array_atom[i][diff_z_col] *= scaling;
    double x_comp = array_atom[i][diff_x_col];
    double y_comp = array_atom[i][diff_y_col];
    double z_comp = array_atom[i][diff_z_col];
    double slope = std::sqrt(x_comp * x_comp + y_comp * y_comp + z_comp * z_comp);
    if (slope < min_slope) {
      array_atom[i][diff_x_col] = -0.5 * min_slope + min_slope * rng->gaussian();
      array_atom[i][diff_y_col] = -0.5 * min_slope + min_slope * rng->gaussian();
      array_atom[i][diff_z_col] = -0.5 * min_slope + min_slope * rng->gaussian();
    }
  }

  phi_sum *= scaling;

  //scalar = num_selected_all ? Q6_sum_all / static_cast<double>(num_selected_all) :0.0;
}

/* --------------------------------------------------------------------- */

int ComputeQ6SmoothAtom::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
                                           int * /*pbc*/)
{
  int m, j;
  m = 0;

  if (forward_mode & Q6_TRANSFER) {
    if (comm_forward != Q6_STRIDE)
      error->one(FLERR, "Wrong value in the comm_foward {}", comm_forward);
    for (int i = 0; i < n; i++) {
      j = list[i];
      for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
        buf[m++] = q6ms_real[j][indx];
        buf[m++] = q6ms_imag[j][indx];
      }
    }
  } else if (forward_mode & N_TRANSFER) {
    if (comm_forward != N_STRIDE)
      error->one(FLERR, "Wrong value in the comm_foward {}", comm_forward);
    for (int i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = Ni[j];
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
    if (comm_forward != Q6_STRIDE)
      error->one(FLERR, "Wrong value in the comm_foward {}", comm_forward);
    for (j = first; j < last; j++) {
      for (int indx = 0; indx < Q6_ARRAY_SIZE; indx++) {
        q6ms_real[j][indx] = buf[m++];
        q6ms_imag[j][indx] = buf[m++];
      }
    }
  } else if (forward_mode & N_TRANSFER) {
    if (comm_forward != N_STRIDE)
      error->one(FLERR, "Wrong value in the comm_foward {}", comm_forward);
    for (j = first; j < last; j++) { Ni[j] = buf[m++]; }
  }
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
  for (i = first; i < last; i++) {
    for (int k = 0; k < N_DIM; k++) buf[m++] = hj[i][k];
  }
  return m;
}

/* ----------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (int k = 0; k < N_DIM; k++) hj[j][k] += buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::orient(const double &input, const double &beta, const double &x0,
                                 double &output, double &diff)
{
  output = 1.0 / (1.0 + std::exp(-beta * (input - x0)));
  diff = beta * output * (1.0 - output);

  // avoiding dead gradient
  if (std::abs(diff) <= min_slope) diff = output >= x0 ? min_slope : -min_slope;
}

/* --------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::dist(const double &input, const double &cutoff, double &output,
                               double &diff)
{
  diff = output = 0.0;
  const double *const coeff = dist_coeffs;
  double r0 = 0.85 * cutoff;
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
  if (std::abs(diff) <= min_slope) diff = x < 0.5 ? min_slope : -min_slope;
}

/* --------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::equal(const double &input, const double &cutoff, double &output,
                                double &diff)
{
  output = input;
  diff = 1.0;
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
   * do not belong to this rank, if it is a ghost atom
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
  if (r < eps || rxy < eps) return Y6m;

  double theta = std::acos(z / r);
  double phi = std::atan2(y, x);    // Accurate azimuthal angle
  double sin_theta = std::sin(theta);
  double cos_theta = std::cos(theta);

  double dtheta_dx = (x * z) / (r * r * rxy);
  double dtheta_dy = (y * z) / (r * r * rxy);
  double dtheta_dz = -rxy / (r * r);

  double dphi_dx = -y / (rxy * rxy);
  double dphi_dy = x / (rxy * rxy);
  double dphi_dz = 0.0;

  std::array<double, 3> dtheta = {dtheta_dx, dtheta_dy, dtheta_dz};
  std::array<double, 3> dphi = {dphi_dx, dphi_dy, dphi_dz};

  double coeff = 0.0;
  double Re = 0.0, Im = 0.0;
  double dRe_dtheta = 0.0, dIm_dtheta = 0.0;
  double dRe_dphi = 0.0, dIm_dphi = 0.0;

  auto calc_params = [&Re, &Im, &dRe_dtheta, &dIm_dtheta, &dRe_dphi, &dIm_dphi, &coeff, phi,
                      sin_theta, cos_theta](const int &deg) {
    auto sin_pow_theta = [&sin_theta](const int &n) {
      return pow_fun(sin_theta, n);
    };

    auto cos_pow_theta = [&cos_theta](const int &n) {
      return pow_fun(cos_theta, n);
    };

    double sin_m_phi = std::sin(deg * phi);
    double cos_m_phi = std::cos(deg * phi);
    double sign = (deg < 0) ? -1.0 : 1.0;

    double first_theta = sin_pow_theta(std::abs(deg));
    double diff_first_theta = std::abs(deg) * sin_pow_theta(std::abs(deg) - 1) * cos_theta;

    double second_theta;
    double diff_second_theta;

    switch (deg) {
      case -6:
      case 6:
        coeff = (1.0 / 64.0) * std::sqrt(3003.0 / M_PI);
        second_theta = 1.0;
        diff_second_theta = 0.0;
        break;

      case -5:
      case 5:
        coeff = -sign * (3.0 / 32.0) * std::sqrt(1001.0 / M_PI);
        second_theta = cos_theta;
        diff_second_theta = -sin_theta;
        break;

      case -4:
      case 4:
        coeff = (3.0 / 32.0) * std::sqrt(91.0 / (2 * M_PI));
        second_theta = 11 * cos_pow_theta(2) - 1;
        diff_second_theta = -sin_theta * 2 * 11 * cos_theta;
        break;

      case -3:
      case 3:
        coeff = -sign * (1.0 / 32.0) * std::sqrt(1365.0 / M_PI);
        second_theta = 11 * cos_pow_theta(3) - 3 * cos_theta;
        diff_second_theta = -sin_theta * (11 * 3 * cos_pow_theta(2) - 3);
        break;

      case -2:
      case 2:
        coeff = (1.0 / 64.0) * std::sqrt(1365.0 / M_PI);
        second_theta = 33 * cos_pow_theta(4) - 18 * cos_pow_theta(2) + 1;
        diff_second_theta = -sin_theta * (4 * 33 * cos_pow_theta(3) - 2 * 18 * cos_theta);
        break;

      case -1:
      case 1:
        coeff = -sign * (1.0 / 16.0) * std::sqrt(273.0 / (2.0 * M_PI));
        second_theta = 33 * cos_pow_theta(5) - 30 * cos_pow_theta(3) + 5 * cos_theta;
        diff_second_theta =
            -sin_theta * (33 * 5 * cos_pow_theta(4) - 30 * 3 * cos_pow_theta(2) + 5);
        break;

      case 0:
        coeff = (1.0 / 32.0) * std::sqrt(13.0 / M_PI);
        second_theta =
            231.0 * cos_pow_theta(6) - 315.0 * cos_pow_theta(4) + 105.0 * cos_pow_theta(2) - 5.0;
        diff_second_theta = -sin_theta *
            (231 * 6 * cos_pow_theta(5) - 315 * 4 * cos_pow_theta(3) + 105 * 2 * cos_theta);
        break;

      default:
        throw std::invalid_argument("m must be between -6 and +6");
    }

    Re = coeff * cos_m_phi * first_theta * second_theta;
    Im = coeff * sin_m_phi * first_theta * second_theta;
    dRe_dtheta =
        coeff * cos_m_phi * (first_theta * diff_second_theta + second_theta * diff_first_theta);
    dIm_dtheta =
        coeff * sin_m_phi * (first_theta * diff_second_theta + second_theta * diff_first_theta);
    dRe_dphi = -coeff * deg * sin_m_phi * first_theta * second_theta;
    dIm_dphi = coeff * deg * cos_m_phi * first_theta * second_theta;
  };

  // Coefficients and angular dependencies for all m
  for (int deg = -6; deg <= 6; deg++) {
    calc_params(deg);
    int offset = (deg + 6) * 8;
    Y6m[offset + 0] = Re;
    Y6m[offset + 1] = Im;
    Y6m[offset + 2] = dRe_dtheta * dtheta[0] + dRe_dphi * dphi[0];
    Y6m[offset + 3] = dIm_dtheta * dtheta[0] + dIm_dphi * dphi[0];
    Y6m[offset + 4] = dRe_dtheta * dtheta[1] + dRe_dphi * dphi[1];
    Y6m[offset + 5] = dIm_dtheta * dtheta[1] + dIm_dphi * dphi[1];
    Y6m[offset + 6] = dRe_dtheta * dtheta[2] + dRe_dphi * dphi[2];
    Y6m[offset + 7] = dIm_dtheta * dtheta[2] + dIm_dphi * dphi[2];
  }

  return Y6m;
}
