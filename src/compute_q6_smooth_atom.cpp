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

enum {
  Q6_TRANSFER = 1<<1,
  N_TRANSFER = 1<<2
};

enum {
  N_MODE = 0,
  PHI_MODE = 1<<1
};

enum {
  Q6_STRIDE = 104, 
  N_STRIDE = 6
};

/* ---------------------------------------------------------------------- */

ComputeQ6SmoothAtom::ComputeQ6SmoothAtom(LAMMPS *lmp, int narg, char **arg) :
    ComputeDiffAtom{lmp, narg, arg},
    mode{N_MODE}, chosen_type{-1}, cutoff{3.2},
    q6ms_real{nullptr}, q6ms_imag{nullptr},
    diff_q6ms_real{nullptr}, diff_q6ms_imag{nullptr},
    diff_q6ms_real_pair{nullptr}, diff_q6ms_imag_pair{nullptr},
    inv_q6_norm_i{nullptr}, inv_nbnum_i{nullptr},
    diff_q6_norm_pair{nullptr},
    Ni{nullptr}, ds2i{nullptr}, 
    diff_Ni{nullptr}, diff_Ni_pair{nullptr}, 
    hj{nullptr},
    forward_mode{Q6_TRANSFER},
    s0j{nullptr}, ds0j{nullptr},
    s1j{nullptr}, ds1j{nullptr}
{
  chosen_type = utils::numeric(FLERR, arg[3], false, lmp);
  cutoff = utils::numeric(FLERR, arg[4], false, lmp);

  comm_forward = 104;
  comm_reverse = 3;

  int iarg=5;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"sigmoid") == 0) {
      if (narg < iarg+5) error->all(FLERR,"Illegal q6-smooth/atom command");
      beta1 = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      x01 = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      beta2 = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      x02 = utils::numeric(FLERR,arg[iarg+4],false,lmp);
      iarg += 5;
    } else if (strcmp(arg[iarg],"phi") == 0) {
      mode = PHI_MODE;
      iarg++;
    } else error->all(FLERR,"Illegal compute q6-smooth/atom command");
  }
}

/* ----------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::init()
{
  ComputeDiffAtom::init();
  memory->create(q6ms_real,nmax,13,"compute_q6_smooth_atom:q6ms_real");
  memory->create(q6ms_imag,nmax,13,"compute_q6_smooth_atom:q6ms_imag");
  memory->create(diff_q6ms_real,nmax,13,3,"compute_q6_smooth_atom:diff_q6ms_real");
  memory->create(diff_q6ms_imag,nmax,13,3,"compute_q6_smooth_atom:diff_q6ms_imag");
  memory->create(inv_q6_norm_i,nmax,"compute_q6_smooth_atom:q6_norm_i");
  memory->create(inv_nbnum_i,nmax,"compute_q6_smooth_atom:inv_nbnum_i");
  memory->create(diff_q6_norm_pair,nmax, "compute_q6_smooth_atom:diff_q6_norm_pair");
  memory->create(Ni,nmax,"compute_q6_smooth_atom:Ni");
  memory->create(ds2i,nmax,"compute_q6_smooth:ds2i");
  memory->create(diff_Ni,nmax,3,"compute_q6_smooth_atom:diff_Ni");
  memory->create(hj,nmax,3,"compute_q6_smooth_atom:hj");
  memory->create(s0j,nmax,"compute_q6_smooth_atom:s0j");
  memory->create(s1j,nmax,"compute_q6_smooth_atom:s1j");
  memory->create(ds0j,nmax,"compute_q6_smooth_atom:ds0j");
  memory->create(ds1j,nmax,"compute_q6_smooth_atom:ds1j");
  update_npair();
  allocate_diff_pairs();
  request->cutoff = cutoff;
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
    if (diff_q6_norm_pair) memory->destroy(diff_q6_norm_pair);
    if (Ni) memory->destroy(Ni);
    if (ds2i) memory->destroy(ds2i);
    if (diff_Ni) memory->destroy(diff_Ni);
    if (hj) memory->destroy(hj);
    if (s0j) memory->destroy(s0j);
    if (s1j) memory->destroy(s1j);
    if (ds0j) memory->destroy(ds0j);
    if (ds1j) memory->destroy(ds1j);
    deallocate_diff_pairs();
}

/* ---------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::compute_all()
{
  //if (last_compute == update->ntimestep) return;
  last_compute = update->ntimestep;
  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int natoms = atom->natoms;


  neighbor->build_one(list);

  if (atom->nmax > nmax)
  {
    nmax = atom->nmax;
    memory->grow(array_atom,nmax,size_peratom_cols,"compute_diff:array_atom");
    memory->grow(q6ms_real,nmax,13,"compute_q6_smooth_atom:q6ms_real");
    memory->grow(q6ms_imag,nmax,13,"compute_q6_smooth_atom:q6ms_imag");
    memory->grow(diff_q6ms_real,nmax,13,3,"compute_q6_smooth_atom:diff_q6ms_real");
    memory->grow(diff_q6ms_imag,nmax,13,3,"compute_q6_smooth_atom:diff_q6ms_imag");
    memory->grow(inv_q6_norm_i,nmax,"compute_q6_smooth_atom:inv_q6_norm_i");
    memory->grow(inv_nbnum_i,nmax,"compute_q6_smooth_atom:inv_nbnum_i");
    memory->grow(diff_q6_norm_pair,nmax,3,"compute_q6_smooth_atom:diff_q6_norm_pair");
    memory->grow(Ni,nmax,"compute_q6_smooth_atom:Ni");
    memory->grow(ds2i,nmax,"compute_q6_smooth_atom:ds2i");
    memory->grow(diff_Ni,nmax,"compute_q6_smooth_atom:diff_Ni");
    memory->grow(hj,nmax,3,"compute_q6_smooth_atom:hj");
    memory->grow(s0j,nmax,"compute_q6_smooth_atom:s0j");
    memory->grow(s1j,nmax,"compute_q6_smooth_atom:s1j");
    memory->grow(ds0j,nmax,"compute_q6_smooth_atom:ds0j");
    memory->grow(ds1j,nmax,"compute_q6_smooth_atom:ds1j");
    update_npair();
    deallocate_diff_pairs();
    allocate_diff_pairs();
  } else if (neighbor->ago == 0) {
    update_npair();
    deallocate_diff_pairs();
    allocate_diff_pairs();
  }


  int i, j, ii, jj, inum, jnum;
  int *ilist, *jlist, *numneigh, **firstneigh;

  
  //it is the same as atom->nlocal and list->gnum is zero since we have not requested for ghost neighbors in the list.
  inum = list->inum;   
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;



  // per rank and summed global variables
  double Q6_sum = 0.0;
  double Q6_sum_all = 0.0;
  double phi_sum = 0.0;
  double phi_sum_all = 0.0;
  int num_selected = 0;
  int num_selected_all = 0;

  for (int i = 0; i < nmax; i++) {
    Ni[i] = 0.0;
    ds2i[i] = 0.0;
    inv_q6_norm_i[i] = 0.0;
    inv_nbnum_i[i] = 0.0;
    for (int k = 0; k < 3; k++) {
      diff_Ni[i][k] = 0.0;
      hj[i][k] = 0.0;
    }
    for (int indx = 0; indx < 13; indx++) {
      q6ms_real[i][indx] = 0.0;
      q6ms_imag[i][indx] = 0.0;
      for (int k = 0; k < 3; k++) {
        diff_q6ms_real[i][indx][k] = 0.0;
        diff_q6ms_imag[i][indx][k] = 0.0;
      }
    }
  }

  for (ii = 0; ii < inum; ii++) {
    for (int jj = 0; jj < jnum; jj++) {
      for (int dim = 0; dim < 3; dim++) {
        diff_Ni_pair[ii][jj][dim] = 0.0;
        for (int indx = 0; indx < 13; indx++)
        {
          diff_q6ms_real_pair[ii][jj][indx][dim] = 0.0;
          diff_q6ms_imag_pair[ii][jj][indx][dim] = 0.0;
        }
      }
    }
  }

  for (int i = 0; i < nmax ; i++)
  {
    array_atom[i][val_col] = 0.0;
    array_atom[i][diff_x_col] = 0.0;
    array_atom[i][diff_y_col] = 0.0;
    array_atom[i][diff_z_col] = 0.0;
    array_atom[i][nbnum_col] = 0.0;
    array_atom[i][second_val_col] = 0.0;
  }

  auto s1 =[&](const double& input, double& output, double& diff) {
    orient(input, beta1, x01, output, diff);
  };
  auto s2 =[&](const double& input, double& output, double& diff) {
    orient(input, beta2, x02, output, diff);
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
      if (r < 1e-8 ||  r >= cutoff) continue;
      nbnum++;
      std::array<double,3> distance = {distx,disty,distz};
      std::array<double,104> Y6m = calculate_Y6m(distance);
      for (int deg = -6; deg <= 6; deg++) {
        int offset = (deg+6)*8;
        int indx = deg + 6;
        q6ms_real[i][indx]    += Y6m[offset + 0];
        q6ms_imag[i][indx]    += Y6m[offset + 1];

        diff_q6ms_real[i][indx][0] += Y6m[offset + 2];
        diff_q6ms_imag[i][indx][0] += Y6m[offset + 3];
        diff_q6ms_real[i][indx][1] += Y6m[offset + 4];
        diff_q6ms_imag[i][indx][1] += Y6m[offset + 5];
        diff_q6ms_real[i][indx][2] += Y6m[offset + 6];
        diff_q6ms_imag[i][indx][2] += Y6m[offset + 7];


        diff_q6ms_real_pair[ii][jj][indx][0] -= Y6m[offset + 2];
        diff_q6ms_imag_pair[ii][jj][indx][0] -= Y6m[offset + 3];
        diff_q6ms_real_pair[ii][jj][indx][1] -= Y6m[offset + 4];
        diff_q6ms_imag_pair[ii][jj][indx][1] -= Y6m[offset + 5];
        diff_q6ms_real_pair[ii][jj][indx][2] -= Y6m[offset + 6];
        diff_q6ms_imag_pair[ii][jj][indx][2] -= Y6m[offset + 7];
      }
    }
    array_atom[i][nbnum_col] = nbnum;

    const double inv_nbnum = nbnum >= nbnum_min ? 1.0/nbnum : 0.0;
    
    inv_nbnum_i[i] = inv_nbnum;
      
    for (int deg = -6; deg <= 6; deg++) {
      int indx = deg + 6;
      q6ms_real[i][indx]    *= inv_nbnum;
      q6ms_imag[i][indx]    *= inv_nbnum;
          
      for (int k = 0; k < 3; k++) {
        diff_q6ms_real[i][indx][k] *= inv_nbnum;
        diff_q6ms_imag[i][indx][k] *= inv_nbnum;
        for (jj = 0; jj < jnum; jj++) {
          diff_q6ms_real_pair[ii][jj][indx][k] *= inv_nbnum;
          diff_q6ms_imag_pair[ii][jj][indx][k] *= inv_nbnum;
        }
      }
    }


    double q6_norm = 0.0;
    double diff_q6_norm[3] = {0.0,0.0,0.0};
    for (jj = 0; jj < jnum; jj++)
      for (int k = 0; k < 3; k++)
        diff_q6_norm_pair[jj][k] = 0.0;
    for (int indx = 0; indx < 13; indx++) {
      q6_norm += q6ms_real[i][indx]*q6ms_real[i][indx]+q6ms_imag[i][indx]*q6ms_imag[i][indx];
      for (int k = 0; k < 3; k++) {
        diff_q6_norm[k] += q6ms_real[i][indx]*diff_q6ms_real[i][indx][k] + q6ms_imag[i][indx]*diff_q6ms_imag[i][indx][k];
        for (jj = 0; jj < jnum; jj++)
          diff_q6_norm_pair[jj][k] += q6ms_real[i][indx]*diff_q6ms_real_pair[ii][jj][indx][k] + q6ms_imag[i][indx]*diff_q6ms_imag_pair[ii][jj][indx][k];
      }
    }

    q6_norm = std::sqrt(q6_norm);
    const double inv_q6_norm = q6_norm >= 1e-20 ? 1.0/q6_norm : 0.0;

    inv_q6_norm_i[i] = inv_q6_norm;


    for (int indx = 0; indx < 13; indx++)
    {
      q6ms_real[i][indx] *= inv_q6_norm;
      q6ms_imag[i][indx] *= inv_q6_norm;
    }
    
    diff_q6_norm[0] *= inv_q6_norm;
    diff_q6_norm[1] *= inv_q6_norm;
    diff_q6_norm[2] *= inv_q6_norm;


    for (jj = 0; jj < jnum; jj++) {
      for (int indx=0; indx<13; indx++) {
        for (int k = 0; k < 3; k++) {
          diff_q6ms_real_pair[ii][jj][indx][k] *= inv_q6_norm;
          diff_q6ms_imag_pair[ii][jj][indx][k] *= inv_q6_norm;
        }
      }
    }
      
    for (int k = 0; k < 3; k++) {
      for (int indx = 0; indx < 13; indx++) {
        diff_q6ms_real[i][indx][k] = (inv_q6_norm)*(diff_q6ms_real[i][indx][k]-q6ms_real[i][indx]*diff_q6_norm[k]);
        diff_q6ms_imag[i][indx][k] = (inv_q6_norm)*(diff_q6ms_imag[i][indx][k]-q6ms_imag[i][indx]*diff_q6_norm[k]);
        for (jj = 0; jj < jnum; jj++) {
          diff_q6ms_real_pair[ii][jj][indx][k] = (inv_q6_norm)*(diff_q6ms_real_pair[ii][jj][indx][k]-q6ms_real[i][indx]*diff_q6_norm_pair[jj][k]);
          diff_q6ms_imag_pair[ii][jj][indx][k] = (inv_q6_norm)*(diff_q6ms_imag_pair[ii][jj][indx][k]-q6ms_real[i][indx]*diff_q6_norm_pair[jj][k]);
        }   
      }
    }
  } 

  // forward_comm the q6ms_real, q6ms_imag, diff_q6ms_real and diff_q6ms_imag to ghost atoms 
  forward_mode = Q6_TRANSFER;
  comm_forward = Q6_STRIDE;

  comm->forward_comm(this);


  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jnum = numneigh[i];
    jlist = firstneigh[i];
    if (type[i] != chosen_type || !(mask[i] & groupbit)) continue;
    
    double Si = 0.0;

    /* Some tests -->>*/
    const double eps = 1e-8;
    double ci_norm = 0.0;
    for (int indx = 0; indx < 13; indx++) {
      ci_norm += q6ms_real[i][indx]*q6ms_real[i][indx]+ q6ms_imag[i][indx]*q6ms_imag[i][indx];
    }
    if (std::abs(ci_norm - 1.0) > eps && std::abs(ci_norm) > eps)
        error->warning(FLERR,"ci normalisation for the atom {} is wrong",i);
    /*<<--Some tests*/

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      double cij = 0.0;
        
      double distx = x[i][0] - x[j][0];
      double disty = x[i][1] - x[j][1];
      double distz = x[i][2] - x[j][2];

      double r = sqrt(distx * distx + disty * disty + distz * distz);
      if (r < 1e-8 ||  r >= cutoff) continue;
    
      for (int indx = 0; indx < 13; ++indx) { 
        cij += q6ms_real[i][indx]*q6ms_real[j][indx] + q6ms_imag[i][indx]*q6ms_imag[j][indx];
      }

      /* Some tests -->>*/
      if (std::abs(cij) + eps > 1.0)
        error->warning(FLERR,"This aint good {},{}",i,j);
      /*<<--Some tests*/

      double s0val, ds0val;
      double s1val, ds1val;
      s1(cij,s1val,ds1val);
      dist(r,cutoff,s0val,ds0val);
      Si += s1val*s0val;

      s0j[j] = s0val;
      s1j[j] = s1val;
      ds0j[j] = ds0val;
      ds1j[j] = ds1val;
    }

    double ds2;
    double s2val;
    s2(Si,s2val,ds2);

    if (mode & N_MODE) {
      array_atom[i][val_col] = s2val; 
    } else if (mode & PHI_MODE) {
      array_atom[i][second_val_col] = s2val;
    }
    Ni[i] = s2val;
    ds2i[i] = ds2;

    Q6_sum += s2val;
    
    /*
     * I need both the ds1 and ds2 in this loop
     * but ds1 is dependent on the cij and the ds2
     * is dependent on the sum of all the values of cij
     * To avoid having to save the cij this loop is repeated.
     * 
     */
     
    /*
     * Another solution is to cache the s0[j], ds0[j],
     * s1[j] and ds1[j] and multiply them by the ds2 or s2 
     * at the second loop to obtain the wpair and wpair2
     * values.
     */


    double sigma_wpair_Rej[13];
    double sigma_wpair_Imj[13];

    // this loop calculated the sum of wpair *Rej and wpair*Imj for all the neighbors
    // as cij = qi*qj and dcij/drk = dqi/drk * qj
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      const double distx = x[i][0] - x[j][0];
      const double disty = x[i][1] - x[j][1];
      const double distz = x[i][2] - x[j][2];
      const double r = sqrt(distx * distx + disty * disty + distz * distz);
      if (r < 1e-8 ||  r >= cutoff) continue;


      const double wpair  = ds2*s0j[j]*ds1j[j]; 
      const double wpair2 = ds2*s1j[j]*ds0j[j];

      
      for (int indx = 0; indx < 13; ++indx) {  // 2l+1 for l=6 → 13 harmonics
        // values I have Rej, Imj and diff_q6s_real/imag for j atoms thank to the forward_comm.
        
        const double Rej  = q6ms_real[j][indx];
        const double Imj  = q6ms_imag[j][indx];

        sigma_wpair_Rej[indx] += wpair*Rej;
        sigma_wpair_Imj[indx] += wpair*Imj;
      }

      // the distance contribution to the dcij/dri
      // the rij distance does not contribute to the dcij/drk if k!=j
      const double diff_x = wpair2*distx/r;
      const double diff_y = wpair2*disty/r;
      const double diff_z = wpair2*distz/r;
      if (mode & N_MODE) {
        array_atom[i][diff_x_col] += 2.0*diff_x; 
        array_atom[i][diff_y_col] += 2.0*diff_y;
        array_atom[i][diff_z_col] += 2.0*diff_z;
      } else if (mode & PHI_MODE) {
        diff_Ni[i][0] += diff_x;
        diff_Ni[i][1] += diff_y;
        diff_Ni[i][2] += diff_z;
      }
    }

    // This part calculates the dcij/dri since cij = sigma (qi*qj)
    // the dcij/drj is calculated when the i = j
    // dcij/dri = dqi/dri * sigma(wpair*qj)
    for (int indx = 0; indx < 13; indx++)
    {
      const double diff_x = diff_q6ms_real[i][indx][0] * sigma_wpair_Rej[indx] + diff_q6ms_imag[i][indx][0] * sigma_wpair_Rej[indx];
      const double diff_y = diff_q6ms_real[i][indx][1] * sigma_wpair_Rej[indx] + diff_q6ms_imag[i][indx][1] * sigma_wpair_Rej[indx];
      const double diff_z = diff_q6ms_real[i][indx][2] * sigma_wpair_Rej[indx] + diff_q6ms_imag[i][indx][2] * sigma_wpair_Rej[indx];
      if (mode & N_MODE) {
        array_atom[i][diff_x_col] += 2.0*diff_x;
        array_atom[i][diff_y_col] += 2.0*diff_y;
        array_atom[i][diff_z_col] += 2.0*diff_z;
      } else if (mode & PHI_MODE) {
        diff_Ni[i][0] += diff_x;
        diff_Ni[i][1] += diff_y;
        diff_Ni[i][2] += diff_z;
      }
    }

    // contribution of the qi to the dcij/drk which is equal to the sigma (qj*dqi/drk)
    // rk can be any neighbors of the i including ghost atoms and that is the reason why 
    // there is a need for a reverse communication of the hj
    for (int kk = 0; kk < jnum; kk++) {
      int k = jlist[kk];
      k &= NEIGHMASK;
      for (int indx= 0; indx<13;indx++) {
        for (int dim = 0; dim < 3; dim++) {
          hj[k][dim] += 2.0*diff_q6ms_real_pair[ii][kk][indx][dim]*sigma_wpair_Rej[indx];
          hj[k][dim] += 2.0*diff_q6ms_imag_pair[ii][kk][indx][dim]*sigma_wpair_Imj[indx];
        }
      }
    }
  }

  // Transfering the Ni, diff_Ni, inv_q6_norm_i and inv_nbnum_i to the ghost atoms 
  if (mode & PHI_MODE) {
    forward_mode = N_TRANSFER;
    comm_forward = N_STRIDE;
    comm->forward_comm(this);
  }

  // Transfering the hj from the ghost atoms
  // I do not need the hj for the diff phi
  if (mode & N_MODE)
    comm->reverse_comm(this);



  /*
   * dCV/dr = sigma(wij*qj*dqi/dr +wij*qi*dqj/dr)s3
   * thanks to the forward comm we have the qj for the ghost atoms in this rank
   * However, we do not have the dgj/dr and it does not make sense to transfer the diff of qj with
   * respect to every possible r since the size would be massively large!
   * So, here we define hj = sigma(wij*qi)over i and then we calculate the hj for every atom
   * including the ghost atoms and reverse comm the values.
   * Then we go through every owned atom and add hj[i]*dqj/dr to every diff.
   */

  if (mode & N_MODE) {
    for (ii= 0; ii < inum; ii++) {
      i = ilist[ii];
      if (type[i] != chosen_type || !(mask[i] & groupbit)) continue;
    
      const double diff_x = hj[i][0];
      const double diff_y = hj[i][1];
      const double diff_z = hj[i][2];
      array_atom[i][diff_x_col] += diff_x;
      array_atom[i][diff_y_col] += diff_y;
      array_atom[i][diff_z_col] += diff_z;
    }
  }


  /*
   * phi = sigma(Ni*Nj)
   * diff phi/diff rk = 2.0*sigma(dNi/drk*Nj) 
   * and Ni = qi * sigma(qj)
   * dNi/drk = dqi/drk * sigma(qj) + qi * dsigma(qj)/drk
   * but we do not have the dsigma(qj)/drk in this rank
   * if the j is a ghost atom.
   * so we need to calculate the dqj/drk on this rank.
   * The good news is that we just need to dqj/drk
   * for the k values belonging to this rank.
   * so we just need the Y6m(jk), the Nb(j), Q6norm(j)
   * which has been transfered thanks to the forward comm.
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
        if (r < 1e-8 ||  r >= cutoff) continue;

        double s0val, ds0;
        double s1val, ds1;
        double s2val, ds2;
        double s3val, ds3;

        dist(r,cutoff,s3val,ds3);
        phi_sum += Ni[i]*Ni[j]*s3val;

        // phi[i] = Ni[i] * sigma(Ni[j])
        array_atom[i][val_col] += Ni[i]*Ni[j]*s3val;


        double cij = 0.0;
        for (int indx = 0; indx < 13; ++indx) { 
          cij += q6ms_real[i][indx]*q6ms_real[j][indx] + q6ms_imag[i][indx]*q6ms_imag[j][indx];
        }

        dist(r,cutoff, s0val,ds0);
        s1(cij, s1val, ds1);
        ds2 = ds2i[i];

        double wpair = ds2*s0val*ds1;
        double wpair2 = ds2*ds0*s1val;
        double drij[3] = {distx/r,disty/r,distz/r};

        for (int indx=0; indx < 13; indx++) {
          const double Rei  = q6ms_real[i][indx];
          const double Rej  = q6ms_real[j][indx];
          const double Imi  = q6ms_imag[i][indx];
          const double Imj  = q6ms_imag[j][indx];


          // diff Ni/diffrj = ds1*(dcij/drj*s0+cij*ds0/drj) = wpair*dcij + wpair2*ds0
          // cij = Rei*Rej+Imi*Imj
          // dcij/rj = dRei/drj*Rej + ... + Rei*dRej/drj
          for (int dim = 0; dim < 3; dim++) {
            diff_Ni_pair[ii][jj][dim] += wpair*(diff_q6ms_real_pair[ii][jj][indx][dim]*Rej+diff_q6ms_imag_pair[ii][jj][indx][dim]*Imj);
            diff_Ni_pair[ii][jj][dim] += wpair*(diff_q6ms_real[j][indx][dim]*Rei + diff_q6ms_imag[j][indx][dim]*Imi);
            diff_Ni_pair[ii][jj][dim] += - wpair2*drij[dim];
          }
        }

        // neighbors other than jj might contribute to the diffNi/drj
        for (int kk = 0; kk < jnum; kk++) {
          int k = jlist[kk];
          k &= NEIGHMASK;
          /* if k!=j just rjk affects the diffqk/diffr
           * We do not have the diffqk/diffr since the k atom might be a ghost atom
           * and transfering the whole diff_q6ms_xxx_pair is expensive so we calculate
           * it on the fly.
           *
           *
           * for the case of k ==j  all the neighbors of the atom j affect
           * the diff which we do not have it if the atom j is a ghost atom
           * thus we have forward communicated the self diff of qj to the ghost atoms
           * and the self diff term is included in the previous section.
           */
          if (k == j) continue;
          double distx = x[j][0] - x[k][0];
          double disty = x[j][1] - x[k][1];
          double distz = x[j][2] - x[k][2];
          double r = sqrt(distx * distx + disty * disty + distz * distz);
          if (r < 1e-8 ||  r >= cutoff) continue;

          double cik = 0.0;
          for (int indx = 0; indx < 13; ++indx) { 
            cik += q6ms_real[i][indx]*q6ms_real[k][indx] + q6ms_imag[i][indx]*q6ms_imag[k][indx];
          }
  
          double s0val, s1val;
          double ds0, ds1;
          dist(r,cutoff, s0val,ds0);
          s1(cij, s1val, ds1);
  
          double wpair  = ds2*ds1*s0val;
          double wpair2 = ds2*ds0*s1val;
          double drjk[3] = {distx/r,disty/r,distz/r};
          
          double dqk_drj_real[13][3];
          double dqk_drj_imag[13][3];
          std::array<double,3> distance = {distx,disty,distz};
          std::array<double,104> Y6m = calculate_Y6m(distance);

          /*
           * It is possible that all of the neighbors of the atom k
           * do not belong to this rank, if it is a ghost atom
           * That is the reason why we cannot have its nbnum and 
           * qnorm.. Accordingly these values have been forward communicated
           * for ghost atoms.
           */
          for (int deg = -6; deg <= 6; deg++) {
            int offset = (deg+6)*8;
            int indx = deg + 6;
            dqk_drj_real[indx][0] = Y6m[offset + 2]*inv_nbnum_i[k];
            dqk_drj_imag[indx][0] = Y6m[offset + 3]*inv_nbnum_i[k];
            dqk_drj_real[indx][1] = Y6m[offset + 4]*inv_nbnum_i[k];
            dqk_drj_imag[indx][1] = Y6m[offset + 5]*inv_nbnum_i[k];
            dqk_drj_real[indx][2] = Y6m[offset + 6]*inv_nbnum_i[k];
            dqk_drj_imag[indx][2] = Y6m[offset + 7]*inv_nbnum_i[k];
          }
          
          double dq_norm_drk[3] = {0.0,0.0,0.0};
          for (int indx = 0; indx < 13; indx++) {
            for (int dim = 0; dim < 3; dim++)
              dq_norm_drk[dim] += dqk_drj_real[indx][dim]*q6ms_real[k][indx] + dqk_drj_imag[indx][dim]*q6ms_imag[k][indx];
          }
          for (int dim= 0; dim < 3; dim++) {
            dq_norm_drk[dim] *= inv_q6_norm_i[k];
          }

          for (int indx = 0; indx< 13; indx++) {
            // these two are just for the indx and all the values for the 13 indexes are summed at the end.
            double dqhatk_dxj_real[3];
            double dqhatk_dxj_imag[3];
            for (int dim = 0; dim < 3; dim++) {
              dqhatk_dxj_real[dim] = dqk_drj_real[indx][dim]*inv_q6_norm_i[k];
              dqhatk_dxj_imag[dim] = dqk_drj_imag[indx][dim]*inv_q6_norm_i[k];
              dqhatk_dxj_real[dim] -= q6ms_real[k][indx]*dq_norm_drk[dim]*inv_q6_norm_i[k]; 
              dqhatk_dxj_imag[dim] -= q6ms_imag[k][indx]*dq_norm_drk[dim]*inv_q6_norm_i[k];
            }
            const double Rei  = q6ms_real[i][indx];
            const double Imi  = q6ms_imag[i][indx];
            // does it need a factor of 2.0?
            for (int dim = 0; dim <3; dim++)
              diff_Ni_pair[ii][jj][dim] +=  wpair*(dqhatk_dxj_real[dim]*Rei + dqhatk_dxj_imag[dim]*Imi)+wpair2*drjk[dim];
          }
        }
      }
    }
  }

  // I have communicated the Nis before with the forward communication.

  if (mode & PHI_MODE) {
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      jnum = numneigh[i];
      jlist = firstneigh[i];
      if (type[i] != chosen_type || !(mask[i] & groupbit)) continue;

      // sum of the term related to the s3val over all the j atoms 
      // and the sum of the term related to the ds3 over all the j atoms
      // these two terms are requird for the calculation of
      // the diff (Ni*Nj) / diffxk
      double sigma_wpair_Nj = 0.0;

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        double distx = x[i][0] - x[j][0];
        double disty = x[i][1] - x[j][1];
        double distz = x[i][2] - x[j][2];
        double r = sqrt(distx * distx + disty * disty + distz * distz);
        if (r < 1e-8 ||  r >= cutoff) continue;
        double s3val, ds3;
        dist(r,cutoff,s3val,ds3);

        array_atom[i][diff_x_col] += 2.0*(s3val*diff_Ni[i][0]*Ni[j]+ds3*Ni[i]*Ni[j]*distx/r);
        array_atom[i][diff_y_col] += 2.0*(s3val*diff_Ni[i][1]*Ni[j]+ds3*Ni[i]*Ni[j]*disty/r);
        array_atom[i][diff_z_col] += 2.0*(s3val*diff_Ni[i][2]*Ni[j]+ds3*Ni[i]*Ni[j]*distz/r);

        sigma_wpair_Nj += s3val*Ni[j];

      }

      for (int jj = 0; jj < jnum; jj++) {
        int j = jlist[jj];
        j &= NEIGHMASK;
        for (int dim = 0; dim < 3; dim++)
          hj[j][dim] += sigma_wpair_Nj*diff_Ni_pair[ii][jj][dim];
      }
    }
  }

  // Transfering the hj from the ghost atoms
  comm->reverse_comm(this);

  if (mode & PHI_MODE) {
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      if (type[i] != chosen_type || !(mask[i] & groupbit)) continue;
    
      const double diff_x = hj[i][0];
      const double diff_y = hj[i][1];
      const double diff_z = hj[i][2];
      array_atom[i][diff_x_col] += 2.0*diff_x;
      array_atom[i][diff_y_col] += 2.0*diff_y;
      array_atom[i][diff_z_col] += 2.0*diff_z;
    }
  }


  MPI_Allreduce(&Q6_sum, &Q6_sum_all, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&phi_sum,&phi_sum_all,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&num_selected, &num_selected_all, 1, MPI_INT, MPI_SUM, world);

  double num_double = static_cast<double>(num_selected_all);
  double scaling = 2.0/(num_double*(num_double-1.0));
  for (ii = 0; ii < inum; ii++)
  {
    i = ilist[ii];
    array_atom[i][val_col] *= scaling;
    array_atom[i][diff_x_col] *= scaling;
    array_atom[i][diff_y_col] *= scaling;
    array_atom[i][diff_z_col] *= scaling;
  }

  phi_sum *= scaling;


  //scalar = num_selected_all ? Q6_sum_all / static_cast<double>(num_selected_all) :0.0;
  
}

/* --------------------------------------------------------------------- */

int ComputeQ6SmoothAtom::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int* /*pbc*/)
{
    int i, m, j;
    m = 0;

    if (forward_mode & Q6_TRANSFER) {
      if (comm_forward != Q6_STRIDE)
        error->one(FLERR,"Wrong value in the comm_foward {}",comm_forward);
      for (int i = 0; i < n; i++) {
        j = list[i];
        for (int indx = 0; indx < 13; indx++)
        {
          buf[m++] = q6ms_real[j][indx];
          buf[m++] = q6ms_imag[j][indx];
          buf[m++] = diff_q6ms_real[j][indx][0];
          buf[m++] = diff_q6ms_imag[j][indx][0];
          buf[m++] = diff_q6ms_real[j][indx][1];
          buf[m++] = diff_q6ms_imag[j][indx][1];
          buf[m++] = diff_q6ms_real[j][indx][2];
          buf[m++] = diff_q6ms_imag[j][indx][2];
        }  
      }
    } else if (forward_mode & N_TRANSFER) {
      if (comm_forward != N_STRIDE)
        error->one(FLERR,"Wrong value in the comm_foward {}",comm_forward);
      for (int i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = Ni[j];
        buf[m++] = diff_Ni[j][0];
        buf[m++] = diff_Ni[j][1];
        buf[m++] = diff_Ni[j][2];
        buf[m++] = inv_q6_norm_i[j];
        buf[m++] = inv_nbnum_i[j];   
      }
    } else 
      error->one(FLERR,"Wrong forward_comm flag");
  

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
        error->one(FLERR,"Wrong value in the comm_foward {}",comm_forward);
      for (j = first; j < last; j++) {
        for (int indx = 0; indx < 13; indx++) {
          q6ms_real[j][indx] = buf[m++];
          q6ms_imag[j][indx] = buf[m++];
          diff_q6ms_real[j][indx][0] = buf[m++];
          diff_q6ms_imag[j][indx][0] = buf[m++];
          diff_q6ms_real[j][indx][1] = buf[m++];
          diff_q6ms_imag[j][indx][1] = buf[m++];
          diff_q6ms_real[j][indx][2] = buf[m++];
          diff_q6ms_imag[j][indx][2] = buf[m++];
        }
      }
    } else if (forward_mode & N_TRANSFER) {
      if (comm_forward != N_STRIDE)
        error->one(FLERR,"Wrong value in the comm_foward {}",comm_forward);
      for (j = first; j < last; j++) {
        Ni[j] = buf[m++];
        diff_Ni[j][0] = buf[m++];
        diff_Ni[j][1] = buf[m++];
        diff_Ni[j][2] = buf[m++];
        inv_q6_norm_i[j] = buf[m++];
        inv_nbnum_i[j] = buf[m++];
      }
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
    for (int k = 0; k < 3; k++)
      buf[m++] = hj[i][k];
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
      for (int k = 0; k < 3; k++)
        hj[j][k] += buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::update_npair()
{
  int inum, *numneigh;
  inum = list->inum;   
  numneigh = list->numneigh;
  npair = std::accumulate(numneigh, numneigh+inum, 0);
}

/* ---------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::allocate_diff_pairs()
{
  int inum = list->inum;   
  int * numneigh = list->numneigh;
  int * ilist = list->ilist;


  // Lambda for allocating 3D-like array (double***)
  auto grow4D = [&](double****& x, const int npairs, const int indx2, const int indx3)
  {
    double*** x_flat;
   
    // Allocate outer pointer array of size inum
    x = (double****) memory->smalloc(inum * sizeof(double***),
                                           "compute_q6_smooth_atom:temp");
   
    // Allocate contiguous memory for npairs × indx2 doubles
    memory->grow(x_flat, npairs, indx2, indx3, "compute_q6_smooth_atom:temp");
   
    int first_index = 0;
    for (int ii = 0; ii < inum; ii++) {
      int i = ilist[ii];
   
      // Each atom gets a slice of x_flat
      x[ii] = &x_flat[first_index];
      first_index += numneigh[i];
    }
  };

  auto grow3D = [&](double***& x, const int npairs, const int indx2)
  {
    double** x_flat;
   
    // Allocate outer pointer array of size inum
    x = (double***) memory->smalloc(inum * sizeof(double**),
                                           "compute_q6_smooth_atom:temp");
   
    // Allocate contiguous memory for npairs × indx2 doubles
    memory->grow(x_flat, npairs, indx2, "compute_q6_smooth_atom:temp");
   
    int first_index = 0;
    for (int ii = 0; ii < inum; ii++) {
      int i = ilist[ii];
   
      // Each atom gets a slice of x_flat
      x[ii] = &x_flat[first_index];
      first_index += numneigh[i];
    }
  };

  grow4D(diff_q6ms_real_pair,npair,13,3);
  grow4D(diff_q6ms_imag_pair,npair,13,3);
  grow3D(diff_Ni_pair,npair,3);

}

/* ---------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::deallocate_diff_pairs()
{
  auto rel4D = [&](double****& x)
  {
    if (!x) return;
    memory->destroy(x[0]);
    memory->sfree(x);
  };

  auto rel3D = [&](double***& x)
  {
    if (!x) return;
    memory->destroy(x[0]);
    memory->sfree(x);
  };
  
  rel4D(diff_q6ms_real_pair);
  rel4D(diff_q6ms_imag_pair);
  rel3D(diff_Ni_pair);
}

/* ---------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::orient(const double& input, const double& beta, const double& x0, double& output, double& diff)
{
    output = 1.0/(1.0+std::exp(-beta*(input-x0)));
    diff = beta*output*(1.0-output);
    if (diff <= 0.02)
    {
      if (output < x0) {
        diff = 0.02;
      } else if (output > x0) {
        diff = -0.02;
      }
    }
}

/* --------------------------------------------------------------------- */

void ComputeQ6SmoothAtom::dist(const double& input, const double& cutoff, double& output, double& diff)
{
  diff = output = 0.0;
  const double * const coeff = dist_coeffs;
  double r0 = 0.85*cutoff;
  double x = (input-r0)/ (cutoff-r0);
  if (x <= 0.0) { 
    output = 1.0;
  } else if (x >= 1.0) { 
    output = 0.0;
  } else {
    output += coeff[0];
    for (int i = 1; i < dist_deg; i++)
    {
      output += coeff[i]*pow(x,i);
      diff += i*coeff[i]*pow(x,i-1);
    }
  }
}

/* ----------------------------------------------------------------------
  computing the Y6m values
------------------------------------------------------------------------- */

std::array<double,104> ComputeQ6SmoothAtom::calculate_Y6m(const std::array<double,3>& dist)
{
  double x = dist[0];
  double y = dist[1];
  double z = dist[2];

  double r = std::sqrt(x*x + y*y + z*z);
  double rxy = std::sqrt(x*x + y*y);

  constexpr double eps = 1e-6;

  double theta = std::acos(z / r);
  double phi = std::atan2(y, x);  // Accurate azimuthal angle
  double sin_theta = std::sin(theta);
  double cos_theta = std::cos(theta);


  double dtheta_dx = (x*z)/(r*r*rxy);
  double dtheta_dy = (y*z)/(r*r*rxy);
  double dtheta_dz = -rxy/(r*r);

  double dphi_dx = -y / (rxy*rxy);
  double dphi_dy =  x / (rxy*rxy);
  double dphi_dz = 0.0;

  if (r < eps || rxy < eps) {
    theta = 0.0;
    phi = 0.0;
    dtheta_dx = 0.0;
    dtheta_dy = 0.0;
    dtheta_dz = 0.0;
    dphi_dx = 0.0;
    dphi_dy = 0.0;
    dphi_dz = 0.0;
  }

  std::array<double,3> dtheta = {dtheta_dx, dtheta_dy, dtheta_dz};
  std::array<double,3> dphi   = {dphi_dx, dphi_dy, dphi_dz};
  std::array<double, 104> Y6m = {0.0};


  double coeff = 0.0;
  double Re = 0.0, Im = 0.0;
  double dRe_dtheta = 0.0, dIm_dtheta = 0.0;
  double dRe_dphi   = 0.0, dIm_dphi   = 0.0;
  


  auto calc_params = [
                      &Re,&Im,
                      &dRe_dtheta, &dIm_dtheta,
                      &dRe_dphi,&dIm_dphi,
                      &coeff,phi,sin_theta,cos_theta
                    ] (const int& deg)
  {
    auto sin_pow_theta = [&sin_theta](const int& n)
    {
      return std::pow(sin_theta,n);
    };
  
    auto cos_pow_theta = [&cos_theta](const int& n)
    {
      return std::pow(cos_theta,n);
    };


    double sin_m_phi = std::sin(deg * phi);
    double cos_m_phi = std::cos(deg * phi);
    double sign = (deg < 0) ? -1.0 : 1.0;


    double first_theta = sin_pow_theta(std::abs(deg));
    double diff_first_theta = std::abs(deg)*sin_pow_theta(std::abs(deg)-1)*cos_theta;
    
    double second_theta;
    double diff_second_theta;

    switch (deg) {
      case -6: case 6:
        coeff = (1.0 / 64.0) * std::sqrt(3003.0 / M_PI);
        second_theta = 1.0;
        diff_second_theta = 0.0;
        break;
  
      case -5: case 5:
        coeff = -sign*(3.0 / 32.0) * std::sqrt(1001.0 /  M_PI);
        second_theta = cos_theta;
        diff_second_theta = -sin_theta;
        break;
  
      case -4: case 4:
        coeff = (3.0 / 32.0) * std::sqrt(91.0 / (2*M_PI));
        second_theta = 11*cos_pow_theta(2)-1;
        diff_second_theta = -sin_theta*2*11*cos_theta;
        break;
  
      case -3: case 3:
        coeff = -sign * (1.0 / 32.0) * std::sqrt(1365.0 / M_PI);
        second_theta = 11*cos_pow_theta(3) - 3*cos_theta;
        diff_second_theta = -sin_theta*(11*3*cos_pow_theta(2)-3);
        break;
  
      case -2: case 2:
        coeff = (1.0 / 64.0) * std::sqrt(1365.0 / M_PI);
        second_theta = 33*cos_pow_theta(4)-18*cos_pow_theta(2)+1;
        diff_second_theta = -sin_theta * (4*33*cos_pow_theta(3) - 2*18*cos_theta);
        break;
  
      case -1: case 1:
        coeff = -sign*(1.0 / 16.0) * std::sqrt(273.0 / (2.0 * M_PI));
        second_theta = 33*cos_pow_theta(5)-30*cos_pow_theta(3)+5*cos_theta;
        diff_second_theta = -sin_theta*(33*5*cos_pow_theta(4)-30*3*cos_pow_theta(2) +5);
        break;
  
      case 0:
        coeff = (1.0 / 32.0) * std::sqrt(13.0 / M_PI);
        second_theta = 231.0 * cos_pow_theta(6) - 315.0 * cos_pow_theta(4) + 105.0 * cos_pow_theta(2) - 5.0;
        diff_second_theta = -sin_theta * (231* 6 * cos_pow_theta(5) - 315* 4* cos_pow_theta(3) + 105* 2 * cos_theta);
        break;
  
      default:
            throw std::invalid_argument("m must be between -6 and +6");
    }


    Re = coeff * cos_m_phi * first_theta*second_theta;
    Im = coeff * sin_m_phi * first_theta*second_theta;
    dRe_dtheta = coeff * cos_m_phi * (first_theta*diff_second_theta + second_theta*diff_first_theta);
    dIm_dtheta = coeff * sin_m_phi * (first_theta*diff_second_theta + second_theta*diff_first_theta);
    dRe_dphi   = -coeff * deg * sin_m_phi * first_theta*second_theta;
    dIm_dphi   =  coeff * deg * cos_m_phi * first_theta*second_theta;

  };


  // Coefficients and angular dependencies for all m
  for (int deg = -6; deg <= 6; deg++)
  {
    calc_params(deg);
    int offset = (deg+6)*8;
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
