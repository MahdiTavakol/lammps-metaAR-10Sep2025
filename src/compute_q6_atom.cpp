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

#include "compute_q6_atom.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeQ6Atom::ComputeQ6Atom(LAMMPS *lmp, int narg, char **arg) :
    ComputeDiffAtom{lmp, narg, arg}, chosen_type{-1}, cutoff{3.2}
{
  chosen_type = utils::numeric(FLERR, arg[3], false, lmp);
  cutoff = utils::numeric(FLERR, arg[4], false, lmp);

  comm_reverse = 3;
}


/* ---------------------------------------------------------------------- */

void ComputeQ6Atom::compute_all()
{
  //if (last_compute == update->ntimestep) return;
  last_compute = update->ntimestep;
  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;

  

  if (atom->nmax > nmax)
  {
    nmax = atom->nmax;
    memory->grow(array_atom,   nmax, size_peratom_cols, "compute_q6_atom:array_atom");
  }

  // constant parameters 
  constexpr int q6_col = 0;
  constexpr double Q6_coeff = 4.0*M_PI/13.0;
  constexpr int nbnum_min = 6; 

  // per rank and summed global variables
  double Q6_sum = 0.0;
  double Q6_sum_all = 0.0;
  int num_selected = 0;
  int num_selected_all = 0;

  // The real part of the q6ms for each m
  std::array<double,13> q6ms_real;
  // The imaginary part
  std::array<double,13> q6ms_imag;
  // diff real part with respect to the x
  std::array<double,13> q6ms_real_dx;
  // diff imaginary part with respect to the x
  std::array<double,13> q6ms_imag_dx;
  // diff real part with respect to the y
  std::array<double,13> q6ms_real_dy;
  // diff imaginary part with respect to the y
  std::array<double,13> q6ms_imag_dy;
  // diff real part with respect to the z
  std::array<double,13> q6ms_real_dz;
  // diff imaginary part with respect to the z
  std::array<double,13> q6ms_imag_dz;


  int i, j, ii, jj, inum, jnum;
  int *ilist, *jlist, *numneigh, **firstneigh;

  neighbor->build_one(list);
  //it is the same as atom->nlocal and list->gnum is zero since we have not requested for ghost neighbors in the list.
  inum = list->inum;   
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (int i = 0; i < nmax ; i++)
  {
    array_atom[i][val_col] = 0.0;
    array_atom[i][diff_x_col] = 0.0;
    array_atom[i][diff_y_col] = 0.0;
    array_atom[i][diff_z_col] = 0.0;
  }


  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jnum = numneigh[i];
    jlist = firstneigh[i];

    if (type[i] != chosen_type || !(mask[i] & groupbit)) continue;
    num_selected++;

    //resetting atom i parameters
    double Q6i = 0.0;
    int nbnum = 0;
    std::fill(q6ms_real.begin(),q6ms_real.end(),0.0);
    std::fill(q6ms_imag.begin(),q6ms_imag.end(),0.0);
    std::fill(q6ms_real_dx.begin(),q6ms_real_dx.end(),0.0);
    std::fill(q6ms_imag_dx.begin(),q6ms_imag_dx.end(),0.0);
    std::fill(q6ms_real_dy.begin(),q6ms_real_dy.end(),0.0);
    std::fill(q6ms_imag_dy.begin(),q6ms_imag_dy.end(),0.0);
    std::fill(q6ms_real_dz.begin(),q6ms_real_dz.end(),0.0);
    std::fill(q6ms_imag_dz.begin(),q6ms_imag_dz.end(),0.0);
    diff_Q6[0] = diff_Q6[1] = diff_Q6[2] = 0.0;


    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      double distx = x[i][0] - x[j][0];
      double disty = x[i][1] - x[j][1];
      double distz = x[i][2] - x[j][2];
      double r = sqrt(distx * distx + disty * disty + distz * distz);
      if (r < 1e-8 ||  r >= cutoff) continue;
      nbnum++;
      std::array<double,3> dist = {distx,disty,distz};
      std::array<double,104> Y6m = calculate_Y6m(dist);
      for (int deg = -6; deg <= 6; deg++) {
        int offset = (deg+6)*8;
        int indx = deg + 6;
        q6ms_real[indx]    += Y6m[offset + 0];
        q6ms_imag[indx]    += Y6m[offset + 1];
        q6ms_real_dx[indx] += Y6m[offset + 2];
        q6ms_imag_dx[indx] += Y6m[offset + 3];
        q6ms_real_dy[indx] += Y6m[offset + 4];
        q6ms_imag_dy[indx] += Y6m[offset + 5];
        q6ms_real_dz[indx] += Y6m[offset + 6];
        q6ms_imag_dz[indx] += Y6m[offset + 7];   
      }
    }
      
    array_atom[i][nbnum_col] = nbnum;

    if (nbnum >= nbnum_min) {
      // Pass 1 --- the central atom [i]
      for (int deg = -6; deg <= 6; deg++)
      {
        int indx = deg + 6;
        q6ms_real[indx]    /= nbnum;
        q6ms_imag[indx]    /= nbnum;
        q6ms_real_dx[indx] /= nbnum;
        q6ms_imag_dx[indx] /= nbnum;
        q6ms_real_dy[indx] /= nbnum;
        q6ms_imag_dy[indx] /= nbnum;
        q6ms_real_dz[indx] /= nbnum;
        q6ms_imag_dz[indx] /= nbnum;
          
        Q6i += Q6_coeff*(q6ms_real[indx]*q6ms_real[indx]+q6ms_imag[indx]*q6ms_imag[indx]);
        diff_Q6[0] += Q6_coeff*(2.0*q6ms_real[indx]*q6ms_real_dx[indx]+2.0*q6ms_imag[indx]*q6ms_imag_dx[indx]);
        diff_Q6[1] += Q6_coeff*(2.0*q6ms_real[indx]*q6ms_real_dy[indx]+2.0*q6ms_imag[indx]*q6ms_imag_dy[indx]);
        diff_Q6[2] += Q6_coeff*(2.0*q6ms_real[indx]*q6ms_real_dz[indx]+2.0*q6ms_imag[indx]*q6ms_imag_dz[indx]);
      }
      Q6i = std::sqrt(Q6i);
      if (Q6i > 1e-8) {
        array_atom[i][diff_x_col] += diff_Q6[0]/(2.0*Q6i);
        array_atom[i][diff_y_col] += diff_Q6[1]/(2.0*Q6i);
        array_atom[i][diff_z_col] += diff_Q6[2]/(2.0*Q6i);
      } 
      Q6_sum += Q6i;
      array_atom[i][val_col] = Q6i;


      // Pass 2 --- the neighbor atoms [j]
      if (Q6i > 1e-8) {
        for (jj = 0; jj < jnum; jj++) {
          diff_Q6[0] = diff_Q6[1] = diff_Q6[2] = 0.0;
          j = jlist[jj];
          j &= NEIGHMASK;
          double distx = x[i][0] - x[j][0];
          double disty = x[i][1] - x[j][1];
          double distz = x[i][2] - x[j][2];
          double r = sqrt(distx * distx + disty * disty + distz * distz);
          if (r < 1e-8 ||  r >= cutoff) continue;
          std::array<double,3> dist = {distx,disty,distz};
          std::array<double,104> Y6m = calculate_Y6m(dist);
          for (int deg = -6; deg <= 6; deg++) {
            int offset = (deg+6)*8;
            int indx = deg + 6;
            double q6m_j_real_dx = -Y6m[offset + 2] / nbnum;
            double q6m_j_imag_dx = -Y6m[offset + 3] / nbnum;
            double q6m_j_real_dy = -Y6m[offset + 4] / nbnum;;
            double q6m_j_imag_dy = -Y6m[offset + 5] / nbnum;;
            double q6m_j_real_dz = -Y6m[offset + 6] / nbnum;;
            double q6m_j_imag_dz = -Y6m[offset + 7] / nbnum;;
            diff_Q6[0] += Q6_coeff*(2.0*q6ms_real[indx]*q6m_j_real_dx+2.0*q6ms_imag[indx]*q6m_j_imag_dx);
            diff_Q6[1] += Q6_coeff*(2.0*q6ms_real[indx]*q6m_j_real_dy+2.0*q6ms_imag[indx]*q6m_j_imag_dy);
            diff_Q6[2] += Q6_coeff*(2.0*q6ms_real[indx]*q6m_j_real_dz+2.0*q6ms_imag[indx]*q6m_j_imag_dz); 
          }
          array_atom[j][diff_x_col] += diff_Q6[0]/(2.0*Q6i);
          array_atom[j][diff_y_col] += diff_Q6[1]/(2.0*Q6i);
          array_atom[j][diff_z_col] += diff_Q6[2]/(2.0*Q6i);
        }
      }
    } 
  }


  MPI_Allreduce(&Q6_sum, &Q6_sum_all, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&num_selected, &num_selected_all, 1, MPI_INT, MPI_SUM, world);


  scalar = num_selected_all ? Q6_sum_all / static_cast<double>(num_selected_all) :0.0;

  comm->reverse_comm(this);
  
}

/* ----------------------------------------------------------------------
   Q6sum = sqrt(sigma(average(Y6mreal)^2+average(Y6mimage)^2)overm)_atomi + 
   sqrt(sigma(average(Y6mreal)^2+average(Y6mimage)^2)overm)_atomj + ...
   even though the dY6m/dx(rij) = -dY6m/dx(ji) since they are under radical
   and powered by two we need to have ghost atom exchange

*/

int ComputeQ6Atom::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m, last;
  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = array_atom[i][diff_x_col];
    buf[m++] = array_atom[i][diff_y_col];
    buf[m++] = array_atom[i][diff_z_col];
  }
  return m;
}

/* ----------------------------------------------------------------------- */

void ComputeQ6Atom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    array_atom[j][diff_x_col] += buf[m++];
    array_atom[j][diff_y_col] += buf[m++];
    array_atom[j][diff_z_col] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
  computing the Y6m values
------------------------------------------------------------------------- */

std::array<double,104> ComputeQ6Atom::calculate_Y6m(const std::array<double,3>& dist)
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
