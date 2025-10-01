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

#include "compute_cparam_atom.h"

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

ComputeCParamAtom::ComputeCParamAtom(LAMMPS *lmp, int narg, char **arg) :
    ComputeDiffAtom{lmp, narg, arg}, n{6}, r0{3.0}
{
  if (narg == 3)
    return;
  else if (narg == 5) {
    n = utils::numeric(FLERR, arg[3], false, lmp);
    r0 = utils::numeric(FLERR, arg[4], false, lmp);
  } else {
    error->all(FLERR, "Illegal compute cparam/atom command");
  }
}

/* ---------------------------------------------------------------------- */

void ComputeCParamAtom::compute_all()
{
  if (last_compute == update->ntimestep) return;
  last_compute = update->ntimestep;


  double **x = atom->x;

  int *mask = atom->mask;
 


 

  if (atom->nmax > nmax)
  {
    nmax = atom->nmax;
    if (array_atom) memory->destroy(array_atom);
    memory->create(array_atom,nmax,size_peratom_cols,"compute_cparam_atom:array_atom");
  }

  int i, j, ii, jj, inum, jnum;
  int *ilist, *jlist, *numneigh, **firstneigh;

  //neighbor->build_one(list);
  //it is the same as atom->nlocal and list->gnum is zero since we have not requested for ghost neighbors in the list.
  inum = list->inum;   
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double Sdt_local = 0.0;
  double Sdt = 0.0;
  int group_atoms_local = 0;
  int group_atoms = 0;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    array_atom[i][diff_x_col]  = 0.0;
    array_atom[i][diff_y_col]  = 0.0;
    array_atom[i][diff_z_col]  = 0.0;
    if (mask[i] & groupbit) {
      group_atoms_local++;
      jnum = numneigh[i];
      jlist = firstneigh[i];

      double Sdi = 0.0;
      double Fxi = 0.0;
      double Fyi = 0.0;
      double Fzi = 0.0;

      for (jj = 0; jj < jnum; jj++) {
        double Sdij = 0.0;
        double Fij = 0.0;

        j = jlist[jj];
        j &= NEIGHMASK;
          
        double distx = x[i][0] - x[j][0];
        double disty = x[i][1] - x[j][1];
        double distz = x[i][2] - x[j][2];
        double r = std::sqrt(distx * distx + disty * disty + distz * distz);
        if (r < 1e-8) continue;

        double denom = 1.0 + pow(r/r0,n);
        Sdij = 1.0/denom;
        Sdi += Sdij;

        Fij = -n*pow(r,n-1)/(pow(r0,n)*denom*denom);

        Fxi += Fij * distx/r;
        Fyi += Fij * disty/r;
        Fzi += Fij * distz/r;
      }
      array_atom[i][val_col] = Sdi;
      array_atom[i][diff_x_col]  = Fxi;
      array_atom[i][diff_y_col]  = Fyi;
      array_atom[i][diff_z_col]  = Fzi;
      Sdt_local += Sdi;
    }
  }

  MPI_Allreduce(&Sdt_local,&Sdt,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&group_atoms_local,&group_atoms,1,MPI_INT,MPI_SUM,world);


  // calculate enthalpy per atom
  Sdt /= static_cast<double>(group_atoms);

  scalar = Sdt;

}
