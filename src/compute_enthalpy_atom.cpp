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

#include "compute_enthalpy_atom.h"

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

ComputeEnthalpyAtom::ComputeEnthalpyAtom(LAMMPS *lmp, int narg, char **arg) :
    ComputeDiffAtom{lmp, narg, arg}
{}

/* ---------------------------------------------------------------------- */

void ComputeEnthalpyAtom::compute_all()
{
  //if (last_compute == update->ntimestep) return;
  last_compute = update->ntimestep;

  double one = 0.0;
  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  
  bool pairflag = true;

  if (atom->nmax > nmax)
  {
    nmax = atom->nmax;
    if (array_atom) memory->destroy(array_atom);
    memory->create(array_atom,nmax,size_peratom_cols,"compute_q6:array_atom");
  }
  
  
  
  int i, j, ii, jj, inum, jnum;
  int *ilist, *jlist, *numneigh, **firstneigh;
  
  neighbor->build_one(list);
  //it is the same as atom->nlocal and list->gnum is zero since we have not requested for ghost neighbors in the list.
  inum = list->inum;   
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  
  double Ent_local = 0.0;
  double Ent = 0.0;
  int group_atoms_local = 0;
  int group_atoms = 0;
  
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    array_atom[i][val_col] = 0.0;
    array_atom[i][diff_x_col] = 0.0;
    array_atom[i][diff_y_col] = 0.0;
    array_atom[i][diff_z_col] = 0.0;
    if (mask[i] & groupbit) {
      group_atoms_local++;
      jnum = numneigh[i];
      jlist = firstneigh[i];
  
  
  
      for (jj = 0; jj < jnum; jj++) {
        double Enji = 0.0;
        double FPji = 0.0;
  
        j = jlist[jj];
        j &= NEIGHMASK;
        if (!(mask[j] & groupbit))
          continue;
            
        double distx = x[j][0] - x[i][0];
        double disty = x[j][1] - x[i][1];
        double distz = x[j][2] - x[i][2];
        double rsq = distx * distx + disty * disty + distz * distz;
  
        if (rsq < 1e-8) continue;
  
        if (pairflag && force->pair) {
          Enji = force->pair->single(j,i,type[j],type[i],rsq,1,1,FPji);
            /*
             * FPji here is the Fji*rji
             * dU/dxi = sigma ((-FPji/rji) * (drji/dxi)) = sigma (FPji*xji/(rji*rji))
             */
          Ent_local += Enji;
          array_atom[i][val_col] += Enji;
          array_atom[i][diff_x_col]  += distx * FPji / rsq;
          array_atom[i][diff_y_col]  += disty * FPji / rsq;
          array_atom[i][diff_z_col]  += distz * FPji / rsq;
        }
      }
    }
  }
  
  MPI_Allreduce(&Ent_local,&Ent,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&group_atoms_local,&group_atoms,1,MPI_INT,MPI_SUM,world);
  
  
  // calculate enthalpy per atom
  Ent /= static_cast<double>(group_atoms);
  
    // keep track of the enthalpy evolution
  scalar = Ent;
  
}