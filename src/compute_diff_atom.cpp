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

#include "compute_diff_atom.h"

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

ComputeDiffAtom::ComputeDiffAtom(LAMMPS *lmp, int narg, char **arg) :
    Compute{lmp, narg, arg}
{}

/* ---------------------------------------------------------------------- */

ComputeDiffAtom::~ComputeDiffAtom()
{
  if (array_atom) memory->destroy(array_atom);
  array_atom = nullptr;
}

/* ---------------------------------------------------------------------- */

void ComputeDiffAtom::init()
{

  scalar_flag = 1;
  peratom_flag = 1;
  size_peratom_cols = 6;
  extscalar = 0;

  nmax = atom->nmax;
  memory->create(array_atom,nmax,size_peratom_cols,"compute_diff:array_atom");
  
  
  // Request a full neighbor list
  int list_flags = NeighConst::REQ_FULL;    // The Full neighbor list is needed to have both the ij and ji pairs

  // need neighbors of the ghost atoms
  list_flags |= NeighConst::REQ_GHOST; // I need the ghost atoms to be included in the neighbor list
  // Occasional neighbor list build
  // list_flags |= NeighConst::REQ_OCCASIONAL;

  // request for a neighbor list
  request = neighbor->add_request(this, list_flags);
}

/* ---------------------------------------------------------------------- */

void ComputeDiffAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}


/* ---------------------------------------------------------------------- */

double ComputeDiffAtom::compute_scalar()
{
  if (last_compute != update->ntimestep) compute_all();
  invoked_scalar = update->ntimestep;  // LAMMPS bookkeeping
  return scalar; 
}

/* ---------------------------------------------------------------------- */

void ComputeDiffAtom::compute_peratom()
{
  if (last_compute != update->ntimestep) compute_all();
  invoked_scalar = update->ntimestep;  // LAMMPS bookkeeping
}

