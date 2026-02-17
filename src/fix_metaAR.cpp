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
/* ---------- v7.06.03 ----------------- */
#include <iostream>

#include "fix_metaAR.h"

#include "atom.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "output.h"
#include "region.h"
#include "respa.h"
#include "update.h"
#include "variable.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "dihedral.h"
#include "force.h"
#include "group.h"
#include "improper.h"
#include "kspace.h"
#include "pair.h"

#include "thermo.h"

#include <numeric>
#include <cmath>
#include <cstring>
#include <stdio.h>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum { NONE, CONSTANT, EQUAL, ATOM };

/* ---------------------------------------------------------------------- */

FixMetaAR::FixMetaAR(LAMMPS *lmp, int narg, char **arg) :
    Fix{lmp, narg, arg}, bias{nullptr}
{
  if (narg < 14) utils::missing_cmd_args(FLERR, "fix metaAR", error);

  dynamic_group_allow = 1;
  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extscalar = 1;    // What is this?
  extvector = 1;    // What is that?
  // It does not contribute to the global energy --> So, it should be zero
  energy_global_flag = 1;
  // It does not contribute to the global and per atom virials --> These should be zero    
  virial_global_flag = virial_peratom_flag = 0;
  // It does not support fix_modify respa --> zero    
  respa_level_support = 1; 
  peratom_flag = 1; 
  size_peratom_cols = 2;  
  ilevel_respa = 0;
  // Writes global info to the restart file
  restart_global = 1;

  nevery = utils::numeric(FLERR, arg[3], false, lmp);
  CV1CId = arg[4];
  CV2CId = arg[5];


  if (utils::strmatch(arg[6], "^v_")) {
    omegastr = std::string(arg[6]).substr(3);
  } else {
    omega = utils::numeric(FLERR, arg[6], false, lmp);
    omegastyle = CONSTANT;
  }
  if (utils::strmatch(arg[7], "^v_")) {
    sigmaCV1str = std::string(arg[7]).substr(3);
  } else {
    sigmaCV1 = utils::numeric(FLERR, arg[7], false, lmp);
    sigmaCV1style = CONSTANT;
  }
  if (utils::strmatch(arg[8], "^v_")) {
    sigmaCV2str = std::string(arg[8]).substr(3);
  } else {
    sigmaCV2 = utils::numeric(FLERR, arg[8], false, lmp);
    sigmaCV2style = CONSTANT;
  }
  if (utils::strmatch(arg[9], "^v_")) {
    deltaTstr = std::string(arg[9]).substr(3);
  } else {
    deltaT = utils::numeric(FLERR, arg[9], false, lmp);
    deltaTstyle = CONSTANT;
  }
  if (utils::strmatch(arg[10], "^v_")) {
    error->all(FLERR, "A variable input for DCV1 is not supported yet");
  } else {
    DCV1 = utils::numeric(FLERR, arg[10], false, lmp);
  }
  if (utils::strmatch(arg[11], "^v_")) {
    error->all(FLERR, "A variable input for DCV2 is not supported yet");
  } else {
    DCV2 = utils::numeric(FLERR, arg[11], false, lmp);
  }
  if (utils::strmatch(arg[12], "^v_")) {
    binsCV1str = std::string(arg[12]).substr(3);
  } else {
    binsCV1 = utils::numeric(FLERR, arg[12], false, lmp);
    binsCV1style = CONSTANT;
  }
  if (utils::strmatch(arg[13], "^v_")) {
    binsCV2str = std::string(arg[13]).substr(3);
  } else {
    binsCV2 = utils::numeric(FLERR, arg[13], false, lmp);
    binsCV2style = CONSTANT;
  }


  int iarg = 14;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "biasevery") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix metaAR biasevery", error);
      biasevery = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (biasevery <= 0) error->all(FLERR, "Invalid fix metaAR every argument: {}", nevery);
      if (biasevery < nevery) error->warning(FLERR, "biasevery is less than every so the printing frequecy will be different");
      iarg += 2;
    } else if (strcmp(arg[iarg], "region") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix metaAR region", error);
      region = domain->get_region_by_id(arg[iarg + 1]);
      if (!region) error->all(FLERR, "Region {} for fix metaAR does not exist", arg[iarg + 1]);
      idregion = utils::strdup(arg[iarg + 1]);
      iarg += 2;
    } else if ((strcmp(arg[iarg], "file") == 0) || (strcmp(arg[iarg], "append") == 0)) {
      if (iarg + 3 > narg)
        utils::missing_cmd_args(FLERR, std::string("fix metaAR ") + arg[iarg], error);
      if (comm->me == 0) {
        if (strcmp(arg[iarg], "file") == 0)
          fp.open(arg[iarg + 1], std::ios::out);
        else
          fp.open(arg[iarg + 1], std::ios::app);
        if (!fp.is_open())
          error->one(FLERR, "Cannot open fix metaAR file {}: {}", arg[iarg + 1],
                     utils::getsyserror());    // Copy from fix print
      }
      printfreq = utils::inumeric(FLERR, arg[iarg + 2], false, lmp);
      if (printfreq <= 0) error->all(FLERR, "Invalid fix metaAR every argument: {}", nevery);
      if (printfreq < nevery) error->warning(FLERR, "printfreq is less than every so the printing frequecy will be different");
      iarg += 3;
    } else if (strcmp(arg[iarg], "Bounds") == 0) {
      if (iarg + 5 > narg)
        utils::missing_cmd_args(FLERR, std::string("fix metaAR") + arg[iarg], error);
      minCV1 = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      maxCV1 = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      minCV2 = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      maxCV2 = utils::numeric(FLERR, arg[iarg + 4], false, lmp);
      setMinMax = false;
      iarg += 5;
    } else if (strcmp(arg[iarg], "welltempered") == 0) {
      // May be I should use just one general flag!
      welltempered_flag = true;
      iarg += 1;
    } else if (strcmp(arg[iarg],"numInstant") == 0) {
      numInstant = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else
      error->all(FLERR, "Unknown fix metaAR keyword: {}", arg[iarg]);
  }

  force_flag = 0;
  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;


  // KOKKOS package

  datamask_read = X_MASK | F_MASK | MASK_MASK | IMAGE_MASK;
  datamask_modify = F_MASK;

  // Just for debugging purposes
  fp3.open("S-dist.dat", std::fstream::out);


}

/* ---------------------------------------------------------------------- */

FixMetaAR::~FixMetaAR()
{
  if (bias)       memory->destroy(bias);
  if (array_atom) memory->destroy(array_atom);
  bias = nullptr;
  array_atom = nullptr;
}

/* ---------------------------------------------------------------------- */

int FixMetaAR::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;    //needs some implementations in the future
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMetaAR::init()
{
  // check computes
  class Compute* CV1ComputeTmp = modify->get_compute_by_id(CV1CId);
  class Compute* CV2ComputeTmp = modify->get_compute_by_id(CV2CId);

  if (CV1ComputeTmp == nullptr || CV2ComputeTmp == nullptr )
    error->all(FLERR, "Illegal computes in fix metaAR");

  // Do I need CVs derived from the ComputeDiffAtom???
  CV1Compute = dynamic_cast<ComputeDiffAtom*>(CV1ComputeTmp);
  CV2Compute = dynamic_cast<ComputeDiffAtom*>(CV2ComputeTmp);

  if (CV1Compute == nullptr || CV2Compute == nullptr )
    error->all(FLERR, "Illegal computes in fix metaAR");


  // check variables
  if (!omegastr.empty()) {
    omegavar = input->variable->find(omegastr.c_str());
    if (omegavar < 0) error->all(FLERR, "Variable {} for fix metaAR does not exist", omegastr);
    if (input->variable->equalstyle(omegavar))
      omegastyle = EQUAL;
    else if (input->variable->atomstyle(omegavar))
      error->all(FLERR, "Atomic style variable is not supported in fix metaAR");
    else
      error->all(FLERR, "Variable {} for fix metaAR is invalid style", omegavar);
  }
  if (!deltaTstr.empty()) {
    deltaTvar = input->variable->find(deltaTstr.c_str());
    if (deltaTvar < 0) error->all(FLERR, "Variable {} for fix metaAR does not exist", deltaTstr);
    if (input->variable->equalstyle(deltaTvar))
      deltaTstyle = EQUAL;
    else if (input->variable->atomstyle(deltaTvar))
      error->all(FLERR, "Atomic style variable is not supported in fix metaAR");
    else
      error->all(FLERR, "Variable {} for fix metaAR is invalid style", deltaTstr);
  }
  if (!sigmaCV1str.empty()) {
    sigmaCV1var = input->variable->find(sigmaCV1str.c_str());
    if (sigmaCV1var < 0) error->all(FLERR, "Variable {} for fix metaAR does not exist", sigmaCV1str);
    if (input->variable->equalstyle(sigmaCV1var))
      sigmaCV1style = EQUAL;
    else if (input->variable->atomstyle(sigmaCV1var))
      error->all(FLERR, "Atomic style variable is not supported in fix metaAR");
    else
      error->all(FLERR, "Variable {} for fix metaAR is invalid style", sigmaCV1var);
  }
  if (!sigmaCV2str.empty()) {
    sigmaCV2var = input->variable->find(sigmaCV2str.c_str());
    if (sigmaCV2var < 0) error->all(FLERR, "Variable {} for fix metaAR does not exist", sigmaCV2str);
    if (input->variable->equalstyle(sigmaCV2var))
      sigmaCV2style = EQUAL;
    else if (input->variable->atomstyle(sigmaCV2var))
      error->all(FLERR, "Atomic style variable is not supported in fix metaAR");
    else
      error->all(FLERR, "Variable {} for fix metaAR is invalid style", sigmaCV2var);
  }
  if (!binsCV1str.empty())
  {
    binsCV1var = input->variable->find(binsCV1str.c_str());
    if (binsCV1var < 0) error->all(FLERR, "Variable {} for fix metaAR does not exist", binsCV1str);
    if (input->variable->equalstyle(binsCV1var))
      binsCV1style = EQUAL;
    else if (input->variable->atomstyle(binsCV1var))
      error->all(FLERR, "Atomic style variable is not supported in fix metaAR");
     else
    error->all(FLERR, "Variable {} for fix metaAR is invalid style", binsCV1str);
  }
  if (!binsCV2str.empty()) {
    binsCV2var = input->variable->find(binsCV2str.c_str());
    if (binsCV2var < 0) error->all(FLERR, "Variable {} for fix metaAR does not exist", binsCV2str);
    if (input->variable->equalstyle(binsCV2var))
      binsCV2style = EQUAL;
    else if (input->variable->atomstyle(binsCV2var))
      error->all(FLERR, "Atomic style variable is not supported in fix metaAR");
    else
      error->all(FLERR, "Variable {} for fix metaAR is invalid style", binsCV2str);
  }

  // set index and check validity of region

  if (!idregion.empty()) {
    region = domain->get_region_by_id(idregion.c_str());
    if (!region) error->all(FLERR, "Region {} for fix metaAR does not exist", idregion);
  }

  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
    error->warning(FLERR, "Fix metaAR has not been tested with respa yet!");
  }
  
  
  // We need values of binsCV1 and binsCV2 so we allocate this after the variable parsing
  memory->create(bias, binsCV1, binsCV2, "metaAR:bias");
  

  nmax = atom->nmax;
  memory->create(array_atom, nmax, size_peratom_cols, "fix_metaAR:array_atom");

  // With every keyword in run the init is called at every N steps
  // so we need to delete locS and locH from previous runs
  locCV1.clear();
  locCV2.clear();


  // Allocating memory for locS and locH
  locCV1.resize(binsCV1);
  locCV2.resize(binsCV2);

  // Only if these vectors have not been set by restart
  if (restart_reset == 0) {
    // set the prev step CV1 and CV2
    CV1prev = 0.0;
    CV2prev = 0.0;

    // set the prev step biast
    biastprev = 0.0;

    // Allocating memory for entropy and enthalpy histories
    // We output the CV1 and CV2 values at step 0
    int nfixes = (update->nsteps) / nevery; 
    int nOut = static_cast<int> (nfixes  / numInstant) + 2;
         

    CV1.clear();
    CV2.clear();

    CV1.reserve(nOut);
    CV2.reserve(nOut);

    biasHistory.clear();
    biasHistory.reserve(nOut);

    // Start allocating CV1 and CV2 from index zero


    pv = 0.0;



    CV1Instant.clear();
    CV2Instant.clear();

    CV1Instant.reserve(numInstant);
    CV2Instant.reserve(numInstant);
  } else {
    // set the prev step CV1 and CV2
    int indx = CV1.size()-1;
    CV1prev = CV1[indx];
    CV2prev = CV2[indx];
    
    // set the prev step biast
    biastprev = biasHistory[indx];
  }
}

/* ----------------------------------------------------------------------- */

void FixMetaAR::setup(int vflag)
{
  for (int i = 0; i < binsCV1; i++)
    for (int j = 0; j < binsCV2; j++) bias[i][j] = 0.0;

  if (utils::strmatch(update->integrate_style, "^verlet"))
    post_force(vflag);
  else {
    (dynamic_cast<Respa *>(update->integrate))->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag, ilevel_respa, 0);
    (dynamic_cast<Respa *>(update->integrate))->copy_f_flevel(ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixMetaAR::post_force(int /*vflag*/)
{
  if (update->ntimestep % nevery) return;

  /* nsteps according to which the entropy and enthalpy size
       has been selected is equal to every value in run N every M
       and the firststep of the current run and the ntimestep
       is the total timestep until the current timestep
    */

  int fixstep = (update->ntimestep - update->firststep) / nevery;

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    memory->destroy(array_atom);
    memory->create(array_atom, nmax, 2, "metaAR:array_atom");
  }


  int CV1instantSize = CV1Instant.size();
  int CV2instantSize = CV2Instant.size();

  if (CV1instantSize != CV2instantSize)
    error->all(FLERR,"Internal Error!");


  if (CV1instantSize  == numInstant) {
    CV1t = std::accumulate(CV1Instant.begin(),CV1Instant.end(),0.0) / static_cast<double>(numInstant);
    CV2t = std::accumulate(CV2Instant.begin(),CV2Instant.end(),0.0) / static_cast<double>(numInstant);

    CV1.push_back(CV1t);
    CV2.push_back(CV2t);

    int CV1size = CV1.size();
    int CV2size = CV2.size();

    if (CV1size != CV2size)
      error->all(FLERR,"Internal Error!");

    calculate_bias(CV1size);

    modify_force();


    CV1Instant.clear();
    CV2Instant.clear();
  }


  CV1Compute->compute_all();
  CV2Compute->compute_all();

  CV1tInstant = CV1Compute->scalar;
  CV2tInstant = CV2Compute->scalar;

  double** CV1_atom = CV1Compute->array_atom;
  double** CV2_atom = CV2Compute->array_atom; 
  const int cv1_col = CV1Compute->val_col;
  const int cv2_col = CV2Compute->val_col;

  
  // We do know that nmax > nlocal
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
  {
    array_atom[i][0] = CV1_atom[i][cv1_col];
  }
  for (int i = 0; i < nlocal; i++)
  {
    array_atom[i][1] = CV2_atom[i][cv2_col];
  }

  CV1Instant.push_back(CV1tInstant);
  CV2Instant.push_back(CV2tInstant);

  //compute_q6_peratom(2,CV1Instant,dCV1_dX,CV1tInstant,0);
  //compute_entropy_peratom(CV1Instant,dCV1_dX,CV1tInstant,0);
  //compute_q6_peratom(3,CV2Instant,dCV2_dX,CV2tInstant,1);
  //compute_enthalpy_peratom(CV2Instant,dCV2_dX,CV2tInstant,1);
  //compute_density_peratom(CV1Instant,dCV1_dX,CV1tInstant,0);
  //compute_density_peratom(CV2Instant,dCV2_dX,CV2tInstant,1);


  if (!fixstep) step0();

  print_debug_1d(fixstep/numInstant);


  if (!(update->ntimestep % printfreq) && fixstep) print_meta();

  if (!(update->ntimestep % biasevery)) writefp2(fixstep/numInstant);
}

/* ---------------------------------------------------------------------- */

void FixMetaAR::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixMetaAR::step0()
{
  if (setMinMax == true) {
    minCV1 = CV1tInstant - DCV1 / 2.0;
    maxCV1 = CV1tInstant + DCV1 / 2.0;
    minCV2 = CV2tInstant - DCV2 / 2.0;
    maxCV2 = CV2tInstant + DCV2 / 2.0;
  }

  // Initializing the locS and locH variables
  init_loc();
  writeheader();

  // Just for debugging purposes
  print_debug_1d_header();

  if (comm->me == 0) error->warning(FLERR, "CV1={},CV2={},DCV1={},DCV2={}", CV1tInstant, CV2tInstant, DCV1, DCV2);
}

/* ----------------------------------------------------------------------
   initializing the locCV1 and locCV2 variables
------------------------------------------------------------------------- */

void FixMetaAR::init_loc()
{
  double loc;
  for (int i = 0; i < binsCV1; i++) {
    loc = minCV1 + (i + 0.5) * (maxCV1 - minCV1) / binsCV1;
    locCV1[i] = loc;
  }
  for (int i = 0; i < binsCV2; i++) {
    loc = minCV2 + (i + 0.5) * (maxCV2 - minCV2) / binsCV2;
    locCV2[i] = loc;
  }
}
  
/* ----------------------------------------------------------------------
   modify_force --> adapted from src/fix_addforce.cpp
------------------------------------------------------------------------- */

void FixMetaAR::modify_force()
{
  double **x = atom->x;
  double **f = atom->f;


  int *mask = atom->mask;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;



  double sforce[3] = {0.0,0.0,0.0};


  double** CV1_atom = CV1Compute->array_atom;
  const int cv1_col_dx = CV1Compute->diff_x_col;
  const int cv1_col_dy = CV1Compute->diff_y_col;
  const int cv1_col_dz = CV1Compute->diff_z_col;
  
  double** CV2_atom = CV2Compute->array_atom; 
  const int cv2_col_dx = CV2Compute->diff_x_col;
  const int cv2_col_dy = CV2Compute->diff_y_col;
  const int cv2_col_dz = CV2Compute->diff_z_col;

  if (update->ntimestep % nevery) return;

  // virial setup

  //v_init(vflag);

  // update region if necessary

  if (region) region->prematch();

  // foriginal[0] = "potential energy" for added force
  // foriginal[123] = force on atoms before extra force added

  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;
  force_flag = 0;

  double unwrap[3];
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (region && !region->match(x[i][0], x[i][1], x[i][2])) continue;
      sforce[0] = fCV1 * CV1_atom[i][cv1_col_dx] + fCV2 * CV2_atom[i][cv2_col_dx];
      sforce[1] = fCV1 * CV1_atom[i][cv1_col_dy] + fCV2 * CV2_atom[i][cv2_col_dy];
      sforce[2] = fCV1 * CV1_atom[i][cv1_col_dz] + fCV2 * CV2_atom[i][cv2_col_dz];
      if (std::isnan(sforce[0]) || std::isnan(sforce[1]) || std::isnan(sforce[2])) 
        error->warning(FLERR,"Nan value in the fix metaAR");
      domain->unmap(x[i], image[i], unwrap);
      foriginal[0] -= sforce[0] * unwrap[0] + sforce[1] * unwrap[1] + sforce[2] * unwrap[2];
      foriginal[1] += sforce[0];
      foriginal[2] += sforce[1];
      foriginal[3] += sforce[2];
      f[i][0] += sforce[0];
      f[i][1] += sforce[1];
      f[i][2] += sforce[2];
      /*if (evflag) {
                e[0] = force[0] * unwrap[0];
                e[1] = force[1] * unwrap[1];
                e[2] = force[2] * unwrap[2];
                e[3] = force[0] * unwrap[1];
                e[4] = force[0] * unwrap[2];
                e[5] = force[1] * unwrap[2];
                v_tally(i, v);
            }*/
    }
}

/* ----------------------------------------------------------------------
   calculating the bias amount
   ---------------------------------------------------------------------- */

void FixMetaAR::calculate_bias(const int &n)
{
  double boltz = force->boltz;
  double welltemperfactor = 1;


  double sigmasq2CV1 = 2 * sigmaCV1 * sigmaCV1;
  double sigmasq2CV2 = 2 * sigmaCV2 * sigmaCV2;


  fCV1 = 0.0;
  fCV2 = 0.0;


  for (int i = 0; i < n - 1; i++) {
    double dCV1 = CV1t - CV1[i];
    double dCV2 = CV2t - CV2[i];
    double A = (dCV1*dCV1) / sigmasq2CV1 + (dCV2*dCV2) / sigmasq2CV2;
    double dA_dCV1 = 2*dCV1 / sigmasq2CV1;
    double dA_dCV2 = 2*dCV2 / sigmasq2CV2;


    if (std::isnan(dCV1)) {
      if (comm->me == 0) error->warning(FLERR, "CV1 is nan");
      dCV1 = 0.0;
      dA_dCV1 = 0.0;
    }
    if (std::isnan(dCV2)) {
      if (comm->me == 0) error->warning(FLERR, "CV2 is nan");
      dCV2 = 0.0;
      dA_dCV2 = 0.0;
    }


    if (welltempered_flag == true) welltemperfactor = exp(-biasHistory[i] / (boltz * deltaT));
    double BCFactor = 1.0;

    if (CV1[i] < minCV1 || CV1[i] > maxCV1 || CV2[i] < minCV2 || CV2[i] > maxCV2) {
      welltemperfactor = 1.0;
      if (welltempered_flag == false) BCFactor = 5.0;
    }
    biast += 0.592 * omega * welltemperfactor * BCFactor * exp(-A);
    fCV1 += -0.592 * omega * welltemperfactor * BCFactor * (-dA_dCV1) * exp(-A);
    fCV2 += -0.592 * omega * welltemperfactor * BCFactor * (-dA_dCV2) * exp(-A);
  }


  // keeping the track of added biasses

  vCV1 = (CV1t - CV1prev) / (numInstant * nevery * update->dt);
  vCV2 = (CV2t - CV2prev) / (numInstant * nevery * update->dt);

  CV1prev = CV1t;
  CV2prev = CV2t;

  // biast velocity calculation
  Vbiast = (biast - biastprev) / (numInstant * nevery * update->dt);
  biastprev = biast;

  //if (Bt > 1e-10 || Bt < -1e-10)
  biasHistory.push_back(biast);

  /* Taking care of the situation in which the systems falls out of the grid
    */

  if (CV1t < minCV1) {
    if (comm->me == 0) error->warning(FLERR, "CV1t={} < {}, out of the grid", CV1t, minCV1);
  } else if (CV1t >= maxCV1) {
    if (comm->me == 0) error->warning(FLERR, "CV1t={} > {}, out of the grid", CV1t, maxCV1);
  } else if (CV2t < minCV2) {
    if (comm->me == 0) error->warning(FLERR, "CV2t={} < {}, out of the grid", CV2t, minCV2);
  } else if (CV2t >= maxCV2) {
    if (comm->me == 0) error->warning(FLERR, "CV2t={} > {}, out of the grid", CV2t, maxCV2);
  } else {
    //bias[Sindex][Hindex] = biast;
  }
}

/* ----------------------------------------------------------------------
   writing the header of the file
   ---------------------------------------------------------------------- */

void FixMetaAR::writeheader()
{
  if (comm->me == 0) {
    fp << "timestep,entropy,enthalpy,entropy_speed,enthalpy_speed,bias,bias_speed\n";
  }
}

/* ----------------------------------------------------------------------
   writing the bias distribution
   ---------------------------------------------------------------------- */

void FixMetaAR::writebiasdist(std::fstream &fp2, const int &n)
{
  double boltz = force->boltz;
  if (comm->me == 0) {
    for (int i = 0; i < binsCV1; i++)
      for (int j = 0; j < binsCV2; j++) {
        double Bt = 0.0;
        bias[i][j] = 0.0;
        for (int k = 0; k < n - 1; k++) {
          double A = 0.5 *
              ((locCV1[i] - CV1[k]) * (locCV1[i] - CV1[k]) / (sigmaCV1 * sigmaCV1) +
               (locCV2[j] - CV2[k]) * (locCV2[j] - CV2[k]) / (sigmaCV2 * sigmaCV2));
               
          Bt += 0.592 * omega * exp(-biasHistory[k]/ (boltz * deltaT))* exp(-A);
        }
        bias[i][j] = Bt;
      }

    fp2 << "binsCV1\\binsCV2" << std::endl;
    for (int j = 0; j < binsCV1; j++) fp2 << "," << locCV1[j];
    fp2 << "\n";
    for (int i = 0; i < binsCV2; i++) {
      fp2 << locCV2[i];
      for (int j = 0; j < binsCV1; j++) fp2 << "," << bias[i][j];
      fp2 << std::endl;
    }
  }
}

/* ----------------------------------------------------------------------
   printing the output
   ---------------------------------------------------------------------- */

void FixMetaAR::print_meta()
{
  bigint ntimestep = update->ntimestep;
  if (comm->me == 0) {
    fp << ntimestep << "," << CV1t << "," << CV2t << "," << vCV1 << "," << vCV2 << "," << biast << ","
         << Vbiast << std::endl;
  }
}

/* ----------------------------------------------------------------------
   Just for debugging
   ---------------------------------------------------------------------- */

void FixMetaAR::print_debug_1d_header()
{
  if (comm->me == 0) {
    fp3 << "timestep";
    double CV1t = minCV1;
    for (int i = 0; i < binsCV1; i++) {
      fp3 << "," << CV1t;
      CV1t += DCV1 / binsCV1;
    }
    fp3 << std::endl;
  }
}

/* ----------------------------------------------------------------------
   Just for debugging
   ---------------------------------------------------------------------- */

void FixMetaAR::print_debug_1d(const int &n)
{
  bigint ntimestep = update->ntimestep;
  if (comm->me == 0) {
    fp3 << ntimestep;
    double CV1t = minCV1;
    for (int j = 0; j < binsCV1; j++) {
      double Bt = 0.0;
      for (int i = 0; i < n - 1 ; i++) {
        double A = 0.5 *((CV1t-CV1[i])*(CV1t-CV1[i])/(sigmaCV1*sigmaCV1));
        Bt += 0.592 * omega * 1 * exp(-A); 
      }
      fp3 << "," << Bt;
      CV1t += DCV1/binsCV1;
    }

    fp3 << std::endl;
  }
}


/* ----------------------------------------------------------------------
   another debugging function 
   ---------------------------------------------------------------------- */

void FixMetaAR::writefp2(const int &fixstep)
{
  if (comm->me == 0) {
    int value = update->ntimestep / biasevery;

    // Create filename using ostringstream
    std::string filename;
    filename = "biasdist-" + std::to_string(value) + "-metaAR.dat";

  

    // Open file stream
    std::fstream fp2(filename,std::ios::out);
    if (!fp2.is_open()) { error->one(FLERR, "Cannot open file for the metaAR bias distribution"); }

    writebiasdist(fp2, fixstep); 
  }
}

/* ----------------------------------------------------------------------
   potential energy of added force
------------------------------------------------------------------------- */

double FixMetaAR::compute_scalar()
{
  return this->biast;
}

/* ----------------------------------------------------------------------
   return components of total force on fix group before force was changed
------------------------------------------------------------------------- */

double FixMetaAR::compute_vector(int n)
{
  MPI_Allreduce(foriginal, foriginal_all, 4, MPI_DOUBLE, MPI_SUM, world);

  foriginal_all[4] =
      sqrt(foriginal_all[1] * foriginal_all[1] + foriginal_all[2] * foriginal_all[2] +
           foriginal_all[3] * foriginal_all[3]);
  return foriginal_all[n + 1];
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixMetaAR::memory_usage()
{
  double bytes = 0.0;
  return bytes;
}


/* -----------------------------------------------------------------------
    writing the restart data
   ----------------------------------------------------------------------- */

void FixMetaAR::write_restart(FILE* fp)
{
    if (comm->me == 0)
    {
        // CV1, CV2, CV1Instant, CV2Instant and biasHistory
        int CV1size = CV1.size();
        int CV2size = CV2.size();
        int CV1Instantsize = CV1Instant.size();
        int CV2Instantsize = CV2Instant.size();
        int biasHistorysize = biasHistory.size();

        if (CV1size != CV2size ||
            CV2size != biasHistorysize ||
            CV1Instantsize != CV2Instantsize)
            error->one(FLERR,"Internal error! ,{}, {}, {}, {}, {}",CV1size,CV2size,biasHistorysize,CV1Instantsize,CV2Instantsize);

        // The first element is for the CV1size and the second for the CV1Instantsize!
        int nsize = 2 + 3*CV1size + 2*CV1Instantsize;
        std::vector<double> data;
        data.reserve(nsize);

        data.push_back(static_cast<double>(CV1size));
        data.push_back(static_cast<double>(CV1Instantsize));
        for (int i = 0; i < CV1size; i++)
        {
            data.push_back(CV1[i]);
            data.push_back(CV2[i]);
            data.push_back(biasHistory[i]);
        }
        for (int i = 0; i < CV1Instantsize; i++)
        {
          data.push_back(CV1Instant[i]);
          data.push_back(CV2Instant[i]);
        }


        int size = nsize * sizeof(double);
        fwrite(&size,sizeof(int),1,fp);
        fwrite(data.data(),sizeof(double),nsize,fp);
    }

    if (comm->me == 0)
    {
        int CV1size = CV1.size();
        int CV1Instantsize = CV1Instant.size();
        error->warning(FLERR,"Debugging the write restart!");
        error->warning(FLERR,"step,CV1,CV2,biasHistory");
        for (int i = 0; i < CV1size; i++)
        {
          error->warning(FLERR,"{},{},{},{}",i,CV1[i],CV2[i],biasHistory[i]);
        }
        error->warning(FLERR,"step,CV1Instant,CV2Instant");
        for (int i = 0; i < CV1Instantsize; i++)
        {
          error->warning(FLERR,"{},{},{}",i,CV1Instant[i],CV2Instant[i]);
        }
    }
}

/* -----------------------------------------------------------------------
    reading the restart data
   ----------------------------------------------------------------------- */

void FixMetaAR::restart(char * buf)
{
  int n = 0;
  auto *list = (double *)buf;
  int CV1size = static_cast<int>(list[n++]);
  int CV1Instantsize = static_cast<int>(list[n++]);

  CV1.clear();
  CV2.clear();
  biasHistory.clear();
  CV1Instant.clear();
  CV2Instant.clear();

  CV1.reserve(CV1size);
  CV2.reserve(CV1size);
  biasHistory.reserve(CV1size);
  CV1Instant.reserve(CV1Instantsize);
  CV2Instant.reserve(CV1Instantsize);


  for (int i = 0; i < CV1size; i++)
  {
    CV1.push_back(list[n++]);
    CV2.push_back(list[n++]);
    biasHistory.push_back(list[n++]);
  }
  for (int i = 0; i < CV1Instantsize ; i++)
  {
    CV1Instant.push_back(list[n++]);
    CV2Instant.push_back(list[n++]);
  }
    
  if (comm->me == 0)
  {
      int CV1size = CV1.size();
      int CV1instantSize = CV1Instant.size();
      error->warning(FLERR,"Debugging the read restart!");
      error->warning(FLERR,"step,CV1,CV2,biasHistory");
      for (int i = 0; i < CV1size; i++)
      {
        error->warning(FLERR,"{},{},{},{}",i,CV1[i],CV2[i],biasHistory[i]);
      }
      error->warning(FLERR,"step,CV1Instant,CV2Instant");
      for (int i = 0; i < CV1instantSize; i++)
      {
        error->warning(FLERR,"{},{},{}",i,CV1Instant[i],CV2Instant[i]);
      }
  }
} 