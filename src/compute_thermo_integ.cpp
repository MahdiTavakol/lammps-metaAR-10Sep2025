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
/* ------------------------------------------------------------------------------------------
   ----  Compute Thermo_Integ written by Mahdi Tavakol (Oxford) mahditavakol90@gmail.com ----
   ----  For some reasons that I do not know why it does not work with the intel package ----
   ------------------------------------------------------------------------------------------ */

   #include "compute_thermo_integ.h"

   #include "atom.h"
   #include "comm.h"
   #include "domain.h"
   #include "error.h"
   #include "fix.h"
   #include "force.h"
   #include "input.h"
   #include "kspace.h"
   #include "memory.h"
   #include "modify.h"
   #include "pair.h"
   #include "pair_hybrid.h"
   #include "timer.h"
   #include "update.h"
   #include "variable.h"
   
   using namespace LAMMPS_NS;
   
   enum { SINGLE = 1 << 0, DUAL = 1 << 1 };
   enum {
       PAIR = 1 << 0,
       CHARGE = 1 << 1,
   };
   enum {POSITIVE,NEGATIVE};
   
   /* ---------------------------------------------------------------------- */
   
   ComputeThermoInteg::ComputeThermoInteg(LAMMPS* lmp, int narg, char** arg) : Compute(lmp, narg, arg)
   {
       if (narg < 11) error->all(FLERR, "Illegal number of arguments in compute ti");
   
   
       peflag = 1;
       peatomflag = 1;
       peratom_flag = 1;
       comm_reverse = 1;
       
       
       scalar_flag = 0;
       vector_flag = 1;
       size_vector = 6;
       peratom_flag = 1; // I need to have per atom energies tallied. 
       
       extvector = 0;
   
       vector = new double[size_vector];
   
       parameter_list = 0;
       mode = 0;
   
   
       lambda = utils::numeric(FLERR,arg[3],false,lmp);
       dlambda = utils::numeric(FLERR,arg[4],false,lmp);
   
       int iarg = 5;
       if (strcmp(arg[iarg], "dual") == 0)
       {
           error->all(FLERR,"Dual topology is not yet supported");
           mode |= DUAL;
           ntypeAs = utils::numeric(FLERR, arg[iarg+1], false, lmp);
           iarg += 2;
           typeAs = new int[ntypeAs];
           for (int i = 0; i < ntypeAs; i++)
           {
              typeAs[i] = utils::numeric(FLERR,arg[iarg],false,lmp);
              if (typeAs[i] > atom->ntypes) error->all(FLERR, "Illegal compute TI atom type {}", typeAs[i]);
              iarg++;
           }
           ntypeBs = utils::numeric(FLERR, arg[iarg], false, lmp);
           iarg++;
           typeBs = new int[ntypeBs];
           for (int i = 0; i < ntypeBs; i++)
           {
              typeBs[i] = utils::numeric(FLERR,arg[iarg],false,lmp);
              if (typeBs[i] > atom->ntypes) error->all(FLERR, "Illegal compute TI atom type {}", typeBs[i]);
              iarg++;
           }
       }
       else if (strcmp(arg[iarg], "single") == 0)
       {
           mode |= SINGLE;
           ntypeAs = utils::numeric(FLERR,arg[iarg+1],false,lmp);
           ntypeBs = 0;
           iarg += 2;
           typeAs = new int[ntypeAs];
           for (int i = 0; i < ntypeAs; i++)
           {
              typeAs[i] = utils::numeric(FLERR,arg[iarg],false,lmp);
              if (typeAs[i] > atom->ntypes) error->all(FLERR, "Illegal compute TI atom type {}", typeAs[i]);
              iarg++;
           }
       }
       else error->all(FLERR, "Unknown compute TI style {}", arg[iarg]);
   
   
       while (iarg < narg)
       {
           if (strcmp(arg[iarg], "pair") == 0)
           {
               parameter_list |= PAIR;
               pstyle = utils::strdup(arg[iarg + 1]);
               iarg += 2;
               lA_ps = new double[ntypeAs];  //NEGATIVE
               lB_ps = new double[ntypeAs];  //POSITIVE
               pparams = new char*[ntypeAs];
   
               if (utils::is_double(arg[iarg]) == false) {
                  for (int k = 0; k < ntypeAs; k++)
                  {
                     per_atom_epsilon = true;
                     double p_initial, p_final;
                     char * pparam = utils::strdup(arg[iarg]);
                     p_initial = utils::numeric(FLERR, arg[iarg+1], false, lmp);
                     p_final = utils::numeric(FLERR, arg[iarg+2], false, lmp);
                     pparams[k] = pparam;
                     lA_ps[k] = -(p_final - p_initial) * dlambda;
                     lB_ps[k] = (p_final - p_initial) * dlambda;
                     iarg+=3;
                  }
               } else {
                  for (int k = 0; k < ntypeAs; k++)
                  {
                     per_atom_epsilon = false;
                     double p_initial, p_final;
                     p_initial = utils::numeric(FLERR, arg[iarg], false, lmp);
                     p_final = utils::numeric(FLERR, arg[iarg+1], false, lmp);
                     lA_ps[k] = -(p_final - p_initial) * dlambda;
                     lB_ps[k] = (p_final - p_initial) * dlambda;
                     iarg+=2;
                  }
               }
           }
           else if (strcmp(arg[iarg], "charge") == 0)
           {
               parameter_list |= CHARGE;
               iarg++;
               lA_qs = new double[ntypeAs];  //NEGATIVE
               lB_qs = new double[ntypeAs];  //POSITIVE
               for (int k = 0; k < ntypeAs; k++)
               {
                  double q_initial, q_final;
                  q_initial = utils::numeric(FLERR, arg[iarg], false, lmp);
                  q_final = utils::numeric(FLERR, arg[iarg+1], false, lmp);
                  lA_qs[k] = -(q_final - q_initial) * dlambda;
                  lB_qs[k] = (q_final - q_initial) * dlambda;
                  iarg+=2;
               }
               typeC = utils::numeric(FLERR, arg[iarg], false, lmp);
               if (typeC > atom->ntypes) error->all(FLERR, "Illegal compute TI atom type {}", typeC);
               iarg++;
           }
           else error->all(FLERR, "Unknown compute TI keyword {}", arg[iarg]);
       }
   
   
       // allocate space for charge, force, energy, virial arrays
   
       f_orig = nullptr;
       q_orig = nullptr;
       peatom_orig = keatom_orig = nullptr;
       pvatom_orig = kvatom_orig = nullptr;
   
       fixgpu = nullptr;
   
       nmax = atom->nmax;
       natoms = 0; // I did this on purpose so that the delta_qC is updated at least in the first step.
   
       allocate_storage();
   }
   
   /* ---------------------------------------------------------------------- */
   ComputeThermoInteg::~ComputeThermoInteg()
   {
       deallocate_storage();
       memory->destroy(epsilon_inits);
       memory->destroy(energy_peratom);
   
       delete[] typeAs;
       if (mode & DUAL)  delete[] typeBs;
       if (parameter_list & PAIR) {
          delete[] lA_ps;
          delete[] lB_ps;
       }
       if (parameter_list & CHARGE) {
          delete[] lA_qs;
          delete[] lB_qs;     
       }
       for (int i = 0; i < ntypeAs; i++) {
          delete [] pparams[i];
       }
       delete[] pparams;
       delete[] pstyle;
       delete[] vector;
   }
   
   /* ---------------------------------------------------------------------- */
   
   void ComputeThermoInteg::setup()
   {
       if (parameter_list & PAIR)
       {
          pair = nullptr;
          if (lmp->suffix_enable)
             pair = force->pair_match(std::string(pstyle) + "/" + lmp->suffix, 1);
          if (pair == nullptr)
             pair = force->pair_match(pstyle, 1); // I need to define the pstyle variable
   
          int ntypes = atom->ntypes;
          memory->create(epsilon_inits, ntypeAs, ntypes + 1, ntypes + 1, "compute_TI:epsilon_inits");
          
          for (int k = 0; k < ntypeAs; k++)
          {
             void* ptr1 = pair->extract(pparams[k], pdim);
             if (ptr1 == nullptr)
                error->all(FLERR, "Compute TI pair style {} was not found", pstyle);
             if (pdim != 2)
                error->all(FLERR, "Pair style parameter {} is not compatible with compute TI", pparams[k]);
   
             epsilons[k] = (double**)ptr1;
          
             for (int i = 0; i < ntypes + 1; i++)
                for (int j = i; j < ntypes + 1; j++)
                    epsilon_inits[k][i][j] = epsilons[k][i][j];
          }
       }
     
       int nmax = atom->nmax;     
       memory->create(energy_peratom,nmax,"compute_TI:energy_peratom");
                
   }
   
   /* ---------------------------------------------------------------------- */
   
   void ComputeThermoInteg::init()
   {
       if ((parameter_list & PAIR) && !per_atom_epsilon)
       {
           std::map<std::string, std::string> pair_params;
          
           pair_params["lj/cut/soft/omp"] = "lambda";
           pair_params["lj/cut/coul/cut/soft/gpu"] = "lambda";
           pair_params["lj/cut/coul/cut/soft/omp"] = "lambda";
           pair_params["lj/cut/coul/long/soft"] = "lambda";
           pair_params["lj/cut/coul/long/soft/gpu"] = "lambda";
           pair_params["lj/cut/coul/long/soft/omp"] = "lambda";
           pair_params["lj/cut/tip4p/long/soft"] = "lambda";
           pair_params["lj/cut/tip4p/long/soft/omp"] = "lambda";
           pair_params["lj/charmm/coul/long/soft"] = "lambda";
           pair_params["lj/charmm/coul/long/soft/omp"] = "lambda";
           pair_params["lj/class2/soft"] = "lambda";
           pair_params["lj/class2/coul/cut/soft"] = "lambda";
           pair_params["lj/class2/coul/long/soft"] = "lambda";
           pair_params["coul/cut/soft"] = "lambda";
           pair_params["coul/cut/soft/omp"] = "lambda";
           pair_params["coul/long/soft"] = "lambda";
           pair_params["coul/long/soft/omp"] = "lambda";
           pair_params["tip4p/long/soft"] = "lambda";
           pair_params["tip4p/long/soft/omp"] = "lambda";
           pair_params["morse/soft"] = "lambda";
   
           pair_params["lj/charmm/coul/charmm"] = "lj14_1";
           pair_params["lj/charmm/coul/charmm/gpu"] = "lj14_1";
           pair_params["lj/charmm/coul/charmm/intel"] = "lj14_1";
           pair_params["lj/charmm/coul/charmm/kk"] = "lj14_1";
           pair_params["lj/charmm/coul/charmm/omp"] = "lj14_1";
           pair_params["lj/charmm/coul/charmm/implicit"] = "lj14_1";
           pair_params["lj/charmm/coul/charmm/implicit/kk"] = "lj14_1";
           pair_params["lj/charmm/coul/charmm/implicit/omp"] = "lj14_1";
           pair_params["lj/charmm/coul/long"] = "epsilon";
           pair_params["lj/charmm/coul/long/gpu"] = "epsilon";
           pair_params["lj/charmm/coul/long/intel"] = "epsilon";
           pair_params["lj/charmm/coul/long/kk"] = "epsilon";
           pair_params["lj/charmm/coul/long/opt"] = "epsilon";
           pair_params["lj/charmm/coul/long/omp"] = "epsilon";
           pair_params["lj/charmm/coul/msm"] = "lj14_1";
           pair_params["lj/charmm/coul/msm/omp"] = "lj14_1";
           pair_params["lj/charmmfsw/coul/charmmfsh"] = "lj14_1";
           pair_params["lj/charmmfsw/coul/long"] = "lj14_1";
           pair_params["lj/charmmfsw/coul/long/kk"] = "lj14_1";
   
           if (pair_params.find(pstyle) == pair_params.end())
               error->all(FLERR, "The pair style {} is not currently supported in fix constant_pH", pstyle);
   
          
           for (int i = 0; i < ntypeAs; i++) {
               pparams[i] = new char[pair_params[pstyle].length() + 1];
               std::strcpy(pparams[i], pair_params[pstyle].c_str());
           }
       }
   
   }
   
   /* ---------------------------------------------------------------------- */
   
   void ComputeThermoInteg::compute_vector()
   {
       vector[0] = 0.0;
       vector[1] = 0.0;
       vector[2] = 0.0;
       vector[3] = 0.0;
       vector[4] = 0.0;
       vector[5] = 0.0;
   
       
       
       if (natoms != atom->natoms)
       {  
          natoms = atom->natoms;
          set_delta_qC();
       }
      
       if (parameter_list & PAIR)
       {
           /* It should be compute_du<PAIR,mode>(delta_p); But that does not work for some reasons!*/
           if (mode & SINGLE)
               vector[0] = compute_du<PAIR, SINGLE>();
           if (mode & DUAL)
               vector[0] = compute_du<PAIR, DUAL>();
       }
       if (parameter_list & CHARGE)
       {
           if (mode & SINGLE)
               vector[1] = compute_du<CHARGE, SINGLE>();
           if (mode & DUAL)
               vector[1] = compute_du<CHARGE, DUAL>();
       }
   
       if (mode & SINGLE)
           vector[2] = compute_du<PAIR|CHARGE,SINGLE>();
       if (mode & DUAL)
           vector[2] = compute_du<PAIR|CHARGE,DUAL>();
   }
   
   /* ---------------------------------------------------------------------- */
   
   template <int parameter, int mode>
   double ComputeThermoInteg::compute_du()
   {
       double uA, uB, uC, du_dl;
   
       
       /* check if there is enough allocated memory */
       if (nmax < atom->nmax)
       {
           nmax = atom->nmax;
           deallocate_storage();
           allocate_storage();
       }
       backup_restore_qfev<1>();      // backup charge, force, energy, virial array values
       
       modify_epsilon_q<parameter, mode, NEGATIVE>();      //
       update_lmp(); // update the lammps force and virial values
       if (groupbit == 1)
           uA = compute_epair(); // I guess it should be more efficient for the group all
       else 
           uA = compute_epair_atom();
       
       
       modify_epsilon_q<parameter, mode, POSITIVE>();
       update_lmp(); // update the lammps force and virial values
   
       if (groupbit == 1)
          uB = compute_epair(); // I guess it should be more efficient for the group all
       else
          uB = compute_epair_atom();
       
       backup_restore_qfev<-1>();      // restore charge, force, energy, virial array values
       restore_epsilon(); // restore epsilon values
   
       du_dl = (uB - uA) / (2*dlambda); //u(x+dx)-u(x-dx) /((x+dx)-(x-dx))
   
       if (groupbit == 1)
          uC = compute_epair();
       else
          uC = compute_epair_atom();
   
       // Just for debugging purposes
       vector[3] = uA;
       vector[4] = uB;
       vector[5] = uC;
       return du_dl;
   }
   
   
   /* --------------------------------------------------------------
   
      -------------------------------------------------------------- */
   
   template <int parameter, int mode, int direction>
   void ComputeThermoInteg::modify_epsilon_q()
   {
       int nlocal = atom->nlocal;
       int* mask = atom->mask;
       int* type = atom->type;
       int ntypes = atom->ntypes;
       double* q = atom->q;
       double _delta_qC;
   
   
   
       // taking care of cases for which epsilon or lambda become negative
       if (parameter & PAIR)
       {
           int bad_i = 0;
           int bad_j = 0;
           bool modified_delta = false;
   
           for (int k = 0; k < ntypeAs; k++)
           {
              int typeA = typeAs[k];
              double& delta_p = (direction == NEGATIVE) ? lA_ps[k] : lB_ps[k];
              for (int j = 1; j < typeA; j++)
                  if (epsilon_inits[k][j][typeA] < -delta_p)
                  {
                        if (epsilons[k][j][j] == 0) continue;
                        bad_i = typeA;
                        bad_j = j;
                        modified_delta = true;
                        delta_p = -epsilon_inits[k][j][typeA];
                  }
              for (int j = typeA; j < ntypes + 1; j++)
                  if (epsilon_inits[k][typeA][j] < -delta_p)
                  {
                        if (epsilons[k][j][j] == 0) continue;
                        bad_i = typeA;
                        bad_j = j;
                        modified_delta = true;
                        delta_p = -epsilon_inits[k][typeA][j];
                  }
              if (modified_delta) 
              {
                  error->warning(FLERR,"The delta value in compute_TI has been modified to {} since it is less than epsilon({},{})", delta_p, bad_i,bad_j);
              }
              epsilons[k][typeA][typeA] = epsilon_inits[k][typeA][typeA] + delta_p;
           }
   
           for (int k = 0; k < ntypeAs; k++)
              for (int i = 0; i < ntypes + 1 ; i++)
                  for (int j = i; j < ntypes + 1; j++)
                      if (i == typeAs[k] || j == typeAs[k])
                          epsilons[k][i][j] = sqrt(epsilons[k][i][i]*epsilons[k][j][j]);
           
           pair->reinit();
       }
      
       if (parameter & CHARGE)
       {   
           _delta_qC = (direction == NEGATIVE) ? delta_qC : -delta_qC;
   
           
           for (int i = 0; i < nlocal; i++)
           {
               for (int k = 0; k < ntypeAs; k++)
               {
                   double delta_q = (direction == NEGATIVE) ? lA_qs[k] : lB_qs[k];
                   if (type[i] == typeAs[k])
                       q[i] += delta_q;
               }
               if (mode & DUAL)
                   for (int k = 0; k < ntypeBs; k++)
                   {
                      // Needs to be implemented!!!
                   }
               if (type[i] == typeC)
                      q[i] += _delta_qC;
           }
   
   
           compute_q_total();
       }
   }
   
   /* ---------------------------------------------------------------------- */
   
   void ComputeThermoInteg::allocate_storage()
   {
       /* It should be nmax since in the case that
          the newton flag is on the force in the
          ghost atoms also must be update and the
          nmax contains the maximum number of nlocal
          and nghost atoms.
       */
       memory->create(vector_atom,nmax, "compute_TI:vector_atom");
       memory->create(f_orig, nmax, 3, "compute_TI:f_orig");
       memory->create(q_orig, nmax, "compute_TI:q_orig");
       memory->create(peatom_orig, nmax, "compute_TI:peatom_orig");
       memory->create(pvatom_orig, nmax, 6, "compute_TI:pvatom_orig");
       if (force->kspace) {
           memory->create(keatom_orig, nmax, "compute_TI:keatom_orig");
           memory->create(kvatom_orig, nmax, 6, "compute_TI:kvatom_orig");
       }
   }
   
   /* ---------------------------------------------------------------------- */
   
   void ComputeThermoInteg::deallocate_storage()
   {
       memory->destroy(vector_atom);
       memory->destroy(q_orig);
       memory->destroy(f_orig);
       memory->destroy(peatom_orig);
       memory->destroy(pvatom_orig);
       memory->destroy(keatom_orig);
       memory->destroy(kvatom_orig);
   
       f_orig = nullptr;
       peatom_orig = keatom_orig = nullptr;
       pvatom_orig = kvatom_orig = nullptr;
   }
   
   /* ----------------------------------------------------------------------
      Forward-reverse copy function to be used in backup_restore_qfev()
      ---------------------------------------------------------------------- */
   
   template  <int direction>
   void ComputeThermoInteg::forward_reverse_copy(double& a, double& b)
   {
       if (direction == 1) a = b;
       if (direction == -1) b = a;
   }
   
   template  <int direction>
   void ComputeThermoInteg::forward_reverse_copy(double* a, double* b, int m)
   {
       for (int i = 0; i < m; i++) forward_reverse_copy<direction>(a[i],b[i]);
   }
   
   template  <int direction>
   void ComputeThermoInteg::forward_reverse_copy(double** a, double** b, int m, int n)
   {
       for (int i = 0; i < m; i++) forward_reverse_copy<direction>(a[i],b[i],n);
   }
   
   /* ----------------------------------------------------------------------
      backup and restore arrays with charge, force, energy, virial
      taken from src/FEP/compute_fep.cpp
      backup ==> direction == 1
      restore ==> direction == -1
   ------------------------------------------------------------------------- */
   
   template <int direction>
   void ComputeThermoInteg::backup_restore_qfev()
   {
       int i;
   
       int nall = atom->nlocal + atom->nghost;
       int natom = atom->nlocal;
       if (force->newton || force->kspace->tip4pflag) natom += atom->nghost;
   
       double** f = atom->f;
       forward_reverse_copy<direction>(f_orig, f, natom, 3);
   
       double* q = atom->q;
       forward_reverse_copy<direction>(q_orig, q, natom);
   
       forward_reverse_copy<direction>(eng_vdwl_orig, force->pair->eng_vdwl);
       forward_reverse_copy<direction>(eng_coul_orig, force->pair->eng_coul);
   
       
       forward_reverse_copy<direction>(pvirial_orig, force->pair->virial, 6);
   
       if (update->eflag_atom) {
           double* peatom = force->pair->eatom;
           forward_reverse_copy<direction>(peatom_orig, peatom, natom);
       }
       if (update->vflag_atom) {
           double** pvatom = force->pair->vatom;
           forward_reverse_copy<direction>(pvatom_orig, pvatom, natom, 6);
       }
   
       if (force->kspace) {
           forward_reverse_copy<direction>(energy_orig, force->kspace->energy);
           forward_reverse_copy<direction>(kvirial_orig, force->kspace->virial, 6);
   
           if (update->eflag_atom) {
               double* keatom = force->kspace->eatom;
               forward_reverse_copy<direction>(keatom_orig, keatom, natom);
           }
           if (update->vflag_atom) {
               double** kvatom = force->kspace->vatom;
               forward_reverse_copy<direction>(kvatom_orig, kvatom, natom, 6);
           }
       }
   }
   
   /* --------------------------------------------------------------------- */
   
   void ComputeThermoInteg::restore_epsilon()
   {
       int ntypes = atom->ntypes;
       
       // I am not sure about the limits of these two loops, please double check them
       for (int k = 0; k < ntypeAs; k++)
           for (int i = 0; i < ntypes + 1; i++)
               for (int j = i; j < ntypes + 1; j++)
                   epsilons[k][i][j] = epsilon_inits[k][i][j];
   }
   
   /* ----------------------------------------------------------------------
      modify force and kspace in lammps according
      ---------------------------------------------------------------------- */
   
   void ComputeThermoInteg::update_lmp() {
       int eflag = ENERGY_ATOM;
       int vflag = 0;
       if (groupbit == 1) eflag = ENERGY_GLOBAL;
   
      
       timer->stamp();
       if (force->pair && force->pair->compute_flag) {
           force->pair->compute(eflag, vflag);
           timer->stamp(Timer::PAIR);
       }
       if (force->kspace && force->kspace->compute_flag) {
           force->kspace->compute(eflag, vflag);
           timer->stamp(Timer::KSPACE);
       }
   
       // accumulate force/energy/virial from /gpu pair styles
       if (fixgpu) fixgpu->post_force(vflag);
   }
   
   
   /* ---------------------------------------------------------------------
      Checking the total system charge
   
      I did not put this in the constructor in purpose since it is possible
      that the number of atoms change during the simulation. So, it is better
      to call this function whenever natoms changes 
      
      Also since this function uses MPI_Allreduce, its usage should be limited
      for the computational efficiency purposes. 
      --------------------------------------------------------------------- */
   
   void ComputeThermoInteg::set_delta_qC()
   {               
      int* selected_counts_local = new int[ntypeAs + ntypeBs + 1] {}; // Inititalize all the elements with the default constructor which is 0 here
      int* selected_counts       = new int[ntypeAs + ntypeBs + 1] {}; // Inititalize all the elements with the default constructor which is 0 here
      
      int nlocal = atom->nlocal;
      double* q = atom->q;
      int* type = atom->type;
         
         
      for (int i = 0; i < nlocal; i++)
      {
         if (type[i] == typeC)
             selected_counts_local[ntypeAs + ntypeBs]++;
         for (int k = 0; k < ntypeAs; k++)
             if (type[i] == typeAs[k])
                 selected_counts_local[k]++;
         for (int k = 0; k < ntypeBs; k++)
             if (type[i] == typeBs[k])
                 selected_counts_local[ntypeAs+k]++;
      }
   
      MPI_Allreduce(selected_counts_local, selected_counts, ntypeAs + ntypeBs + 1, MPI_INT, MPI_SUM, world);
   
      for (int k = 0; k < ntypeAs; k++)
         if (selected_counts[k] == 0) error->warning(FLERR, "Total number of atoms of {} in compute ti is zero",typeAs[k]);
      for (int k = 0; k < ntypeBs; k++)
         if (selected_counts[ntypeAs + k] == 0  && (mode & DUAL) ) error->warning(FLERR, "Total number of atoms of {} in compute ti is zero",typeBs[k]);
      if (selected_counts[ntypeAs + ntypeBs] == 0) error->all(FLERR, "Total number of atoms of type {} in compute ti is zero", typeC);
   
      delta_qC = 0.0;
      for (int k = 0; k < ntypeAs; k++)
         delta_qC -= lA_qs[k] * static_cast<double>(selected_counts[k]);
      for (int k = 0; k < ntypeBs; k++)
         delta_qC += lA_qs[k] * static_cast<double>(selected_counts[ntypeAs+k]);
      delta_qC /= static_cast<double>(selected_counts[ntypeAs+ntypeBs]);
          
      delete [] selected_counts_local;
      delete [] selected_counts;
   }
   
   /* --------------------------------------------------------------------- */
   
   void ComputeThermoInteg::compute_q_total()
   {
       double* q = atom->q;
       double nlocal = atom->nlocal;
       double q_local = 0.0;
       double q_total;
       double tolerance = 0.001;
   
       for (int i = 0; i < nlocal; i++)
           q_local += q[i];
   
       MPI_Allreduce(&q_local, &q_total, 1, MPI_DOUBLE, MPI_SUM, world);
   
       if ((q_total >= tolerance || q_total <= -tolerance) && comm->me == 0)
           error->warning(FLERR, "q_total in compute TI is non-zero: {}", q_total);
   }
   
   /* --------------------------------------------------------------------- */
   
   double ComputeThermoInteg::compute_epair()
   {
       if (update->eflag_global != update->ntimestep)
          error->all(FLERR,"Energy was not tallied on the needed timestep");
   
       bigint natoms = atom->natoms;
   
       double energy_local = 0.0;
       double energy = 0.0;
       if (force->pair) energy_local += (force->pair->eng_vdwl + force->pair->eng_coul);
   
       /* As the bond, angle, dihedral and improper energies
          do not change with the espilon, we do not need to
          include them in the energy. We are interested in
          their difference afterall */
   
       MPI_Allreduce(&energy_local, &energy, 1, MPI_DOUBLE, MPI_SUM, world);
       
       if (force->pair && force->kspace) energy += force->kspace->energy;
       energy /= static_cast<double> (natoms); // To convert to kcal/mol the total energy must be devided by the number of atoms
       return energy;
   }
   
   /* --------------------------------------------------------------------- 
      Taken from src/compute_pe_atom.cpp
      --------------------------------------------------------------------- */
      
   double ComputeThermoInteg::compute_epair_atom()
   {
      //invoked_scalar = update->ntimestep;
      //if (update->eflag_atom != invoked_scalar)
      //   error->all(FLERR,"Per-atom energy was not tallied on needed timestep");
         
      int nlocal = atom->nlocal;
      int npair = atom->nlocal;
      int ntotal = atom->nlocal;
      int nkspace = atom->nlocal;
      if (force->newton) npair += atom->nghost;
      if (force->newton) ntotal += atom->nghost;
      if (force->kspace && force->kspace->tip4pflag) nkspace += atom->nghost;
   
      int *mask = atom->mask;
      
      for (int i = 0; i < ntotal; i++) energy_peratom[i] = 0.0;
      
      if (force->pair && force->pair->compute_flag)
      {
         double *eatom = force->pair->eatom;
         for (int i = 0; i < npair; i++)
             energy_peratom[i] += eatom[i];
      } 
      if (force->kspace && force->kspace->compute_flag)
      {
         double *eatom = force->kspace->eatom;
         for (int i = 0; i < nkspace; i++)
             energy_peratom[i] += eatom[i];
      }
      
      if (force->newton || (force->kspace && force->kspace->tip4pflag)) comm->reverse_comm(this);
      
      
      double energy_local = 0.0;
      double natom_local = 0.0;
      
      for (int i = 0; i < nlocal; i++)
         if (mask[i] & groupbit)
         {
             natom_local += 1.0;
             energy_local += energy_peratom[i];
         }
       
      double *local = new double[2];
      double *total = new double[2];
      
      
      local[0] = energy_local;
      local[1] = natom_local;
      
      MPI_Allreduce(local,total,2,MPI_DOUBLE,MPI_SUM,world);
      double energy = total[0];
      double natom = total[1];
      energy = energy / natom;
      
      
      delete [] local;
      delete [] total;
   
      return energy;
   }
   
   /* ----------------------------------------------------------
      Ghost atom contributions
      from src/compute_pe_atom.cpp
      ---------------------------------------------------------- */
      
   int ComputeThermoInteg::pack_reverse_comm(int n, int first, double *buf)
   {
      int i, m, last;
      
      m = 0; 
      last = first + n;
      for (i = first; i < last; i++) buf[m++] = energy_peratom[i];
      return m;
   }
   
   /* ---------------------------------------------------------- */ 
   
   void ComputeThermoInteg::unpack_reverse_comm(int n, int *list, double * buf)
   {
      int i, j , m;
      
      m = 0;
      for (i = 0; i < n; i++) {
         j = list[i];
         energy_peratom[j] += buf[m++];
      }
   }
