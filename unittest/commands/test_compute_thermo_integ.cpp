/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
------------------------------------------------------------------------- */

#include "../testing/core.h"
#include "info.h"
#include "lammps.h"
#include "library.h"
#include "utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <mpi.h>
#include <cstring>
#include <string>
#include <sstream>

bool verbose = false;

namespace LAMMPS_NS {

#define STRINGIFY(val) XSTR(val)
#define XSTR(val) #val

// Helper that does not depend on a fixture method
static void setupBox(lammps* lmp) {
  lammps_command(lmp, "units          real");
  lammps_command(lmp, "atom_style     full");
  lammps_command(lmp, "boundary       p p p");
  lammps_command(lmp, "neigh_modify   delay 1");
  lammps_command(lmp, "neighbor       2 bin");

  lammps_command(lmp, "bond_style     harmonic");
  lammps_command(lmp, "angle_style    harmonic");
  lammps_command(lmp, "dihedral_style harmonic");
  lammps_command(lmp, "improper_style harmonic");
  lammps_command(lmp, "pair_style     lj/charmm/coul/long 10 12");
  lammps_command(lmp, "kspace_style   pppm 1.0e-6");
}

// -------------------- Fixture 1 --------------------
class PeVsUCTest : public LAMMPSTest {
protected:
  void SetUp() override {
    testbinary = "PeVsUCTest";
    LAMMPSTest::SetUp();

    if (!info->has_style("atom", "full")) GTEST_SKIP();
    if (!lammps_config_has_package("KSPACE")) GTEST_SKIP();
    if (!info->has_style("compute", "thermo_integ")) GTEST_SKIP();

    BEGIN_HIDE_OUTPUT();
    command("variable input_dir index \"" STRINGIFY(TEST_INPUT_FOLDER) "\"");
    setupBox(lmp);
    // Adjust the relative path to where you installed/copied your data file
    command("read_data ${input_dir}/data.30000000-0.46");
    command("group HAp type <= 5");
    END_HIDE_OUTPUT();
  }

  // Convenience wrappers
  static double get_scalar_compute(lammps* lmp, const char* id) {
    return *static_cast<double*>(
        lammps_extract_compute(lmp, id, LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR));
  }
  static double get_equal(lammps* lmp, const char* name) {
    return *static_cast<double*>(lammps_extract_variable(lmp, name, nullptr));
  }
};

TEST_F(PeVsUCTest, ComparePeWithUc_PerAtomAndTotals)
{
  if (lammps_get_natoms(lmp) == 0.0) GTEST_SKIP();

  BEGIN_HIDE_OUTPUT();
  // PE over ALL and HAp (pair+kspace only)
  command("compute pe_atom HAp pe/atom pair kspace");
  command("compute pe_all  all pe      pair kspace");
  command("compute pe_HAp  HAp reduce  sum c_pe_atom");

  // Counts
  command("variable cAll equal count(all)");
  command("variable cHAp equal count(HAp)");

  // Averages we want to compare
  command("variable peA equal c_pe_all/v_cAll"); // average PE over all atoms
  command("variable peB equal c_pe_HAp/v_cHAp"); // average PE over HAp

  // thermo_integ -> U_C is index [6]
  command("compute TIAll all thermo_integ 0.46 0.02 "
          "single 2 10 11 "
          "pair lj/charmm/coul/long 0.0800014955 0.0400000000 0.0 0.0 "
          "charge -1.1 -1 0.2 0 3");
  command("compute TIHAp HAp thermo_integ 0.46 0.02 "
          "single 2 10 11 "
          "pair lj/charmm/coul/long 0.0800014955 0.0400000000 0.0 0.0 "
          "charge -1.1 -1 0.2 0 3");

  command("variable uC_All equal c_TIAll[6]");
  command("variable uC_HAp equal c_TIHAp[6]");

  command("thermo_style custom v_peA v_peB v_uC_All v_uC_HAp");
  command("run 0 post no");
  END_HIDE_OUTPUT();

  const double peA    = get_equal(lmp, "peA");
  const double peB    = get_equal(lmp, "peB");
  const double uC_All = get_equal(lmp, "uC_All");
  const double uC_HAp = get_equal(lmp, "uC_HAp");

  // Compare averages directly (they’re all global scalars here)
  const double tol = 1e-6;
  EXPECT_NEAR(peA, uC_All, tol);
  EXPECT_NEAR(peB, uC_HAp, tol);
}

// -------------------- Fixture 2 --------------------
class LJChangeTest : public LAMMPSTest {
protected:
  void SetUp() override {
    testbinary = "LJChangeTest";
    LAMMPSTest::SetUp();


    if (!info->has_style("atom", "full")) GTEST_SKIP();
    if (!lammps_config_has_package("KSPACE")) GTEST_SKIP();
    if (!info->has_style("compute", "thermo_integ")) GTEST_SKIP();

    // No data yet; each phase below sets its own state
  }

  static double get_equal(lammps* lmp, const char* name) {
    return *static_cast<double*>(lammps_extract_variable(lmp, name, nullptr));
  }
  static double get_compute_scalar(lammps* lmp, const char* id) {
    return *static_cast<double*>(
        lammps_extract_compute(lmp, id, LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR));
  }
};

TEST_F(LJChangeTest, UA_UB_Equals_PE_From_Separate_Inputs)
{
  // ---- Phase 1: compute UA/UB on your base data ----
  BEGIN_HIDE_OUTPUT();
  command("clear");
  command("variable input_dir index \"" STRINGIFY(TEST_INPUT_FOLDER) "\"");
  setupBox(lmp);
  command("read_data ${input_dir}/data.30000000-0.46");
  command("group HAp type <= 5");

  const double lambda  = 0.46;
  const double dlambda = 0.02;
  const double LJ0 = 0.0800014955;
  const double LJ1 = 0.0400000000;

  std::ostringstream c1, c2;
  c1.setf(std::ios::fixed); c1.precision(4);
  c2.setf(std::ios::fixed); c2.precision(4);

  c1 << "compute TIAll all thermo_integ " << lambda << " " << dlambda
     << " single 2 10 11 "
     << "pair lj/charmm/coul/long " << std::setprecision(10) << LJ0 << " " << LJ1 << " 0.0 0.0 "
     << "charge -1.1 -1 0.2 0 3";

  c2 << "compute TIHAp HAp thermo_integ " << std::setprecision(4) << lambda << " " << dlambda
     << " single 2 10 11 "
     << "pair lj/charmm/coul/long " << std::setprecision(10) << LJ0 << " " << LJ1 << " 0.0 0.0 "
     << "charge -1.1 -1 0.2 0 3";

  command(c1.str().c_str());
  command(c2.str().c_str());

  command("variable UA_All equal c_TIAll[4]");
  command("variable UA_HAp equal c_TIHAp[4]");
  command("variable UB_All equal c_TIAll[5]");
  command("variable UB_HAp equal c_TIHAp[5]");

  command("thermo_style custom v_UA_All v_UA_HAp v_UB_All v_UB_HAp");
  command("run 0 post no");
  END_HIDE_OUTPUT();

  const double UA_All = get_equal(lmp, "UA_All");
  const double UA_HAp = get_equal(lmp, "UA_HAp");
  const double UB_All = get_equal(lmp, "UB_All");
  const double UB_HAp = get_equal(lmp, "UB_HAp");

  // ---- Phase 2: load reference A-state input, measure PE_A ----
  BEGIN_HIDE_OUTPUT();
  command("clear");
  command("variable input_dir index \"" STRINGIFY(TEST_INPUT_FOLDER) "\"");
  // NOTE: this file must be an **input script** that sets units/styles and read_data
  command("read_data ${input_dir}/data.30000000-0.46-LJA-0.02");
  command("group HAp type <= 5");
  command("compute pe_atom HAp pe/atom pair kspace");
  command("compute pe_all  all pe      pair kspace");
  command("compute pe_HAp  HAp reduce  sum c_pe_atom");
  command("variable cAll equal count(all)");
  command("variable cHAp equal count(HAp)");
  command("variable RefA_All equal c_pe_all/v_cAll");
  command("variable RefA_HAp equal c_pe_HAp/v_cHAp");
  command("thermo_style custom v_RefA_All v_RefA_HAp");
  command("run 0 post no");
  END_HIDE_OUTPUT();

  const double RefA_All = get_equal(lmp, "RefA_All");
  const double RefA_HAp = get_equal(lmp, "RefA_HAp");

  // ---- Phase 3: load reference B-state input, measure PE_B ----
  BEGIN_HIDE_OUTPUT();
  command("clear");
  command("variable input_dir index \"" STRINGIFY(TEST_INPUT_FOLDER) "\"");
  command("read_data ${input_dir}/data.30000000-0.46-LJA-0.02");
  command("group HAp type <= 5");
  command("compute pe_atom HAp pe/atom pair kspace");
  command("compute pe_all  all pe      pair kspace");
  command("compute pe_HAp  HAp reduce  sum c_pe_atom");
  command("variable cAll equal count(all)");
  command("variable cHAp equal count(HAp)");
  command("variable RefB_All equal c_pe_all/v_cAll");
  command("variable RefB_HAp equal c_pe_HAp/v_cHAp");
  command("thermo_style custom v_RefB_All v_RefB_HAp");
  command("run 0 post no");
  END_HIDE_OUTPUT();

  const double RefB_All = get_equal(lmp, "RefB_All");
  const double RefB_HAp = get_equal(lmp, "RefB_HAp");

  // ---- Compare ----
  const double tol = 1e-3;
  EXPECT_NEAR(UA_All, RefA_All, tol);
  EXPECT_NEAR(UA_HAp, RefA_HAp, tol);
  EXPECT_NEAR(UB_All, RefB_All, tol);
  EXPECT_NEAR(UB_HAp, RefB_HAp, tol);
}

} // namespace LAMMPS_NS

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleMock(&argc, argv);

  if ((argc > 1) && (strcmp(argv[1], "-v") == 0)) verbose = true;
  if (const char* env = getenv("TEST_ARGS")) {
    for (auto& tok : LAMMPS_NS::utils::split_words(env))
      if (tok == "-v") verbose = true;
  }

  const int rv = RUN_ALL_TESTS();
  MPI_Finalize();
  return rv;
}
