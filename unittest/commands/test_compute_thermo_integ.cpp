/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
------------------------------------------------------------------------- */

#include "../testing/core.h"
#include "info.h"
#include "input.h"
#include "lammps.h"
#include "library.h"
#include "utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <mpi.h>
#include <cstring>
#include <string>

bool verbose = false;

namespace LAMMPS_NS {

#define STRINGIFY(val) XSTR(val)
#define XSTR(val) #val

static void setupBox() {
    command("units          real");
    command("atom_style     full");
    command("boundary       p p p");
    command("neigh_modify   delay 1");
    command("neighbor       2 bin");

    command("bond_style     harmonic");
    command("angle_style    harmonic");
    command("dihedral_style harmonic");
    command("improper_style harmonic");
    command("pair_style     lj/charmm/coul/long 10 12");
    command("kspace_style   pppm 1.0e-6");
}

// Test fixture matching the LAMMPS unittest harness
class PeVsUCTest : public LAMMPSTest {
protected:
    void SetUp() override
    {
        testbinary = "PeVsUCTest";
        LAMMPSTest::SetUp();

        // Require atom_style full (adjust if your data uses another style)
        if (!info->has_style("atom", "full")) GTEST_SKIP();

        BEGIN_HIDE_OUTPUT();
        // TEST_INPUT_FOLDER is provided by the LAMMPS test harness
        command("variable input_dir index \"" STRINGIFY(TEST_INPUT_FOLDER) "\"");

        setupBox();

        command("read_data ${input_dir}/ti_uc/data.dat");
)
        command("group HAp type <= 5");
        END_HIDE_OUTPUT();
    }

    // Convenience wrappers
    static double get_scalar_compute(lammps *lmp, const char *id) {
        return *static_cast<double *>(
            lammps_extract_compute(lmp, id, LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR));
    }
    static double get_var_equal(lammps *lmp, const char *name) {
        // equal-style variables return a pointer to a double
        return *static_cast<double *>(lammps_extract_variable(lmp, name, nullptr));
    }
};

class LJChangeTest : public LAMMPSTest {
protected:
    void SetUp() override {
        testbinary = "LJChangeTest";
        LAMMPSTest::SetUp();
   
        // Skip gracefully if required pieces aren't built
        if (!info->has_style("compute", "thermo_integ")) GTEST_SKIP();

        setupBox();
        
    }
    
    static double get_equal(lammps* lmp, const char* name) {
        return *static_cast<double*>(lammps_extract_variable(lmp, name, nullptr));
    }

    static double get_compute_scalar(lammps* lmp, const char* id) {
        return *static_cast<double*>(
            lammps_extract_compute(lmp, id, LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR));
    }
};



TEST_F(PeVsUCTest, ComparePeWithUc_PerAtomAndTotals)
{
    // Skip when there are no atoms (e.g., if in.fourmol wasn't included)
    if (lammps_get_natoms(lmp) == 0.0) GTEST_SKIP();

    // Only run if your compute style is compiled in
    if (!info->has_style("compute", "thermo_integ")) GTEST_SKIP();

    const int has_kspace = lammps_config_has_package("KSPACE");

    BEGIN_HIDE_OUTPUT();


    // --- Potential energy over ALL and over HAp (pair + kspace only)
    command("compute pe_atom HAp pe/atom pair kspace");
    command("compute pe_all  all pe      pair kspace");
    command("compute pe_HAp  HAp reduce  sum c_pe_atom");

    // Counts
    command("variable cAll equal count(all)");
    command("variable cHAp equal count(HAp)");

    // Averages we want to compare
    command("variable peA equal c_pe_all/v_cAll"); // average PE over all atoms
    command("variable peB equal c_pe_HAp/v_cHAp"); // average PE over HAp

    // --- Your compute that provides U_C (index [6] below):
    //     (Keep the arguments identical to your run; adjust if needed.)
    command("compute TIAll all thermo_integ 0.46 0.02 "
            "single 2 10 11 "
            "pair lj/charmm/coul/long 0.0800014955 0.0400000000 0.0 0.0 "
            "charge -1.1 -1 0.2 0 3");
    command("compute TIHAp HAp thermo_integ 0.46 0.02 "
            "single 2 10 11 "
            "pair lj/charmm/coul/long 0.0800014955 0.0400000000 0.0 0.0 "
            "charge -1.1 -1 0.2 0 3");

    // Expose U_C (total) and per-atom versions via variables
    command("variable uC_All equal c_TIAll[6]");
    command("variable uC_HAp equal c_TIHAp[6]");

    // Make sure everything is evaluated deterministically
    command("thermo_style custom v_peA v_peB v_uC_All v_uC_HAp v_uC_All_pa v_uC_HAp_pa");
    command("run 0 post no");
    END_HIDE_OUTPUT();

    // Grab equal-style variables
    const double peA        = get_var_equal(lmp, "peA");
    const double peB        = get_var_equal(lmp, "peB");
    const double uC_All     = get_var_equal(lmp, "uC_All");
    const double uC_HAp     = get_var_equal(lmp, "uC_HAp");

    // Also pull the raw totals from the computes for completeness
    const double peAll_tot = get_scalar_compute(lmp, "pe_all");
    const double peHAp_tot = get_scalar_compute(lmp, "pe_HAp");

    // Tolerances: PPPM / compiler / math-lib noise → be modest
    const double tol_pa  = 1e-6;  // per-atom comparison
    // Per-atom consistency: <pe> should match U_C
    EXPECT_NEAR(peA, uC_All, tol_pa);
    EXPECT_NEAR(peB, uC_HAp, tol_pa);

}


TEST_F(LJChangeTest, UA_UB_Equals_PE_From_Separate_Input_ChangeLJ)
{
    BEGIN_HIDE_OUTPUT();

    // your test data
    command("read_data ${input_dir}/ti_uc/data.dat");

    // thermo_integ: U_A is index [4]  and U_B is index [5]
    command("compute TIAll all thermo_integ 0.46 0.02 "
            "single 2 10 11 "
            "pair lj/charmm/coul/long 0.0800014955 0.0400000000 0.0 0.0 ");
    command("compute TIHAp HAp thermo_integ 0.46 0.02 "
            "single 2 10 11 "
            "pair lj/charmm/coul/long 0.0800014955 0.0400000000 0.0 0.0 ");
    command("variable UA_All equal c_TIAll[4]");
    command("variable UA_HAp equal c_TIHAp[4]");
    command("variable UB_HAp equal c_TIHAp[5]");
    command("variable UB_HAp equal c_TIHAp[5]");

    command("thermo_style custom v_UA_All v_UA_HAp v_UB_All v_UB_HAp");
    command("run 0 post no");
    END_HIDE_OUTPUT();

    const double UA_All = get_equal(lmp, "UA_All");
    const double UA_HAp = get_equal(lmp, "UA_HAp");
    const double UB_All = get_equal(lmp, "UB_All");
    const double UB_HAp = get_equal(lmp, "UB_HAp");

    // -------- 2) Second simulation: UA from another input --------------
    BEGIN_HIDE_OUTPUT();
    command("clear");
    command("variable input_dir index \"" STRINGIFY(TEST_INPUT_FOLDER) "\"");

    // include your separate reference input (sets units/styles/read_data)
    // file path: unittest/inputs/ti_uc/pe_ref.in
    command("include \"${input_dir}/ti_uc/pe_ref.in\"");


    // --- Potential energy over ALL and over HAp (pair + kspace only)
    command("compute pe_atom HAp pe/atom pair kspace");
    command("compute pe_all  all pe      pair kspace");
    command("compute pe_HAp  HAp reduce  sum c_pe_atom");
    
    // Counts
    command("variable cAll equal count(all)");
    command("variable cHAp equal count(HAp)");
    
    // Averages we want to compare
    command("variable RefA_All equal c_pe_all/v_cAll"); // average PE over all atoms
    command("variable RefA_HAp equal c_pe_HAp/v_cHAp"); // average PE over HAp

    command("thermo_style custom v_RefA_All v_RefA_HAp");
    command("run 0 post no");

    END_HIDE_OUTPUT();

    const double RefA_All = get_equal(lmp,"RefA_All");
    const double RefA_HAp = get_equal(lmp,"RefA_HAp");


    // -------- 2) Second simulation: UA from another input --------------
    BEGIN_HIDE_OUTPUT();
    command("clear");
    command("variable input_dir index \"" STRINGIFY(TEST_INPUT_FOLDER) "\"");
    
    // include your separate reference input (sets units/styles/read_data)
    // file path: unittest/inputs/ti_uc/pe_ref.in
    command("include \"${input_dir}/ti_uc/pe_ref.in\"");
    
    
    // --- Potential energy over ALL and over HAp (pair + kspace only)
    command("compute pe_atom HAp pe/atom pair kspace");
    command("compute pe_all  all pe      pair kspace");
    command("compute pe_HAp  HAp reduce  sum c_pe_atom");
        
    // Counts
    command("variable cAll equal count(all)");
    command("variable cHAp equal count(HAp)");
        
    // Averages we want to compare
    command("variable RefB_All equal c_pe_all/v_cAll"); // average PE over all atoms
    command("variable RefB_HAp equal c_pe_HAp/v_cHAp"); // average PE over HAp
    
    command("thermo_style custom v_RefB_All v_RefB_HAp");
    command("run 0 post no");
    
    END_HIDE_OUTPUT();
    
    const double RefB_All = get_equal(lmp,"RefA_All");
    const double RefB_HAp = get_equal(lmp,"RefB_HAp");

    // -------- 3) Compare ------------------------------------------------
    const double tol_tot = 1e-3;
    EXPECT_NEAR(UA_All, RefA_All, tol_tot);
    EXPECT_NEAR(UA_HAp, RefA_HAp, tol_tot);
    EXPECT_NEAR(UB_All, RefB_All, tol_tot);
    EXPECT_NEAR(UB_HAp, RefB_HAp, tol_tot);
}

} // namespace LAMMPS_NS

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleMock(&argc, argv);

    if ((argc > 1) && (strcmp(argv[1], "-v") == 0)) verbose = true;
    if (const char *env = getenv("TEST_ARGS")) {
        for (auto &tok : LAMMPS_NS::utils::split_words(env))
            if (tok == "-v") verbose = true;
    }

    const int rv = RUN_ALL_TESTS();
    MPI_Finalize();
    return rv;
}
