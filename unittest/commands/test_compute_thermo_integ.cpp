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

// Test fixture matching the LAMMPS unittest harness
class PeVsUcTest : public LAMMPSTest {
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

        // --- match your production script BEFORE read_data ---
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

TEST_F(PeVsUcTest, ComparePeWithUc_PerAtomAndTotals)
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
