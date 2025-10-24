/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
------------------------------------------------------------------------- */

#include "../testing/core.h"
#include "info.h"
#include "lammps.h"
#include "library.h"
#include "utils.h"

#include "compute.h"
#include "compute_diff_atom.h"
#include "style_compute.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <mpi.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sstream>

bool verbose = false;

namespace LAMMPS_NS {

#define STRINGIFY(val) XSTR(val)
#define XSTR(val) #val

enum MockType {MOCK_A,MOCK_B,MOCK_C,MOCK_D};

template<int mockType>
class ComputeMockDiffAtom : public ComputeDiffAtom 
{
public:
    ComputeMockDiffAtom(LAMMPS* lmp, int narg, char **argv):
        ComputeDiffAtom{lmp,narg,argv} {}
    ~ComputeMockDiffAtom() override = default;

    void init() override {
        // Ensure the per-atom array matches what FixMetaAR expects
        val_col        = 0;
        diff_x_col     = 1;
        diff_y_col     = 2;
        diff_z_col     = 3;
        size_peratom_cols = 4;            
        scalar_flag    = 1;
        peratom_flag   = 1;
        extscalar      = 0;

        nmax = atom->nmax;
        memory->create(array_atom, nmax, size_peratom_cols,
                   "compute_mock_diff_atom:array_atom");
    }

    void compute_all() override {
        nmax = atom->nmax;
        memory->grow(array_atom,nmax,4,"compute_mock_diff_atom:array_atom");
        int nlocal = atom->nlocal;
        double ** x = atom->x;
        scalar = 0.0;

        for (int i = 0; i < nlocal; i++)
        {
            switch (mockType)
            {
                case MOCK_A:
                {
                    array_atom[i][0] = x[i][0] + x[i][1] + x[i][2];
                    array_atom[i][1] = 1.0;
                    array_atom[i][2] = 1.0;
                    array_atom[i][3] = 1.0;
                    break;
                }
                case MOCK_B:
                {
                    array_atom[i][0] = x[i][0]*x[i][0] + x[i][1]*x[i][1] + x[i][2]*x[i][2];
                    array_atom[i][1] = 2.0*x[i][0];
                    array_atom[i][2] = 2.0*x[i][1];
                    array_atom[i][3] = 2.0*x[i][2];
                    break;
                }
                case MOCK_C:
                {
                    array_atom[i][0] = x[i][0]*x[i][1];
                    array_atom[i][1] = x[i][1];
                    array_atom[i][2] = x[i][0];
                    array_atom[i][3] = 0.0;
                    break;
                }
                case MOCK_D:
                {
                    array_atom[i][0] = x[i][0]*x[i][0]*x[i][0];
                    array_atom[i][1] = 3*x[i][0]*x[i][0];
                    array_atom[i][2] = 0.0;
                    array_atom[i][3] = 0.0;
                    break;
                }
                default:
                    throw std::invalid_argument("Wrong mock type!");
            }
            
            scalar += array_atom[i][0];
        }
    }

};

// Register as LAMMPS styles
ComputeStyle(mock/diff/a, ComputeMockDiffAtom<MOCK_A>);
ComputeStyle(mock/diff/b, ComputeMockDiffAtom<MOCK_B>);
ComputeStyle(mock/diff/c, ComputeMockDiffAtom<MOCK_C>);
ComputeStyle(mock/diff/d, ComputeMockDiffAtom<MOCK_D>);

// ------------------------ Helpers -------------------------------------

static inline double get_equal(LAMMPS* lmp, const char *name)
{
    return *static_cast<double*>(lammps_extract_variable(lmp,name,nullptr));
}

// Helper that does not depend on a fixture method
static void setupBox(LAMMPS* lmp) {
    lammps_command(lmp, "units          lj");
    lammps_command(lmp, "atom_style     atomic");
    lammps_command(lmp, "boundary       p p p");
    lammps_command(lmp, "region         box block 0 2 0 2 0 2");
    lammps_command(lmp, "create_box     1 box");
    lammps_command(lmp, "create_atoms   1 single 0.1 0.2 0.3");
    lammps_command(lmp, "mass           1 1.0");
    lammps_command(lmp, "pair_style     none");
    lammps_command(lmp, "timestep       1.0");
    lammps_command(lmp, "neighbor       0.1 bin");
    lammps_command(lmp, "neigh_modify   every 1 delay 0 check yes");
}

// Runs: run 1 → displace → run 10; returns current bias scalar variable.
static double run_two_step_bias(LAMMPS *lmp,
                                double omega,
                                double sigma1,
                                double sigma2,
                                double deltaT,
                                double DCV1, double DCV2,
                                int bins1, int bins2,
                                std::string cv1_style,
                                std::string cv2_style,
                                std::string fix_id,
                                double dx, double dy, double dz,
                                bool cleanup = true)
{
    // define CVs
    std::string cmd;
    cmd = std::string("compute c_cv1 all ") + cv1_style;
    lammps_command(lmp,cmd.c_str());
    cmd = std::string("compute c_cv2 all ") + cv2_style;
    lammps_command(lmp,cmd.c_str());

    // define the fix_metaAR
    std::stringstream oss;
    oss << "fix " << fix_id << " all metaAR 1 c_cv1 c_cv2 "
        << omega << " " << sigma1 << " " << sigma2 << " " << deltaT << " "
        << DCV1 << " " << DCV2 << " " << bins1 << " " << bins2 
        << " file metaAR.csv 1";
    cmd = oss.str();
    lammps_command(lmp,cmd.c_str());

    // metaAR output
    cmd = std::string("variable bias equal f_") + fix_id;
    lammps_command(lmp,cmd.c_str());
    lammps_command(lmp,"thermo_style custom step v_bias");

    // run 
    lammps_command(lmp,"run 1 post no");

    // displacing atoms
    cmd = std::string("displace_atoms all move ") + 
          std::to_string(dx) + std::string(" ") +
          std::to_string(dy) + std::string(" ") +
          std::to_string(dz) + std::string(" units box");
    lammps_command(lmp,cmd.c_str());
    
    // running for 10 steps
    lammps_command(lmp,"run 10 post no");

    // collecting the bias
    const double bias = get_equal(lmp,"bias");

    if (cleanup) {
        // cleaning up
        cmd = std::string("unfix ") + fix_id;
        lammps_command(lmp,cmd.c_str());
        lammps_command(lmp,"uncompute cv1");
        lammps_command(lmp,"uncompute cv2");
    }

    // retuning the bias
    return bias;
}

static constexpr double etol = 1e-12;

struct CVPair {
    std::string cv1,
    std::string cv2
} static const CVPair settings[] = {
    {"mock/diff/a","mock/diff/a"},
    {"mock/diff/b","mock/diff/b"},
    {"mock/diff/c","mock/diff/c"},
    {"mock/diff/d","mock/diff/d"},
    {"mock/diff/a","mock/diff/b"},
    {"mock/diff/a","mock/diff/c"},
    {"mock/diff/a","mock/diff/d"},
    {"mock/diff/b","mock/diff/c"},
    {"mock/diff/b","mock/diff/d"},
    {"mock/diff/c","mock/diff/d"}
};


// -------------------- Fixture  --------------------
class FixMetaARTest : public LAMMPSTest {
protected:
  void SetUp() override {
    testbinary = "FixMetaARTest";
    LAMMPSTest::SetUp();
    BEGIN_HIDE_OUTPUT();
    setupBox(lmp);
    END_HIDE_OUTPUT();
  }
};

// ------------------------- Tests ---------------------------

TEST_F(FixMetaARTest, BiasIsFiniteAndNonNegative)
{
    if (lammps_get_natoms(lmp) == 0) GTEST_SKIP();

    for (const auto& setting : settings)
    {
        BEGIN_HIDE_OUTPUT();
        const double bias = run_two_step_bias(
            lmp,
            /*omega*/ 1.0,
            /*sigma1*/ 0.5, /*sigma2*/ 0.5,
            /*deltaT*/ 1000.0,
            /*DCV1*/ 1.0, /*DCV2*/ 1.0,
            /*bins1*/ 8, /*bins2*/ 8,
            /*cv1_style*/ setting.cv1,
            /*cv2_style*/ setting.cv2,
            /*fixid*/ std::string("meta"),
            /*dx,dy,dz*/ 0.05, 0.0, 0.0);
        END_HIDE_OUTPUT();
        ASSERT_TRUE(std::isfinite(bias));
        EXPECT_GE(bias,0.0);
    }   
}

TEST_F(FixMetaARTest,BiasScalesLinearlyWithOmega)
{
    if (lammps_get_natoms(lmp) == 0) GTEST_SKIP();

    for (const auto& setting: settings)
    {
        BEGIN_HIDE_OUTPUT();
        const double b1 = run_two_step_bias(
            lmp,
            /*omega*/ 1.0,
            /*sigma1*/ 0.5, /*sigma2*/ 0.5,
            /*deltaT*/ 1000.0,
            /*DCV1*/ 1.0, /*DCV2*/ 1.0,
            /*bins1*/ 8, /*bins2*/ 8,
            /*cv1_style*/ setting.cv1,
            /*cv2_style*/ setting.cv2,
            /*fixid*/ std::string("meta1"),
            /*dx,dy,dz*/ 0.05, 0.0, 0.0);
        const double b2 = run_two_step_bias(
            lmp,
            /*omega*/ 2.0,
            /*sigma1*/ 0.5, /*sigma2*/ 0.5,
            /*deltaT*/ 1000.0,
            /*DCV1*/ 1.0, /*DCV2*/ 1.0,
            /*bins1*/ 8, /*bins2*/ 8,
            /*cv1_style*/ setting.cv1,
            /*cv2_style*/ setting.cv2,
            /*fixid*/ std::string("meta2"),
            /*dx,dy,dz*/ 0.05, 0.0, 0.0);
        END_HIDE_OUTPUT();
        ASSERT_TRUE(std::isfinite(b1));
        ASSERT_TRUE(std::isfinite(b2));
        EXPECT_NEAR(b2,2.0*b1,etol);
    }


}

TEST_F(FixMetaARTest,WiderSigmaIncreaseBias)
{
    if (lammps_get_natoms(lmp) == 0.0) GTEST_SKIP();

    for (const auto& setting: settings)
    {
        BEGIN_HIDE_OUTPUT();
        const double b_narrow = run_two_step_bias(
            lmp,
            /*omega*/ 1.0,
            /*sigma1*/ 0.3, /*sigma2*/ 0.3,
            /*deltaT*/ 1000.0,
            /*DCV1*/ 1.0, /*DCV2*/ 1.0,
            /*bins1*/ 8, /*bins2*/ 8,
            /*cv1_style*/ setting.cv1,
            /*cv2_style*/ setting.cv2,
            /*fixid*/ std::string("meta1"),
            /*dx,dy,dz*/ 0.05, 0.0, 0.0);
        const double b_wide = run_two_step_bias(
            lmp,
            /*omega*/ 1.0,
            /*sigma1*/ 5.0, /*sigma2*/ 5.0,
            /*deltaT*/ 1000.0,
            /*DCV1*/ 1.0, /*DCV2*/ 1.0,
            /*bins1*/ 8, /*bins2*/ 8,
            /*cv1_style*/ setting.cv1,
            /*cv2_style*/ setting.cv2,
            /*fixid*/ std::string("meta2"),
            /*dx,dy,dz*/ 0.05, 0.0, 0.0);

        END_HIDE_OUTPUT();
        ASSERT_TRUE(std::isfinite(b_narrow));
        ASSERT_TRUE(std::isfinite(b_wide));
        EXPECT_GE(b_narrow,b_wide);
    }

}

TEST_F(FixMetaARTest,ForceComponentsEqualForMockA)
{
    if (lammps_get_natoms(lmp) == 0) GTEST_SKIP();

    BEGIN_HIDE_OUTPUT();

    const double bias = run_two_step_bias(
        lmp, 1.0, 0.5, 0.5, 1000.0, 1.0, 1.0, 8, 8,
        "mock/diff/a", "mock/diff/a", "fmetaF",
        0.05, 0.05, 0.05, false);
    
    lammps_command(lmp,"variable fx equal f_fmetaF[1]");
    lammps_command(lmp,"variable fy equal f_fmetaF[2]");
    lammps_command(lmp,"variable fz equal f_fmetaF[3]");
    lammps_command(lmp,"run 0 post no");

    auto fx = get_equal(lmp,"fx");
    auto fy = get_equal(lmp,"fy");
    auto fz = get_equal(lmp,"fz");

    BEGIN_HIDE_OUTPUT();
    lammps_command(lmp,"unfix fmetaF");
    lammps_command(lmp,"uncompute cv1");
    lammps_command(lmp,"uncompute cv2");
    END_HIDE_OUTPUT();

    ASSERT_TRUE(std::isfinite(bias));
    ASSERT_TRUE(std::isfinite(fx));
    ASSERT_TRUE(std::isfinite(fy));
    ASSERT_TRUE(std::isfinite(fz));
    EXPECT_NEAR(fx,fy,etol);
    EXPECT_NEAR(fy,fz,etol);
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