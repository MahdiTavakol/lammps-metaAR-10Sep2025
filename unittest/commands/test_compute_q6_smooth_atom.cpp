#include "../testing/core.h"
#include "atom.h"
#include "input.h"
#include "lammps.h"
#include "library.h"
#include "utils.h"
#include "gtest/gtest.h"

using namespace LAMMPS_NS;

namespace {

struct Move { int atomid; double dx; };
// Adjust if your compute writes gradients to different columns:
enum { DIFF_X_COL = 1, DIFF_Y_COL = 2, DIFF_Z_COL = 3 };

class ComputeQ6SmoothAtomDiffTest : public LAMMPSTest {
protected:
  void SetUp() override {
    testbinary = "ComputeQ6SmoothAtomDiffTest";
    LAMMPSTest::SetUp();
    if (info->has_style("atom", "full")) {
      BEGIN_HIDE_OUTPUT();
      command("variable input_dir index \"" STRINGIFY(TEST_INPUT_FOLDER) "\"");
      command("include \"${input_dir}/in.fourmol\"");
      command("atom_modify map array");               // fast tag→local lookup
      command("group allwater molecule 3:6");
      command("region half block 0.0 INF INF INF INF INF");
      END_HIDE_OUTPUT();
    }
  }

  double get_scalar(const char *id) {
    return *(double*) lammps_extract_compute(lmp, id, LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR);
  }
  double** get_atom_array(const char *id) {
    return (double**) lammps_extract_compute(lmp, id, LMP_STYLE_ATOM, LMP_TYPE_ARRAY);
  }

  int find_local_index_by_tag(LAMMPS *lmp, tagint aid, bool include_ghost=false) {
    Atom *atom = lmp->atom;
    if (atom->map_style) {
      int ilocal = atom->map(aid);
      if (ilocal >= 0) {
        if (!include_ghost && ilocal >= atom->nlocal) return -1;
        return ilocal;
      }
    }
    int nall = include_ghost ? atom->nlocal + atom->nghost : atom->nlocal;
    for (int i = 0; i < nall; ++i) if (atom->tag[i] == aid) return i;
    return -1;
  }
};

TEST_F(ComputeQ6SmoothAtomDiffTest, CentralDiffMatchesAnalytic) {
  if (lammps_get_natoms(lmp) == 0.0) GTEST_SKIP();

  BEGIN_HIDE_OUTPUT();
  command("compute cv1 q6-smooth/atom 2 4.2 phi S1 3.0 0.3 min_slope 0.0");
  command("compute cv2 q6-smooth/atom 3 6.3 phi S1 3.0 0.3 min_slope 0.0");
  command("run 0 post no");
  END_HIDE_OUTPUT();

  Atom* atom = lmp->atom;
  const char* cidA = "cv1";
  const char* cidB = "cv2";

  // Replace with IDs you know reside on rank 0 for single-rank testing
  const int atomidsA[] = {19772,16691,10873,21459};
  const int atomidsB[] = {11650,18480,12787,21905};
  const double eps_list[] = {1e-4, 3e-4};  // small enough to avoid neighbor flips

  // Templated lambda to accept C arrays by reference (keeps size)
  auto test_lambda = [&](auto const& atomids, const char* cid) {
    double** arr = get_atom_array(cid);
    for (int aid : atomids) {
      int i = find_local_index_by_tag(lmp, aid, /*include_ghost=*/false);
      if (i < 0) continue;  // not on this rank

      const int cols[3] = {DIFF_X_COL, DIFF_Y_COL, DIFF_Z_COL};
      for (int dim = 0; dim < 3; ++dim) {
        double g_ana = arr[i][cols[dim]];

        for (double eps : eps_list) {
          // backup
          double x0 = atom->x[i][dim];

          // −eps
          atom->x[i][dim] = x0 - eps;
          BEGIN_HIDE_OUTPUT(); command("run 0 post no"); END_HIDE_OUTPUT();
          double phim = get_scalar(cid);

          // +eps (from original, not from −eps)
          atom->x[i][dim] = x0 + eps;
          BEGIN_HIDE_OUTPUT(); command("run 0 post no"); END_HIDE_OUTPUT();
          double phip = get_scalar(cid);

          // restore
          atom->x[i][dim] = x0;
          BEGIN_HIDE_OUTPUT(); command("run 0 post no"); END_HIDE_OUTPUT();

          double g_num = (phip - phim) / (2.0*eps);
          EXPECT_NEAR(g_num, g_ana, 1e-3)
            << "cid="<<cid<<" aid="<<aid<<" dim="<<dim<<" eps="<<eps
            << " num="<<g_num<<" ana="<<g_ana;
        }
      }
    }
  };

  test_lambda(atomidsA, cidA);
  test_lambda(atomidsB, cidB);
}

} // namespace
