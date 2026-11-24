#include "eigen.h"
#include <stdio.h>
#include <math.h>

static int fails = 0;

#define CHECK(c, msg) do { \
    if (c) \
        printf("PASS: %s\n", msg); \
    else { \
        printf("FAIL: %s\n", msg); \
        fails++; \
    } \
} while (0)

int main(void) {
    /* T1: symmetric tridiagonal 3x3:
       eigenvalues: 2-sqrt(2), 2, 2+sqrt(2) */
    Matrix *a = mat_create(3, 3);
    double d1[] = {
        2, -1, 0,
        -1, 2, -1,
        0, -1, 2
    };

    for (int i = 0; i < 9; i++)
        a->data[i] = d1[i];

    double v[3];
    Matrix *V = mat_create(3, 3);

    CHECK(eig_jacobi(a, 1e-13, 100, v, V) == EIG_OK,
          "T1 jacobi runs");

    CHECK(
        fabs(v[0] - (2 - sqrt(2))) < 1e-10 &&
        fabs(v[1] - 2) < 1e-10 &&
        fabs(v[2] - (2 + sqrt(2))) < 1e-10,
        "T1 eigenvalues");

    int good = 1;

    for (int k = 0; k < 3; k++) {
        double c[3] = {
            MAT(V, 0, k),
            MAT(V, 1, k),
            MAT(V, 2, k)
        };

        if (mat_residual(a, v[k], c) > 1e-10)
            good = 0;
    }

    CHECK(good, "T1 residuals < 1e-10");

    /* T2: general 2x2 [[4,1],[2,3]], eigenvalues 2 and 5 */
    Matrix *b = mat_create(2, 2);
    double d2[] = {4, 1, 2, 3};

    for (int i = 0; i < 4; i++)
        b->data[i] = d2[i];

    double w[2];

    CHECK(
        eig_qr(b, 1e-12, 10000, w) == EIG_OK &&
        fabs(w[0] - 2) < 1e-9 &&
        fabs(w[1] - 5) < 1e-9,
        "T2 QR eigenvalues 2 and 5");

    double lam;
    double pv[2];

    CHECK(
        eig_power(b, 1e-10, 10000, &lam, pv) == EIG_OK &&
        fabs(lam - 5) < 1e-8,
        "T2 power method lambda=5");

    Matrix *W = mat_create(2, 2);
    double ww[2];

    CHECK(eig_general(b, 1e-8, ww, W) == EIG_OK,
          "T2 eig_general runs");

    double c0[2] = {
        MAT(W, 0, 0),
        MAT(W, 1, 0)
    };

    CHECK(mat_residual(b, ww[0], c0) < 1e-7,
          "T2 residual lambda=2");

    /* T3: rotation matrix, complex eigenvalues */
    Matrix *r = mat_create(2, 2);
    double d3[] = {0, -1, 1, 0};

    for (int i = 0; i < 4; i++)
        r->data[i] = d3[i];

    double z[2];

    CHECK(eig_qr(r, 1e-12, 2000, z) == EIG_ERR_NOCONV,
          "T3 complex eigenvalues detected");

    /* T4: nonsymmetric matrix passed to Jacobi */
    CHECK(eig_jacobi(b, 1e-13, 100, w, W) == EIG_ERR_ARG,
          "T4 jacobi rejects nonsymmetric");

    /* T5: diagonal matrix */
    Matrix *dg = mat_create(3, 3);

    MAT(dg, 0, 0) = 3;
    MAT(dg, 1, 1) = -1;
    MAT(dg, 2, 2) = 7;

    CHECK(
        eig_jacobi(dg, 1e-13, 100, v, V) == EIG_OK &&
        v[0] == -1 &&
        v[1] == 3 &&
        v[2] == 7,
        "T5 diagonal matrix");

    mat_free(a);
    mat_free(V);
    mat_free(b);
    mat_free(W);
    mat_free(r);
    mat_free(dg);

    printf(fails ? "\n%d test(s) FAILED\n" : "\nAll tests passed\n",
           fails);

    return fails;
}
