/* Console application: eigenvalues/eigenvectors of a matrix.
   Usage: eigapp [file] [auto|jacobi|power|qr]

   File format:
       n
       n*n matrix entries

   Without a file, read the matrix from stdin.
*/
#include "eigen.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static Matrix *read_matrix(FILE *f) {
    int n;

    if (fscanf(f, "%d", &n) != 1 || n <= 0)
        return NULL;

    Matrix *m = mat_create(n, n);
    if (!m)
        return NULL;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fscanf(f, "%lf", &MAT(m, i, j)) != 1) {
                mat_free(m);
                return NULL;
            }
        }
    }

    return m;
}

static void print_pairs(const Matrix *a, const double *vals,
                        const Matrix *vecs) {
    int n = a->rows;

    for (int k = 0; k < n; k++) {
        double *v = malloc((size_t)n * sizeof(double));

        if (!v) {
            fprintf(stderr, "Error: allocation failure\n");
            return;
        }

        printf("lambda_%d = %.10g\n  v = (", k + 1, vals[k]);

        for (int i = 0; i < n; i++) {
            v[i] = MAT(vecs, i, k);
            printf("%s%.6f", i ? ", " : "", v[i]);
        }

        printf(")\n  ||Av - lambda*v|| = %.3e\n",
               mat_residual(a, vals[k], v));

        free(v);
    }
}

int main(int argc, char **argv) {
    FILE *f = stdin;

    if (argc > 1 && strcmp(argv[1], "-") != 0) {
        f = fopen(argv[1], "r");

        if (!f) {
            fprintf(stderr, "Cannot open %s\n", argv[1]);
            return 1;
        }
    }

    const char *method = argc > 2 ? argv[2] : "auto";
    Matrix *a = read_matrix(f);

    if (f != stdin)
        fclose(f);

    if (!a) {
        fprintf(stderr, "Input error: bad matrix\n");
        return 1;
    }

    int n = a->rows;
    int rc = EIG_OK;
    double *vals = calloc((size_t)n, sizeof(double));
    Matrix *vecs = mat_create(n, n);

    if (!vals || !vecs) {
        fprintf(stderr, "Error: allocation failure\n");
        free(vals);
        mat_free(vecs);
        mat_free(a);
        return EIG_ERR_ALLOC;
    }

    if (!strcmp(method, "power")) {
        double lam;
        double *v = malloc((size_t)n * sizeof(double));

        if (!v) {
            rc = EIG_ERR_ALLOC;
        } else {
            rc = eig_power(a, 1e-10, 100000, &lam, v);

            if (rc == EIG_OK) {
                vals[0] = lam;

                for (int i = 0; i < n; i++)
                    MAT(vecs, i, 0) = v[i];

                printf(
                    "Dominant eigenpair (power method)\n"
                    "lambda = %.10g\n  v = (",
                    lam);

                for (int i = 0; i < n; i++)
                    printf("%s%.6f", i ? ", " : "", v[i]);

                printf(")\n  ||Av - lambda*v|| = %.3e\n",
                       mat_residual(a, lam, v));
            }

            free(v);
        }
    } else {
        int symmetric = mat_is_symmetric(a, 1e-12);

        if (!strcmp(method, "jacobi") ||
            (!strcmp(method, "auto") && symmetric)) {
            printf("Method: Jacobi\n");
            rc = eig_jacobi(a, 1e-13, 100, vals, vecs);
        } else if (!strcmp(method, "qr") ||
                   !strcmp(method, "auto")) {
            printf("Method: QR algorithm + inverse iteration\n");
            rc = eig_general(a, 1e-8, vals, vecs);
        } else {
            rc = EIG_ERR_ARG;
        }

        if (rc == EIG_OK)
            print_pairs(a, vals, vecs);
    }

    if (rc == EIG_ERR_ARG)
        fprintf(stderr,
                "Error: invalid arguments or unsupported method\n");

    if (rc == EIG_ERR_ALLOC)
        fprintf(stderr, "Error: allocation failure\n");

    if (rc == EIG_ERR_NOCONV)
        fprintf(stderr,
                "Error: no convergence "
                "(complex or equal-modulus eigenvalues?)\n");

    free(vals);
    mat_free(vecs);
    mat_free(a);

    return rc;
}
