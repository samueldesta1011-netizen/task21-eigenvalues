#include "eigen.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Matrix utilities ---------- */

Matrix *mat_create(int rows, int cols) {
    if (rows <= 0 || cols <= 0)
        return NULL;

    Matrix *m = malloc(sizeof *m);
    if (!m)
        return NULL;

    m->rows = rows;
    m->cols = cols;
    m->data = calloc((size_t)rows * cols, sizeof(double));

    if (!m->data) {
        free(m);
        return NULL;
    }

    return m;
}

void mat_free(Matrix *m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

Matrix *mat_copy(const Matrix *a) {
    if (!a)
        return NULL;

    Matrix *m = mat_create(a->rows, a->cols);
    if (m)
        memcpy(m->data, a->data,
               (size_t)a->rows * a->cols * sizeof(double));

    return m;
}

Matrix *mat_identity(int n) {
    Matrix *m = mat_create(n, n);
    if (!m)
        return NULL;

    for (int i = 0; i < n; i++)
        MAT(m, i, i) = 1.0;

    return m;
}

int mat_is_symmetric(const Matrix *a, double eps) {
    if (!a || a->rows != a->cols)
        return 0;

    for (int i = 0; i < a->rows; i++)
        for (int j = i + 1; j < a->cols; j++)
            if (fabs(MAT(a, i, j) - MAT(a, j, i)) > eps)
                return 0;

    return 1;
}

double mat_residual(const Matrix *a, double lambda, const double *v) {
    if (!a || !v)
        return HUGE_VAL;

    double sum = 0.0;

    for (int i = 0; i < a->rows; i++) {
        double residual = -lambda * v[i];

        for (int j = 0; j < a->cols; j++)
            residual += MAT(a, i, j) * v[j];

        sum += residual * residual;
    }

    return sqrt(sum);
}

static double norm2(const double *v, int n) {
    double sum = 0.0;

    for (int i = 0; i < n; i++)
        sum += v[i] * v[i];

    return sqrt(sum);
}

/* ---------- Jacobi method ---------- */

int eig_jacobi(const Matrix *a, double eps, int max_sweeps,
               double *vals, Matrix *vecs) {
    if (!a || !vals || !vecs ||
        a->rows != a->cols ||
        vecs->rows != a->rows ||
        vecs->cols != a->cols ||
        eps <= 0.0 ||
        max_sweeps <= 0 ||
        !mat_is_symmetric(a, 1e-12))
        return EIG_ERR_ARG;

    int n = a->rows;
    Matrix *b = mat_copy(a);
    if (!b)
        return EIG_ERR_ALLOC;

    /* Initialize eigenvector matrix to the identity matrix. */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            MAT(vecs, i, j) = (i == j) ? 1.0 : 0.0;

    int converged = 0;

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        double off = 0.0;

        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                off += MAT(b, i, j) * MAT(b, i, j);

        if (sqrt(off) < eps) {
            converged = 1;
            break;
        }

        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double apq = MAT(b, p, q);

                if (fabs(apq) < 1e-300)
                    continue;

                double theta =
                    (MAT(b, q, q) - MAT(b, p, p)) / (2.0 * apq);
                double t =
                    (theta >= 0.0 ? 1.0 : -1.0) /
                    (fabs(theta) + sqrt(theta * theta + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0);
                double s = t * c;

                /* Apply similarity transformation to columns. */
                for (int k = 0; k < n; k++) {
                    double x = MAT(b, k, p);
                    double y = MAT(b, k, q);
                    MAT(b, k, p) = c * x - s * y;
                    MAT(b, k, q) = s * x + c * y;
                }

                /* Apply similarity transformation to rows. */
                for (int k = 0; k < n; k++) {
                    double x = MAT(b, p, k);
                    double y = MAT(b, q, k);
                    MAT(b, p, k) = c * x - s * y;
                    MAT(b, q, k) = s * x + c * y;
                }

                /* Accumulate eigenvectors. */
                for (int k = 0; k < n; k++) {
                    double x = MAT(vecs, k, p);
                    double y = MAT(vecs, k, q);
                    MAT(vecs, k, p) = c * x - s * y;
                    MAT(vecs, k, q) = s * x + c * y;
                }
            }
        }
    }

    if (!converged) {
        mat_free(b);
        return EIG_ERR_NOCONV;
    }

    for (int i = 0; i < n; i++)
        vals[i] = MAT(b, i, i);

    mat_free(b);

    /* Sort eigenvalues and corresponding eigenvector columns. */
    for (int i = 1; i < n; i++) {
        for (int j = i;
             j > 0 && vals[j - 1] > vals[j];
             j--) {
            double temp = vals[j];
            vals[j] = vals[j - 1];
            vals[j - 1] = temp;

            for (int k = 0; k < n; k++) {
                double temp_vec = MAT(vecs, k, j);
                MAT(vecs, k, j) = MAT(vecs, k, j - 1);
                MAT(vecs, k, j - 1) = temp_vec;
            }
        }
    }

    return EIG_OK;
}

/* ---------- Power method ---------- */

int eig_power(const Matrix *a, double eps, int max_iter,
              double *lambda, double *v) {
    if (!a || !lambda || !v ||
        a->rows != a->cols ||
        eps <= 0.0 ||
        max_iter <= 0)
        return EIG_ERR_ARG;

    int n = a->rows;
    double *w = malloc((size_t)n * sizeof(double));
    if (!w)
        return EIG_ERR_ALLOC;

    /* Generic initial vector. */
    for (int i = 0; i < n; i++)
        v[i] = 1.0 + 0.1 * i;

    double nv = norm2(v, n);
    if (nv == 0.0) {
        free(w);
        return EIG_ERR_NOCONV;
    }

    for (int i = 0; i < n; i++)
        v[i] /= nv;

    for (int iteration = 0; iteration < max_iter; iteration++) {
        for (int i = 0; i < n; i++) {
            w[i] = 0.0;
            for (int j = 0; j < n; j++)
                w[i] += MAT(a, i, j) * v[j];
        }

        double current_lambda = 0.0;
        for (int i = 0; i < n; i++)
            current_lambda += v[i] * w[i];

        double residual = 0.0;
        for (int i = 0; i < n; i++) {
            double difference = w[i] - current_lambda * v[i];
            residual += difference * difference;
        }

        residual = sqrt(residual);
        *lambda = current_lambda;

        if (residual < eps * fmax(1.0, fabs(current_lambda))) {
            free(w);
            return EIG_OK;
        }

        double nw = norm2(w, n);
        if (nw < 1e-300) {
            *lambda = 0.0;
            free(w);
            return EIG_OK;
        }

        for (int i = 0; i < n; i++)
            v[i] = w[i] / nw;
    }

    free(w);
    return EIG_ERR_NOCONV;
}

/* ---------- QR algorithm ---------- */

int eig_qr(const Matrix *a, double eps, int max_iter, double *vals) {
    if (!a || !vals ||
        a->rows != a->cols ||
        eps <= 0.0 ||
        max_iter <= 0)
        return EIG_ERR_ARG;

    int n = a->rows;
    Matrix *m = mat_copy(a);
    if (!m)
        return EIG_ERR_ALLOC;

    size_t count = (size_t)n * (n - 1) / 2;
    double *cs = malloc(count * 2 * sizeof(double));

    if (!cs) {
        mat_free(m);
        return EIG_ERR_ALLOC;
    }

    int converged = 0;

    for (int iteration = 0;
         iteration < max_iter && !converged;
         iteration++) {
        double lower = 0.0;

        for (int i = 1; i < n; i++)
            for (int j = 0; j < i; j++)
                lower = fmax(lower, fabs(MAT(m, i, j)));

        if (lower < eps) {
            converged = 1;
            break;
        }

        size_t k = 0;

        /* Compute A = Q^T A = R. */
        for (int j = 0; j < n - 1; j++) {
            for (int i = j + 1; i < n; i++) {
                double x = MAT(m, j, j);
                double y = MAT(m, i, j);
                double r = hypot(x, y);
                double c = 1.0;
                double s = 0.0;

                if (r > 0.0) {
                    c = x / r;
                    s = y / r;
                }

                cs[2 * k] = c;
                cs[2 * k + 1] = s;
                k++;

                for (int col = 0; col < n; col++) {
                    double p = MAT(m, j, col);
                    double q = MAT(m, i, col);
                    MAT(m, j, col) = c * p + s * q;
                    MAT(m, i, col) = -s * p + c * q;
                }
            }
        }

        k = 0;

        /* Compute A = R Q. */
        for (int j = 0; j < n - 1; j++) {
            for (int i = j + 1; i < n; i++) {
                double c = cs[2 * k];
                double s = cs[2 * k + 1];
                k++;

                for (int row = 0; row < n; row++) {
                    double p = MAT(m, row, j);
                    double q = MAT(m, row, i);
                    MAT(m, row, j) = c * p + s * q;
                    MAT(m, row, i) = -s * p + c * q;
                }
            }
        }
    }

    if (converged) {
        for (int i = 0; i < n; i++)
            vals[i] = MAT(m, i, i);

        for (int i = 1; i < n; i++)
            for (int j = i;
                 j > 0 && vals[j - 1] > vals[j];
                 j--) {
                double temp = vals[j];
                vals[j] = vals[j - 1];
                vals[j - 1] = temp;
            }
    }

    mat_free(m);
    free(cs);

    return converged ? EIG_OK : EIG_ERR_NOCONV;
}

/* ---------- Inverse iteration ---------- */

static int solve_lu(Matrix *m, double *x) {
    int n = m->rows;

    for (int k = 0; k < n; k++) {
        int pivot = k;

        for (int i = k + 1; i < n; i++)
            if (fabs(MAT(m, i, k)) > fabs(MAT(m, pivot, k)))
                pivot = i;

        if (pivot != k) {
            for (int j = 0; j < n; j++) {
                double temp = MAT(m, k, j);
                MAT(m, k, j) = MAT(m, pivot, j);
                MAT(m, pivot, j) = temp;
            }

            double temp = x[k];
            x[k] = x[pivot];
            x[pivot] = temp;
        }

        if (fabs(MAT(m, k, k)) < 1e-300)
            MAT(m, k, k) = 1e-300;

        for (int i = k + 1; i < n; i++) {
            double factor =
                MAT(m, i, k) / MAT(m, k, k);

            for (int j = k; j < n; j++)
                MAT(m, i, j) -= factor * MAT(m, k, j);

            x[i] -= factor * x[k];
        }
    }

    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++)
            x[i] -= MAT(m, i, j) * x[j];

        x[i] /= MAT(m, i, i);
    }

    return 0;
}

int eig_inverse_iteration(const Matrix *a, double lambda, double eps,
                          int max_iter, double *v) {
    if (!a || !v ||
        a->rows != a->cols ||
        eps <= 0.0 ||
        max_iter <= 0)
        return EIG_ERR_ARG;

    int n = a->rows;
    double shift =
        lambda + 1e-10 * fmax(1.0, fabs(lambda));

    for (int i = 0; i < n; i++)
        v[i] = 1.0 + 0.1 * i;

    double nv = norm2(v, n);
    for (int i = 0; i < n; i++)
        v[i] /= nv;

    for (int iteration = 0; iteration < max_iter; iteration++) {
        Matrix *m = mat_copy(a);
        if (!m)
            return EIG_ERR_ALLOC;

        for (int i = 0; i < n; i++)
            MAT(m, i, i) -= shift;

        solve_lu(m, v);
        mat_free(m);

        nv = norm2(v, n);

        if (!(nv > 0.0) || !isfinite(nv))
            return EIG_ERR_NOCONV;

        for (int i = 0; i < n; i++)
            v[i] /= nv;

        if (mat_residual(a, lambda, v) <
            eps * fmax(1.0, fabs(lambda)))
            return EIG_OK;
    }

    return EIG_ERR_NOCONV;
}

int eig_general(const Matrix *a, double eps,
                double *vals, Matrix *vecs) {
    if (!a || !vals || !vecs ||
        a->rows != a->cols ||
        vecs->rows != a->rows ||
        vecs->cols != a->cols ||
        eps <= 0.0)
        return EIG_ERR_ARG;

    int n = a->rows;
    int rc = eig_qr(a, 1e-12, 20000, vals);

    if (rc != EIG_OK)
        return rc;

    double *v = malloc((size_t)n * sizeof(double));
    if (!v)
        return EIG_ERR_ALLOC;

    for (int k = 0; k < n; k++) {
        rc = eig_inverse_iteration(a, vals[k], eps, 50, v);

        if (rc != EIG_OK) {
            free(v);
            return rc;
        }

        for (int i = 0; i < n; i++)
            MAT(vecs, i, k) = v[i];
    }

    free(v);
    return EIG_OK;
}
