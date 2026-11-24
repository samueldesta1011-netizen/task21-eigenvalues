/* eigen.h - library for eigenvalues and eigenvectors of real matrices */
#ifndef EIGEN_H
#define EIGEN_H

typedef struct {
    int rows, cols;
    double *data;                 /* row-major storage */
} Matrix;

#define MAT(m, i, j) ((m)->data[(size_t)(i) * (m)->cols + (j)])

enum {
    EIG_OK = 0,
    EIG_ERR_ARG = 1,
    EIG_ERR_ALLOC = 2,
    EIG_ERR_NOCONV = 3
};

/* Matrix utilities */
Matrix *mat_create(int rows, int cols);          /* zero-filled, NULL on error */
void mat_free(Matrix *m);
Matrix *mat_copy(const Matrix *a);
Matrix *mat_identity(int n);
int mat_is_symmetric(const Matrix *a, double eps);
double mat_residual(const Matrix *a, double lambda, const double *v);

/* Jacobi: all eigenpairs of a real symmetric matrix.
   Eigenvalues are ascending and eigenvectors are stored in columns. */
int eig_jacobi(const Matrix *a, double eps, int max_sweeps,
               double *vals, Matrix *vecs);

/* Power method: dominant eigenpair by absolute eigenvalue. */
int eig_power(const Matrix *a, double eps, int max_iter,
              double *lambda, double *v);

/* QR algorithm: all real eigenvalues of a general matrix, ascending. */
int eig_qr(const Matrix *a, double eps, int max_iter, double *vals);

/* Inverse iteration: eigenvector for a known eigenvalue. */
int eig_inverse_iteration(const Matrix *a, double lambda, double eps,
                          int max_iter, double *v);

/* QR eigenvalues followed by inverse-iteration eigenvectors. */
int eig_general(const Matrix *a, double eps,
                double *vals, Matrix *vecs);

#endif
