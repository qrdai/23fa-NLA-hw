#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// Declaring BLAS and LAPACK functions
void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
            double *alpha, double *a, int *lda, double *b, int *ldb,
            double *beta, double *c, int *ldc);

void dgemv_(char *trans, int *m, int *n, double *alpha, double *a,
            int *lda, double *x, int *incx, double *beta, double *y,
            int *incy);

void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);

void dtrsm_(char *side, char *uplo, char *transa, char *diag, 
            int *m, int *n, double *alpha, 
            double *a, int *lda, double *b, int *ldb);



// Function to read data from a file
void readData(const char *filename, double **x, double **y, int *n)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Failed to open file");
        exit(1);
    }

    // n is an int pointer, *n is an int number
    // *x and *y are both level-1 pointers, and match the x and y in the main function
    fscanf(file, "%d", n);
    *x = (double *)malloc(*n * sizeof(double));
    *y = (double *)malloc(*n * sizeof(double));

    for (int i = 0; i < *n; ++i)
    {
        fscanf(file, "%lf %lf", &(*x)[i], &(*y)[i]);
    }

    fclose(file);
}


int main(int argc, char *argv[])
{
    int d = atoi(argv[1]);  // d stands for "degree"
    int d_plus_one = d + 1;
    double *x, *y; // x and y are both level one pointers, and will be malloc in readData function
    int n;
    readData("data.dat", &x, &y, &n);

    // Building matrices X and y for the normal equations
    double *X = (double *)malloc(n * (d_plus_one) * sizeof(double));
    double *Y = (double *)malloc(n * sizeof(double));
    double *XT_X = (double *)malloc((d_plus_one) * (d_plus_one) * sizeof(double));
    double *XT_y = (double *)malloc((d_plus_one) * sizeof(double)); // namely the y_tilde used when solving triangular systems

    // initialize X and Y using column major ordering
    for (int j = 0; j < d_plus_one; j++)
        for (int i = 0; i < n; i++)
            X[i + n*j] = pow(x[i], j);
    for (int i = 0; i < n; i++)
        Y[i] = y[i];

    // arguments to be passed into BLAS and LAPACK functions
    char trans = 'T', notrans = 'N';
    double alpha = 1.0, beta = 0.0;
    int inc = 1;

    // dgemm: output in XT_X
    dgemm_(&trans, &notrans, &d_plus_one, &d_plus_one, &n, &alpha, X, &n, X, &n, &beta, XT_X, &d_plus_one);
    // dgemv: output in XT_y
    dgemv_(&trans, &n, &d_plus_one, &alpha, X, &n, Y, &inc, &beta, XT_y, &inc);


    char lower = 'L', left = 'L', nodiag = 'N';
    int info = 0, one = 1;

    // dpotrf: output in the lower triangular part of XT_X
    // XT_y is equal to y_tilde
    dpotrf_(&lower, &d_plus_one, XT_X, &d_plus_one, &info);

    // dtrsm * 2: output both in XT_y (y_tilde)
    dtrsm_(&left, &lower, &notrans, &nodiag, &d_plus_one, &one, &alpha, XT_X, &d_plus_one, XT_y, &d_plus_one);
    dtrsm_(&left, &lower, &trans, &nodiag, &d_plus_one, &one, &alpha, XT_X, &d_plus_one, XT_y, &d_plus_one);

    // Finally, print out the calculated coefficient XT_y
    printf("Coefficients of the polynomial fit:\n");
    for (int i = 0; i < d_plus_one; i++) {
        printf("b[%d] = %lf\n", i, XT_y[i]);
    }

    free(x);
    free(y);
    free(X);
    free(Y);
    free(XT_X);
    free(XT_y);

    return 0;
}
