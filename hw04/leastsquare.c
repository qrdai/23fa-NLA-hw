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
void read_data(const char *filename, double **x, double **y, int *n)
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


void print_whole_matrix(const double *X, int m, int n)
{
    // m is row number and n is column number
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++)
            printf("%20.6lf ", X[i + m*j]);
        printf("\n");
    }
    printf("\n\n");
}


void print_whole_vector(const double *x, int n)
{
    // n is the length of vector x
    for (int i = 0; i < n; i++)
        printf("%20.6lf\n", x[i]);
    printf("\n\n");
}


void print_lower_triangular_matrix(const double *X, int n)
{
    // n is the order of square matrix X
    for (int i = 0; i < n; i++){
        for (int j = 0; j <= i; j++)
            printf("%20.6lf ", X[i + n*j]);
        printf("\n");
    }
    printf("\n\n");
}


int main(int argc, char *argv[])
{
    int d = atoi(argv[1]);  // d stands for "degree"
    int d_plus_one = d + 1;
    double *x, *y; // x and y are both level one pointers, and will be malloc in read_data function
    int n;
    read_data("data.dat", &x, &y, &n);

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
    
    // print initial matrix X: n * (d+1)
    printf("Initial Matrix X:\n");
    print_whole_matrix(X, n, d_plus_one);

    // arguments to be passed into BLAS and LAPACK functions
    char trans = 'T', notrans = 'N';
    double alpha = 1.0, beta = 0.0;
    int inc = 1;

    // dgemm: output in XT_X: (d+1) * (d+1)
    dgemm_(&trans, &notrans, &d_plus_one, &d_plus_one, &n, &alpha, X, &n, X, &n, &beta, XT_X, &d_plus_one);
    printf("Matrix XT_X:\n");
    print_whole_matrix(XT_X, d_plus_one, d_plus_one);

    // dgemv: output in XT_y: (d+1) * 1
    dgemv_(&trans, &n, &d_plus_one, &alpha, X, &n, Y, &inc, &beta, XT_y, &inc);
    printf("Vector XT_y:\n");
    print_whole_vector(XT_y, d_plus_one);


    char lower = 'L', left = 'L', nodiag = 'N';
    int info = 0, one = 1;

    // dpotrf: output in the lower triangular part of XT_X: (d+1) * (d+1)
    dpotrf_(&lower, &d_plus_one, XT_X, &d_plus_one, &info);
    printf("Lower Triangular Matrix L:\n");
    print_lower_triangular_matrix(XT_X, d_plus_one);

    // dtrsm * 2: output both in XT_y (y_tilde)
    dtrsm_(&left, &lower, &notrans, &nodiag, &d_plus_one, &one, &alpha, XT_X, &d_plus_one, XT_y, &d_plus_one);
    dtrsm_(&left, &lower, &trans, &nodiag, &d_plus_one, &one, &alpha, XT_X, &d_plus_one, XT_y, &d_plus_one);

    // Finally, print out the calculated coefficient XT_y
    printf("Coefficients of the polynomial fit:\n");
    for (int i = 0; i < d_plus_one; i++) {
        // printf("b[%d] = %lf\n", i, XT_y[i]);
        printf("%lf, ", XT_y[i]);
    }
    printf("\n\n");

    free(x);
    free(y);
    free(X);
    free(Y);
    free(XT_X);
    free(XT_y);

    return 0;
}
