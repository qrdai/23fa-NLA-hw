#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void dgemv_(char *trans, int *m, int *n, double *alpha, double *a,
            int *lda, double *x, int *incx, double *beta, double *y,
            int *incy);

double dasum_(const int *n, const double *dx, const int *incx);

void dscal_(const int *n, const double *alpha, double *dx, const int *incx);

void daxpy_(const int *n, const double *alpha, const double *dx, 
                   const int *incx, double *dy, const int *incy);

int idamax_(const int *n, const double *dx, const int *incx);

const double EPSILON = 1.0e-6;


void read_link_matrix(const char *filename, double **X, int *n)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Failed to open file");
        exit(1);
    }

    // read the first line: matrix order *n
    // then allocate space for matrix *X
    fscanf(file, "%d", n);
    // printf("Value of *n: %d\n", *n);
    // printf("\n\n");
    *X = (double *)malloc((*n) * (*n) * sizeof(double));

    // initialize matrix *X column-wise
    for (int i = 0; i < *n; i++){
        // read the ith column of *X:
        int n_i = 0;
        fscanf(file, "%d", &n_i);
        // printf("Value of n_i: %d\n", n_i);

        // read 0th row -> (n-1)th row of column i
        int j = 0, idx = 0;
        double col_value = 1.0 / (double)n_i;
        while (j < *n && n_i > 0){
            // read the next index of nonzero value in column i
            fscanf(file, "%d", &idx);
            // printf("Value of idx: %d\n", idx);
            n_i -= 1;
            // then fill in the value from j -> idx
            // j -> (idx - 1) are 0, idx is col_value
            for (int k = j; k < idx; k++)
                (*X)[k + (*n)*i] = 0;
            (*X)[idx + (*n)*i] = col_value;
            j = idx + 1;
        }
        // printf("\n\n");
    }

    fclose(file);
}


void print_whole_matrix(const double *X, int m, int n)
{
    // m is row number and n is column number
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++)
            printf("%15.6lf ", X[i + m*j]);
        printf("\n");
    }
    printf("\n\n");
}


void print_whole_vector(const double *x, int n)
{
    // n is the length of vector x
    for (int i = 0; i < n; i++)
        printf("%15.6lf\n", x[i]);
    printf("\n\n");
}


// normalize vector x using 1-norm
void normalize(double *x, int n)
{
    // n is the length of vector x
    int inc = 1;

    double x_norm1 = dasum_(&n, x, &inc);
    double da = 1.0 / x_norm1;
    dscal_(&n, &da, x, &inc);
}


// return r_max of given 2 vectors
double calculate_r_max(double *x, double *y, int n, double lambda)
{
    // lambda is the targeted eigenvalue
    double neg_lambda = -lambda;
    double r_max;
    int inc = 1, max_index = -1;

    daxpy_(&n, &neg_lambda, x, &inc, y, &inc); // y <- y - lambda*x
    max_index = idamax_(&n, y, &inc) - 1; // BLAS array indexing starts at 1, not 0
    r_max = fabs(y[max_index]);

    return r_max;
}


int main()
{
    double *A; // matrix A
    int n; // matrix order n

    read_link_matrix("link_matrix_chain.dat", &A, &n);
    printf("Link Matrix A:\n");
    print_whole_matrix(A, n, n);

    // initialize x0
    double *x = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        // x[i] = 1.0;  // [1,1,1,1] orthogonal to eigenvector of lambda=-1: [1/6, -1/3, 1/3, -1/6], therefore cannot converge
        x[i] = (double)i + 1.0;

    // normalize x, and initialize r_max and y
    double r_max = 1.0, shifted_value = -0.25, lambda = -1.0;   // after shifting -0.25, absmax_lambda = -1 - 0.25 = -1.25

    double alpha = 1.0, beta = 0.0;
    int inc = 1;
    char notrans = 'N';

    normalize(x, n);
    // printf("Normalized Initial Vector x0:");
    // print_whole_vector(x, n);
    double *y = (double *)malloc(n * sizeof(double));
    double *temp = (double *)malloc(n * sizeof(double));

    // power method iteration with normalization (no shifting)
    while (r_max >= EPSILON)
    {
        // y <- A*x
        dgemv_(&notrans, &n, &n, &alpha, A, &n, x, &inc, &beta, y, &inc);

        // copy y into temp, and calculate r_max with (normalized(temp), x)
        for (int i = 0; i < n; i++)
            temp[i] = y[i];
        normalize(temp, n);
        r_max = calculate_r_max(x, temp, n, lambda);
        printf("r_max: %15.9lf\n\n", r_max);

        // shift y, then update x
        daxpy_(&n, &shifted_value, x, &inc, y, &inc); // y <- y + sigma*x, sigma default to -0.25
        normalize(y, n);
        // printf("normalized y:\n");
        // print_whole_vector(y, n);
        for (int i = 0; i < n; i++)
            x[i] = y[i];
    }

    // print normalized vector x(components sum to 1)
    printf("Normalized eigenvector x:\n");
    print_whole_vector(x, n);

    free(A);
    free(x);
    free(y);
    free(temp);
}