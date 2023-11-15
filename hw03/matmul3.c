/*
 * matmul_3.c: matrix multiplication implemented by dgemm_
 */

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>

// BLAS dgemm function prototype
void dgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N,
            const int *K, const double *ALPHA, const double *A, const int *LDA,
            const double *B, const int *LDB, const double *BETA, double *C,
            const int *LDC);


volatile double gtod(void)
{
  static struct timeval tv; // holds two members: tv_sec (seconds) and tv_usec (microseconds)
  static struct timezone tz;  // hold timezone information, but often unused and obsolete in many systems
  gettimeofday(&tv,&tz);
  return tv.tv_sec + 1.e-6*tv.tv_usec;
}


int main(int argc, char** argv) 
{
    int n = atoi(argv[1]), num_runs = 3;
    printf("Matrix Order n: %d\n", n);
    double t, t_elapsed, fl_performance, avg_t_elapsed = 0, avg_fl_performance = 0;

    double *a = (double *)malloc(n*n*sizeof(double));
    double *b = (double *)malloc(n*n*sizeof(double));
    double *c = (double *)malloc(n*n*sizeof(double));

    // test if dynamic allocation succeeds
    if (a == NULL || b == NULL || c == NULL){
        printf("Memory allocation failed.\n");
        exit(1);
    }

    // arguments for calling dgemm_
    double alpha = 1.0, beta = 0.0;
    char trans_a = 'N', trans_b = 'N';

    // set random seed
    srand(time(NULL));

    for (int run = 0; run < num_runs; run++)
    {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++){
                a[i + n*j] = (double)rand() / RAND_MAX;
                b[i + n*j] = (double)rand() / RAND_MAX;
                c[i + n*j] = 0;
            }

        // timing starts
        t = gtod();

        dgemm_(&trans_a, &trans_b, &n, &n, &n, &alpha, a, &n, b, &n, &beta, c, &n);

        // timing stops
        t_elapsed = gtod() - t;

        fl_performance = 2.0 * n * n * n / (1e9 * t_elapsed);
        printf("Elapsed Time: %15.6f s\n", t_elapsed);
        printf("Floating-Point Performance: %15.6f GFlops/s\n", fl_performance);

        // update average t_elapsed and average fl_performance
        avg_t_elapsed += t_elapsed / num_runs;
        avg_fl_performance += fl_performance / num_runs;
    }

    printf("Average Elapsed Time: %15.6f s\n", avg_t_elapsed);
    printf("Average Floating-Point Performance: %15.6f GFlops/s\n", avg_fl_performance);

    free(a);
    free(b);
    free(c);

    exit(0);
}
