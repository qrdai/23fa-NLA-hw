/*
 * matmul_2.c: The 6 loop orderings for matrix multiplication
 */

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>

volatile double gtod(void)
{
  static struct timeval tv; // holds two members: tv_sec (seconds) and tv_usec (microseconds)
  static struct timezone tz;  // hold timezone information, but often unused and obsolete in many systems
  gettimeofday(&tv,&tz);
  return tv.tv_sec + 1.e-6*tv.tv_usec;
}


void matmul_dot_prod_1(double *a, double *b, double * restrict c, int n, int num_runs);

void matmul_dot_prod_2(double *a, double *b, double * restrict c, int n, int num_runs);

void matmul_outer_prod_1(double *a, double *b, double * restrict c, int n, int num_runs);

void matmul_outer_prod_2(double *a, double *b, double * restrict c, int n, int num_runs);

void matmul_gaxpy_column(double *a, double *b, double * restrict c, int n, int num_runs);

void matmul_gaxpy_row(double *a, double *b, double * restrict c, int n, int num_runs);


int main(int argc, char **argv)
{
  int n = atoi(argv[1]), num_runs = 3;
  printf("Matrix Order n: %d\n", n);

  double *a = malloc(n*n*sizeof(double));
  double *b = malloc(n*n*sizeof(double));
  double * restrict c = calloc(n*n, sizeof(double)); // initialize c with all 0

  // test if dynamic allocation succeeds
  if (a == NULL || b == NULL || c == NULL){
    printf("Memory allocation failed.\n");
    exit(1);
  }

  // set random seed
  srand(time(NULL));

  matmul_dot_prod_1(a, b, c, n, num_runs);
  matmul_dot_prod_2(a, b, c, n, num_runs);
  matmul_outer_prod_1(a, b, c, n, num_runs);
  matmul_outer_prod_2(a, b, c, n, num_runs);
  matmul_gaxpy_column(a, b, c, n, num_runs);
  matmul_gaxpy_row(a, b, c, n, num_runs);

  free(a);
  free(b);
  free(c);

  exit(0);
}


void matmul_dot_prod_1(double *a, double *b, double * restrict c, int n, int num_runs)
{
  double t, t_elapsed, fl_performance, avg_t_elapsed = 0, avg_fl_performance = 0;

  printf("Loop Ordering: dot product 1\n");

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

    // loop ordering: dot product 1
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                c[i + n*j] += a[i + n*k] * b[k + n*j];

    // timing stops
    t_elapsed = gtod() - t;

    fl_performance = 2.0 * n * n * n / (1e9 * t_elapsed);
    printf("\tElapsed Time: %15.6f s\n", t_elapsed);
    printf("\tFloating-Point Performance: %15.6f GFlops/s\n", fl_performance);

    // update avg_t_elapsed and avg_fl_performance
    avg_t_elapsed += t_elapsed / num_runs;
    avg_fl_performance += fl_performance / num_runs;
  }

  printf("\tAverage Elapsed Time: %15.6f s\n", avg_t_elapsed);
  printf("\tAverage Floating-Point Performance: %15.6f GFlops/s\n\n", avg_fl_performance);
}


void matmul_dot_prod_2(double *a, double *b, double * restrict c, int n, int num_runs)
{
  double t, t_elapsed, fl_performance, avg_t_elapsed = 0, avg_fl_performance = 0;

  printf("Loop Ordering: dot product 2\n");

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

    // loop ordering: dot product 2
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
                c[i + n*j] += a[i + n*k] * b[k + n*j];

    // timing stops
    t_elapsed = gtod() - t;

    fl_performance = 2.0 * n * n * n / (1e9 * t_elapsed);
    printf("\tElapsed Time: %15.6f s\n", t_elapsed);
    printf("\tFloating-Point Performance: %15.6f GFlops/s\n", fl_performance);

    // update avg_t_elapsed and avg_fl_performance
    avg_t_elapsed += t_elapsed / num_runs;
    avg_fl_performance += fl_performance / num_runs;
  }

  printf("\tAverage Elapsed Time: %15.6f s\n", avg_t_elapsed);
  printf("\tAverage Floating-Point Performance: %15.6f GFlops/s\n\n", avg_fl_performance);
}


void matmul_outer_prod_1(double *a, double *b, double * restrict c, int n, int num_runs)
{
  double t, t_elapsed, fl_performance, avg_t_elapsed = 0, avg_fl_performance = 0;

  printf("Loop Ordering: outer product 1\n");

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

    // loop ordering: outer product 1
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                c[i + n*j] += a[i + n*k] * b[k + n*j];

    // timing stops
    t_elapsed = gtod() - t;

    fl_performance = 2.0 * n * n * n / (1e9 * t_elapsed);
    printf("\tElapsed Time: %15.6f s\n", t_elapsed);
    printf("\tFloating-Point Performance: %15.6f GFlops/s\n", fl_performance);

    // update avg_t_elapsed and avg_fl_performance
    avg_t_elapsed += t_elapsed / num_runs;
    avg_fl_performance += fl_performance / num_runs;
  }

  printf("\tAverage Elapsed Time: %15.6f s\n", avg_t_elapsed);
  printf("\tAverage Floating-Point Performance: %15.6f GFlops/s\n\n", avg_fl_performance);
}


void matmul_outer_prod_2(double *a, double *b, double * restrict c, int n, int num_runs)
{
  double t, t_elapsed, fl_performance, avg_t_elapsed = 0, avg_fl_performance = 0;

  printf("Loop Ordering: outer product 2\n");

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

    // loop ordering: outer product 2
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                c[i + n*j] += a[i + n*k] * b[k + n*j];

    // timing stops
    t_elapsed = gtod() - t;

    fl_performance = 2.0 * n * n * n / (1e9 * t_elapsed);
    printf("\tElapsed Time: %15.6f s\n", t_elapsed);
    printf("\tFloating-Point Performance: %15.6f GFlops/s\n", fl_performance);

    // update avg_t_elapsed and avg_fl_performance
    avg_t_elapsed += t_elapsed / num_runs;
    avg_fl_performance += fl_performance / num_runs;
  }

  printf("\tAverage Elapsed Time: %15.6f s\n", avg_t_elapsed);
  printf("\tAverage Floating-Point Performance: %15.6f GFlops/s\n\n", avg_fl_performance);
}


void matmul_gaxpy_column(double *a, double *b, double * restrict c, int n, int num_runs)
{
  double t, t_elapsed, fl_performance, avg_t_elapsed = 0, avg_fl_performance = 0;

  printf("Loop Ordering: gaxpy column\n");

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

    // loop ordering: gaxpy column
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++)
            for (int i = 0; i < n; i++)
                c[i + n*j] += a[i + n*k] * b[k + n*j];

    // timing stops
    t_elapsed = gtod() - t;

    fl_performance = 2.0 * n * n * n / (1e9 * t_elapsed);
    printf("\tElapsed Time: %15.6f s\n", t_elapsed);
    printf("\tFloating-Point Performance: %15.6f GFlops/s\n", fl_performance);

    // update avg_t_elapsed and avg_fl_performance
    avg_t_elapsed += t_elapsed / num_runs;
    avg_fl_performance += fl_performance / num_runs;
  }

  printf("\tAverage Elapsed Time: %15.6f s\n", avg_t_elapsed);
  printf("\tAverage Floating-Point Performance: %15.6f GFlops/s\n\n", avg_fl_performance);
}


void matmul_gaxpy_row(double *a, double *b, double * restrict c, int n, int num_runs)
{
  double t, t_elapsed, fl_performance, avg_t_elapsed = 0, avg_fl_performance = 0;

  printf("Loop Ordering: gaxpy row\n");

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

    // loop ordering: gaxpy row
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                c[i + n*j] += a[i + n*k] * b[k + n*j];

    // timing stops
    t_elapsed = gtod() - t;

    fl_performance = 2.0 * n * n * n / (1e9 * t_elapsed);
    printf("\tElapsed Time: %15.6f s\n", t_elapsed);
    printf("\tFloating-Point Performance: %15.6f GFlops/s\n", fl_performance);

    // update avg_t_elapsed and avg_fl_performance
    avg_t_elapsed += t_elapsed / num_runs;
    avg_fl_performance += fl_performance / num_runs;
  }

  printf("\tAverage Elapsed Time: %15.6f s\n", avg_t_elapsed);
  printf("\tAverage Floating-Point Performance: %15.6f GFlops/s\n\n", avg_fl_performance);
}