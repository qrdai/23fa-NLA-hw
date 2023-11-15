/*
 * matmul_1.c: The dot product loop ordering for matrix multiplication
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

int main(int argc, char **argv)
{
    int n = atoi(argv[1]), num_runs = 3;
    printf("Matrix Order n: %d\n", n);
    double t, t_elapsed, fl_performance, avg_t_elapsed = 0, avg_fl_performance = 0;

    double *a = malloc(n*n*sizeof(double));
    double *b = malloc(n*n*sizeof(double));
    double * restrict c = calloc(n*n, sizeof(double)); // initialize c with all 0

    if (a == NULL || b == NULL || c == NULL){
      printf("Memory allocation failed.\n");
      exit(1);
    }

    srand(time(NULL));

    for (int run = 0; run < num_runs; run++)
    {
      // (re)initialize A and B with random floating point numbers
      for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++){
          a[i + n*j] = (double)rand() / RAND_MAX;
          b[i + n*j] = (double)rand() / RAND_MAX;
          c[i + n*j] = 0;
        }

      // timing starts
      t = gtod();

      // dot product loop ordering for matrix multiplication
      for (int i = 0; i < n; i++)
          for (int j = 0; j < n; j++)
              for (int k = 0; k < n; k++)
                  c[i + n*j] += a[i + n*k] * b[k + n*j];

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