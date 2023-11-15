/*
 * dotblas.c: dot product using the BLAS library 
 * The BLAS library is available on CSIF computers
 * Compile with gcc -o dotblas dotblas.c -lblas
 */

#include<stdio.h>
#include<stdlib.h>

/* dot product */
/* BLAS ddot function prototype */
/* Note the added '_' character */
double ddot_(int *n, double *x, int *incx, double *y, int *incy);

int main(int argc, char** argv)
{
  /* first argument on the command line is the vector size */
  int n = atoi(argv[1]);
  int i;

  double *x = (double*)malloc(n*sizeof(double));
  double *y = (double*)malloc(n*sizeof(double));

  for ( i=0; i<n; i++ )
  {
    x[i] = 1.0/(i+1);
    y[i] = 2.0/(i+2);
  }

  double sum = 0.0;
  for ( i=0; i<n; i++ )
    sum += x[i] * y[i];
  printf("loop sum: %12.8f\n",sum);

  int inc=1;
  sum = ddot_(&n,x,&inc,y,&inc);
  printf("ddot sum: %12.8f\n",sum);

  return 0;
}
