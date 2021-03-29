#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "utils.h"
#include <math.h>
//
void Jacobi( int N, int iterations, double *u, double *u_kp1, double h){

  // f array, all ones
  double f = 1.0;
  // extra padding on the arrays for edge cases
  long N2 = N+2;
  int k = 0;
  
  // set number of threads
  int p = 16;
  #ifdef _OPENMP
  omp_set_num_threads(p);
  #endif
  while( k < iterations ){
    #pragma omp parallel for schedule(static) shared(u, u_kp1)
    // 
    for( long i = 1; i < N+1; i++ ){
      for( long j = 1; j < N+1; j++ ){
	u_kp1[i*N2+j] = 0.25*(h*h*f + u[(i-1)*N2+j] + u[i*N2+(j-1)] + u[(i+1)*N2+j] + u[i*N2+(j+1)]); 
      }
    }
    double residual = 0.0;
#pragma omp parallel for schedule(static) shared(u, u_kp1) reduction(+:residual)
    for( long i = 1; i < N+1; i++ ){
      for( long j = 1; j < N+1; j++ ){
	residual += pow(u[i*N2+j]-u_kp1[i*N2+j],2);
      }
    };
    //printf("%f \n", pow(residual,0.5));
    // entering update to u vector
#pragma omp parallel for schedule(static) shared(u, u_kp1)
    for( long ij = 0; ij < N2*N2; ij++){
      u[ij] = u_kp1[ij];
    }
    // update iteration number
    k += 1;
  }
  //double time = t.toc();
}

int main(int argc, char** argv){

  // number of grid points
  int N = atof(argv[1]);
  printf("N = %d\n", N);
  long N2 = N + 2;

  // grid step size
  double N_d = 1.0*N;
  double h = 1/(N_d+1);
  
  double* u = (double*) malloc(N2*N2*sizeof(double));
  for( long i = 0; i < N*N; i++) u[i] = 0.0;
  
  double* u_kp1 = (double*) malloc(N2*N2*sizeof(double));

  // number of iterations
  int iterations = 1000;
  double time;
  #ifdef _OPENMP
  time = omp_get_wtime();
  #endif
  Jacobi(N, iterations, u, u_kp1, h);
  #ifdef _OPENMP
  printf("Jacobi runtime = %f \n", omp_get_wtime() - time);
  #endif
  free(u);
  free(u_kp1);

  return 0;
}
