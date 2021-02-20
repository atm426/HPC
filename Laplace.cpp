#include <stdio.h>
#include "utils.h"
#include <math.h>

void Jacobi( int N, int iterations, double *u, double *u_kp1, double *A){

  //printf("%f \n", h);
  // f array, all ones
  double f = 1.0;
  // printf("f = %d\n", f[9]);

  // u array, all zeros
  // printf("u = %d\n", u[9]);
  int k = 0;

  Timer t;
  t.tic();
  while( k < iterations ){

    for( int i = 0; i < N-1; i++ ){
      double sum_term = 0.0;
      for( int j = 0; j < N-1; j++ ){
	if( i == j ){
	  continue;
	}
	else{
	  sum_term += A[j+N*i]*u[j];
	  //printf("%f \n", A[i+N*j]); 
	}
	
      }
      u_kp1[i] = (1/(A[i+i*N]))*(f - sum_term);
      //printf("%f \n", A[i+N*i]);
    }
    
    for( int i = 0; i < N-1; i++){
      u[i] = u_kp1[i];
      //printf("%f \n", u[i]);
    }
    
    double residual = 0.0;
    //printf("start res = %f \n", residual);
    for( int i = 0; i < N-1; i++ ){
      double Auk = 0.0;
      for( int j = 0; j < N-1; j++ ){
	Auk += A[j+N*i]*u[j];
      }
      //printf("%f \n", Auk);
      residual += pow(Auk-1, 2.0);
    }
    printf("%f \n", pow(residual,0.5));
    k += 1;
    
  }
  double time = t.toc();
  printf("Jacobi runtime = %f \n", time);
}

void GS(int N, int iterations, double *u, double *u_kp1, double *A){

  //printf("%f \n", h);
  // f array, all ones
  double f = 1.0;
  // printf("f = %d\n", f[9]);

  // u array, all zeros
  // printf("u = %d\n", u[9]);
  int k = 0;

  Timer t;
  t.tic();
  while( k < iterations ){

    for( int i = 0; i < N-1; i++ ){
      //printf("%d \n", i);
      double second_sum_term = 0.0;
      double first_sum_term = 0.0;
      for( int j = 0; j < N-1; j++){
	if( j < i ){
	  first_sum_term += A[j+N*i]*u_kp1[j];
	}
	else{
	  if( j > i ){
	    second_sum_term += A[j+N*i]*u[j];
	  }
	}
      }
      u_kp1[i] = 1/A[i+N*i]*(1 - first_sum_term - second_sum_term);
	}
    for( int i = 0; i < N-1; i++){
      u[i] = u_kp1[i];
      //printf("%f \n", u[i]);
    }
    double residual = 0.0;
    //printf("start res = %f \n", residual);
    for( int i = 0; i < N-1; i++ ){
      double Auk = 0.0;
      for( int j = 0; j < N-1; j++ ){
	Auk += A[j+N*i]*u[j];
      }
      //printf("%f \n", Auk);
      residual += pow(Auk-1, 2.0);
    }
    printf("%f \n", pow(residual,0.5));
    k += 1;

  }
  double time = t.toc();
  printf("GS runtime = %f \n", time);
}

int main(int argc, char** argv){

  // number of grid points
  int N = atof(argv[1]);
  printf("N = %d\n", N);

  char* method = argv[2];

  double N_d = 1.0*N;
  double h = 1/(N_d+1);
  
  double* u = (double*) malloc(N*sizeof(double));
  for( long i = 0; i < N; i++) u[i] = 0.0;
  
  double* u_kp1 = (double*) malloc(N*sizeof(double));
  //for( long i = 0; i < N; i++) u_kp1[i] = 0.0;

  double* A = (double*) malloc(N*N*sizeof(double));
  for( long i = 0; i < N*N; i++ ) A[i] = 0.0;
  for( long i = 0; i < N; i++ ){
    if( i == N-1 ){
      continue;
    }
    A[i + i*N] = 2.0/(h*h);
    A[i+(i+1)*N] = -1.0/(h*h);
    A[i+1+i*N] = -1.0/(h*h);
  }
  

  // grid step size
  int iterations = 5000;
  
  //Jacobi(N, iterations, u, u_kp1, A);
  GS(N, iterations, u, u_kp1, A);
  free(u);
  free(u_kp1);
  free(A);

  return 0;
}
