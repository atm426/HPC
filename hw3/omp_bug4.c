/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

int main (int argc, char *argv[]){
  int nthreads;
  double* a = (double*) malloc(N*N*sizeof(double));
  // had to explicitly mallocate memory for a
 
/* Fork a team of threads with explicit variable scoping */
printf("N = %d \n", N);
#pragma omp parallel shared(nthreads)
  {
    int tid;
    long i, j;
  /* Obtain/print thread info */
    
    tid = omp_get_thread_num();
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  
  printf("Thread %d starting...\n", tid);
  /* Each thread works on its own private copy of the array */

  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      a[i*N+j] = tid + i + j;
      // had to change indexing for a since it is mallocated
    }
  }
  
  /* For confirmation */
  printf("Thread %d done. Last element= %d\n",tid,a[(N-1)*N+(N-1)]);
  }
}  /* All threads join master thread and disband */



