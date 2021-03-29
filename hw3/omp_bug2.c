/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
int nthreads, i, tid;
double total;

/*** Spawn parallel region ***/
#pragma omp parallel private(tid, i)
// i made tid private so that it was specific to each thread
// the tid was not initially private and was being set by the last thread 
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

#pragma omp barrier

  /* do some work */
  total = 0.0;
#pragma omp for schedule(dynamic,10)
  for (i=0; i<1000000; i++){
    #pragma omp critical
    {
    
    // making sure the write to total is only done one thread at a timee
    // i'm not sure if this is what the intended solution is, but i get the correct answer to the sum. 
     total = total + i*1.0;
  
    }
  }
  printf ("Thread %d is done! Total= %e\n",tid,total);
  
  
  } /*** End of parallel region ***/
}
