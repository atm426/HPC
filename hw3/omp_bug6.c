/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

float dotprod ()
{
int i,tid;
float sum;
// sum=0.0;

//#pragma omp parallel
tid = omp_get_thread_num();  
#pragma omp parallel shared(sum)
// had to declare "sum" shared inside the function as well

 #pragma omp for reduction(+: sum)
  for (i=0; i < VECLEN; i++)
    {
      
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d sum=%f\n",tid,i,sum);
      
    }
  return sum;
}


int main (int argc, char *argv[]) {
int i;
float sum;
//sum=0.0;
 for (i=0; i < VECLEN; i++){
  a[i] = b[i] = 1.0 * i;
 }
 #pragma omp parallel shared(sum)
 // had to return "sum" to print the correct value
  sum=dotprod();

printf("Sum = %f\n",sum);

}

