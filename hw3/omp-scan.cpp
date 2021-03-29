#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
    //printf("serial %d \n", prefix_sum[i]);
  }
}
void scan_omp(long* prefix_sum, const long* A, long n) {
  if (n==0) return;
  int p=10;
  omp_set_num_threads(p);
#pragma omp parallel shared(p,prefix_sum,A,n)
  {
    int tid = omp_get_thread_num();
    int start;
    start = tid*n/p;
    //printf("thread %d starts at index %d \n", tid, start);
    prefix_sum[start] = A[start];
    //printf("thread %d starts at value %d \n", tid, prefix_sum[start]);
    for ( int i = 1; i < n/p; i++){
      prefix_sum[start+i] = prefix_sum[start+i-1] + A[start+i];   
    }
  }
  //for (int i =0; i<n; i++) printf("sum[%d] = %d \n", i, prefix_sum[i]);
  double S[p];
  for( int k = 1; k<p; k++){
    for ( long i = 0; i < k*n/p; i++){
      S[k] += A[i];
    }
    //printf("%f \n", S[k]);
  }

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    for (long i = tid*n/p; i<tid*n/p+n/p; i++){
	prefix_sum[i] += S[tid];
      }
  }
}

int main() {
  long N = 1000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
