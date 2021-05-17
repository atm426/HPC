////////////////////////////////////////////////////////////
/// vectorized spherical coordinate transformation
/// adapted from function in coords.cpp in boxfit
/// for HPC spring 2021 final project
/// author: austin mcdowell (170521)
/// NOTE: ** must be compiled with Intel compiler ** 
////////////////////////////////////////////////////////////

#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "intrin-wrapper.h"
#include "coords.h"


#ifdef __AVX__
#include <immintrin.h>
#endif
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif


void sphericalfromcartesian_novec(s_coordinates &cor){
  // non-vectorized version for comparison
  // x^2 + y^2
  double xysqrd = cor.x * cor.x + cor.y * cor.y;

  // r from cartesian
  cor.r = sqrt(xysqrd + cor.z * cor.z);
  
  // taking care of singularities near origin or along z-axis
  if (cor.r < 1e-8 or xysqrd < 1e-8){
    cor.phi = 0.0;
    cor.theta = 0.0;
    cor.sin_theta = sin(cor.theta);
    cor.cos_theta = cos(cor.theta);
    cor.cos_phi = cos(cor.phi);
  }
  else{
    cor.theta = acos(cor.z / cor.r);
    cor.sin_theta = sin(cor.theta);
    cor.cos_theta = cos(cor.theta);
    
    // dealing with first quadrant
    if (cor.x > 0 and cor.y >= 0){
      if (cor.y < 1e-8) cor.phi = 0.0;
      else cor.phi = atan(cor.y/cor.x);
    }
    // dealing with second quadrant
    if (cor.x <= 0 and cor.y > 0){
      if (fabs(cor.x) < 1e-8) cor.phi = 0.5 * PI;
      else cor.phi = PI - atan(-cor.y/cor.x);
    }
    // dealing with third quadrant
    if (cor.x < 0 and cor.y <= 0){
      if (fabs(cor.y) < 1e-8) cor.phi = PI;
      else cor.phi = PI + atan(cor.y / cor.x);
    }
    // dealing with fourth quadrant
    if (cor.x >= 0 and cor.y < 0){
      if (cor.x < 1e-8) cor.phi = 1.5 * PI;
      else cor.phi = 2.0 * PI - atan(-cor.y/cor.x);
    }
    cor.cos_phi = cos(cor.phi);
  }
}

void sphericalfromcartesian_vec(s_coordinates &cor, long i){
  // vectorized function
  // coordinates are assumed to be all in the same quadrant
  // coordinates are assumed to be not too close to origin
#if defined(__AVX__)
  //printf("ayyyy \n");
  __m256d x, y, z, r, theta, stheta, ctheta, phi, cphi;

  // loading to vectors
  x = _mm256_load_pd(cor.p_x+i);
  y = _mm256_load_pd(cor.p_y+i);
  z = _mm256_load_pd(cor.p_z+i);

  // spherical transformation
  r = _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(z,z), _mm256_add_pd(_mm256_mul_pd(x,x),_mm256_mul_pd(y,y))));
  theta = _mm256_acos_pd( _mm256_div_pd(z,r) );
  stheta = _mm256_sin_pd(theta);
  ctheta = _mm256_cos_pd(theta);
  phi = _mm256_atan_pd( _mm256_div_pd(y,x) );
  cphi = _mm256_cos_pd(phi);
  // storing back to cor
  _mm256_store_pd(cor.p_r+i, r);
  _mm256_store_pd(cor.p_theta+i, theta);
  _mm256_store_pd(cor.p_stheta+i, stheta);
  _mm256_store_pd(cor.p_ctheta+i, ctheta);
  _mm256_store_pd(cor.p_phi+i, phi);
  _mm256_store_pd(cor.p_cphi+i, cphi);

#elif defined(__SSE2__)
  //printf("yooooo \n");
  constexpr int sse_length = 2;
  for (int j = 0; j < 4; j+=sse_length){
    __m128d x, y, z, r, theta, stheta, ctheta, phi, cphi;
    // loading to vectors
    x = _mm_load_pd(cor.p_x+i);
    y = _mm_load_pd(cor.p_y+i);
    z = _mm_load_pd(cor.p_z+i);

    // spherical transformation
    r = _mm_sqrt_pd(_mm_add_pd(_mm_mul_pd(z,z), _mm_add_pd(_mm_mul_pd(x,x),_mm_mul_pd(y,y))));
    theta = _mm_acos_pd( _mm_div_pd(z,r) );
    theta = _mm_acos_pd( _mm_div_pd(z,r) );
    stheta = _mm_sin_pd(theta);
    ctheta = _mm_cos_pd(theta);
    phi = _mm_atan_pd( _mm_div_pd(y,x) );
    cphi = _mm_cos_pd(phi);
    
    // storing back to cor
    _mm_store_pd(cor.p_r+i, r);
    _mm_store_pd(cor.p_theta+i, theta);
    _mm_store_pd(cor.p_stheta+i, stheta);
    _mm_store_pd(cor.p_ctheta+i, ctheta);
    _mm_store_pd(cor.p_phi+i, phi);
    _mm_store_pd(cor.p_cphi+i, cphi);
  }
#endif
  
}

int main( int argc, char *argv[]){

  // timing variables
  Timer tt;
  long rep;

  // setting up fake ray array
  // filled with random number between 0 and 1
  // has to be between 0 and 1 for vectorized part
  long N = atol(argv[1]);
  double *rays = (double*) aligned_malloc(N*sizeof(double));
  for( long i = 0; i < N; i++) rays[i] = drand48();

  // setting up cor struct
  s_coordinates cor;
  cor.N = N;

  // timing non-vectorized function for reference
  tt.tic();
  for (rep = 0; rep <= 10000; rep++){
    for (long i = 0; i < N; i++){
      cor.x = 0.25*rays[i];
      cor.y = 0.13*rays[i];
      cor.z = 0.666*rays[i];
      sphericalfromcartesian_novec(cor);
    }
  }
  printf("non-vectorized time for %lu repititions, grid size %lu: %6.5f \n", rep-1, N, tt.toc());
  printf("non-vectorized cosine(phi): %f \n", cor.cos_phi);
  printf("////////////// \n");

  // timing vectorized version
  // points are always in the first quadrant
  // I think I have to allocate these
  cor.p_x = (double*) aligned_malloc(N*sizeof(double));
  cor.p_y = (double*) aligned_malloc(N*sizeof(double));
  cor.p_z = (double*) aligned_malloc(N*sizeof(double));
  cor.p_r = (double*) aligned_malloc(N*sizeof(double));
  cor.p_phi = (double*) aligned_malloc(N*sizeof(double));
  cor.p_theta = (double*) aligned_malloc(N*sizeof(double));
  cor.p_stheta = (double*) aligned_malloc(N*sizeof(double));
  cor.p_ctheta = (double*) aligned_malloc(N*sizeof(double));
  cor.p_cphi = (double*) aligned_malloc(N*sizeof(double));

  // filling pointer arrays
  for (long i = 0; i < N; i++){
    cor.p_x[i] = 0.25*rays[i];
    cor.p_y[i] = 0.13*rays[i];
    cor.p_z[i] = 0.666*rays[i];
  }
  // timing vectorized version
  tt.tic();
  for (rep = 0; rep <= 10000; rep++){
    for (long i = 0; i < N; i+=4){
      sphericalfromcartesian_vec(cor, i);
    }
  }

  // printing timing and cos(phi) for comparison to non-vectorized
  printf("vectorized time for %lu repititions, grid size %lu: %6.5f \n", rep-1, N, tt.toc());
  printf("vectorized cosine(phi): %f \n", cor.p_cphi[N-1]);

  aligned_free(rays);
  aligned_free(cor.p_x);
  aligned_free(cor.p_y);
  aligned_free(cor.p_z);
  aligned_free(cor.p_r);
  aligned_free(cor.p_phi);
  aligned_free(cor.p_theta);
  aligned_free(cor.p_stheta);
  aligned_free(cor.p_ctheta);
  aligned_free(cor.p_cphi);
  return 0;
}


