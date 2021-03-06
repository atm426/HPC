////////////////////////////////////////////////////////////////////////////////
//
// coords.h
//
// Created Jun 23, 2011 by HJvE
// Last modified April 24, 2015 by HJvE
//
// provides coord structs and functions
//
// this is a working file, trying to implement vectorization
////////////////////////////////////////////////////////////////////////////////

#ifndef COORDS_H_
#define COORDS_H_

#include "extramath.h"
#include "intrin-wrapper.h"

// include r_res and phi_res somehow 
struct s_coordinates
{
  // these coordinates and time will always be in the LAB frame
  double x, y, z; // z also cylindrical coordinate
  double r, theta, phi; // phi also cylindrical coordinate
  double h; // cylindrial coordinate perpendicular to jet axis
  double sin_theta, cos_theta;
  double cos_phi;
  long N;
  double *p_x;
  double *p_y;
  double *p_z;
  double *p_r;
  double *p_phi;
  double *p_theta;
  double *p_stheta;
  double *p_ctheta;
  double *p_cphi;

  
  double t; // local lab frame time
};

////////////////////////////////////////////////////////////////////////////////

void sphericalfromcartesian_novec(s_coordinates &cor);
// uses the Cartesian elements of the struct to calculate the spherical ones,
// taking care to avoid singularities in the transformation

void sphericalfromcartesian_vec(s_coordinates &cor);

void hfromcartesian(s_coordinates &cor);
// only calculate the cylindrical h coordinate

void cylindricalfromspherical(s_coordinates &cor);

#endif // COORDS_H_
