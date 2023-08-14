#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cstdlib>
#include <random>
#include <fstream>
#include <algorithm> 
#include <iostream>

#include "utils.hpp"

//create matrix from file 
func_ret_t create_matrix_from_file(float **mp, const char* filename, int *size_p){
  int i, j, size;
  float *m;
  FILE *fp = NULL;

  fp = fopen(filename, "rb");
  if ( fp == NULL) {
    return RET_FAILURE;
  }

  fscanf(fp, "%d\n", &size);

  m = (float*) malloc(sizeof(float)*size*size);
  if ( m == NULL) {
    fclose(fp);
    return RET_FAILURE;
  }

  for (i=0; i < size; i++) {
    for (j=0; j < size; j++) {
      fscanf(fp, "%f ", m+i*size+j);
    }
  }

  fclose(fp);

  *size_p = size;
  *mp = m;

  return RET_SUCCESS;
}


//create sparse matrix from file 
func_ret_t create_sparse_matrix_from_file(int **rp, int **cp, float **vp, const char* filename, int *nnz, int *size_p){
  int i, j, size;
  int *r, *c ;
  float *v;

  std::ifstream fin(filename);

  if ( fin.fail()) {
    return RET_FAILURE;
  }

  fin >> i >> size >> *nnz;

  j = *nnz;
  r = (int*) malloc(sizeof(int)*j);
  c = (int*) malloc(sizeof(int)*j);
  v = (float*) malloc(sizeof(float)*j);

  if ( r == NULL || c == NULL || v == NULL ) {
    fin.close();
    return RET_FAILURE;
  }

  for (i=1; i <= j; i++) {
    fin >> c[i-1] >> r[i-1] >> v[i-1];
    
  }

  fin.close();
  

  *size_p = size;
  *rp = r;
  *cp = c;
  *vp = v;


  return RET_SUCCESS;
}

// create dense random matrix

func_ret_t create_matrix(float * __restrict__ * mp, int size){
  float * m;
  int i,j;
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<float> dist_uniform(0,1);
  
  float lamda = -0.001;
  float coe[2*size-1];
  float coe_i =0.0;


  for (i=0; i < size; i++)
  {
    coe_i = 10*exp(lamda*i); 
    j=size-1+i;     
    coe[j]=coe_i;
    j=size-1-i;     
    coe[j]=coe_i;
  }

  m = (float*) malloc(sizeof(float)*size*size);
  if ( m == NULL) {
    return RET_FAILURE;
  }

  for (i=0; i < size; i++) {
    for (j=i; j < size; j++) {
      m[i*size+j]=coe[size-1-i+j];
//      dtype ran = dist_uniform(rng);
//      m[i*size+j] = ran;
//     m[j*size+i] = ran;
    }
  }

  *mp = m;

  return RET_SUCCESS;
}

 
  

// create dense vector 

func_ret_t create_vector(float **vp, int size){
  float *m;
  int i,j;
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<float> dist_uniform(0,1);

  float lamda = -0.001;
  float coe[2*size-1];
  float coe_i =0.0;

  for (i=0; i < size; i++)
  {
    coe_i = 10*exp(lamda*i); 
    j=size-1+i;     
    coe[j]=coe_i;
    j=size-1-i;     
    coe[j]=coe_i;
  }
  
  m = (float*) malloc(sizeof(float)*size);
  if ( m == NULL) {
    return RET_FAILURE;
  }

  for (i=0; i < size; i++) {
      m[i]=coe[size-1-i];
//    float ran = dist_uniform(rng);
//    m[i] = ran;

  }

  *vp = m;


  return RET_SUCCESS;
}

//create sparse random matrix
func_ret_t create_sparse_matrix(float **mp, int size){
  float *m;
  int i,j;
  
  float lamda = -0.001;
  float coe[2*size-1];
  float coe_i =0.0;


  for (i=0; i < size; i++)
  {
    coe_i = 10*exp(lamda*i); 
    j=size-1+i;     
    coe[j]=coe_i;
    j=size-1-i;     
    coe[j]=coe_i;
  }

  m = (float*) malloc(sizeof(float)*size*size);
  if ( m == NULL) {
    return RET_FAILURE;
  }
  time_t t;

  srand((unsigned) time(&t));

  for (i=0; i < size; i++) {
    for (j=i; j < size; j++) {
      int a = 100;
      if (rand()%a == 0)
      {
        m[i*size+j] = rand()%128;
        m[j*size+i] = rand()%128;
      }
      else
      {
        m[i*size+j] = 0;
        m[j*size+i] = 0;
      }
      
    }
  }

  *mp = m;

  return RET_SUCCESS;
}


