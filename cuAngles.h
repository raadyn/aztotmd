#ifndef CUANGLES
#define CUANGLES

void init_cuda_angles(int mxAt, int nsize, Field* fld, cudaMD* h_md, hostManagMD* man);

__global__ void refresh_angles(int iStep, int atPerBlock, int atPerThread, cudaMD* md);
__global__ void clear_angles(cudaMD* md);
__global__ void apply_angles(int iStep, int angPerBlock, int angPerThread, cudaMD* md);
__global__ void define_ang_potential(cudaMD* md);


#endif