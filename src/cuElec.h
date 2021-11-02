#ifndef CUELEC_H
#define CUELEC_H

// functions for calculating part of electrostatic interaction which can be calculated during pairs bypass (return forces and modify energy)
//__device__ float no_coul(float r2, float& r, float chprd, float alpha, float& eng);
//__device__ float direct_coul(float r2, float& r, float chprd, float alpha, float& eng);
//__device__ float real_ewald(float r2, float& r, float chprd, float alpha, float& eng);
__device__ float no_coul(float r2, float& r, float chprd, cudaMD *md, float& eng);
__device__ float direct_coul(float r2, float& r, float chprd, cudaMD* md, float& eng);
__device__ float real_ewald(float r2, float& r, float chprd, cudaMD* md, float& eng);
__device__ float fennel(float r2, float& r, float chprd, cudaMD* md, float& eng);
__device__ float real_ewald_tex(float r2, float& r, float chprd, float alpha, float& eng);

__global__ void recip_ewald(int atPerBlock, int atPerThread, cudaMD* md);
__global__ void ewald_force(int atPerBlock, int atPerThread, cudaMD* md);

void init_realEwald_tex(cudaMD* md, float mxRange, float alpha);
void free_realEwald_tex(cudaMD* md);


#endif  /* CUELEC_H */
