#ifndef CUVDW_H
#define CUVDW_H

__global__ void define_vdw_func(cudaMD* md);

/*
__device__ float cu_fer_lj(float r2, float& r, cudaVdW* vdw, float& eng);
__device__ float cu_fe_lj(float r2, cudaVdW* vdw, float& eng);
__device__ float cu_e_lj(float r2, cudaVdW* vdw);
__device__ float cu_er_lj(float r2, float r, cudaVdW* vdw);

__device__ float cu_fer_buck(float r2, float& r, cudaVdW* vdw, float& eng);
__device__ float cu_fe_buck(float r2, cudaVdW* vdw, float& eng);
__device__ float cu_e_buck(float r2, cudaVdW* vdw);
__device__ float cu_er_buck(float r2, float r, cudaVdW* vdw);

__device__ float cu_fer_bmh(float r2, float& r, cudaVdW* vdw, float& eng);
__device__ float cu_fe_bmh(float r2, cudaVdW* vdw, float& eng);
__device__ float cu_e_bmh(float r2, cudaVdW* vdw);
__device__ float cu_er_bmh(float r2, float r, cudaVdW* vdw);

__device__ float cu_fer_elin(float r2, float& r, cudaVdW* vdw, float& eng);
__device__ float cu_fe_elin(float r2, cudaVdW* vdw, float& eng);
__device__ float cu_e_elin(float r2, cudaVdW* vdw);
__device__ float cu_er_elin(float r2, float r, cudaVdW* vdw);

__device__ float cu_fer_einv(float r2, float& r, cudaVdW* vdw, float& eng);
__device__ float cu_fe_einv(float r2, cudaVdW* vdw, float& eng);
__device__ float cu_e_einv(float r2, cudaVdW* vdw);
__device__ float cu_er_einv(float r2, float r, cudaVdW* vdw);
*/



#endif  /* CUVDW_H */
