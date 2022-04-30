#ifndef CUMDFUNC_H
#define CUMDFUNC_H

//__global__ void verify_parents(int id, cudaMD* dmd);

// box functions:
__device__ float3 get_shift(int shift_type, cudaMD* md);
__device__ void put_periodic(float z0, float3& xyz, float3 vel, float mass, int type, cudaMD* md);
__device__ void put_halfperiodic(float z0, float3& xyz, float3 vel, float mass, int type, cudaMD* md);	// x,y - periodic, z - not
__device__ void delta_periodic_orth(float& dx, float& dy, float& dz, cudaMD* md);
__device__ void delta_periodic_half(float& dx, float& dy, float& dz, cudaMD* md);
__device__ float dist2_periodic_orth(int i, int j, cudaMD* md);
__device__ float dist2_periodic_half(int i, int j, cudaMD* md);
//__device__ float r2_periodic(int id1, int id2, cudaMD* md);
__device__ void pass_periodic(int id1, int id2, cudaMD* md, int& px, int& py, int& pz);


__device__ void keep_in_cell(int index, float3 xyz, cudaMD* md);
__global__ void reset_quantities(cudaMD* md);
__global__ void print_stat(int iStep, cudaMD* md);
__global__ void clear_clist(/*int cellPerBlock, int cellPerThread, */cudaMD* md);

__global__ void verlet_1stage(int iStep, int atPerBlock, int atPerThread, cudaMD* md);
__global__ void verlet_2stage(int atPerBlock, int atPerThread, int iStep, cudaMD* md);
__global__ void zero_vel(int atPerBlock, int atPerThread, int iStep, cudaMD* md);


__global__ void verify_forces(int atPerBlock, int atPerThread, int iStep, cudaMD* md, int id);



#endif //  CUMDFUNC_H