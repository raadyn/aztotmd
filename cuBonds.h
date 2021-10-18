#ifndef CUBONDS_H
#define CUBONDS_H

__device__ void try_to_bind(float r2, int id1, int id2, int spec1, int spec2, cudaMD* md);
__device__ cudaBond* evol_bondtype_addr(cudaBond* old_bnd, int spec1, int spec2, cudaMD* md);
__global__ void apply_bonds(int iStep, int bndPerBlock, int bndPerThread, cudaMD* md);
__global__ void apply_const_bonds(int iStep, int bndPerBlock, int bndPerThread, cudaMD* md);
__global__ void fix_bonds(int bndPerBlock, int bndPerThread, cudaMD* md);
__global__ void clear_bonds(cudaMD* md);
__global__ void create_bonds(int iStep, int atPerBlock, int atPerThread, cudaMD* md);
__global__ void refresh_atomTypes(int iStep, int atPerBlock, int atPerThread, cudaMD* md);

__global__ void define_bond_potential(cudaMD* md);

/*
__device__ float bond_harm(float r2, float r, float& eng, cudaBond* bnd);
__device__ float bond_morse(float r2, float r, float& eng, cudaBond* bnd);
__device__ float bond_pedone(float r2, float r, float& eng, cudaBond* bnd);
__device__ float bond_buck(float r2, float r, float& eng, cudaBond* bnd);
__device__ float bond_e6812(float r2, float r, float& eng, cudaBond* bnd);
*/



#endif // CUBONDS_H