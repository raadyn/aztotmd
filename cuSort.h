#ifndef CUSORT_H
#define CUSORT_H

__device__ void count_cell(int index, float3 xyz, cudaMD* md);
void alloc_sort(int nAt, int nCell, cudaMD* hmd);
void free_sort(cudaMD* hmd);

__global__ void refresh_arrays(int use_bnd, int use_ang, cudaMD* md);
__global__ void calc_firstAtomInCell(cudaMD* md);
__global__ void sort_atoms(int use_bnd, int use_ang, int atPerBlock, int atPerThread, cudaMD* md);
__global__ void sort_dependent(int atPerBlock, int atPerThread, cudaMD* md);
__global__ void sort_bonds(int bndPerBlock, int bndPerThread, cudaMD* md);
__global__ void sort_angles(int angPerBlock, int angPerThread, cudaMD* md);


#endif // CUSORT_H