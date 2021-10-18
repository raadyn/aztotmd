#ifndef CUEJUMP_H
#define CUEJUMP_H

void init_cuda_ejump(Sim* sim, Atoms* atm, cudaMD* hmd);
void free_cuda_ejump(cudaMD* hmd);

__device__ void try_to_jump(float r2, int id1, int id2, int spec1, int spec2, cudaMD* md);
__global__ void cuda_ejump(int bndPerThread, cudaMD* md);

#endif // CUEJUMP_H
