// general cuda functions, not related to MD
#ifndef CUUTILS_H
#define CUUTILS_H

//sizes for memory allocation
const int float_size = sizeof(float);
const int float3_size = sizeof(float3);
const int int3_size = sizeof(int3);
const int int4_size = sizeof(int4);

void data_to_device(void** adrPtr, void* source, int size);
void cuda2DFree(void** addr, int n);

__device__ float3 float3_dif(float3 a, float3 b);		// a - b
__device__ void atomic_incFloat3(float3* var, float3 inc);
__device__ void inc_float3(float3* var, float3 inc);
__device__ float float3_length(float3 vect);
__device__ float float3_sqr(float3 vect);


__device__ int devNpairs(int n);	// return the number of pairs
void cuda_info();

// random number generators:
__device__ unsigned int rand1();
__device__ unsigned int rnd_xor128(uint4& s);


#endif // CUUTILS_H
