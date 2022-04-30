// some useful functions for CUDA

#include <curand.h>

#include "cuUtils.h"
#include "utils.h"

void data_to_device(void** adrPtr, void* source, int size)
// allocate array on device and copy data
{
    cudaMalloc(adrPtr, size);
    cudaMemcpy(*adrPtr, source, size, cudaMemcpyHostToDevice);
}

void cuda2DFree(void** addr, int n)
// delete array of arrays located on device (n - the number of elements in the first dimension)
{
    int i;
    int sz = n * pointer_size;

    void** ptrs = (void**)malloc(sz);
    cudaMemcpy(ptrs, addr, sz, cudaMemcpyDeviceToHost);
    for (i = 0; i < n; i++)
        cudaFree(ptrs[i]);
    cudaFree(addr);
    free(ptrs);
}

__device__ void inc_float3(float3 *var, float3 inc)
// increase value of float3 variable
{
    var->x += inc.x;
    var->y += inc.y;
    var->z += inc.z;
}

__device__ void inc_float3_coef(float3* var, float3 inc, float k)
// increase value of float3 variable by inc with coefficien
{
    var->x += inc.x * k;
    var->y += inc.y * k;
    var->z += inc.z * k;
}


__device__ void atomic_incFloat3(float3* var, float3 inc)
// atomic increase value of float3 variable
{
    atomicAdd(&(var->x), inc.x);
    atomicAdd(&(var->y), inc.y);
    atomicAdd(&(var->z), inc.z);
}

__device__ int dev_npair(int n)
// number of pairs for n elements (device version)
{
    return n * (n - 1) / 2;
}

/*
__device__ float sqr_summ(float3 val)
// return x^2 + y^2 + z^2, where x, y and z are components of float3 value
{
    return val.x * val.x + val.y * val.y + val.z + val.z;
}
*/

__device__ float3 float3_dif(float3 a, float3 b)
// difference between two float3 values (return a - b)
{
    float3 res = make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    return res;
}

__device__ float float3_length(float3 vect)
//  return length of a vector presented as float3 variable
{
    return sqrt(vect.x * vect.x + vect.y * vect.y + vect.z * vect.z);
}

__device__ float float3_sqr(float3 vect)
// summ of squares for components
{
    return vect.x * vect.x + vect.y * vect.y + vect.z * vect.z;
}

__device__ float sc_prod(float3 a, float3 b)
// scalar production of a and b
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// random number generators:
__device__ unsigned int rand1()
// https://stackoverflow.com/questions/837955/random-number-generator-in-cuda
// ! не работает
{
    unsigned int m_w = 150;
    unsigned int m_z = 40;

    m_z = 36969 * (m_z & 65535) + (m_z >> 16);
    m_w = 18000 * (m_w & 65535) + (m_w >> 16);
    return unsigned int ((m_z << 16) + m_w) % 1000;   // 32-bit result
}

__device__ unsigned int rnd_xor128(uint4 &s)
// https://habr.com/ru/post/99876/ (XorShift)
// variant with blockIdx and threadIdx
{
    unsigned int t;

    //atomicAdd(&s.x, blockIdx.x * blockDim.x + threadIdx.x);
    t = s.x ^ (s.x << 11);

    s.x = s.y;
    s.y = s.z;
    s.z = s.w;

    s.w = (s.w ^ (s.w >> 19)) ^ (t ^ (t >> 8));
    atomicAdd(&s.w, blockIdx.x * blockDim.x + threadIdx.x);
    printf("sw=%d\n", s.w);

    return s.w;
}

__device__ unsigned int rnd_xor128_single(uint4& s)
// https://habr.com/ru/post/99876/ (XorShift)
// simple variant with individual buffer variable
{
    unsigned /*long long*/ int t;

    //atomicAdd(&s.x, blockIdx.x * blockDim.x + threadIdx.x);
    //s.x += blockIdx.x * blockDim.x + threadIdx.x;
    //if (s.x == 0)
      //  s.x = 1;
    t = s.x ^ (s.x << 11);

    s.x = s.y;
    s.y = s.z;
    s.z = s.w;

    s.w = (s.w ^ (s.w >> 19)) ^ (t ^ (t >> 8));
    //printf("s=%d t=%d x=%d y=%d z=%d sw=%d\n", sizeof(t), t, s.x, s.y, s.z, s.w);
    return s.w;
    
/*
    t = s.x;
    if (t == 0)
        t = 1;
    unsigned long long int const w = s.y;
    s.x = w;
    t ^= t << 23;		// a
    t ^= t >> 18;		// b -- Again, the shifts and the multipliers are tunable
    t ^= w ^ (w >> 5);	// c
    s.y = t;
    printf("w=%d t=%d\n", w, t);
    return t + w;
*/

}

__device__ float3 rand_usphere_single(uint4& var)
// get random vector on unit sphere
// based on Daan Frenkel "understanding of moleular dynamics..." p. 578
{
    float ran1, ran2;
    float ransq = 2.f;
    while (ransq > 1.f)
    {
        ran1 = 1.f - 2.f * (float)(rnd_xor128_single(var) % 128) / 127.f;     //rnd_xor128(var) % 128) / 127.f = random number from 0 to 1
        ran2 = 1.f - 2.f * (float)(rnd_xor128_single(var) % 128) / 127.f;     //rnd_xor128(var) % 128) / 127.f = random number from 0 to 1
        //printf("%f %f\n", ran1, ran2);
        ransq = ran1 * ran1 + ran2 * ran2;
    }

    float ranh = 2.f * sqrt(1.f - ransq);
    float x = ran1 * ranh;
    float y = ran2 * ranh;
    float z = (1.f - 2.f * ransq);

    return make_float3(x, y, z);
}

void cuda_info()
{
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("%s CC=%d.%d GlobMem=%IuGb shMem/Block=%IukB constMem=%IukB clockRate=%dMHz nMultProc=%d maxThrPerMP=%d\n", devProp.name, devProp.major, devProp.minor, devProp.totalGlobalMem / 1024 / 1024 / 1024, devProp.sharedMemPerBlock / 1024, devProp.totalConstMem / 1024, devProp.clockRate / 1000, devProp.multiProcessorCount, devProp.maxThreadsPerMultiProcessor);
    printf("maxBlock=%d\n", devProp.maxGridSize[0]);
}

