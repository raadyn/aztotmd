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

__device__ void atomic_incFloat3(float3* var, float3 inc)
// atomic increase value of float3 variable
{
    atomicAdd(&(var->x), inc.x);
    atomicAdd(&(var->y), inc.y);
    atomicAdd(&(var->z), inc.z);
}

__device__ int devNpairs(int n)
// number of pairs for n elements
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
// difference between to float3 values (return a - b)
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
{
    unsigned int t;

    //atomicAdd(&s.x, blockIdx.x * blockDim.x + threadIdx.x);
    t = s.x ^ (s.x << 11);

    s.x = s.y;
    s.y = s.z;
    s.z = s.w;

    s.w = (s.w ^ (s.w >> 19)) ^ (t ^ (t >> 8));
    atomicAdd(&s.w, blockIdx.x * blockDim.x + threadIdx.x);

    return s.w;
}

void cuda_info()
{
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("%s CC=%d.%d GlobMem=%IuGb shMem/Block=%IukB constMem=%IukB clockRate=%dMHz nMultProc=%d maxThrPerMP=%d\n", devProp.name, devProp.major, devProp.minor, devProp.totalGlobalMem / 1024 / 1024 / 1024, devProp.sharedMemPerBlock / 1024, devProp.totalConstMem / 1024, devProp.clockRate / 1000, devProp.multiProcessorCount, devProp.maxThreadsPerMultiProcessor);
    printf("maxBlock=%d\n", devProp.maxGridSize[0]);
    // вроде 64 процессора в мультипроцессоре этой видеокарты(4352 ядра)
}

