#ifndef CUTEMP_H
#define CUTEMP_H

const int nUvect = 3072;
__constant__ const int dnUvect = 3072;
__constant__ const int ctTermRadi = 2;    // radiative thermostat (prefix ct means "Cuda type")

void init_cuda_tstat(int nAt, Atoms* atm, Field* fld, TStat* tstat, cudaMD* hmd, hostManagMD* man);
void apply_pre_tstat(int iStep, TStat* tstat, Sim* sim, cudaMD* devMD, hostManagMD* man);
void apply_tstat(int iStep, TStat* tstat, Sim* sim, cudaMD* devMD, hostManagMD* man);
void free_cuda_tstat(TStat* tstat, Field* fld, cudaMD* hmd);

__global__ void before_nose(cudaMD* md);
__global__ void tstat_nose(int atPerBlock, int atPerThread, cudaMD* md);
__global__ void after_nose(int refresh_kin, cudaMD* md);

__global__ void zero_engKin(cudaMD* md);



#endif // CUTEMP_H
