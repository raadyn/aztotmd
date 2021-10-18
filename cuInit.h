#ifndef CUINIT_H
#define CUINIT_H

cudaMD* init_cudaMD(Atoms* atm, Field* fld, Sim* sim, TStat* tstat, Box* bx, Elec* elec, hostManagMD* man, cudaMD *h_md);

void md_to_host(Atoms* atm, Field* fld, cudaMD* hmd, cudaMD* dmd, hostManagMD* man);
void free_device_md(cudaMD* dmd, hostManagMD* man, Sim* sim, Field* fld, TStat* tstat, cudaMD* hmd);



#endif // CUINIT_H