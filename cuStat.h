#ifndef CUSTAT_H
#define CUSTAT_H

//__global__ void write_stat(int iStep, int ind0, int shift, cudaMD* md);
void start_stat(hostManagMD* man, Field* fld, Sim* sim);
void end_stat(hostManagMD* man, Field* fld, Sim* sim, cudaMD* hmd, double dt);
void stat_iter(int step, hostManagMD* man, statStruct* stat, cudaMD* dmd, cudaMD* hmd, double dt);


//void copy_stat(FILE* f, cudaMD* hmd, hostManagMD* man, int size, int type_shift, int nstep, int ndata, int dstep, double dtime, int step0, double time0);
void init_cuda_stat(cudaMD* hmd, hostManagMD* man, Sim* sim, Field* fld);
__global__ void prepare_stat_addr(cudaMD* md);
void free_cuda_stat(cudaMD* hmd, hostManagMD* man);

void init_cuda_rdf(Field* fld, Sim* sim, hostManagMD* man, cudaMD* hmd);
void free_cuda_rdf(hostManagMD* man, cudaMD* hmd);
void rdf_iter(int step, Field* fld, Sim* sim, hostManagMD* man, cudaMD* hmd, cudaMD* dmd);
void copy_rdf(Field* fld, Sim* sim, hostManagMD* man, cudaMD* hmd, char* fname);

// nuclei rdf functions:
void init_cuda_nrdf(Field* fld, Sim* sim, hostManagMD* man, cudaMD* hmd);
void free_cuda_nrdf(hostManagMD* man, cudaMD* hmd);
__global__ void brute_nrdf(int nSpec, int nNucl, int nPair, int n_nPair, float idRDF, float r2max, cudaMD* md);
void copy_nrdf(Field* fld, Sim* sim, hostManagMD* man, cudaMD* hmd, char* fname, char* nfname);
void nrdf_iter(int step, Field* fld, Sim* sim, hostManagMD* man, cudaMD* hmd, cudaMD* dmd);

// trajectories
void init_cuda_trajs(Atoms* atm, cudaMD* hmd, hostManagMD* man);
void start_traj(Atoms* atm, hostManagMD* man, Field* fld, Sim* sim);
void traj_iter(int step, hostManagMD* man, cudaMD* dmd, cudaMD* hmd, Sim* sim, Atoms* atm);
void end_traj(hostManagMD* man, cudaMD* hmd, Sim* sim, Atoms* atm);
void free_cuda_trajs(cudaMD* hmd, hostManagMD* man);

// "bind" trajectories
void init_cuda_bindtrajs(Sim* sim, cudaMD* hmd, hostManagMD* man);
void start_bindtraj(hostManagMD* man, Field* fld, Sim* sim);
void bindtraj_iter(int step, hostManagMD* man, cudaMD* dmd, cudaMD* hmd, Sim* sim, Box* bx);
void end_bindtraj(hostManagMD* man, cudaMD* hmd, Sim* sim, Box* bx);
void free_cuda_bindtrajs(cudaMD* hmd, hostManagMD* man);


#endif
