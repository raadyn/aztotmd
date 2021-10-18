//### cudaVDW functions #####
// Here and next fer_, fe_ and e_ functions must contfloatain the same code
//   prefexis fer_, fe_ and e_ means force and energy calculation by r also, force and energy and energy
#include <stdio.h>

#include "dataStruct.h"
#include "vdw.h"
#include "cuStruct.h"
#include "cuVdW.h"

/*
struct cudaVdW
{
    // all fields are constant during the simulation
    int type;
    float p0, p1, p2, p3, p4;
    float r2cut;    //square of cutoff
    float (*eng)(float r2, cudaVdW* vdw); // function to calculate energy
    float (*feng)(float r2, cudaVdW* vdw, float& eng); // function to calculate force (return) & energy (save in eng)
    float (*feng_r)(float r2, float& r, cudaVdW* vdw, float& eng); // function to calculate force (return) & energy (save in eng) if r may be knonw
    float (*eng_r)(float r2, float r, cudaVdW* vdw); // return energy by r and r^2
};
*/

__device__ float cu_fer_lj(float r2, float& r, cudaVdW* vdw, float& eng)
// calculate force and energy by Lennard-Jones pair potential: U = 4e[(s/r)^12 - (s/r)^6]
//  r2 - square of distance, vdw - parameters (the same for next PP functions)
{
    float r2i = 1.f / r2;
    float sr2 = vdw->p1 * r2i;
    float sr6 = sr2 * sr2 * sr2;

    eng += vdw->p0 * sr6 * (sr6 - 1.f);
    return vdw->p2 * r2i * sr6 * (2.f * sr6 - 1.f);
}

__device__ float cu_fe_lj(float r2, cudaVdW* vdw, float& eng)
// calculate force and energy by Lennard-Jones pair pfloatotential: U = 4e[(s/r)^12 - (s/r)^6]
//  r2 - square of distance, vdw - parameters (the same for next PP functions)
{
    float r2i = 1.f / r2;
    float sr2 = vdw->p1 * r2i;
    float sr6 = sr2 * sr2 * sr2;

    eng += vdw->p0 * sr6 * (sr6 - 1.f);
    return vdw->p2 * r2i * sr6 * (2.f * sr6 - 1.f);
}

__device__ float cu_e_lj(float r2, cudaVdW* vdw)
// calculate energy by Lennard-Jones pair potentialfloat
{
    float r2i = 1.f / r2;
    float sr2 = vdw->p1 * r2i;
    float sr6 = sr2 * sr2 * sr2;

    return vdw->p0 * sr6 * (sr6 - 1.f);
}

__device__ float cu_er_lj(float r2, float &r, cudaVdW* vdw)
// calculate energy by Lennard-Jones pair potential
{
    float r2i = 1.f / r2;
    float sr2 = vdw->p1 * r2i;
    float sr6 = sr2 * sr2 * sr2;

    return vdw->p0 * sr6 * (sr6 - 1.f);
}

__device__ float cu_fer_buck(float r2, float& r, cudaVdW* vdw, float& eng)
// calculate force and energy (with r) by Buckingham pair potential: U = A exp(-r/ro) - C/r^6
{
    //printf("begin cu_fer_buck\n");
    float r2i = 1.f / r2;
    float r4i = r2i * r2i;
    if (r == 0.f)    // calculate r if unkonwn (zero) and use it otherwise
        r = sqrt(r2);

    eng += vdw->p0 * exp(-r / vdw->p1) - vdw->p2 * r4i * r2i;
    return vdw->p0 * exp(-r / vdw->p1) / r / vdw->p1 - 6.f * vdw->p2 * r4i * r4i;
}

__device__ float cu_fe_buck(float r2, cudaVdW* vdw, float& eng)
// calculate force and energy by Buckingham pair potential: U = A exp(-r/ro) - C/r^6
{
    float r2i = 1.f / r2;
    float r = sqrt(r2);
    float r4i = r2i * r2i;

    eng += vdw->p0 * exp(-r / vdw->p1) - vdw->p2 * r4i * r2i;
    return vdw->p0 * exp(-r / vdw->p1) / r / vdw->p1 - 6.0 * vdw->p2 * r4i * r4i;
}

__device__ float cu_e_buck(float r2, cudaVdW* vdw)
// calculate energy by Buckingham pair potential: U = A exp(-r/ro) - C/r^6
{
    float r2i = 1.f / r2;
    float r = sqrt(r2);
    float r4i = r2i * r2i;

    return vdw->p0 * exp(-r / vdw->p1) - vdw->p2 * r4i * r2i;
}

__device__ float cu_er_buck(float r2, float &r, cudaVdW* vdw)
// calculate energy by Buckingham pair potential: U = A exp(-r/ro) - C/r^6
{
    float r2i = 1.f / r2;
    float r4i = r2i * r2i;

    if (r == 0.f)    // calculate r if unkonwn (zero) and use it otherwise
        r = sqrt(r2);

    return vdw->p0 * exp(-r / vdw->p1) - vdw->p2 * r4i * r2i;
}

__device__ float cu_fer_bmh(float r2, float& r, cudaVdW* vdw, float& eng)
// calculate force and energy (with r) by Born–Mayer–Huggins pair potential: U = Aexp[B(s-r)] - C/r^6 - D/r^8
{
    float r2i = 1.f / r2;
    float r4i = r2i * r2i;
    if (r == 0.f)    // calculate if unkonwn (zero) and use otherwise
        r = sqrt(r2);

    //printf("r(bmh)=%f\n", r);

    eng += vdw->p0 * exp(vdw->p1 * (vdw->p2 - r)) - vdw->p3 * r4i * r2i - vdw->p4 * r4i * r4i;
    return vdw->p0 * vdw->p1 * exp(vdw->p1 * (vdw->p2 - r)) / r - 6.f * vdw->p3 * r4i * r4i - 8.f * vdw->p4 * r4i * r4i * r2i;
}

__device__ float cu_fe_bmh(float r2, cudaVdW* vdw, float& eng)
// calculate force and energy by Born–Mayer–Huggins pair potential: U = Aexp[B(s-r)] - C/r^6 - D/r^8
{
    float r2i = 1.f / r2;
    float r = sqrt(r2);
    float r4i = r2i * r2i;

    eng += vdw->p0 * exp(vdw->p1 * (vdw->p2 - r)) - vdw->p3 * r4i * r2i - vdw->p4 * r4i * r4i;
    return vdw->p0 * vdw->p1 * exp(vdw->p1 * (vdw->p2 - r)) / r - 6.f * vdw->p3 * r4i * r4i - 8.f * vdw->p4 * r4i * r4i * r2i;
}

__device__ float cu_e_bmh(float r2, cudaVdW* vdw)
// calculate energy by Born–Mayer–Huggins pair potential: U = Aexp[B(s-r)] - C/r^6 - D/r^8
{
    float r2i = 1.f / r2;
    float r = sqrt(r2);
    float r4i = r2i * r2i;

    return vdw->p0 * exp(vdw->p1 * (vdw->p2 - r)) - vdw->p3 * r4i * r2i - vdw->p4 * r4i * r4i;
}

__device__ float cu_er_bmh(float r2, float &r, cudaVdW* vdw)
// calculate energy by Born–Mayer–Huggins pair potential: U = Aexp[B(s-r)] - C/r^6 - D/r^8
{
    float r2i = 1.f / r2;
    float r4i = r2i * r2i;

    if (r == 0.f)    // calculate r if unkonwn (zero) and use it otherwise
        r = sqrt(r2);

    return vdw->p0 * exp(vdw->p1 * (vdw->p2 - r)) - vdw->p3 * r4i * r2i - vdw->p4 * r4i * r4i;
}

__device__ float cu_fer_elin(float r2, float& r, cudaVdW* vdw, float& eng)
// calculate force and energy by "elin" potential: U = A * exp(-x/ro) + C*x
//  r2 - square of distance, vdw - parameters (the same for next PP functions)
{
    if (r == 0.f)    // calculate r if unkonwn (zero) and use it otherwise
        r = sqrt(r2);

    eng += vdw->p0 * exp(-r / vdw->p1) + vdw->p2 * r;
    return vdw->p0 * exp(-r / vdw->p1) / r / vdw->p1 - vdw->p2 / r;
}

__device__ float cu_fe_elin(float r2, cudaVdW* vdw, float& eng)
// calculate force and energy by "elin" potential: U = A * exp(-x/ro) + C*x
//  r2 - square of distance, vdw - parameters (the same for next PP functions)
{
    float r = sqrt(r2);

    eng += vdw->p0 * exp(-r / vdw->p1) + vdw->p2 * r;
    return vdw->p0 * exp(-r / vdw->p1) / r / vdw->p1 - vdw->p2 / r;
}

__device__ float cu_e_elin(float r2, cudaVdW* vdw)
// calculate energy by "elin" potential: U = A * exp(-x/ro) + C*x
{
    float r = sqrt(r2);

    return vdw->p0 * exp(-r / vdw->p1) + vdw->p2 * r;
}

__device__ float cu_er_elin(float r2, float &r, cudaVdW* vdw)
// calculate energy by "elin" potential: U = A * exp(-x/ro) + C*x
{
    if (r == 0.f)    // calculate r if unkonwn (zero) and use it otherwise
        r = sqrt(r2);
    return vdw->p0 * exp(-r / vdw->p1) + vdw->p2 * r;
}


__device__ float cu_fer_einv(float r2, float& r, cudaVdW* vdw, float& eng)
// calculate force and energy by "einv" potential: U = A * exp(-x/ro) - C/x
//  r2 - square of distance, vdw - parameters (the same for next PP functions)
{
    if (r == 0.f)    // calculate r if unkonwn (zero) and use it otherwise
        r = sqrt(r2);
    eng += vdw->p0 * exp(-r / vdw->p1) - vdw->p2 / r;
    return vdw->p0 * exp(-r / vdw->p1) / r / vdw->p1 - vdw->p2 / r / r2;
}

__device__ float cu_fe_einv(float r2, cudaVdW* vdw, float& eng)
// calculate force and energy by "einv" potential: U = A * exp(-x/ro) - C/x
//  r2 - square of distance, vdw - parameters (the same for next PP functions)
{
    float r = sqrt(r2);

    eng += vdw->p0 * exp(-r / vdw->p1) - vdw->p2 / r;
    return vdw->p0 * exp(-r / vdw->p1) / r / vdw->p1 - vdw->p2 / r / r2;
}

__device__ float cu_e_einv(float r2, cudaVdW* vdw)
// calculate energy by "einv" potential: U = A * exp(-x/ro) - C/x
{
    float r = sqrt(r2);

    return vdw->p0 * exp(-r / vdw->p1) - vdw->p2 / r;
}

__device__ float cu_er_einv(float r2, float &r, cudaVdW* vdw)
// calculate energy by "einv" potential: U = A * exp(-x/ro) - C/x
{
    if (r == 0.f)    // calculate r if unkonwn (zero) and use it otherwise
        r = sqrt(r2);
    return vdw->p0 * exp(-r / vdw->p1) - vdw->p2 / r;
}

__device__ float surk_pot(float r2, float rad1, float rad2, cudaVdW* vdw, float& eng)
// potential derived by Platon Surkov:
//   U = ri*rj*(C1 ri^2 rj^2 / rij^7 - C2 / (ki*ri + kj * rj) / r^6
//   p0 = C1, p1 = C2, p2 = ki, p3 = kj
{
    float Ñ2ir_sum = vdw->p1 / (vdw->p2 * rad1 + vdw->p3 * rad2);   // C2 / (ka + lb)
    float r_prod = rad1 * rad2;
    float C1ab2 = r_prod * r_prod * vdw->p0;        // C1 * a^2 b^2
    float r6 = r2 * r2 * r2;
    float r = sqrt(r2);
    float ir6 = 1.f / r6;
    float ir = 1.f / r;

    float val = r_prod * ir6 * (C1ab2 * ir - Ñ2ir_sum);
    //printf("U=%f: ra=%f rb=%f (%f %f %f %f)\n", val, rad1, rad2, vdw->p0, vdw->p1, vdw->p2, vdw->p3);

    eng += val;
    return r_prod * ir6 / r2 * (7.f * C1ab2 * ir - 6.f * Ñ2ir_sum);
}


__global__ void define_vdw_func(cudaMD* md)
{
    cudaVdW* vdw;
    vdw = &(md->pairpots[threadIdx.x]);

    switch (vdw->type)
    {
    case lj_type:
        vdw->eng = &cu_e_lj;
        vdw->eng_r = &cu_er_lj;
        vdw->feng = &cu_fe_lj;
        vdw->feng_r = &cu_fer_lj;
        break;
    case bh_type:
        vdw->eng = &cu_e_buck;
        vdw->eng_r = &cu_er_buck;
        vdw->feng = &cu_fe_buck;
        vdw->feng_r = &cu_fer_buck;
        break;
    case BHM_type:
        vdw->eng = &cu_e_bmh;
        vdw->eng_r = &cu_er_bmh;
        vdw->feng = &cu_fe_bmh;
        vdw->feng_r = &cu_fer_bmh;
        break;
    case elin_type:
        vdw->eng = &cu_e_elin;
        vdw->eng_r = &cu_er_elin;
        vdw->feng = &cu_fe_elin;
        vdw->feng_r = &cu_fer_elin;
        break;
    case einv_type:
        vdw->eng = &cu_e_einv;
        vdw->eng_r = &cu_er_einv;
        vdw->feng = &cu_fe_einv;
        vdw->feng_r = &cu_fer_einv;
        break;
    case surk_type:
        vdw->radi_func = &surk_pot;
        break;

    }
}