#include "cuda_runtime.h"

#include "cuStruct.h"
#include "dataStruct.h"
#include "dataStruct.h"
#include "cuTemp.h"
#include "temperature.h"
#include "cuUtils.h"
#include "utils.h"

void init_cuda_tstat(int nAt, Atoms *atm, Field *fld, TStat *tstat, cudaMD *hmd, hostManagMD *man)
{
    int i, t, size;

    hmd->tstat = tstat->type;
    hmd->temp = (float)tstat->Temp;
    hmd->teKin = (float)tstat->tKin;

    switch (tstat->type)
    {
    case tpTermNose:
        hmd->chit = 0.f;
        hmd->consInt = 0.f;
        hmd->rQmass = (float)tstat->rQmass;
        hmd->qMassTau2 = (float)tstat->qMassTau2;
        break;
    case tpTermRadi:
        hmd->curEng = 0;
        hmd->curVect = 0;
        man->tstat_sign = 1;

        float *phs = (float*)malloc(nAt * float_size);
        float3 *vecs = (float3*)malloc(nAt * float3_size);
        float* engs = (float*)malloc(nAt * float_size);
        int* radstep = (int*)malloc(nAt * int_size);
        for (i = 0; i < nAt; i++)
        {
            phs[i] = (float)tstat->photons[i];
            vecs[i] = make_float3((float)tstat->randVx[i], (float)tstat->randVy[i], (float)tstat->randVz[i]);
            engs[i] = 0.f;
            radstep[i] = 0;
        }
        data_to_device((void**)(&hmd->engPhotons), phs, nAt * float_size);
        data_to_device((void**)(&hmd->randVects), vecs, nAt * float3_size);
        data_to_device((void**)(&hmd->engs), engs, nAt * float_size);
        data_to_device((void**)(&hmd->radstep), radstep, nAt * int_size);
        free(phs);
        free(vecs);
        free(engs);
        free(radstep);

        //preset unit vectors:
        float3* uvs = (float3*)malloc(nUvect * float3_size);
        for (i = 0; i < nUvect; i++)
            uvs[i] = make_float3((float)tstat->uvectX[i], (float)tstat->uvectY[i], (float)tstat->uvectZ[i]);
        data_to_device((void**)(&hmd->uvects), uvs, nUvect * float3_size);
        free(uvs);

        // for output statistics
        hmd->numbPhEngBin = 20;
        hmd->phEngBin = (float)tstat->mxEng / (float)hmd->numbPhEngBin;
        size = hmd->numbPhEngBin * int_size;
        int* numbs = (int*)malloc(size);
        for (i = 0; i < hmd->numbPhEngBin; i++)
            numbs[i] = 0;
        data_to_device((void**)(&hmd->phEngs), numbs, size);

        break;
    }

    if (tstat->type != tpTermRadi && fld->is_tdep)
        printf("WARNING[b011] Temperature-dependent potentials are used without radiative thermostat! In this case potentials will not depend on temperature!\n");

    if (tstat->type == tpTermRadi || fld->is_tdep)
    {
        float* radii = (float*)malloc(nAt * float_size);
        for (i = 0; i < nAt; i++)
        {
            t = atm->types[i];
            if (fld->species[t].radB != 0.0)
                radii[i] = float(fld->species[t].radA / fld->species[t].radB);
            else
                radii[i] = 1.f;
            //radii[i] = 0.577f + rand01() * 0.0001f;    // values must be initialized to avoid division by zero in some radii-dependet pair potential
            //radii[i] = 0.577f;
            //if (i % 200 == 0)
              //  radii[i] = 0.578;
            //radii[i] = 0.6f;
        }
        data_to_device((void**)(&hmd->radii), radii, nAt * float_size);
        free(radii);
    }
}

void free_cuda_tstat(TStat* tstat, Field* fld, cudaMD* hmd)
{
    if (tstat->type == tpTermRadi)
    {
        cudaFree(hmd->engPhotons);
        cudaFree(hmd->randVects);
        cudaFree(hmd->engs);
        //cudaFree(hmd->sort_engs);
        cudaFree(hmd->uvects);
        cudaFree(hmd->phEngs);
    }

    if (tstat->type == tpTermRadi || fld->is_tdep)
        cudaFree(hmd->radii);
}

__global__ void temp_scale(int atPerBlock, int atPerThread, cudaMD* md)
// naive scale velocities to target kinetic energy
{
    int i;
    float k, c;

    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);

    if (md->engKin == 0.f)
        return;

    // correction for radiative thermostat (according to our theory the correct kinetic energy at temperature T is sufficienlty lower that it follows from MKT)
    if (md->tstat == tpTermRadi)
        c = 0.25f;
    else
        c = 1.f;
    k = sqrt(c * md->teKin / md->engKin);    //! it the same value for all threads
    /*
    if (blockIdx.x == 0)
        if (threadIdx.x == 0)   
        {
            printf("vel sclaling k=%f\n", k);
        }
    */

    for (i = id0; i < N; i++)
    {
        md->vls[i].x *= k;
        md->vls[i].y *= k;
        md->vls[i].z *= k;
#ifdef DEBUG_MODE
        atomicAdd(&(md->nCult[i]), 1);
#endif 
    }
    
    /*
    //set kinetic energy equal to the target value
    if (blockIdx.x == 0)
        if (threadIdx.x == 0)   // 0th thread
        {
            md->engKin = md->teKin;
        }

     */
}
//end 'temp_scale' function

__global__ void after_tscale(cudaMD* md)
// reset kinetic energy to target value
{
    md->engKin = md->teKin;
}

__global__ void before_nose(cudaMD* md)
// some general calculations for appliyng Nose-Hoover thermostat scaling
{
#ifdef DEBUG1_MODE
    float old_chit = md->chit;
#endif
    md->chit += md->tSt * (md->engKin - md->teKin) * md->rQmass;
    md->tscale = 1.f - md->tSt * md->chit;
    //printf("tscale=%f\n", md->tscale);
#ifdef DEBUG1_MODE
    if ((md->tscale > 4.f)||(md->tscale < 0.f))
    {
        printf("tscale=%f chit=%f->%f Kin=%f int=%f | targ Kin=%f rQmass=%f\n", md->tscale, old_chit, md->chit, md->engKin, md->consInt, md->teKin, md->rQmass);
        if (md->tscale > 4.f)
            md->tscale = 0.98f;
        if (md->tscale < 0.f)
            md->tscale = 0.2;
        md->chit = 0.f;
    }
#endif
}

__global__ void tstat_nose(int atPerBlock, int atPerThread, cudaMD* md)
// apply Nose-Hoover thermostat, save new kinetic energy in md struct
{
    int i;

    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);
#ifdef DEBUG1_MODE
    float3 old_vel;
#endif

    //if (threadIdx.x == 0)
      //  if (blockIdx.x == 0)
        //    printf("tscale=%f\n", md->tscale);

    for (i = id0; i < N; i++)
    {
#ifdef DEBUG1_MODE
        old_vel = md->vls[i];
#endif
        md->vls[i].x *= md->tscale;
        md->vls[i].y *= md->tscale;
        md->vls[i].z *= md->tscale;
#ifdef DEBUG1_MODE
        if ((md->vls[i].x > MX_VEL) || (md->vls[i].x < -MX_VEL))
            printf("tstat_nose: vls[%d].x=%f old_vel=%f tscale=%f\n", i, md->vls[i].x, old_vel.x, md->tscale);
#endif
#ifdef DEBUG_MODE
        md->nCult[i]++;
#endif
    }
}
// end 'tstat_nose' function

__global__ void after_nose(int refresh_kin, cudaMD* md)
// some general calculations after appliyng Nose-Hoover thermostat scaling
//! maybe remove this function to calc_quantites() ???
{
    if (refresh_kin)
        md->engKin = md->engKin * md->tscale * md->tscale;
    md->consInt += md->tSt * md->chit * md->qMassTau2;
    md->chit += md->tSt * (md->engKin - md->teKin) * md->rQmass; // new kinetic energy (отличие от первого действия этой процедуры)

#ifdef DEBUG_MODE
    int i;
    if (threadIdx.x == 0)
        for (i = 1; i < md->nBndTypes; i++)
            if ((md->bondTypes[i].spec1 < 0) || (md->bondTypes[i].spec2 < 0) || (md->bondTypes[i].spec1 >= MX_SPEC) || (md->bondTypes[i].spec2 >= MX_SPEC))
            {
                printf("aft after_nose: bnd[%d] spec1=%d spec2=%d\n", i, md->bondTypes[i].spec1, md->bondTypes[i].spec2);
                md->xyz[9999999999].x = 15.f;  // crash cuda
            }
#endif
}

__global__ void zero_engKin(cudaMD* md)
// reset kinetic energy to 0
{
    md->engKin = 0.f;
}


/*
__device__ inline unsigned int rand1(unsigned int& seed)
{
    seed = seed * 1664525 + 1013904223UL;
    return seed;
}
*/
/*
__global__ void get_random(cudaMD *md)
{
    int i, j;
    //unsigned int seed = blockIdx.x * blockDim.x + threadIdx.x;
    //int k = rand1(seed);
    //int k = 10;
    int k = (5 * 1664525 + 1013904223UL) % 9;

    __shared__ int int_mom, int_vect;
    __shared__ int block_mom, block_vect;
    switch (threadIdx.x)
    {
    case 0:
        for (i = 0; i < k; i++)
            md->rnd++;
        block_mom = atomicAdd(&md->idMom, 1);
        break;
    case 1:
        block_vect = atomicAdd(&md->idVect, 1);
        break;
    case 2:
        for (i = 0; i < k; i++)
            md->rnd++;
        int_mom = 0;
        break;
    case 3:
        for (i = 0; i < k; i++)
            md->rnd++;
        int_vect = 0;
        break;
    }
    __syncthreads(); 

    for (i = 0; i < k; i++)
        j++;
    int mom = atomicAdd(&int_mom, 1);
    for (i = 0; i < k; i++)
        j++;
    int vect = atomicAdd(&int_vect, 1);
    if (blockIdx.x == 0)
        if (threadIdx.x == 0)
            printf("block: %d mom: %d vect:%d int_mom:%d int_vect:%d    k=%d\n", blockIdx.x, block_mom, block_vect, mom, vect, k);
}
*/

__constant__ const float revLight = 3.33567e-5f; // 1/c, where c is lightspeed, 2.9979e4 A/ps
//__constant__ const float Light = 2.9979e4f;      // lightspeed, 2.9979e4 A/ps
//__constant__ const float revPlank = 241.55f;    // 1 / plank constant (4.14 eV*ps)
__constant__ const float numPi = 3.14159f;    // pi

__device__ float3 rand_uvect(uint4 &var, cudaMD* md)
{
    int rnd = rnd_xor128(var) % dnUvect;
    return md->uvects[rnd];

/*
    float3 res;
    double x, y, z, cost;
    unsigned int rnd, rnd1;
    rnd = rnd_xor128(var) % 2;
    //rnd = (rnd_xor128(md->ui4rnd) + blockIdx.x * blockDim.x + threadIdx.x) % 16;
    rnd1 = rnd_xor128(var) % 4;
    //rnd1 = (rnd_xor128(md->ui4rnd) + blockIdx.x * blockDim.x + threadIdx.x) % 16;

    double theta = 3.14159265 * double(rnd) / 2.0;
    double phi = 2 * 3.14159265 * double(rnd1) / 4.0;
    sincos(theta, &z, &cost);
    sincos(phi, &y, &x);
    x *= cost;
    y *= cost;
    res = make_float3((float)x, (float)y, (float)z);
*/

    /*
    rnd = rnd_xor128(var) % 2;
    if (rnd)
    {
        res = make_float3(1.f, 0.f, 0.f);
    }
    else
        res = make_float3(-1.f, 0.f, 0.f);
    */
    //printf("[%d, %d] rnd=%d rnd2=%d theta=%f phi=%f(%f, %f, %f)\n", blockIdx.x, threadIdx.x, rnd, rnd1, theta, phi, res.x, res.y, res.z);
    //return res;
}

__device__ float3 rand_usphere(uint4& var, cudaMD* md)
// get random vector on unit sphere
// from Frenkel p. 578
{
    float ran1, ran2;
    float ransq = 2.f;
    while (ransq > 1.f)
    {
        ran1 = 1.f - 2.f * (rnd_xor128(var) % 128) / 127.f;     //rnd_xor128(var) % 128) / 127.f = random number from 0 to 1
        ran2 = 1.f - 2.f * (rnd_xor128(var) % 128) / 127.f;     //rnd_xor128(var) % 128) / 127.f = random number from 0 to 1
        ransq = ran1 * ran1 + ran2 * ran2;
    }

    float ranh = 2.f * sqrt(1.f - ransq);
    float x = ran1 * ranh;
    float y = ran2 * ranh;
    float z = (1.f - 2.f * ransq);

    return make_float3(x, y, z);
}

__device__ float3 rand_neg_vect(float3 vect, uint4& var /*cudaMD* md*/)
// generate random unit vector at an obtuse angle to the given one
// use that x1*x2 + y1*y2 + z1*z2 < 0 (there x,y and z are vectors component, 1 and 2 - given and resulting vector)
// only for non-zero vectors!
{
    float3 res;
    unsigned int rnd, rnd1, rnd2;

    rnd = rnd_xor128(var) % 64 + 1;  // 1 - 64
    rnd1 = rnd_xor128(var) % 64 + 1;  // 1 - 64
    rnd2 = rnd_xor128(var) % 64 + 1;  // 1 - 64

    if (vect.x > 0.f)
        res.x = -1.f * (float)rnd;
    else
        res.x = (float)rnd;

    if (vect.y > 0.f)
        res.y = -1.f * (float)rnd1;
    else
        res.y = (float)rnd1;

    if (vect.z > 0.f)
        res.z = -1.f * (float)rnd2;
    else
        res.z = (float)rnd2;

    double leng = float3_length(res);
    res.x /= leng;
    res.y /= leng;
    res.z /= leng;

    //printf("vls(%f %f %f) and res(%f %f %f) leng=%f\n", vect.x, vect.y, vect.z, res.x, res.y, res.z, leng);

    //printf("[%d, %d] rnd=%d rnd2=%d theta=%f phi=%f(%f, %f, %f)\n", blockIdx.x, threadIdx.x, rnd, rnd1, theta, phi, res.x, res.y, res.z);
    return res;
}

__device__ float3 rand_pos_vect(float3 vect, uint4& var /*cudaMD* md*/)
// generate random unit vector at an obtuse angle to the given one
// use that x1*x2 + y1*y2 + z1*z2 < 0 (there x,y and z are vectors component, 1 and 2 - given and resulting vector)
// only for non-zero vectors!
{
    float3 res;
    unsigned int rnd, rnd1, rnd2;

    rnd = rnd_xor128(var) % 64 + 1;  // 1 - 64
    rnd1 = rnd_xor128(var) % 64 + 1;  // 1 - 64
    rnd2 = rnd_xor128(var) % 64 + 1;  // 1 - 64

    if (vect.x < 0.f)
        res.x = -1.f * (float)rnd;
    else
        res.x = (float)rnd;

    if (vect.y < 0.f)
        res.y = -1.f * (float)rnd1;
    else
        res.y = (float)rnd1;

    if (vect.z < 0.f)
        res.z = -1.f * (float)rnd2;
    else
        res.z = (float)rnd2;

    double leng = float3_length(res);
    res.x /= leng;
    res.y /= leng;
    res.z /= leng;

    //printf("vls(%f %f %f) and res(%f %f %f) leng=%f\n", vect.x, vect.y, vect.z, res.x, res.y, res.z, leng);

    //printf("[%d, %d] rnd=%d rnd2=%d theta=%f phi=%f(%f, %f, %f)\n", blockIdx.x, threadIdx.x, rnd, rnd1, theta, phi, res.x, res.y, res.z);
    return res;
}

__global__ void laser_cooling(int sign, int atPerBlock, int atPerThread, cudaMD* md)
// adsorb photon with direction to atom and then radiate photon in random direction
{
    __shared__ int indEng, indVect;
    //__shared__ uint4 randVar;
    if (threadIdx.x == 0)
    {
        indEng = atomicAdd(&(md->curEng), 1);   // get current index in photon energies array
        indVect = atomicAdd(&(md->curVect), 1);   // get current index in photon energies array
        //randVar = md->ui4rnd;
    }
    __syncthreads();
    uint4 randVar = md->ui4rnd;

    double rmc, leng;
    double pe;      // photon energy
    double vls0;
    int i, e0, v0;
    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);

    e0 = indEng * atPerBlock + threadIdx.x * atPerThread - id0;
    v0 = indVect * atPerBlock + threadIdx.x * atPerThread - id0;

    int ei, vi;
    float3 vect;    // vector for velocity adding

    // v = v +– E/(c*m) * u
    //! пока принебрежем рандомизацией внтури блока
    for (i = id0; i < N; i++)
    {
        ei = e0 + i;
        vi = v0 + i;
        if (ei >= md->nAt)
            ei -= md->nAt;
        if (vi >= md->nAt)
            vi -= md->nAt;
        rmc = revLight / md->masses[i];
        vls0 = md->vls[i].x;
        if (rmc * md->engPhotons[ei] * md->randVects[vi].x > 1.f)
            printf("too high momentum: rmc=%f photon[%d]=%f rand[%d]=%f i=%d id0=%d\n", rmc, e0 + i, md->engPhotons[ei], v0 + i, md->randVects[vi].x, i, id0);

        //if (i == 0)
          //  printf("therm eng[0]=%f sign=%d\n", md->engs[i], sign);

        //if (i == 0)
          //  printf("rand=%d\n", rnd_xor128(md->ui4rnd) % 10);
        
        if (sign == 1)
        {
            //if (md->engs[i] < 4.2)
            {
                //vect = md->randVects[vi];
                //vect = rand_uvect(randVar);
                pe = md->engPhotons[ei];
                md->engs[i] += pe;
                leng = float3_length(md->vls[i]);
                if (leng > 0.f)
                    vect = make_float3(-md->vls[i].x / leng, -md->vls[i].y / leng, -md->vls[i].z / leng);
            }
        }
        else
        {
            //continue;
            if (md->engs[i] > 0.f) // 64.2 = enthalpy of argon at 298 K and 1 atm in eV/particle
            {
                pe = md->engs[i];
                md->engs[i] = 0.f;
                vect = rand_uvect(randVar, md);
            }
            else
                continue;
            // единичный вектор скорости, противоположный данному движению атома
            //leng = float3_length(md->vls[i]);
            //vect = make_float3(-md->vls[i].x / leng, -md->vls[i].y / leng, -md->vls[i].z / leng);
            //vect = md->randVects[vi];
            vect = rand_uvect(randVar, md);
        }

        // sign убрал из произведения, он определяет вектор направления
        md->vls[i].x = md->vls[i].x + rmc * pe * vect.x;
        md->vls[i].y = md->vls[i].y + rmc * pe * vect.y;
        md->vls[i].z = md->vls[i].z + rmc * pe * vect.z;
        //printf("%d->%d,%d(%d, %d)) rmc=%f e=%f, v=%f: vel=%f -> %f\n", i, e0 + i, v0 + i, e0, v0, rmc, md->engPhotons[e0 + i], md->randVects[v0 + i].x, vls0, md->vls[i].x);
        if (isnan(md->vls[i].x))
            printf("v[%d]=%f -> %f, rmc=%f e=%f, v=%f sign=%d leng=%f\n", i, vls0, md->vls[i].x, rmc, md->engPhotons[ei], vect.x, sign, leng);
    }
    if (threadIdx.x == 0)
        rnd_xor128(md->ui4rnd);

}


__device__ float3 get_angled_vector(float3 invec, float cos_phi, float theta)
// return unit vector at angle from given and with rotation angle theta
// phi is angle between old and result vector
// theta is any angle
{
    float3 v1, v2, v3;
    float leng1 = float3_length(invec);
    //! только для ненулевых векторов!
    v1 = make_float3(invec.x / leng1, invec.y / leng1, invec.z / leng1);

    // find v2, which is perpendicular to v1
    if (v1.x != 0.f)
    {
        v2.y = 1.f; v2.z = 1.f; // any coordinates
        v2.x = -(v1.y * v2.y + v1.z * v2.z) / v1.x;
    }
    else
        if (v1.y != 0.f)
        {
            v2.x = 1.f; v2.z = 1.f; // any coordinates
            v2.y = -(v1.z * v2.z) / v1.y;     // a1 = 0 !
        }
        else // a1=0, b1=0, c1 <> 0:
        {
            v2.x = 1.f; v2.y = 0.f; v2.z = 0.f;
        }

    // v3 is perpendicular to both v1 and v2
    v3.x = v1.y * v2.z - v1.z * v2.y;
    v3.y = -v1.x * v2.z + v1.z * v2.x;
    v3.z = v1.x * v2.y - v1.y * v2.x;

    float leng2 = float3_length(v2);
    float leng3 = float3_length(v3);


    v2.x /= leng2;
    v2.y /= leng2;
    v2.z /= leng2;

    v3.x /= leng3;
    v3.y /= leng3;
    v3.z /= leng3;

    //printf("verifiyng: v1^2=%f v2^2=%f v3^2=%f v1*v2=%f v1*v3=%f v2*v3=%f\n", float3_leng(v1), float3_leng(v2), float3_leng(v1), v1.x * v2.x + v1.y * v2.y + v1.z * v2.z, v1.x * v3.x + v1.y * v3.y + v1.z * v3.z, v3.x * v2.x + v3.y * v2.y + v3.z * v2.z);

    float sinPhi, sinTh, cosTh; //, cosPhi;
    //sincos(phi, &sinPhi, &cosPhi);
    sinPhi = sqrt(1 - cos_phi * cos_phi);
    sincos(theta, &sinTh, &cosTh);

    //float x0 = cos_phi;// cosPhi;
    //float y0 = sinPhi * sinTh;
    //float z0 = sinPhi * cosTh;

    // parameteric equation of circle (+ point at initial vector, but with length of cosine phi):
    //r = r0 + R cos φ i1 / |i1 | +R sin φ j1 / |j1 |
    float x = v1.x * cos_phi + sinPhi * (cosTh * v2.x + sinTh * v3.x);
    float y = v1.y * cos_phi + sinPhi * (cosTh * v2.y + sinTh * v3.y);
    float z = v1.z * cos_phi + sinPhi * (cosTh * v2.z + sinTh * v3.z);

    float3 res = make_float3(x, y, z);
    return res;
}

__device__ int modify_vel(float3* vel, float mass, int sign, float eng, uint4 &rnd_var)
// modify atom velocity by radiation/adsorption of photon
{
    float rm = 1.f / mass;
    float v0 = float3_length(*vel);

    //! отдельно обработать ситуацию, когда v0 == 0
    float in_sqrt = v0 * v0 + sign * 2.f * rm * eng;
    if (in_sqrt < 0.f)
        return 0;
    float v1 = sqrt(in_sqrt);
    // cosine theorem:
    //float cosPhi = (mass * mass * (in_sqrt + v0 * v0) - eng * eng * revLight * revLight) / (2.f * mass * mass * v0 * v1); // in_sqrt = v1^2
    float cosPhi = (mass * mass * v1 * v1 + mass * mass * v0 * v0  - eng * eng * revLight * revLight) / (2.f * mass * mass * v0 * v1); // in_sqrt = v1^2
    if ((cosPhi < -1.f) || (cosPhi > 1.f))
    {
        printf("wrong cosine: %f v0=%f v1=%f mass=%f ph_eng=%f kin0=%f sign=%d ph_mom=%f mv=%f\n", cosPhi, v0, v1, mass, eng, mass * v0 * v0, sign, eng*revLight, mass * v0);
        return 0;
    }
    printf("ok\n");


    // random angle from 0 to 2PI:
    float theta = (rnd_xor128(rnd_var) % 32) / 16 * 3.1415926f;
    float3 new_v = get_angled_vector(*vel, cosPhi, theta);

    vel->x = new_v.x * v1;
    vel->y = new_v.y * v1;
    vel->z = new_v.z * v1;

    return 1;
}

__device__ void adsorb_rand_photon(float3 *vel, float *int_eng, float mass, float eng, uint4 &rand_var, cudaMD *md, int out)
// adsorb photon with energy = eng and random direction by a given atom
{
    float u0 = *int_eng;
    float v02 = float3_sqr(*vel);   // square of initial velocity
    float3 rand_vect = rand_uvect(rand_var, md);
    //float3 rand_vect = rand_usphere(rand_var, md);
    //if (rand_vect.x < 0)
      //  printf("rand(%f %f %f).x < 0\n", rand_vect.x, rand_vect.y, rand_vect.z);

    // momentum conservation:
    float ermc = eng * revLight / mass;
    vel->x += ermc * rand_vect.x;   // -= or += doesn't matter, because random vector
    vel->y += ermc * rand_vect.y;
    vel->z += ermc * rand_vect.z;

    float v12 = float3_sqr(*vel);
    // energy conservation: old kinetic energy + photon energy = new kinetic energy + 'internal' energy
    *int_eng += eng + 0.5f * mass * (v02 - v12);
    //if (blockIdx.x == 0)
      //  if (threadIdx.x == 0)
        //    printf("adsorb photon U=%f->%f: ph_e=%f, K0=%f K1=%f\n", u0, *int_eng, eng, eng + 0.5f * mass * (v02 - v12), eng, 0.5f * mass * v02, 0.5f * mass * v12);

    if (out)
        printf("adsorb photon U=%f->%f: ph_e=%f, K0=%f K1=%f v0=%f v1=%f\n", u0, *int_eng, eng, eng + 0.5f * mass * (v02 - v12), eng, 0.5f * mass * v02, 0.5f * mass * v12, sqrt(v02), sqrt(v12));

    if (isnan(*int_eng))
        printf("ads rand photon: U is nan\n");
    if (isnan(vel->x))
        printf("ads rand photon: v is nan\n");
}

__device__ void radiate_photon(float3 *vel, float *int_eng, float mass, uint4& rand_var, cudaMD* md)
{
    float v02 = float3_sqr(*vel);   // square of initial velocity

    const float enth = 0.032f;// 0.0477f;     // constant enthalpy, temporary

    // define photon energy
    float ph_eng;       
    int rnd = rnd_xor128(rand_var) % 2048;
    float x = (float)rnd / 2048.f;       // x = 0..1
    float delt = *int_eng - enth;
    if (delt > 0.f)
    {
        if (x < (exp(-(*int_eng) / enth)))
            ph_eng = delt;
        else
            ph_eng = delt * x;
    }
    else // 'internal energy' <= enth
    {
        if (x < (1.0 - (*int_eng) / enth))
            return; // no radiation, exit routine
        else
            ph_eng = delt * x;
    }


    float ermc = ph_eng * revLight / mass;

    float3 rand_vect = rand_uvect(rand_var, md);
    // momentum conservation:
    vel->x -= ermc * rand_vect.x;   // -= or += doesn't matter, because random vector
    vel->y -= ermc * rand_vect.y;
    vel->z -= ermc * rand_vect.z;

    float v12 = float3_sqr(*vel);   // square of initial velocity
    // energy conservation:
    *int_eng -= (ph_eng + 0.5f * mass * (v12 - v02));
    printf("energy deleting: %f. ph_e=%f, K0=%f K1=%f cur_eng=%f\n", ph_eng + 0.5f * mass * (v12 - v02), ph_eng, 0.5f * mass * v02, 0.5f * mass * v12, *int_eng);
}

__device__ void radiate_photon2(float3* vel, float* int_eng, float mass, uint4& rand_var, cudaMD* md, int out)
// radiate photon do decrease kinetic energy
{
    float u0 = *int_eng;
    float v02 = float3_sqr(*vel);   // square of initial velocity
    float v0 = sqrt(v02);           // module of initial velocity

    int sign = -1;

    const float enth = 0.032f;// 0.0477f;     // constant enthalpy, temporary

    // define radiate or not
    int rnd = rnd_xor128(rand_var) % 2048;
    float x = (float)rnd / 2048.f;
    float eh = u0 / enth;
    
    if (u0 <= enth)
    {
        if (x > (0.5f * eh))
            return;
            //sign = 1;
    }
    else
        if (x > (1.f - exp(-2.f * eh)))
            return;
            //sign = 1;
    

    // chose random negative cosine
    rnd = rnd_xor128(rand_var) % 2048;
    float cos_phi = 2.f * ((float)rnd / 2048.f) - 1.f;       // x = 0..1, then multiple to -1  // try -1 to 1

    // define photon energy
    rnd = rnd_xor128(rand_var) % 2048;
    
    //! попробуем наоборот, выразим энергию как часть от внутренней, а потом сравним с 2vmc * cos
    float ph_eng = (float)rnd / 2048.f * (*int_eng * 0.99f);    // 0.99f - нужно подстраховаться, чтобы осталась энергия на изменение кинетической энергии
    float ermc = ph_eng * revLight / mass;

    // old variant
    //float ermc = (float)rnd / 2048.f;       // x = 0..1         // ermc - photon_energy / atom mass / c
    //ermc *= 2.f * v0 * cos_phi;
    //float ph_eng = ermc * Light * mass;                 // eng = 0.. 2*v*|cos phi|*m*c

    //if (ph_eng > (*int_eng))
    if (ermc > (2.f * v0 * abs(cos_phi)))
        return;               // no such energy in atom

    /*
    float delt = *int_eng - enth;
    if (delt > 0.f)
    {
        if (x < (exp(-(*int_eng) / enth)))
            ph_eng = delt;
        else
            ph_eng = delt * x;
    }
    else // 'internal energy' <= enth
    {
        if (x < (1.0 - (*int_eng) / enth))
            return; // no radiation, exit routine
        else
            ph_eng = delt * x;
    }
    */

    // get unit vector at given cosine and random rotation angle
    rnd = rnd_xor128(rand_var) % 2048;
    float theta = (float)rnd / 1024.f * numPi;  // 0 .. 2PI
    float3 rand_vect = get_angled_vector(*vel, cos_phi, theta);
    //printf("cos: %f\n", cos_phi);

    // momentum conservation:
    vel->x += ermc * rand_vect.x;   // -  because we chose positive cosine
    vel->y += ermc * rand_vect.y;
    vel->z += ermc * rand_vect.z;

    float v12 = float3_sqr(*vel);   // square of initial velocity
    // energy conservation:
    *int_eng -= (ph_eng + 0.5f * mass * (v12 - v02));
    //if (out)
       // printf("energy decreasing %f -> %f(%f). ph_e=%f, v0=%f v1=%f\n", u0, *int_eng, ph_eng + 0.5f * mass * (v12 - v02), ph_eng, v0, sqrt(v12));

}

__device__ void radiate_photon3(float3* vel, float* int_eng, float mass, uint4& rand_var, cudaMD* md, int out)
// radiate photon do decrease kinetic energy
{
    float u0 = *int_eng;
    float v02 = float3_sqr(*vel);   // square of initial velocity
    float v0 = sqrt(v02);           // module of initial velocity


    float ph_eng = 0.9f * u0;
    float ermc = ph_eng * revLight / mass;

    // count radiated photon energy for output statistics
    int nbin = ph_eng / md->phEngBin;
    if (nbin >= md->numbPhEngBin)
        nbin = md->numbPhEngBin - 1;
    atomicAdd(&(md->phEngs[nbin]), 1);



    // random radiation
    // or not so random
    int flag = 0;
    float3 rand_vect;
    /*
    int rnd = rnd_xor128(rand_var) % 2048;
    //printf("rnd=%d\n", rnd);
    if (rnd < 2048)
        rand_vect = rand_uvect(rand_var, md);
    else
    {
        rnd = rnd_xor128(rand_var) % 2048;
        float cos_phi = (float)rnd / 2048.f * (-1.f);

        rnd = rnd_xor128(rand_var) % 2048;
        float theta = (float)rnd / 1024.f * numPi;  // 0 .. 2PI
        rand_vect = get_angled_vector(*vel, cos_phi, theta);
        flag = 1;
    }
    */
    // new variant: random cosine between   (1-2a/v0) .. -1 with mean = -a/v0, where a = e/mc
    float ermcv0 = ermc / v0;       // ph_eng/(m*c*v0)
    float cos_phi;
    if (ermcv0 >= 1.f)
        cos_phi = -1.f;
    else
    {
        int rnd = rnd_xor128(rand_var) % 2048;
        float cos_phi = (float)rnd / 1024.f * (1.f - ermcv0);   // 1024 because () need to be multiplified by 2
        cos_phi -= 1.f;
        rnd = rnd_xor128(rand_var) % 2048;
        float theta = (float)rnd / 1024.f * numPi;  // 0 .. 2PI
        rand_vect = get_angled_vector(*vel, cos_phi, theta);
    }

    // momentum conservation:
    vel->x += ermc * rand_vect.x;   
    vel->y += ermc * rand_vect.y;
    vel->z += ermc * rand_vect.z;

    float v12 = float3_sqr(*vel);   // square of initial velocity
    // energy conservation:
    *int_eng -= (ph_eng + 0.5f * mass * (v12 - v02));
    if (out)
       printf("energy decreasing %f -> %f(%f). ph_e=%f, v0=%f v1=%f dK=%f\n", u0, *int_eng, ph_eng + 0.5f * mass * (v12 - v02), ph_eng, sqrt(v02), sqrt(v12), 0.5f * mass * (v12 - v02));

    if (flag)
        printf("negative cosine %f -> %f(%f). ph_e=%f, v0=%f v1=%f dK=%f\n", u0, *int_eng, ph_eng + 0.5f * mass * (v12 - v02), ph_eng, sqrt(v02), sqrt(v12), 0.5f*mass*(v12-v02));

    if (isnan(*int_eng))
        printf("rad_photon3: U is nan: u0=%f ph_en=%f v02=%f v12=%f\n", u0, ph_eng, v02, v12);
    if (isnan(vel->x))
        printf("rad_photon3: v is nan\n");
}



__global__ void tstat_radi9(int iStep, int atPerBlock, int atPerThread, cudaMD* md)
// use func adsorb_photon and radiate_photon
// with timestep accounting
{
    __shared__ int indEng, indVect;
    __shared__ float engTemp;
    if (threadIdx.x == 0)
    {
        indEng = atomicAdd(&(md->curEng), 1);   // get current index in photon energies array
        engTemp = 0.f;
    }
    __syncthreads();
    uint4 randVar = md->ui4rnd;

    //double leng;
    double pe;      // photon energy
    //double vls0;
    int i, e0;// , v0;
    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);

    e0 = indEng * atPerBlock + threadIdx.x * atPerThread - id0;

    int ei;// , vi;
    //float3 vect;    // vector for velocity adding
    //const float enth = 0.032f;// 0.0477f;     // constant enthalpy, temporary
    int rnd;
    double x, freq, teng = 0.f;
    //double tm;  // time from previous radiation

    // v = v +– E/(c*m) * u
    //! пока принебрежем рандомизацией внтури блока
    for (i = id0; i < N; i++)
    {
        // calculate time from previous photon adsorption/radiation:
        //tm = (iStep - md->radstep[i]) * md->tSt;

        ei = e0 + i;
        if (ei >= md->nAt)
            ei -= md->nAt;

        // photon frequency
        pe = md->engPhotons[ei];
        //if (isnan(pe))
        //    printf("step:%d, i=%d, photon energy is nan!\n", iStep, i);
/*
        freq = pe * revPlank;

        if (freq * tm < 1.f)
        {
            rnd = rnd_xor128(randVar) % 2048;
            x = (double)rnd / 2048.0;       // x = 0..1
            if (x > freq * tm)
                continue;
        }

*/
        //if (i == 0)
          //  printf("iStep=%d dStep=%d eng=%f\n", iStep, iStep - md->radstep[i], md->engs[i]);

        //rnd = rnd_xor128(randVar) % 8;
        //if (rnd < 2)
        //if ((iStep) % 2 == 0)
          adsorb_rand_photon(&(md->vls[i]), &(md->engs[i]), md->masses[i], pe, randVar, md, 0/*i == 0*/);
        //else
          //  radiate_photon(&(md->vls[i]), &(md->engs[i]), md->masses[i], randVar, md);
           //radiate_photon2(&(md->vls[i]), &(md->engs[i]), md->masses[i], randVar, md, i == 0);
        //rnd = rnd_xor128(randVar) % 2048;
        //if (rnd < 1024)
        if (md->engs[i] > 1e-4f)
        {
            //rnd = rnd_xor128(randVar) % 8;
            //if (rnd < 8)
            //if ((iStep) % 2 == 1)
                radiate_photon3(&(md->vls[i]), &(md->engs[i]), md->masses[i], randVar, md, 0/*(i == 0)&&(iStep % 200 == 0)*/);
        }

        //if (isnan(md->vls[i].x))
          //  printf("aft tstat 9 vls[%d].x is nan", i);
        //if (isnan(md->vls[i].y))
          //  printf("aft tstat 9 vls[%d].y is nan", i);
        //if (isnan(md->vls[i].z))
          //  printf("aft tstat 9 vls[%d].z is nan", i);

        // calculate atom radius (thermal exicated)
           // r = A/(B - eng)
        int tp = md->types[i];
        float restrE = min(md->engs[i], md->specs[tp].mxEng);
        md->radii[i] = md->specs[tp].radA / (md->specs[tp].radB - restrE);
        //md->radii[i] = 0.577;
        //printf("ra = %f\n", rad);


        teng += md->engs[i];
        //if (isnan(teng))
          //  printf("aft tstat 9 teng is nan, step:%d", iStep);

        // refresh last radiation time
        //md->radstep[i] = iStep;


        //printf("%d->%d,%d(%d, %d)) rmc=%f e=%f, v=%f: vel=%f -> %f\n", i, e0 + i, v0 + i, e0, v0, rmc, md->engPhotons[e0 + i], md->randVects[v0 + i].x, vls0, md->vls[i].x);
        //if (isnan(md->vls[i].x))
          //  printf("v[%d]=%f -> %f, rmc=%f e=%f, v=%f step=%d leng=%f\n", i, vls0, md->vls[i].x, rmc, md->engPhotons[ei], vect.x, iStep, leng);
    }

    atomicAdd(&engTemp, teng);
    if (threadIdx.x == 0)
    {
        rnd_xor128(md->ui4rnd);
        atomicAdd(&(md->engTemp), engTemp);
    }
}

void apply_pre_tstat(int iStep, TStat* tstat, Sim* sim, cudaMD* devMD, hostManagMD* man)
// apply first stage of thermostat (used before the first stage of the Velocity Verlet integrator)
{
    if (tstat->type == tpTermNose)
        {
            before_nose << <1, 1 >> > (devMD);
            cudaThreadSynchronize();
            tstat_nose << <  man->nAtBlock, man->nAtThread >> > (man->atPerBlock, man->atPerThread, devMD);
            cudaThreadSynchronize();
            after_nose << <1, 1 >> > (1, devMD);
            cudaThreadSynchronize();
        }
}


void apply_tstat(int iStep, TStat *tstat, Sim *sim, cudaMD *devMD, hostManagMD *man)
// apply thermostat (used after the second stage of the Velocity Verlet integrator)
{
    if (sim->nEq)   //! only if nEq != 0
        if (iStep <= sim->nEq)
            if ((iStep % sim->freqEq) == 0)
            {
                temp_scale << <man->nAtBlock, man->nAtThread >> > (man->atPerBlock, man->atPerThread, devMD);
                cudaThreadSynchronize();
                after_tscale << <1, 1 >> > (devMD);
                cudaThreadSynchronize();
            }


    switch (tstat->type)
    {
    case tpTermNose:
        before_nose << <1, 1 >> > (devMD);
        cudaThreadSynchronize();
        tstat_nose << <  man->nAtBlock, man->nAtThread/*man->nMultProc, man->nSingProc*/ >> > (man->atPerBlock, man->atPerThread, devMD);
        cudaThreadSynchronize();
        after_nose<<<1, 1>>>(1, devMD);
        cudaThreadSynchronize();
        //get_random << <man->nAtBlock, man->nAtThread >> > (devMD);
        break;
    case tpTermRadi:

        tstat_radi9 << <man->nAtBlock, man->nAtThread >> > (iStep, man->atPerBlock, man->atPerThread, devMD);
        cudaThreadSynchronize();

        if (iStep < 50)
        {
            tstat_radi9 << <man->nAtBlock, man->nAtThread >> > (iStep, man->atPerBlock, man->atPerThread, devMD);
            cudaThreadSynchronize();
        }
        
        /*
        if (iStep < 5000)
        {
            tstat_radi9 << <man->nAtBlock, man->nAtThread >> > (iStep, man->atPerBlock, man->atPerThread, devMD);
            cudaThreadSynchronize();
        }
        if (iStep % 50 == 0)
        {
            tstat_radi9 << <man->nAtBlock, man->nAtThread >> > (iStep, man->atPerBlock, man->atPerThread, devMD);
            cudaThreadSynchronize();
        }
        */
        //tstat_radi9 << <man->nAtBlock, man->nAtThread >> > (iStep, man->atPerBlock, man->atPerThread, devMD);
        //cudaThreadSynchronize();
        break;
    }
}