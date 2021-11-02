#include "cuda_runtime.h"

#include "cuStruct.h"
#include "dataStruct.h"
#include "dataStruct.h"
#include "cuTemp.h"
#include "temperature.h"
#include "cuUtils.h"
#include "utils.h"

void init_cuda_tstat(int nAt, TStat *tstat, cudaMD *hmd, hostManagMD *man)
{
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

        int i;
        float *phs = (float*)malloc(nAt * float_size);
        float3 *vecs = (float3*)malloc(nAt * float3_size);
        float* engs = (float*)malloc(nAt * float_size);
        float* radii = (float*)malloc(nAt * float_size);
        int* radstep = (int*)malloc(nAt * int_size);
        for (i = 0; i < nAt; i++)
        {
            phs[i] = (float)tstat->photons[i];
            vecs[i] = make_float3((float)tstat->randVx[i], (float)tstat->randVy[i], (float)tstat->randVz[i]);
            engs[i] = 0.f;
            radii[i] = 0.577f + rand01() * 0.0001f;    // values must be initialized to avoid division by zero in some radii-dependet pair potential
            radstep[i] = 0;
        }
        data_to_device((void**)(&hmd->engPhotons), phs, nAt * float_size);
        data_to_device((void**)(&hmd->randVects), vecs, nAt * float3_size);
        data_to_device((void**)(&hmd->engs), engs, nAt * float_size);
        data_to_device((void**)(&hmd->radii), radii, nAt * float_size);
        data_to_device((void**)(&hmd->radstep), radstep, nAt * int_size);

        //preset unit vectors:
        float3* uvs = (float3*)malloc(nUvect * float3_size);
        for (i = 0; i < nUvect; i++)
            uvs[i] = make_float3((float)tstat->uvectX[i], (float)tstat->uvectY[i], (float)tstat->uvectZ[i]);
        data_to_device((void**)(&hmd->uvects), uvs, nUvect * float3_size);
        free(uvs);

        // for sorting (if applied):
        free(phs);
        free(vecs);
        break;
    }
}

void free_cuda_tstat(TStat* tstat, cudaMD* hmd)
{
    if (tstat->type == tpTermRadi)
    {
        cudaFree(hmd->engPhotons);
        cudaFree(hmd->randVects);
        cudaFree(hmd->engs);
        cudaFree(hmd->radii);
        //cudaFree(hmd->sort_engs);
        cudaFree(hmd->uvects);
    }
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

    for (i = id0; i < N; i++)
    {
        md->vls[i].x *= k;
        md->vls[i].y *= k;
        md->vls[i].z *= k;
#ifdef DEBUG_MODE
        atomicAdd(&(md->nCult[i]), 1);
#endif 
    }
  
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

__constant__ const float revLight = 3.33567e-5; // 1/c, where c is lightspeed, 2.9979e4 A/ps
__constant__ const float Light = 2.9979e4;      // lightspeed, 2.9979e4 A/ps
__constant__ const float revPlank = 241.55f;    // 1 / plank constant (4.14 eV*ps)
__constant__ const float numPi = 3.14159f;    // pi

__device__ float3 rand_uvect(uint4 &var, cudaMD* md)
{
    int rnd = rnd_xor128(var) % dnUvect;
    return md->uvects[rnd];
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

    return res;
}

__global__ void laser_cooling(int sign, int atPerBlock, int atPerThread, cudaMD* md)
// adsorb photon with direction to atom and then radiate photon in random direction
{
    __shared__ int indEng, indVect;
    if (threadIdx.x == 0)
    {
        indEng = atomicAdd(&(md->curEng), 1);   // get current index in photon energies array
        indVect = atomicAdd(&(md->curVect), 1);   // get current index in photon energies array
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

        if (sign == 1)
        {
                pe = md->engPhotons[ei];
                md->engs[i] += pe;
                leng = float3_length(md->vls[i]);
                if (leng > 0.f)
                    vect = make_float3(-md->vls[i].x / leng, -md->vls[i].y / leng, -md->vls[i].z / leng);
        }
        else
        {
            if (md->engs[i] > 0.f) // 64.2 = enthalpy of argon at 298 K and 1 atm in eV/particle
            {
                pe = md->engs[i];
                md->engs[i] = 0.f;
                vect = rand_uvect(randVar, md);
            }
            else
                continue;
            vect = rand_uvect(randVar, md);
        }

        md->vls[i].x = md->vls[i].x + rmc * pe * vect.x;
        md->vls[i].y = md->vls[i].y + rmc * pe * vect.y;
        md->vls[i].z = md->vls[i].z + rmc * pe * vect.z;
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


    float sinPhi, sinTh, cosTh; //, cosPhi;
    //sincos(phi, &sinPhi, &cosPhi);
    sinPhi = sqrt(1 - cos_phi * cos_phi);
    sincos(theta, &sinTh, &cosTh);

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

    float in_sqrt = v0 * v0 + sign * 2.f * rm * eng;
    if (in_sqrt < 0.f)
        return 0;
    float v1 = sqrt(in_sqrt);
    // cosine theorem:
    float cosPhi = (mass * mass * v1 * v1 + mass * mass * v0 * v0  - eng * eng * revLight * revLight) / (2.f * mass * mass * v0 * v1); // in_sqrt = v1^2
    if ((cosPhi < -1.f) || (cosPhi > 1.f))
    {
        printf("wrong cosine: %f v0=%f v1=%f mass=%f ph_eng=%f kin0=%f sign=%d ph_mom=%f mv=%f\n", cosPhi, v0, v1, mass, eng, mass * v0 * v0, sign, eng*revLight, mass * v0);
        return 0;
    }

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

    // momentum conservation:
    float ermc = eng * revLight / mass;
    vel->x += ermc * rand_vect.x;   // -= or += doesn't matter, because random vector
    vel->y += ermc * rand_vect.y;
    vel->z += ermc * rand_vect.z;

    float v12 = float3_sqr(*vel);
    // energy conservation: old kinetic energy + photon energy = new kinetic energy + 'internal' energy
    *int_eng += eng + 0.5f * mass * (v02 - v12);
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

    // momentum conservation:
    vel->x += ermc * rand_vect.x;   // -  because we chose positive cosine
    vel->y += ermc * rand_vect.y;
    vel->z += ermc * rand_vect.z;

    float v12 = float3_sqr(*vel);   // square of initial velocity
    // energy conservation:
    *int_eng -= (ph_eng + 0.5f * mass * (v12 - v02));

}

__device__ void radiate_photon3(float3* vel, float* int_eng, float mass, uint4& rand_var, cudaMD* md, int out)
// radiate photon do decrease kinetic energy
{
    float u0 = *int_eng;
    float v02 = float3_sqr(*vel);   // square of initial velocity
    float v0 = sqrt(v02);           // module of initial velocity


    float ph_eng = 0.9f * u0;
    float ermc = ph_eng * revLight / mass;

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

    double pe;      // photon energy
    int i, e0;// , v0;
    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);

    e0 = indEng * atPerBlock + threadIdx.x * atPerThread - id0;

    int ei;// , vi;
    int rnd;
    double x, freq, teng = 0.f;

    // v = v +– E/(c*m) * u
    for (i = id0; i < N; i++)
    {
        // calculate time from previous photon adsorption/radiation:
        //tm = (iStep - md->radstep[i]) * md->tSt;

        ei = e0 + i;
        if (ei >= md->nAt)
            ei -= md->nAt;

        // photon frequency
        pe = md->engPhotons[ei];
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

        int rnd;
        rnd = rnd_xor128(randVar) % 8;
        //if (rnd < 2)
        //if ((iStep) % 2 == 0)
          adsorb_rand_photon(&(md->vls[i]), &(md->engs[i]), md->masses[i], pe, randVar, md, 0/*i == 0*/);
        //else
          //  radiate_photon(&(md->vls[i]), &(md->engs[i]), md->masses[i], randVar, md);
           //radiate_photon2(&(md->vls[i]), &(md->engs[i]), md->masses[i], randVar, md, i == 0);
        //rnd = rnd_xor128(randVar) % 2048;
        //if (rnd < 1024)
        if (md->engs[i] > 1e-4)
        {
            //rnd = rnd_xor128(randVar) % 8;
            //if (rnd < 8)
            //if ((iStep) % 2 == 1)
                radiate_photon3(&(md->vls[i]), &(md->engs[i]), md->masses[i], randVar, md, 0/*(i == 0)&&(iStep % 200 == 0)*/);
        }

        // calculate atom radius (thermal exicated)
           // r = A/(B - eng)
        int tp = md->types[i];
        float restrE = min(md->engs[i], md->specs[tp].mxEng);
        md->radii[i] = md->specs[tp].radA / (md->specs[tp].radB - restrE);

        teng += md->engs[i];

        // refresh last radiation time
        //md->radstep[i] = iStep;
    }

    atomicAdd(&engTemp, teng);
    if (threadIdx.x == 0)
    {
        rnd_xor128(md->ui4rnd);
        atomicAdd(&(md->engTemp), engTemp);
    }
}

void apply_tstat(int iStep, TStat *tstat, Sim *sim, cudaMD *devMD, hostManagMD *man)
{
    if (sim->nEq)   //! only if nEq != 0
        if (iStep <= sim->nEq)
            if ((iStep % sim->freqEq) == 0)
            {
                temp_scale << <man->nAtBlock, man->nAtThread/*man->nMultProc, man->nSingProc*/ >> > (man->atPerBlock, man->atPerThread, devMD);
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
        break;
    case tpTermRadi:
        tstat_radi9 << <man->nAtBlock, man->nAtThread >> > (iStep, man->atPerBlock, man->atPerThread, devMD);
        cudaThreadSynchronize();
        break;
    }
}
