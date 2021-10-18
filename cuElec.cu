// Electrostatic calculations
#include "cuda_runtime.h"
#include <stdio.h>

#include "cuStruct.h"
#include "cuElec.h"
#include "defines.h"
#include "const.h"
#include "cuUtils.h"

texture<float, 1, cudaReadModeElementType> texCoulEng;	// coulomb energy as a function of distance
texture<float, 1, cudaReadModeElementType> texCoulFrc;  // coulomb force ...


extern __constant__ const float d_Fcoul_scale;  //! не помню точно
extern __constant__ const float d_sqrtpi;  //sqrt(PI); - не поддерживается динамическая инициализация
extern __constant__ const float d_2pi;  //2 * (PI);

//__device__ float no_coul(float r2, float& r, float chprd, float alpha, float& eng)
__device__ float no_coul(float r2, float& r, float chprd, cudaMD *md, float& eng)
// if there are no Coulomb interaction  in the system
{
    return 0.f;
}

//__device__ float direct_coul(float r2, float& r, float chprd, float alpha, float& eng)
__device__ float direct_coul(float r2, float& r, float chprd, cudaMD* md, float& eng)
//  r2 - square of distance, chprd - production of charges
// parameter alpha - for compability with real part of Ewald summation
//! тут надо ещё ввести эпсилон в закон кулона
{
    float kqq = chprd * d_Fcoul_scale;  //! нужно ещё добавить 1/epsilon
    r = sqrt(r2);

    eng += kqq / r;
    return kqq / r / r2;
}

void init_realEwald_tex(cudaMD *md, float mxRange, float alpha)
// create textures for evaluation of real part of Ewald summ
{
    int i;
    float r, ar;
    int n = mxRange * 100 + 1;        // each angstrom is 100 points and plus additional point to avoid out of range
    int size = n * sizeof(float);
    float *eng = (float*)malloc(size);
    float* frc = (float*)malloc(size);
    float erfcar;
    for (i = 1; i < n; i++)
    {
        r = i * 0.01f;
        ar = alpha * r;
        erfcar = erfc(ar);
        eng[i] = erfcar / r;
        frc[i] =  (erfcar + 2 * ar / sqrtpi * exp(-ar * ar)) / (r * r * r);
    }
    //data_to_device(engArr, eng, size);
    //data_to_device(frcArr, frc, size);

    //! к сожалению так и не получилось приручить cudaBindTexture, правда написано, что оно и не поддерживает фильтрацию
    //! прийдется использовать bindTextureArray

    cudaChannelFormatDesc cform = cudaCreateChannelDesc<float>();// (32, 32, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&(md->coulEng), &cform, n, 1, cudaArrayDefault);
    cudaMemcpyToArray(md->coulEng, 0, 0, eng, size, cudaMemcpyHostToDevice);
    texCoulEng.normalized = 1;
    texCoulEng.filterMode = cudaFilterModeLinear;
    texCoulEng.addressMode[0] = cudaAddressModeClamp;
    cudaBindTextureToArray(&texCoulEng, md->coulEng, &cform);

    cudaMallocArray(&(md->coulFrc), &cform, n, 1, cudaArrayDefault);
    cudaMemcpyToArray(md->coulFrc, 0, 0, frc, size, cudaMemcpyHostToDevice);
    texCoulFrc.normalized = 1;
    texCoulFrc.filterMode = cudaFilterModeLinear;
    texCoulFrc.addressMode[0] = cudaAddressModeClamp;
    cudaBindTextureToArray(&texCoulFrc, md->coulFrc, &cform);



    delete[] eng;
    delete[] frc;
}

void free_realEwald_tex(cudaMD* md)
{
    cudaUnbindTexture(&texCoulEng);
    cudaUnbindTexture(&texCoulFrc);
    cudaFreeArray(md->coulEng);
    cudaFreeArray(md->coulFrc);
}


//__device__ float real_ewald(float r2, float& r, float chprd, float alpha, float& eng)
__device__ float real_ewald(float r2, float& r, float chprd, cudaMD* md, float& eng)
// calculate energy and return force of real part Coulombic iteraction via Ewald procedure
//  r2 - square of distance, chprd - production of charges
//  eng - for saving energy
//! тут надо ещё ввести эпсилон в закон кулона
{
    //double r;

    float ar; //alpha * r
    float erfcar;   // erfc(alpha * r);
    float kqq = chprd * d_Fcoul_scale; // q[i]*q[j]*1/4pie0;

    //if (r == 0)
    r = sqrt(r2);  // if r is unknown, calculate it
    ar = md->alpha * r;
    erfcar = erfc(ar);

    eng += kqq * erfcar / r;       //! save energy (не всегда она нужна)
    return kqq / r / r2 * (erfcar + 2 * ar / d_sqrtpi * exp(-ar * ar));
}

__device__ float fennel(float r2, float& r, float chprd, cudaMD* md, float& eng)
// calculate energy and return force of real part Coulombic iteraction via Ewald procedure
//  r2 - square of distance, chprd - production of charges
//  eng - for saving energy
//! тут надо ещё ввести эпсилон в закон кулона
{
    //double r;

    float kqq = chprd * d_Fcoul_scale; // q[i]*q[j]*1/4pie0;

    //if (r == 0)
    r = sqrt(r2);  // if r is unknown, calculate it
    double ir = 1.f / r;

    float ar = md->alpha * r;
    float erfcar = erfc(ar);    // erfc(alpha * r);

    eng += kqq * (erfcar * ir - md->elC1 + md->elC2 * (r - md->rElec));    //! save energy (не всегда она нужна)
    return kqq * ir * ((erfcar / r2 + md->daipi2 * exp(-ar * ar) * ir) - md->elC2);
}

__device__ float real_ewald_tex(float r2, float& r, float chprd, float alpha, float& eng)
// texture variant of real_ewald
{
    float kqq = chprd * d_Fcoul_scale; // q[i]*q[j]*1/4pie0;

    //if (r == 0)
    r = sqrt(r2);  // if r is unknown, calculate it
    float norm = r / (8.f + 0.01f);    // ! temp, it's defined by coul distance
    //int n = r / 0.01;
    //printf("r=%f e=%f calc=%f f=%f x=%f n=%d e=%f\n", norm, tex1D(texCoulEng, norm), erfc(alpha * r) / r, tex1D(texCoulFrc, norm), tex1Dfetch(texCoulEng, 0.5 ), n, tex1D(texCoulEng, n));

    eng += kqq * tex1D(texCoulEng, norm);
    return kqq * tex1D(texCoulFrc, norm);
}

__global__ void recip_ewald(int atPerBlock, int atPerThread, cudaMD* md)
// calculate reciprocal part of Ewald summ and corresponding forces
// the first part : summ (qiexp(kr)) evaluation
{
    int i;      // for atom loop
    int ik;     // index of k-vector
    int l, m, n;
    int mmin = 0;
    int nmin = 1;
    float tmp, ch;
    float rkx, rky, rkz, rk2;   // component of rk-vectors

    int nkx = md->nk.x;
    int nky = md->nk.y;
    int nkz = md->nk.z;
    // double rkcut2 = sim->rkcut2;  //! В DL_POLY это вычисляемая величина
    //printf("ewald_rec Nat=%d kx=%d ky=%d kz=%d  rkut2=%f\n", Nat, kx, ky, kz, ew->rkcut2);
    
    // arrays for keeping iexp(k*r) Re and Im part
    float2 el[2];
    float2 em[NKVEC_MX];
    float2 en[NKVEC_MX];

    float2 sums[NTOTKVEC];          // summ (q iexp (k*r)) for each k
    extern __shared__ float2 sh_sums[];     // the same in shared memory

    float2 lm;     // temp var for keeping el*em
    float2 ck;     // temp var for keeping q * el * em * en (q iexp (kr))

    // invert length of box cell
    float ra = md->revLeng.x;
    float rb = md->revLeng.y;
    float rc = md->revLeng.z;

    if (threadIdx.x == 0)
        for (i = 0; i < md->nKvec; i++)
            sh_sums[i] = make_float2(0.0f, 0.0f);
    __syncthreads();

    for (i = 0; i < md->nKvec; i++)
        sums[i] = make_float2(0.0f, 0.0f);

    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);

    //! тут всё верно для прямоугольной геометрии. Если ячейка будет кривая, код нужно править
    ik = 0;
    for (i = id0; i < N; i++)
    {
        // save charge
        ch = md->specs[md->types[i]].charge;

        // iexp (2pi * 0 * l) for em- and en- arrays this step omitted as they set in 'init_ewald'
        //   el- arrays need to refresh (according cycle by l)
        el[0] = make_float2(1.0f, 0.0f);    // .x - real part (or cos) .y - imagine part (or sin)
        
        // в оригинале эти две переменные определены на этапе инициализации
        em[0] = make_float2(1.0f, 0.0f);
        en[0] = make_float2(1.0f, 0.0f);

        // iexp (ikr)
        sincos(d_2pi * md->xyz[i].x * ra, &(el[1].y), &(el[1].x));
        sincos(d_2pi * md->xyz[i].y * rb, &(em[1].y), &(em[1].x));
        sincos(d_2pi * md->xyz[i].z * rc, &(en[1].y), &(en[1].x));

        // fil exp(iky) array by complex multiplication
        for (l = 2; l < nky; l++)
        {
             em[l].x = em[l - 1].x * em[1].x - em[l - 1].y * em[1].y;
             em[l].y = em[l - 1].y * em[1].x + em[l - 1].x * em[1].y;
        }

        // fil exp(ikz) array by complex multiplication
        for (l = 2; l < nkz; l++)
        {
             en[l].x = en[l - 1].x * en[1].x - en[l - 1].y * en[1].y;
             en[l].y = en[l - 1].y * en[1].x + en[l - 1].x * en[1].y;
        }

        // MAIN LOOP OVER K-VECTORS:
        for (l = 0; l < nkx; l++)
        {
            //! its kept in rk array!
            rkx = l * d_2pi * ra; // only for rect geometry!
            
            // move exp(ikx[l]) to ikx[0] for memory saving (ikx[i>1] are not used)
            if (l == 1)
                el[0] = el[1];
            else if (l > 1)
                {
                    // exp(ikx[0]) = exp(ikx[0]) * exp(ikx[1])
                    tmp = el[0].x;
                    el[0].x = tmp * el[1].x - el[0].y * el[1].y;
                    el[0].y = el[0].y * el[1].x + tmp * el[1].y;
                }

            //ky - loop:
            for (m = mmin; m < nky; m++)
            {
                //! its kept in rk array!
                rky = m * d_2pi * rb;

                //set temporary variable lm = e^ikx * e^iky
                if (m >= 0)
                {
                        lm.x = el[0].x * em[m].x - el[0].y * em[m].y;       // [0] - потому что мы перезаписываем элемент [l] в [0]
                        lm.y = el[0].y * em[m].x + em[m].y * el[0].x;
                }
                else // for negative ky give complex adjustment to positive ky:
                {
                        lm.x = el[0].x * em[-m].x + el[0].y * em[-m].y;
                        lm.y = el[0].y * em[-m].x - em[-m].x * el[0].x;
                }

                //kz - loop:
                for (n = nmin; n < nkz; n++)
                {
                    //! its kept in rk array!
                    rkz = n * d_2pi * rc;
                    rk2 = rkx * rkx + rky * rky + rkz * rkz;
                    //rk2 = md->rk[ik].x * md->rk[ik].x + md->rk[ik].y * md->rk[ik].y + md->rk[ik].z * md->rk[ik].z;

                    //! у нас cuttof и rk2 возможно в разных единицах измерения, надо это провентилировать
                    //printf("rk2 * rkcut2 :  %f  *  %f\n", rk2, ew->rkcut2);
                    if (rk2 < md->rKcut2) // cutoff
                    {
                        // calculate summ[q iexp(kr)]   (local part)
                        if (n >= 0)
                         {
                                ck.x = ch * (lm.x * en[n].x - lm.y * en[n].y);
                                ck.y = ch * (lm.y * en[n].x + lm.x * en[n].y);
                         }
                        else // for negative kz give complex adjustment to positive kz:
                         {
                                ck.x = ch * (lm.x * en[-n].x + lm.y * en[-n].y);
                                ck.y = ch * (lm.y * en[-n].x - lm.x * en[-n].y);
                        }
                        sums[ik].x += ck.x;
                        sums[ik].y += ck.y;
                        
                        // save qiexp(kr) for each k for each atom:
                        md->qiexp[i][ik] = ck;
                        ik++;
                    }
                } // end n-loop (over kz-vectors)

                nmin = 1 - nkz;

            } // end m-loop (over ky-vectors)

            mmin = 1 - nky;

        }  // end l-loop (over kx-vectors)
#ifdef DEBUG_MODE
        //atomicAdd(&(md->nCult[i]), 1);
        md->nCult[i]++;
#endif
    } // end loop by atoms

    // split sum into shared memory
    for (i = 0; i < md->nKvec; i++)
    {
        atomicAdd(&(sh_sums[i].x), sums[i].x);
        atomicAdd(&(sh_sums[i].y), sums[i].y);
    }
    __syncthreads();

    //...and to global
    int step = ceil((double)md->nKvec / (double)blockDim.x);
    id0 = threadIdx.x * step;
    N = min(id0 + step, md->nKvec);
    for (i = id0; i < N; i++)
    {
        atomicAdd(&(md->qDens[i].x), sh_sums[i].x);
        atomicAdd(&(md->qDens[i].y), sh_sums[i].y);
    }
}
// end 'ewald_rec' function

__global__ void ewald_force(int atPerBlock, int atPerThread, cudaMD* md)
// calculate reciprocal part of Ewald summ and corresponding forces
// the second part : enegy and forces
{
    int i;      // for atom loop
    int ik;     // index of k-vector
    float tmp;

    // accumulator for force components
    float3 force;

    // constant factors for energy and force
    float eScale = md->ewEscale;
    float fScale = md->ewFscale;

    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);
    //printf("range: %d..%d\n", id0, N - 1);
    for (i = id0; i < N; i++)
    {
        force = make_float3(0.0f, 0.0f, 0.0f);

        // summ by k-vectors
        for (ik = 0; ik < md->nKvec; ik++)
        {
            // in DLPOLY there is a factor  4 * pi / V / (4piee0):
            tmp = fScale * md->exprk2[ik] * (md->qiexp[i][ik].y * md->qDens[ik].x - md->qiexp[i][ik].x * md->qDens[ik].y);

            force.x += tmp * md->rk[ik].x;
            force.y += tmp * md->rk[ik].y;
            force.z += tmp * md->rk[ik].z;
        }

        md->frs[i].x += force.x;
        md->frs[i].y += force.y;
        md->frs[i].z += force.z;
#ifdef DEBUG_MODE
        //atomicAdd(&(md->nCult[i]), 1);
        md->nCult[i]++;
#endif
    } // end loop by atoms


    // one block calculate energy
    if (blockIdx.x == 0)
        if (threadIdx.x == 0)
        {
            for (ik = 0; ik < md->nKvec; ik++)
                md->engCoul2 += eScale * md->exprk2[ik] * (md->qDens[ik].x * md->qDens[ik].x + md->qDens[ik].y * md->qDens[ik].y);
        }

}
// end 'ewald_force' function