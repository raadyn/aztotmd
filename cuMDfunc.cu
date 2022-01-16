#include <stdio.h>

#include "cuStruct.h"
#include "cuMDfunc.h"
#include "cuSort.h"
#include "defines.h"
#include "cuUtils.h"
#ifdef USE_FASTLIST
//#include "cuFastList.h"
#endif

__global__ void verify_parents(int id, cudaMD* dmd)
{
    int i;
    printf("verify autoparents %d: ", id);
    for (i = 0; i < dmd->nAt; i++)
        if (dmd->parents[i] == i)
            printf(" %d", i);
    printf("\n");
}

__device__ float3 get_shift(int shift_type, cudaMD* md)
{
    int x, y, xy, z;

    z = shift_type / 9;
    xy = shift_type % 9;
    y = xy / 3;
    x = xy % 3;
    x--; y--; z--;

    float3 res = make_float3(x * md->leng.x, y * md->leng.y, z * md->leng.z);
    return res;
}

__device__ void put_periodic(float3& xyz, float3 vel, float mass, int type, cudaMD* md)
// apply the periodic conditions to a particle coordinates and save some info if the particle cross a box
{
    int nx, ny, nz;
    //float x0 = xyz.x;

    //! Only for rectangular geometry!
    //! momentum is kept without factor 2!
    //xyz.x = xyz.x * md->revLeng.x - floor(xyz.x * md->revLeng.x) * md->leng.x;

    /*
    xyz.x -= floor(xyz.x * md->revLeng.x) * md->leng.x;
    xyz.y -= floor(xyz.y * md->revLeng.y) * md->leng.y;
    xyz.z -= floor(xyz.z * md->revLeng.z) * md->leng.z;
    */
    nx = floor((double)xyz.x * (double)md->revLeng.x);
    ny = floor((double)xyz.y * (double)md->revLeng.y);
    nz = floor((double)xyz.z * (double)md->revLeng.z);
    xyz.x -= (float)(nx * md->leng.x);
    xyz.y -= (float)(ny * md->leng.y);
    xyz.z -= (float)(nz * md->leng.z);
    //! раньше было так
    /*
    xyz.x -= (float)(floor((double)xyz.x * (double)md->revLeng.x) * (double)md->leng.x);
    xyz.y -= (float)(floor((double)xyz.y * (double)md->revLeng.y) * (double)md->leng.y);
    xyz.z -= (float)(floor((double)xyz.z * (double)md->revLeng.z) * (double)md->leng.z);
    */

    // подстраховка
    if (xyz.x >= md->leng.x)
        xyz.x = 0.f;
    if (xyz.y >= md->leng.y)
        xyz.y = 0.f;
    if (xyz.z >= md->leng.z)
        xyz.z = 0.f;

    // счетчики
    if (nx > 0)
    {
        atomicAdd(&md->specAcBoxPos[type].x, 1); // counter of crossing in a positive direction
        atomicAdd(&md->posMom.x, mass * vel.x); // add momentum (mv)
    }
    else 
        if (nx < 0)
        {
            atomicAdd(&md->specAcBoxNeg[type].x, 1); // counter of crossing in a negative direction
            atomicAdd(&md->negMom.x, mass * (-vel.x)); // we suppose that vx in this case is negative
        }

    if (ny > 0)
    {
        atomicAdd(&md->specAcBoxPos[type].y, 1); // counter of crossing in a positive direction
        atomicAdd(&md->posMom.y, mass * vel.y); // add momentum (mv)
    }
    else
        if (ny < 0)
        {
            atomicAdd(&md->specAcBoxNeg[type].y, 1); // counter of crossing in a negative direction
            atomicAdd(&md->negMom.y, mass * (-vel.y)); // we suppose that vx in this case is negative
        }

    if (nz > 0)
    {
        atomicAdd(&md->specAcBoxPos[type].z, 1); // counter of crossing in a positive direction
        atomicAdd(&md->posMom.z, mass * vel.z); // add momentum (mv)
    }
    else
        if (nz < 0)
        {
            atomicAdd(&md->specAcBoxNeg[type].z, 1); // counter of crossing in a negative direction
            atomicAdd(&md->negMom.z, mass * (-vel.z)); // we suppose that vx in this case is negative
        }

    //    if (blockIdx.x == 0)
      //      if (threadIdx.x == 0)
        //        printf("%f -> %f rev=%f cor=%f\n", x0, xyz.x, md->revLeng.x, xyz.x * md->revLeng.x - floor(xyz.x * md->revLeng.x) * md->leng.x);

    /*
        if (xyz.x < 0)
        {
            // (variant 1): not farhter than box length:
            //atm->xs[index] += box->la;

            // (variant 2): any length
            xyz.x += ((int)(-xyz.x * md->revLeng.x) + 1) * md->leng.x;
            //xyz.x += (ceil(-xyz.x * md->revLeng.x) + 1) * md->leng.x;

            atomicAdd(&md->specAcBoxNeg[type].x, 1); // counter of crossing in a negative direction
            atomicAdd(&md->negMom.x, mass * (-vel.x)); // we suppose that vx in this case is negative
        }
        else
            if (xyz.x >= md->leng.x)
            {
                //atm->xs[index] -= box->la;
                //xyz.x -= ((int)(xyz.x * md->revLeng.x)) * md->leng.x;
                xyz.x -= (ceil(xyz.x * md->revLeng.x)) * md->leng.x;
                if (xyz.x >= md->leng.x)
                    printf("again delt=%f\n", ((int)(xyz.x * md->revLeng.x)) * md->leng.x);

                atomicAdd(&md->specAcBoxPos[type].x, 1);
                atomicAdd(&md->posMom.x, mass * vel.x);
            }


        if (xyz.y < 0)
        {
            //atm->ys[index] += box->lb;
            xyz.y += ((int)(-xyz.y * md->revLeng.y) + 1) * md->leng.y;

            atomicAdd(&md->specAcBoxNeg[type].y, 1);
            atomicAdd(&md->negMom.y, mass * (-vel.y));    // we suppose that vy in this case is negative
        }
        else
            if (xyz.y >= md->leng.y)
            {
                //atm->ys[index] -= box -> lb;
                xyz.y -= ((int)(xyz.y * md->revLeng.y)) * md->leng.y;

                atomicAdd(&md->specAcBoxPos[type].y, 1);
                atomicAdd(&md->posMom.y, mass * vel.y);
            }

        if (xyz.z < 0)
        {
            //atm->zs[index] += box -> lc;
            xyz.z += ((int)(-xyz.z * md->revLeng.z) + 1) * md->leng.z;

            atomicAdd(&md->specAcBoxNeg[type].z, 1);
            atomicAdd(&md->negMom.z, mass * (-vel.z)); // we suppose that vz in this case is negative
        }
        else
            if (xyz.z >= md->leng.z)
            {
                //atm->zs[index] -= box -> lc;
                xyz.z -= ((int)(xyz.z * md->revLeng.z)) * md->leng.z;

                atomicAdd(&md->specAcBoxPos[type].z, 1);
                atomicAdd(&md->posMom.z, mass * vel.z);
            }
    */
}
// end 'put_periodic' function

__device__ void delta_periodic(float& dx, float& dy, float& dz, cudaMD* md)
// apply periodic boundary to coordinate differences: dx, dy, dz
{
    //!Only for rectangular geometry!

    // x
    if (dx > md->halfLeng.x)
        dx -= md->leng.x;
    else
        if (dx < -md->halfLeng.x)
            dx += md->leng.x;

    // y
    if (dy > md->halfLeng.y)
        dy -= md->leng.y;
    else
        if (dy < -md->halfLeng.y)
            dy += md->leng.y;

    // z
    if (dz > md->halfLeng.z)
        dz -= md->leng.z;
    else
        if (dz < -md->halfLeng.z)
            dz += md->leng.z;
}
// end 'delta_periodic' function

__device__ float r2_periodic(int id1, int id2, cudaMD *md)
// return square of distance between atoms id1 and id2 with account of periodic boundaries
{
    float dx = md->xyz[id1].x - md->xyz[id2].x;
    float dy = md->xyz[id1].y - md->xyz[id2].y;
    float dz = md->xyz[id1].z - md->xyz[id2].z;
    delta_periodic(dx, dy, dz, md);
    return dx * dx + dy * dy + dz * dz;
}

__device__ void pass_periodic(int id1, int id2, cudaMD *md, int& px, int& py, int& pz)
// function for electron jumps, determine there was jumps through box edge or not
//! only for rectangular geometry
{
    float dx = md->xyz[id1].x - md->xyz[id2].x;
    //double dy = atm->ys[iat] - atm->ys[jat];
    //double dz = atm->zs[iat] - atm->zs[jat];

    if (dx > md->halfLeng.x) // второй атом в отрицательном отображении
    {
        px = -1;
    }
    else
        if (dx < -md->halfLeng.x) //  второй атом в положительном отображении
        {
            px = 1;
        }
        else
            px = 0;

    //! add y- and z- directions
}

__device__ void keep_in_cell(int index, float3 xyz, cudaMD* md)
// save atom index with coordinates xyz in the cell list
{
    int c, j;

    //printf("keep_in_cell: rev=(%f %f %f)\n", md->cRevSize.x, md->cRevSize.y, md->cRevSize.z);

    // determine cell index
    //c = floor(xyz.x / md->cSize.x) * md->cnYZ + floor(xyz.y / md->cSize.y) * md->cNumber.z + floor(xyz.z / md->cSize.z);
    //c = floor((double)xyz.x / (double)md->cSize.x) * md->cnYZ + floor((double)xyz.y / (double)md->cSize.y) * md->cNumber.z + floor((double)xyz.z / (double)md->cSize.z);
    c = floor((double)xyz.x * (double)md->cRevSize.x) * md->cnYZ + floor((double)xyz.y * (double)md->cRevSize.y) * md->cNumber.z + floor((double)xyz.z * (double)md->cRevSize.z);
    if (c >= md->nCell)
        printf("keep in cell: xyz=(%f; %f; %f)revsizes:[%f %f %f] c = %d\n", xyz.x, xyz.y, xyz.z, md->cRevSize.x, md->cRevSize.y, md->cRevSize.z, c);
    if (c < 0)
        printf("keep in cell: xyz=(%f; %f; %f) c = %d\n", xyz.x, xyz.y, xyz.z, c);
    //! или может не вычислять это, а хранить индекс ячейки в трехмерном массиве в texture_memory
    //!    и обращаться по приведненным координатм [x / md->cSize.x][y / md->cSize.y]
    //!    есть ли в этом смысл?


    //! традиционный способ
    //md->clist[index] = atomicExch(&md->chead[c], index);

    //! прямой способ
    j = atomicAdd(&(md->cells[c][0]), 1);    // increase the number of particles in cell[c] (it keeps in the 0th element of cell[cell_index] array)
    if (j >= md->maxAtPerCell)
        printf("at[%d](%f,%f,%f) too many atoms (%d) in the cell[%d]\n", index, xyz.x, xyz.y, xyz.z, j, c);
    md->cells[c][j + 1] = index;                 // and save index. j+1 because the first element is reserved for quantity
}
// end 'keep_in_cell' function

__global__ void reset_quantities(cudaMD* md)
// set some values, as energies to zero
{
    int i, j;

    md->engCoul1 = 0.f;
    md->engCoul2 = 0.f;
    md->engCoul3 = 0.f;
    md->engElecField = 0.f;
    //md->engKin = 0.f;
    md->engTot = 0.f;
    md->engTemp = 0.f;      // radiative thermostat
    md->engVdW = 0.f;
    md->engBond = 0.f;
    md->engAngl = 0.f;
    md->virial = 0.f;
    md->G = 0.f;

    if (md->use_coul == 2) // Ewald
        for (i = 0; i < md->nKvec; i++)
            md->qDens[i] = make_float2(0.f, 0.f);

    md->curEng = 0;
    md->curVect = 0;

    md->nFreeEls = 0;       // for electron jumps
/*
    for (i = 0; i < 10; i++)
    {
        for (j = 0; j < md->nFirstPairs[i]; j++)
            printf("pair[%d][%d]=(%d, %d, %d) shift=(%f, %f, %f)\n", i, j, md->firstPairs[i][j].x, md->firstPairs[i][j].y, md->firstPairs[i][j].z, md->firstShifts[i][j].x, md->firstShifts[i][j].y, md->firstShifts[i][j].z);
    }
*/
    //for (i = 0; i < 10; i++)
      //  printf("pair[%d]=(%d, %d, %d) shift=(%f, %f, %f)\n", i, md->cellBlocks[i].x, md->cellBlocks[i].y, md->cellBlocks[i].z, md->secShifts[i][0].x, md->secShifts[i][0].y, md->secShifts[i][0].z);

#ifdef DEBUG_MODE
    if (threadIdx.x == 0)
        for (i = 1; i < md->nBndTypes; i++)
            if ((md->bondTypes[i].spec1 < 0) || (md->bondTypes[i].spec2 < 0) || (md->bondTypes[i].spec1 >= MX_SPEC) || (md->bondTypes[i].spec2 >= MX_SPEC))
            {
                printf("aft reset: bnd[%d] spec1=%d spec2=%d\n", i, md->bondTypes[i].spec1, md->bondTypes[i].spec2);
                //md->xyz[9999999999].x = 15.f;  // crash cuda
            }
#endif

}

__global__ void print_stat(int iStep, cudaMD* md)
{
    printf("%d x1=%.2f", iStep, md->xyz[0].x);
    if (md->use_coul)
        printf(" C1=%.3G, C2=%.3G ", md->engCoul1, md->engCoul2);
    if (md->use_bnd == 2)
        printf(" nBnd=%d", md->nBond);
    if (md->use_bnd)
        printf(" bndEng=%.3G", md->engBond);
    printf(" Kin=%.3G Vdw=%.3G", md->engKin, md->engVdW);
    if (md->tstat == 2) // radiative thermostat
        printf(" K+P=%.3G U=%.3G", md->engPotKin, md->engTemp);
    printf(" Tot=%.3G P=%.0f V=%f G=%f", md->engTot, md->pressure, md->virial, md->G);
    printf("\n");
}

__global__ void verlet_1stage(int iStep, int atPerBlock, int atPerThread, cudaMD* md)
// the first stage of velocity verlet algorithm with neighbors saving
{
    int i, t;
    float charge;
    float engElecField = 0.f;
    __shared__ float shEngElField;   // shared copy of energy field variable

    //float rm;
    float x0, y0, z0;   // to debug


    // reset accumulators of energy field
    if (threadIdx.x == 0)   // 0th thread of 0th block, reset some energies
    {
        shEngElField = 0;
        if (blockIdx.x == 0)
        {
            //     md->engElecField = 0;  //! убрать в reset
#ifdef TX_CHARGE
            /*
            float p1, p2, p3, p4;
            p1 = tex2D(qProd, 0, 0);
            //p1 = (float)(md->texChProd[0]);
            p2 = tex2D(qProd, 0, 1);
            p3 = tex2D(qProd, 1, 0);
            p4 = tex2D(qProd, 1, 1);
            printf("prod of charges are: %f %f %f %f\n", p1, p2, p3, p4);
            */
#endif
        }
    }
    __syncthreads();

#ifdef DEBUG_MODE
    if (threadIdx.x == 0)
        for (i = 1; i < md->nBndTypes; i++)
            if ((md->bondTypes[i].spec1 < 0) || (md->bondTypes[i].spec2 < 0) || (md->bondTypes[i].spec1 >= MX_SPEC) || (md->bondTypes[i].spec2 >= MX_SPEC))
            {
                printf("bef verlete1: bl[%d] step %d bnd[%d] spec1=%d spec2=%d\n", blockIdx.x, iStep, i, md->bondTypes[i].spec1, md->bondTypes[i].spec2);
                md->xyz[9999999999].x = 15.f;  // crash cuda
            }
#endif


    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);
    //printf("start verlet1[%d,%d] atoms=%d .. <%d\n", blockIdx.x, threadIdx.x, id0, N);
    for (i = id0; i < N; i++)
    {
        //if (i == 0)
          //  printf("v1_0(%d) x=%f vls=%f PosMomX=%f\n", iStep, md->xyz[0].x, md->vls[0].x, md->posMom.x);
        
        t = md->types[i];
#ifdef DEBUG_MODE
        if ((t < 0) || (t >= MX_SPEC))
        {
            printf("WRONG SPECIE %d of atom[%d]!\n", t, i);
            continue;
        }
#endif
        charge = md->specs[t].charge;


        //the first stage of velocity update:
        //  v = v + f/m * 0.5 dt
        md->vls[i].x += md->rMasshdT[i] * md->frs[i].x;
        md->vls[i].y += md->rMasshdT[i] * md->frs[i].y;
        md->vls[i].z += md->rMasshdT[i] * md->frs[i].z;
        //if (i == 0)
          //  printf("v05_1(%d) x=%f vls=%f frc=%f\n", iStep, md->xyz[0].x, md->vls[0].x, md->frs[0].x);

        //rm = tex1Dfetch(rMassHdt, i);
        //md->vls[i].x += rm * md->frs[i].x;
        //md->vls[i].y += rm * md->frs[i].y;
        //md->vls[i].z += rm * md->frs[i].z;


        x0 = md->xyz[i].x;
        y0 = md->xyz[i].y;
        z0 = md->xyz[i].z;

        //if (isnan(x0))
          //  printf("th(%d,%d): x0[%d]=%f, v=%f f=%f\n", blockIdx.x, threadIdx.x, i, x0, md->vls[i].x, md->frs[i].x);

        // x = x + v * dt
#ifdef USE_CONST    // use constant memory for timestep
        if (!md->specs[t].frozen)
        {
            md->xyz[i].x += md->vls[i].x * tStep;
            md->xyz[i].y += md->vls[i].y * tStep;
            md->xyz[i].z += md->vls[i].z * tStep;
        }
#else
        if (!md->specs[t].frozen)
        {
            md->xyz[i].x += md->vls[i].x * md->tSt;
            md->xyz[i].y += md->vls[i].y * md->tSt;
            md->xyz[i].z += md->vls[i].z * md->tSt;
        }
#endif
        //if (i == 0)
          //  printf("v1_1(%d) x=%f\n", iStep, md->xyz[0].x);

        // apply periodic boundaries
        put_periodic(md->xyz[i], md->vls[i], md->masses[i], md->types[i], md);

        if (md->xyz[i].x >= md->leng.x)
            printf("verl1(%d) th(%d,%d): x[%d]=%f->%f, v=%f f=%f\n", iStep, blockIdx.x, threadIdx.x, i, x0, md->xyz[i].x, md->vls[i].x, md->frs[i].x);
        else
            if (md->xyz[i].x < 0)
                printf("verl1(%d) th(%d,%d): x[%d]=%f->%f, v=%f f=%f\n", iStep, blockIdx.x, threadIdx.x, i, x0, md->xyz[i].x, md->vls[i].x, md->frs[i].x);
        if (md->xyz[i].y >= md->leng.y)
            printf("verl1(%d) th(%d,%d): y[%d]=%f->%f, v=%f f=%f\n", iStep, blockIdx.x, threadIdx.x, i, y0, md->xyz[i].y, md->vls[i].y, md->frs[i].y);
        else
            if (md->xyz[i].y < 0)
                printf("verl1(%d) th(%d,%d): y[%d]=%f->%f, v=%f f=%f\n", iStep, blockIdx.x, threadIdx.x, i, y0, md->xyz[i].y, md->vls[i].y, md->frs[i].y);
        if (md->xyz[i].z >= md->leng.z)
            printf("verl1(%d) th(%d,%d): z[%d]=%f->%f, v=%f f=%f\n", iStep, blockIdx.x, threadIdx.x, i, z0, md->xyz[i].z, md->vls[i].z, md->frs[i].z);
        else
            if (md->xyz[i].z < 0)
                printf("verl1(%d) th(%d,%d): z[%d]=%f->%f, v=%f f=%f\n", iStep, blockIdx.x, threadIdx.x, i, z0, md->xyz[i].z, md->vls[i].z, md->frs[i].z);


        //if (i == 0)
          //  printf("v1_2(%d) x=%f\n", iStep, md->xyz[0].x);

        //save the atom in cell list
#ifndef USE_ALLPAIR
        //printf("verlet1: rev=(%f %f %f)\n", md->cRevSize.x, md->cRevSize.y, md->cRevSize.z);
#ifdef USE_FASTLIST
        //if (isnan(md->xyz[i].x))
          //  printf("(%d) try call count_cell with nan value, i=%d, vx=%f\n", iStep, i, md->vls[i].x);
        count_cell(i, md->xyz[i], md);
#else
        keep_in_cell(i, md->xyz[i], md);
#endif
#endif
        //external fields:
        //  Eng = q * x * dU/dx
        engElecField += charge * (md->xyz[i].x * md->elecField.x + md->xyz[i].y * md->elecField.y + md->xyz[i].z * md->elecField.z);

        //! сброшу здесь, хотя при использовании cell_list это не обязательно, это нужно если мы используем all_pair
        //! теперь обязательно, поскольку apply_bonds выполняется до cell_list
        //! а ещё обязательно, поскольку используется внешнее эл. поле
        md->frs[i] = make_float3(-charge * md->elecField.x, -charge * md->elecField.y, -charge * md->elecField.z);    // F = -q * dU/dx

        if (md->use_bnd == 2)   // variable bonds
        {
            md->neighToBind[i] = 0;
            md->canBind[i] = 0;
            md->r2Min[i] = 999999;
        }
        if (md->use_ejump)
        {
            md->r2Jumps[i] = 999999999;
            md->accIds[i] = -1;
        }
#ifdef DEBUG_MODE
        md->nCult[i]++;
#endif
    }


    // copy energy of elec field, at the first to the shared variable...
    atomicAdd(&shEngElField, engElecField);
    __syncthreads();

    //.. then to global variable
    if (threadIdx.x == 0)
    {
        atomicAdd(&md->engElecField, shEngElField);
    }
    //printf("end verlet1[%d,%d] atoms=%d .. <%d\n", blockIdx.x, threadIdx.x, id0, N);

#ifdef DEBUG_MODE
    if (threadIdx.x == 0)
        for (i = 1; i < md->nBndTypes; i++)
            if ((md->bondTypes[i].spec1 < 0) || (md->bondTypes[i].spec2 < 0) || (md->bondTypes[i].spec1 >= MX_SPEC) || (md->bondTypes[i].spec2 >= MX_SPEC))
            {
                printf("aft verlete1: bl[%d] step %d bnd[%d] spec1=%d spec2=%d\n", blockIdx.x, iStep, i, md->bondTypes[i].spec1, md->bondTypes[i].spec2);
                md->xyz[9999999999].x = 15.f;  // crash cuda
            }
#endif

}
// end 'verlet_1stage' function

__global__ void verlet_2stage(int atPerBlock, int atPerThread, int iStep, cudaMD* md)
// the second part of verlet integrator (v = v + 0.5 f/m dt), save kinetic energy in sim
{
    //double rm;  // rmasshdt[i]
    int i;

    float kinE = 0.f;  // kinetic energy
    float vir = 0.f;    // virial
    float G = 0.f;      // virial analog for momentum
    // their shared versions:
    __shared__ float shKinE;  // shared version of kinetic energy
    __shared__ float shvir;
    __shared__ float shG;

    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);

    // for debugging
    //float vx0, vx01;
    int auto_cap = 1;
    float scale, mx, mx_force = 1e4;
    int j;

    //the first thread reset system kinetic energy
    if (threadIdx.x == 0)
    {
        shKinE = 0.f;
        shvir = 0.f;
        shG = 0.f;
    }
    __syncthreads();

    //vx0 = md->vls[id0].x;

    //the second stage of the velocities update
    for (i = id0; i < N; i++)
    {
        //if (i == 0)
          //  printf("vlt2: vel[0].x=%f\n", md->vls[0].x);

        //vx01 = md->vls[i].x;

        if (auto_cap)
        {
            mx = 0.f;
            if (md->frs[i].x > mx_force)
                mx = md->frs[i].x;
            else if (md->frs[i].x < -mx_force)
                mx = -md->frs[i].x;

            if (md->frs[i].y > mx_force)
                mx = max(mx, md->frs[i].y);
            else if (md->frs[i].y < -mx_force)
                mx = max(mx, -md->frs[i].y);

            if (md->frs[i].z > mx_force)
                mx = max(mx, md->frs[i].z);
            else if (md->frs[i].z < -mx_force)
                mx = max(mx, -md->frs[i].z);

            if (mx > 1.f)   // that automatically means that |mx| > mx_force
            {
                scale = 0.1f * mx_force / mx;
                md->frs[i].x *= scale;
                md->frs[i].y *= scale;
                md->frs[i].z *= scale;
            }

        }

        //if (md->frs[i].x > 30.f)
          //  printf("vel[%d].x=%f dv=%f f=%f\n", i, md->vls[i].x, md->rMasshdT[i] * md->frs[i].x, md->frs[i].x);
        md->vls[i].x += md->rMasshdT[i] * md->frs[i].x;
        md->vls[i].y += md->rMasshdT[i] * md->frs[i].y;
        md->vls[i].z += md->rMasshdT[i] * md->frs[i].z;

        //rm = tex1Dfetch(rMassHdt, i);
        //md->vls[i].x += rm * md->frs[i].x;
        //md->vls[i].y += rm * md->frs[i].y;
        //md->vls[i].z += rm * md->frs[i].z;

        //if (isnan(md->vls[i].x))
          //  printf("v2nan: %d: vx[%d]=%f -> %f rm=%f f=%f\n", iStep, id0, vx01, md->vls[id0].x, md->rMasshdT[id0], md->frs[id0].x);


        // saving mv2, virial and G
        kinE += (md->vls[i].x * md->vls[i].x + md->vls[i].y * md->vls[i].y + md->vls[i].z * md->vls[i].z) * md->masses[i];
        vir += sc_prod(md->xyz[i], md->frs[i]);
        G += sc_prod(md->xyz[i], md->vls[i]) * md->masses[i];

#ifdef DEBUG_MODE
        md->nCult[i]++;
#endif
    }

    //accumulate kinetic energy, at first to shared memory...
    atomicAdd(&shKinE, 0.5f * kinE);   // kinE = mv2/2
    atomicAdd(&shvir, vir);
    atomicAdd(&shG, G);
    __syncthreads();
    //... then to global memory
    if (threadIdx.x == 0) // 0th thread
    {
        atomicAdd(&md->engKin, shKinE);
        atomicAdd(&md->virial, shvir);
        atomicAdd(&md->G, shG);
    }

#ifdef DEBUG_MODE
    if (threadIdx.x == 0)
        for (i = 1; i < md->nBndTypes; i++)
            if ((md->bondTypes[i].spec1 < 0) || (md->bondTypes[i].spec2 < 0) || (md->bondTypes[i].spec1 >= MX_SPEC) || (md->bondTypes[i].spec2 >= MX_SPEC))
            {
                printf("aft verlet2: bl[%d] step %d bnd[%d] spec1=%d spec2=%d\n", blockIdx.x, iStep, i, md->bondTypes[i].spec1, md->bondTypes[i].spec2);
                md->xyz[9999999999].x = 15.f;  // crash cuda
            }
#endif

}
// end 'verlet_2stage' function

__global__ void zero_vel(int atPerBlock, int atPerThread, int iStep, cudaMD* md)
// set all velocities as zero
{
    int i;

    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);


    for (i = id0; i < N; i++)
    {
        md->vls[i] = make_float3(0.f, 0.f, 0.f);
    }

    if (blockIdx.x == 0)
        if (threadIdx.x == 0) // 0th thread
        {
            md->engKin = 0.f;
        }

}
// end 'zero_vel' function

#ifdef DEBUG_MODE
__global__ void clear_ncult(cudaMD* md)
{
    int i;
    for (i = 0; i < md->nAt; i++)
        md->nCult[i] = 0;

    for (i = 0; i < md->nPair; i++)
        md->nPairCult[i] = 0;

    for (i = 0; i < md->nCell; i++)
        md->nCelCult[i] = 0;

    md->atInList = 0;
    md->dublInList = 0;
    md->nFCall = 0;
    md->nVdWcall = 0;
    md->sqrCoul = 0.0f;
    md->nAllPair = 0;
}

__global__ void verify_ncult(int n, int nP, int nC, cudaMD* md)
{
    int i;
    int fl = 1;

    for (i = 0; i < md->nAt; i++)
        if (md->nCult[i] != n)
        {
            printf("n atom cult[%d]=%d\n", i, md->nCult[i]);
            fl = 0;
        }
    if (fl)
        printf("ncult ok\n");
    /*
        fl = 1;
        for (i = 0; i < md->nPair; i++)
            if (md->nPairCult[i] != nP)
            {
                //printf("nPair[%d]=%d\n", i, md->nPairCult[i]);
                fl = 0;
            }
        if (fl)
            printf("n pair(%d) cult ok\n", md->nPair);

        fl = 1;
        for (i = 0; i < md->nCell; i++)
            if (md->nCelCult[i] != nC)
            {
                //printf("nCellCult[%d]=%d\n", i, md->nCelCult[i]);
                fl = 0;
            }
        if (fl)
            printf("n cell(%d) cult ok\n", md->nCell);
    */
    //printf("force func call %d(%d) times (vdw). eVdW = %f. eCoul1 = %f eCoul^2=%f all_pair=%d\n", md->nFCall, md->nVdWcall, md->engVdW, md->engCoul1, md->sqrCoul, md->nAllPair);

}
#endif


__global__ void some_info(int iStep, cudaMD* md)
{
#ifdef DEBUG_MODE
    printf("%d: atm in list: %d (%d dublicates, pure: %d)\n", iStep, md->atInList, md->dublInList, md->atInList - md->dublInList);
#endif
}

__global__ void clear_clist(/*int cellPerBlock, int cellPerThread, */cudaMD* md)
{
    int step = ceil((double)md->nCell / (double)blockDim.x / (double)gridDim.x);
    int i0 = step * (blockIdx.x * blockDim.x + threadIdx.x);
    int N = min(i0 + step, md->nCell);
    int i;

    /*
        int i;
        int id0 = blockIdx.x * cellPerBlock + threadIdx.x * cellPerThread;
        int N = min(id0 + cellPerThread, md->nCell);
    */

    for (i = i0; i < N; i++)
    {
#ifdef USE_FASTLIST
        md->nAtInCell[i] = 0;
#else
        md->cells[i][0] = 0;
#endif
    }
}

__global__ void verify_clist(cudaMD* md)
{
    int step = ceil((double)md->nCell / (double)blockDim.x / (double)gridDim.x);
    int i0 = step * (blockIdx.x * blockDim.x + threadIdx.x);
    int N = min(i0 + step, md->nCell);
    int i, j, k, nAt, ta, nd;

    __shared__ int totAt;
    __shared__ int nDubl;
    if (threadIdx.x == 0)
    {
        totAt = 0;
        nDubl = 0;
    }

    ta = 0;
    nd = 0;
    for (i = i0; i < N; i++)
    {
        nAt = md->cells[i][0];
        for (j = 0; j < nAt - 1; j++)
            for (k = j + 1; k < nAt; k++)
            {
                if (md->cells[i][j + 1] >= md->nAt)
                    printf("cell[%d](nAt=%d) out range! [%d]==%d\n", i, nAt, j, md->cells[i][j + 1]);
                if (md->cells[i][j + 1] < 0)
                    printf("cell[%d](nAt=%d) out range! [%d]==%d\n", i, nAt, j, md->cells[i][j + 1]);
                if (md->cells[i][j + 1] == md->cells[i][k + 1])
                {
                    //  printf("cell[%d](nAt=%d) [%d]=[%d]=%d\n", i, nAt, j, k, md->cells[i][j + 1]);
                    nd++;
                }

            }
        ta += nAt;
    }
    atomicAdd(&totAt, ta);
    atomicAdd(&nDubl, nd);
    __syncthreads();

#ifdef DEBUG_MODE
    if (threadIdx.x == 0)
    {
        //printf("tA[%d]=%d\n", blockIdx.x, totAt);
        atomicAdd(&(md->atInList), totAt);
        atomicAdd(&(md->dublInList), nDubl);
        //if (blockIdx.x == 0)
          //  printf("%d in cell list (%d dublicates)\n", md->atInList, md->dublInList);
    }
#endif
}

__global__ void verify_forces(int atPerBlock, int atPerThread, int iStep, cudaMD* md, int id)
{
    int i;

    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);

    for (i = id0; i < N; i++)
    {
        if (isnan(md->frs[i].x))
            printf("%d step(%d) atm[%d]type[%d] has force(%f %f %f)\n", id, iStep, i, md->types[i], md->frs[i].x, md->frs[i].y, md->frs[i].z);
        else
            if (isnan(md->frs[i].y))
                printf("%d step(%d) atm[%d]type[%d] has force(%f %f %f)\n", id, iStep, i, md->types[i], md->frs[i].x, md->frs[i].y, md->frs[i].z);
            else
                if (isnan(md->frs[i].z))
                    printf("%d step(%d) atm[%d]type[%d] has force(%f %f %f)\n", id, iStep, i, md->types[i], md->frs[i].x, md->frs[i].y, md->frs[i].z);
    }
}
