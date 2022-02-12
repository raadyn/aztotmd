// unit for preparation data for CUDA
//#include <math.h>

#include "defines.h"
#include "cuStruct.h"
#include "dataStruct.h" 
#include "sys_init.h"
#include "vdw.h"
#include "cuVdW.h"
#include "cuBonds.h"
#include "cuAngles.h"
#include "cuElec.h"
#include "const.h"  // kB
#include "cuInit.h"
#include "utils.h"  // int_size..
#include "temperature.h"
#include "cuStat.h" // init_cuda_stat, free_cuda_stat
#include "cuUtils.h"
#ifdef USE_FASTLIST
//#include "cuFastList.h"
#endif
#include "cuSort.h"
#include "cuCellList.h"
#include "cuTemp.h"
#include "cuEjump.h"

void bonds_to_host(int4* buffer, cudaMD* hmd, /*int number,*/ Field* fld, hostManagMD *man)
// copy bonds from device to host
{
    int i;

    cudaMemcpy(&(fld->nBonds), man->nBndPtr, int_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(buffer, hmd->bonds, fld->nBonds * int4_size, cudaMemcpyDeviceToHost);
    for (i = 0; i < fld->nBonds; i++)
    {
        fld->at1[i] = buffer[i].x;
        fld->at2[i] = buffer[i].y;
        fld->bTypes[i] = buffer[i].z;
    }

    // we also need to copy bond quantities:
    int sz = (fld->nBdata - 1) * sizeof(cudaBond);
    cudaBond* btypes = (cudaBond*)malloc(sz);   // no need to copy [0] element, as it's reserved for 'no bond'
    cudaMemcpy(btypes, (void*)(&hmd->bondTypes[1]), sz, cudaMemcpyDeviceToHost);
    for (i = 1; i < fld->nBdata; i++)
        fld->bdata[i].number = btypes[i - 1].count;
    free(btypes);
 }

void angles_to_host(int4* buffer, cudaMD* md, Field* fld, hostManagMD* man)
// copy angles from device to host
{
    int i;

    cudaMemcpy(&(fld->nAngles), man->nAngPtr, int_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(buffer, md->angles, fld->nAngles * int4_size, cudaMemcpyDeviceToHost);
    for (i = 0; i < fld->nAngles; i++)
    {
        fld->centrs[i] = buffer[i].x;
        fld->lig1[i] = buffer[i].y;
        fld->lig2[i] = buffer[i].z;
        fld->angTypes[i] = buffer[i].w;
    }
}

__global__ void save_ptrs(int **ptr_arr, cudaMD *dmd)
// dmd - pointer device exemplar of cudaMD struct
{
    ptr_arr[0] = &(dmd->nAt);
    ptr_arr[1] = &(dmd->nBond);
    ptr_arr[2] = &(dmd->nAngle);
}

int read_cuda(Field *fld, cudaMD *hmd, hostManagMD *man)
// read cuda settings from "cuda.txt"
{
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    
    FILE *f = fopen("cuda.txt", "r");
    if (f == NULL)
    {
        printf("ERROR[a001]! Fatal Error. Can't open cuda.txt file\n");
        return 0;
    }

    if (!find_int_def(f, " multproc %d", man->nMultProc, 0))
    {
        printf("WARNING[b012]: 'multproc' directive is not specified in cuda.txt, default value of 0 (=maximal number of multiprocessors) is used\n");
    }
    if (man->nMultProc == 0)    // the value of 0 means use all multiprocessors
        man->nMultProc = devProp.multiProcessorCount;

    // number of the single processor can be found only in documentation
    if (!find_int_def(f, " singproc %d", man->nSingProc, 64))
    {
        printf("WARNING[b013]: 'singproc' directive is not specified in cuda.txt, default value of 64 is used\n");
    }
    man->totCores = man->nMultProc * man->nSingProc;

    //! перенести это в cuStat.cu
    if (!find_int_def(f, " nstep stat %d", man->stat.nstep, 10))
    {
        printf("WARNING[a003]: 'nstep stat' directive is not specified in cuda.txt, default value of 10 is used\n");
    }
    //! добавить проверку, что траектории нужны
    if (!find_int_def(f, " nstep traj %d", man->nstep_traj, 10))
    {
        printf("WARNING[b007]: 'nstep traj' directive is not specified in cuda.txt, default value of 10 is used\n");
    }

    //! добавить проверку, что связанные траектории нужны
    if (!find_int_def(f, " nstep bindtraj %d", man->nstep_bindtraj, 40))
    {
        printf("WARNING[b009]: 'nstep bindtraj' directive is not specified in cuda.txt, default value of 40 is used\n");
    }
    if (!find_int_def(f, " bindtraj threads %d", man->bindTrajPerThread, 1))
    {
        printf("WARNING[b010]: 'bindtraj thread' directive is not specified in cuda.txt, default values of 1 and 32 are used\n");
        man->nBindTrajThread = 32;
    }
    else
    {
        fscanf(f, "%d", &man->nBindTrajThread);
    }
    

    //! temp!
    man->sjmp.nstep = 10;

    if (!find_int_def(f, " nstep msdstat %d", man->smsd.nstep, 10))
    {
        printf("WARNING[a004]: 'nstep msdstat' directive is not specified in cuda.txt, default value of 10 is used\n");
    }

    if (fld->nBdata)
        if (!find_int_def(f, " nstep bondstat %d", man->sbnd.nstep, 10))
        {
            printf("WARNING[a005]: 'nstep bondstat' directive is not specified in cuda.txt, default value of 10 is used\n");
        }

    // number of threads for cell list routines
    if (!find_int_def(f, " nthread a %d", man->pairThreadA, 16))
    {
        printf("WARNING[a006]: 'nthread a ' directive is not specified in cuda.txt, default value of 16 is used\n");
    }
    if (!find_int_def(f, " nthread b %d", man->pairThreadB, 32))
    {
        printf("WARNING[a006]: 'nthread b ' directive is not specified in cuda.txt, default value of 32 is used\n");
    }
    // maximal number of cell in cellBlock for bypassing according to the last type (with sorting)
    if (!find_int_def(f, " maxcellinblock %d", man->mxCellInBlock, 10))
    {
        printf("WARNING[b014]: 'maxcellinblock' directive is not specified in cuda.txt, default value of 10 is used\n");
    }



    fclose(f);
    return 1;
}

void init_cuda_box(Box *box, cudaMD *h_md)
{
    //! only for rectangular geometry!
    h_md->leng = make_float3((float)box->la, (float)box->lb, (float)box->lc);

    h_md->halfLeng = make_float3(0.5f * h_md->leng.x, 0.5f * h_md->leng.y, 0.5f * h_md->leng.z);   
    h_md->revLeng = make_float3(1.f / h_md->leng.x, 1.f / h_md->leng.y, 1.f / h_md->leng.z);       
    h_md->edgeArea = make_float3(h_md->leng.y * h_md->leng.z, h_md->leng.x * h_md->leng.z, h_md->leng.x * h_md->leng.y);    //! are they neccessary?
    h_md->revEdgeArea = make_float3(1.f / h_md->edgeArea.x, 1.f / h_md->edgeArea.y, 1.f / h_md->edgeArea.z);
    h_md->volume = h_md->leng.x * h_md->leng.y * h_md->leng.z;
}

cudaMD* init_cudaMD(Atoms* atm, Field* fld, Sim* sim, TStat* tstat, Box* bx, Elec* elec, hostManagMD* man, cudaMD *h_md)
// prepare everything for CUDA execution based on loaded data for serial execution
{
    int i, j, k;
    int xyzsize = atm->mxAt * float3_size;
    int nsize = atm->mxAt * int_size;
    int flsize = atm->mxAt * float_size;

    // 0 CUDA SETTINGS
    if (!read_cuda(fld, h_md, man))
        return NULL;

    // GENERAL SETTINGS: timestep, external field...
    h_md->tSt = (float)sim->tSt;
#ifdef USE_CONST
    cudaMemcpyToSymbol(&tStep, &(h_md->tSt), sizeof(float), 0, cudaMemcpyHostToDevice);
#endif
    h_md->elecField = make_float3((float)sim->Ux, (float)sim->Uy, (float)sim->Uz);

    // 1 ATOMS DATA
    h_md->nAt = atm->nAt;
    h_md->mxAt = atm->mxAt;
    float3* h_xyz = (float3*)malloc(xyzsize);
    float3* h_vls = (float3*)malloc(xyzsize);
    float3* h_frc = (float3*)malloc(xyzsize);
    float* h_rMasshdT = (float*)malloc(flsize);
    float* h_masses = (float*)malloc(flsize);
    for (i = 0; i < atm->nAt; i++)
    {
        h_xyz[i] = make_float3((float)atm->xs[i], (float)atm->ys[i], (float)atm->zs[i]);
        h_vls[i] = make_float3((float)atm->vxs[i], (float)atm->vys[i], (float)atm->vzs[i]);
        h_frc[i] = make_float3((float)atm->fxs[i], (float)atm->fys[i], (float)atm->fzs[i]);
        h_masses[i] = (float)fld->species[atm->types[i]].mass;
        h_rMasshdT[i] = 0.5f * (float)sim->tSt / h_masses[i];
    }
    data_to_device((void**)&(h_md->xyz), h_xyz, xyzsize);
    data_to_device((void**)&(h_md->vls), h_vls, xyzsize);
    data_to_device((void**)&(h_md->frs), h_frc, xyzsize);
    data_to_device((void**)&(h_md->types), atm->types, nsize);
    data_to_device((void**)&(h_md->masses), h_masses, flsize);
    data_to_device((void**)&(h_md->rMasshdT), h_rMasshdT, flsize);
    free(h_xyz);
    free(h_vls);
    free(h_frc);
    free(h_masses);
    free(h_rMasshdT);

    // 2 FORCE FIELD
    h_md->tdep_force = fld->is_tdep;
    cudaSpec* h_specs = (cudaSpec*)malloc(fld->nSpec * sizeof(cudaSpec));
    cudaVdW* h_ppots = (cudaVdW*)malloc(fld->nVdW * sizeof(cudaVdW));
    cudaVdW*** h_vdw = (cudaVdW***)malloc(fld->nSpec * sizeof(void*));  // 2d array to pointer to cudaVdW

    cudaVdW* d_ppots;
    cudaMalloc((void**)&d_ppots, fld->nVdW * sizeof(cudaVdW));

    cudaVdW*** d_vdw;
    cudaVdW** vdw_i;
    cudaMalloc((void**)&d_vdw, fld->nSpec * sizeof(cudaVdW**));

    // charges
#ifdef TX_CHARGE
    float* chprods = (float*)malloc(sizeof(float) * fld->nSpec * fld->nSpec);
#endif
    //float* qiqj = (float*)malloc(fld->nSpec * sizeof(float));
    float** h_chProd = (float**)malloc(fld->nSpec * pointer_size);
    float** d_chProd;
    float* chProd_i;
    cudaMalloc((void**)&d_chProd, fld->nSpec * pointer_size);

    // species propreties
    h_md->nSpec = fld->nSpec;
    for (i = 0; i < fld->nSpec; i++)
    {
        h_chProd[i] = (float*)malloc(fld->nSpec * float_size);
        cudaMalloc((void**)&chProd_i, fld->nSpec * float_size);
        h_vdw[i] = (cudaVdW**)malloc(fld->nSpec * pointer_size);
        cudaMalloc((void**)&vdw_i, fld->nSpec * pointer_size);
        for (j = 0; j < fld->nSpec; j++)
        {
            h_chProd[i][j] = (float)(fld->species[i].charge * fld->species[j].charge);
            //qiqj[j] = (float)(fld->species[i].charge * fld->species[j].charge);
            h_vdw[i][j] = NULL;
#ifdef TX_CHARGE
            chprods[i * fld->nSpec + j] = (float)(fld->species[i].charge * fld->species[j].charge);
            chprods[j * fld->nSpec + i] = (float)(fld->species[i].charge * fld->species[j].charge);
#endif
            for (k = 0; k < fld->nVdW; k++)
                if (fld->vdws[i][j] == &fld->pairpots[k])
                    h_vdw[i][j] = &d_ppots[k];  //! {...; break;} ???
        }
        //array_to_device((void**)&(h_md->chProd[i]), qiqj, fld->nSpec * sizeof(float));
        cudaMemcpy(chProd_i, h_chProd[i], fld->nSpec * float_size, cudaMemcpyHostToDevice);
        free(h_chProd[i]);
        h_chProd[i] = chProd_i;
        cudaMemcpy(vdw_i, h_vdw[i], fld->nSpec * pointer_size, cudaMemcpyHostToDevice);
        free(h_vdw[i]);
        h_vdw[i] = vdw_i;
        h_specs[i].number = fld->species[i].number;
        h_specs[i].displ = 0.0;
        h_specs[i].vaf = 0.0;
        h_specs[i].mass = fld->species[i].mass;
        h_specs[i].charge = fld->species[i].charge;
        h_specs[i].energy = fld->species[i].energy;
        h_specs[i].charged = fld->species[i].charged;
        h_specs[i].donacc = fld->species[i].donacc;
        h_specs[i].oxForm = fld->species[i].oxForm;
        h_specs[i].redForm = fld->species[i].redForm;
        h_specs[i].varNumber = fld->species[i].varNumber;
        h_specs[i].nFreeEl = fld->species[i].nFreeEl;
        //h_specs[i].canBond = fld->species[i].canBond;
        //h_specs[i].angleType = fld->species[i].angleType;
        h_specs[i].idCentral = fld->species[i].idCentral;
        h_specs[i].idCounter = fld->species[i].idCounter;
        h_specs[i].frozen = fld->species[i].frozen;
        h_specs[i].nuclei = fld->species[i].nuclei;
        h_specs[i].radA = fld->species[i].radA;
        h_specs[i].radB = fld->species[i].radB;
        h_specs[i].mxEng = fld->species[i].mxEng;
    }
    cudaMemcpy(d_chProd, h_chProd, fld->nSpec * pointer_size, cudaMemcpyHostToDevice);
    free(h_chProd);
    cudaMemcpy(d_vdw, h_vdw, fld->nSpec * pointer_size, cudaMemcpyHostToDevice);

    // counters for species crossing box
    //! fill by zero?
    cudaMalloc((void**)&(h_md->specAcBoxPos), fld->nSpec * int3_size);
    cudaMalloc((void**)&(h_md->specAcBoxNeg), fld->nSpec * int3_size);

    //van der Waals
    h_md->pairpots = d_ppots;
    h_md->vdws = d_vdw;

    data_to_device((void**)&(h_md->specs), h_specs, fld->nSpec * sizeof(cudaSpec));
    free(h_specs);
    h_md->chProd = d_chProd;

    data_to_device((void**)&(h_md->nnumbers), fld->nnumbers, fld->nNucl * int_size);

    // EXPERIMENTAL: put charges in texture memory
#ifdef TX_CHARGE
    cudaChannelFormatDesc cform = cudaCreateChannelDesc<float>();// (32, 32, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&(h_md->texChProd), &cform, fld->nSpec, fld->nSpec, cudaArrayDefault);
    cudaMemcpyToArray(h_md->texChProd, 0, 0, chprods, fld->nSpec * fld->nSpec * sizeof(float), cudaMemcpyHostToDevice);
    qProd.normalized = 0;
    //qProd.addressMode[0] = cudaAddressModeWrap;
    //qProd.addressMode[1] = cudaAddressModeWrap;
    cudaBindTextureToArray(&qProd, h_md->texChProd, &cform);
    delete[] chprods;
#endif 

    if (fld->nVdW)
    {
        for (i = 0; i < fld->nVdW; i++)
        {
            h_ppots[i].type = fld->pairpots[i].type;
            h_ppots[i].p0 = (float)fld->pairpots[i].p0;
            h_ppots[i].p1 = (float)fld->pairpots[i].p1;
            h_ppots[i].p2 = (float)fld->pairpots[i].p2;
            h_ppots[i].p3 = (float)fld->pairpots[i].p3;
            h_ppots[i].p4 = (float)fld->pairpots[i].p4;
            h_ppots[i].r2cut = (float)fld->pairpots[i].r2cut;
            h_ppots[i].use_radii = fld->pairpots[i].use_radii;
        }
        cudaMemcpy(d_ppots, h_ppots, fld->nVdW * sizeof(cudaVdW), cudaMemcpyHostToDevice);
    }
    free(h_ppots);
    free(h_vdw);

    // 3 RESET COUNTERS
    // energies
    h_md->engKin = 0.f;
    h_md->engTot = 0.f;
    h_md->engTemp = 0.f;    // radiative thermostat
    h_md->engElecField = 0.f;
    h_md->engCoul1 = 0.f;
    h_md->engCoul2 = 0.f;
    h_md->engCoul3 = 0.f;
    h_md->engVdW = 0.f;
    h_md->engPotKin = 0.f;

    // momentum
    h_md->posMom = make_float3(0.f, 0.f, 0.f);
    h_md->negMom = make_float3(0.f, 0.f, 0.f);

    // for pressure:
    h_md->nMom = 20;    //! temp, then from directives
    h_md->iMom = 0;
    //h_md->jMom = 0;
    h_md->posPres = make_float3(0.f, 0.f, 0.f);
    h_md->negPres = make_float3(0.f, 0.f, 0.f);
    h_md->pressure = 0.f;
    cudaMalloc((void**)&(h_md->posMomBuf), h_md->nMom * float_size);
    cudaMalloc((void**)&(h_md->negMomBuf), h_md->nMom * float_size);

    // 4 THEREMOSTAT & TEMPERATURE DATA
    init_cuda_tstat(atm->mxAt, atm, fld, tstat, h_md, man);

    // 5 BOX
    init_cuda_box(bx, h_md);

    // 6 ELEC
    init_cuda_elec(atm, elec, sim, man, h_md);


    // 7 DEFINTION OF hostManagMD variables
    man->atStep = ceil((double)atm->mxAt / man->totCores);   // >= 1
    int mxAtPerBlock = man->atStep * man->nSingProc;
    man->nAtBlock = ceil((double)atm->mxAt / mxAtPerBlock);
    man->nAtThread = man->nSingProc;

    //!  загрузить все МП поровну, 1блок = 1МП, опустим ситуацию, когда число МП меньше кол-ва атомов, ведь для моделирования нам нужны тысячи атомов
    man->atPerBlock = ceil((double)atm->mxAt / man->nMultProc); // число атомов на МП
    man->atPerThread = ceil((double)man->atPerBlock / man->nSingProc);    //! число атомов на поток
    if (man->atPerBlock < (man->atPerThread * man->nSingProc))
        man->atPerBlock = man->atPerThread * man->nSingProc;    // но не меньше, чем число атомов во всех потоках блока

    // some settings for old variant of cell list...
    //dim3 dim;
    //dim.x = 32;
    //dim.y = 2;
    //dim.z = 1;
    //int nB1 = ceil((double)man->nPair1Block / (double)man->pairPerBlock);
    //int nB2 = ceil((double)man->nPair2Block / (double)man->pairPerBlock);

    // 8 CELL LIST
    init_cellList(1, 1, 6, (float)sim->desired_cell_size, atm, fld, elec, h_md, man, bx->type);

    // 9 STATISTICS
    init_cuda_stat(h_md, man, sim, fld, tstat);
    init_cuda_rdf(fld, sim, man, h_md);
    //! нафиг, уже внутри init_cuda_rdf решаем делать n_ или обычный
    if (sim->nuclei_rdf)
        init_cuda_nrdf(fld, sim, man, h_md);
    if (sim->frTraj)    // trajectories
        init_cuda_trajs(atm, sim, h_md, man);
    if (sim->nBindTrajAtoms)    // bind trajectories
        init_cuda_bindtrajs(sim, h_md, man);

    // 10 BONDS AND ANGLES
    h_md->use_angl = sim->use_angl;
    h_md->use_bnd = sim->use_bnd;
    if (h_md->use_bnd)
        init_cuda_bonds(atm, fld, sim, h_md, man);
    if (h_md->use_angl)
        init_cuda_angles(atm->mxAt, nsize, fld, h_md, man);
    // create oldTypes array (used for bonds and angles)
    if (sim->use_angl || sim->use_bnd)  //! maybe == 2, ==2 ??
    {
        int* int_array;
        int_array = (int*)malloc(nsize);
        for (i = 0; i < atm->mxAt; i++)
        {
            int_array[i] = -1;  // oldTypes[i] = -1
        }
        data_to_device((void**)&(h_md->oldTypes), int_array, nsize);
        free(int_array);
    }

    // electron jumps
    init_cuda_ejump(sim, atm, h_md);
    man->bndPerThreadEjump = ceil((double)h_md->mxBond / 32);   //! move into init_cude_ejump

    // for debugging
#ifdef DEBUG_MODE
    cudaMalloc((void**)&(h_md->nCult), sizeof(int) * atm->mxAt);
    cudaMalloc((void**)&(h_md->nPairCult), sizeof(int) * h_md->nPair);
    cudaMalloc((void**)&(h_md->nCelCult), sizeof(int) * h_md->nCell);
#endif

    cudaMD* d_md;
    //data_to_device((void**)&(d_md), h_md, sizeof(cudaMD));
    cudaMalloc((void**)&d_md, sizeof(cudaMD));
    cudaMemcpy(d_md, h_md, sizeof(cudaMD), cudaMemcpyHostToDevice);

    // read pointers to some data
    int sizePtrs = 3 * pointer_size;
    int** d_somePtrs;
    int** h_somePtrs = (int**)malloc(sizePtrs);
    cudaMalloc((void**)&d_somePtrs, sizePtrs);
    save_ptrs<<<1,1>>>(d_somePtrs, d_md);
    cudaMemcpy(h_somePtrs, d_somePtrs, sizePtrs, cudaMemcpyDeviceToHost);
    cudaFree(d_somePtrs);
    man->nAtPtr = h_somePtrs[0];
    man->nBndPtr = h_somePtrs[1];
    man->nAngPtr = h_somePtrs[2];
    free(h_somePtrs);

    return d_md;
}
// end 'init_cudaMD' function

void md_to_host(Atoms* atm, Field* fld, cudaMD *hmd, cudaMD *dmd, hostManagMD* man)
// copy md results from device to host (hmd - host exemplar of cudaMD)
{
    int i;

    //!!!!!!!!!
    // у нас СОРТИРОВКА! следовательно массивы чередуются местами сортированный и нет! т.е. с хоста мы можем ссылаться не на тот массив девайса
    // ! обновляем структуру
    cudaMemcpy(hmd, dmd, sizeof(cudaMD), cudaMemcpyDeviceToHost);

    // read number of atoms
    cudaMemcpy(&(atm->nAt), man->nAtPtr, int_size, cudaMemcpyDeviceToHost);

    // copy atoms data
    int xyz_size = atm->nAt * float3_size;
    float3* xyz = (float3*)malloc(xyz_size);
    float3* vls = (float3*)malloc(xyz_size);
    cudaMemcpy(xyz, hmd->xyz, xyz_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(vls, hmd->vls, xyz_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(atm->types, hmd->types, int_size * atm->nAt, cudaMemcpyDeviceToHost);

    // convert from xyz to native format    (for compability with serial code)
    for (i = 0; i < atm->nAt; i++)
    {
        atm->xs[i] = xyz[i].x;
        atm->ys[i] = xyz[i].y;
        atm->zs[i] = xyz[i].z;
        atm->vxs[i] = vls[i].x;
        atm->vys[i] = vls[i].y;
        atm->vzs[i] = vls[i].z;
    }
    free(xyz);
    free(vls);

    // copy correct number of particles (if variable number)
    cudaSpec* hsps = (cudaSpec*)malloc(fld->nSpec * sizeof(cudaSpec));
    cudaMemcpy(hsps, hmd->specs, fld->nSpec * sizeof(cudaSpec), cudaMemcpyDeviceToHost);
    for (i = 0; i < fld->nSpec; i++)
        fld->species[i].number = hsps[i].number;
    free(hsps);

    // copy bonds and angles
    int mx_int4 = fld->mxBonds;     // one buffer variable for both bonds and angles
    if (fld->mxAngles > mx_int4)
        mx_int4 = fld->mxAngles;
    int4* int4_arr = (int4*)malloc(int4_size * mx_int4);
    if (hmd->use_bnd)
       bonds_to_host(int4_arr, hmd, fld, man);
    if (hmd->use_angl)
        angles_to_host(int4_arr, hmd, fld, man);
    free(int4_arr);
}

void free_device_md(cudaMD* dmd, hostManagMD* man, Sim* sim, Field* fld, TStat *tstat, cudaMD *hmd)
// free all md-arrays on device
// hmd - host exemplar of cudaMD, dmd - on device
{
    cudaMemcpy(hmd, dmd, sizeof(cudaMD), cudaMemcpyDeviceToHost);

    //! внимение, везде, где используется nCell и nAt в будущем надо перезагрузить эти поля

    cudaFree(hmd->xyz);
    cudaFree(hmd->vls);
    cudaFree(hmd->frs);
    cudaFree(hmd->types);
    cudaFree(hmd->masses);
    cudaFree(hmd->rMasshdT);
    //cudaUnbindTexture(rMassHdt);

    cudaFree(hmd->pairpots);
    cudaFree(hmd->specs);
    cudaFree(hmd->nnumbers);
    cudaFree(hmd->specAcBoxNeg);
    cudaFree(hmd->specAcBoxPos);

    cudaFree(hmd->cellPairs);
    cudaFree(hmd->cellShifts);

    cuda2DFree((void**)hmd->vdws, fld->nSpec);    // 2d to pointer
    cuda2DFree((void**)hmd->chProd, fld->nSpec);  //nSpec*nSpec

#ifdef USE_FASTLIST
    //free_fastCellList(hmd, man);
    //free_sort(hmd);
    //free_singleAtomCellList(hmd);

    free_cellList(hmd, man);
#else
    cuda2DFree((void**)hmd->cells, hmd->nCell);   // cells[nCell][maxAtincell + 1]
#endif

    // for pressure:
    cudaFree(hmd->posMomBuf);
    cudaFree(hmd->negMomBuf);

    free_cuda_tstat(tstat, fld, hmd);

#ifdef TX_CHARGE
    cudaUnbindTexture(&qProd);
    cudaFreeArray(hmd.texChProd);
#endif

    free_cuda_elec(hmd);
    
    // bonds:
    if (fld->nBdata)
    {
        cudaFree(hmd->bondTypes);
        cudaFree(hmd->bonds);
        cuda2DFree((void**)hmd->def_bonds, fld->nSpec);
        cudaFree(hmd->parents);
        cudaFree(hmd->nbonds);
        if (hmd->use_bnd == 2)  // binding
        {
            cudaFree(hmd->neighToBind);
            cudaFree(hmd->canBind);
            cudaFree(hmd->r2Min);
            cuda2DFree((void**)hmd->bindBonds, fld->nSpec);
            cuda2DFree((void**)hmd->bindR2, fld->nSpec);
        }
    }
    // end bonds

    // angles:
    if (fld->nAdata)
    {
        cudaFree(hmd->angles);
        cudaFree(hmd->angleTypes);
        cudaFree(hmd->nangles);
        cudaFree(hmd->oldTypes);
        if (hmd->use_angl == 2)
            cudaFree(hmd->specAngles);
    }
    // end angles

    free_cuda_stat(hmd, man);
    free_cuda_rdf(man, hmd);
    if (sim->nuclei_rdf)
        free_cuda_nrdf(man, hmd);

    // electron jumps
    free_cuda_ejump(hmd);

    // trajectories
    if (sim->frTraj)
        free_cuda_trajs(hmd, man);

    // bind trajectories
    if (sim->nBindTrajAtoms)
        free_cuda_bindtrajs(hmd, man);


#ifdef DEBUG_MODE
    cudaFree(hmd->nCult);
#endif
    cudaFree(dmd);
    free(hmd);
}
// end 'final_cudaMD' function
