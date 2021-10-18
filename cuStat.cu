// unit for statistics preparation and output
// to add parameter to output you need to correct functions init_cuda_stat, prepare_stat_addr, start_stat, end_stat and read_cuda (cuInit.cu)

#include "dataStruct.h"
#include "const.h"
#include "utils.h"
#include "cuStruct.h"
#include "cuMDfunc.h"
#include "cuStat.h"
#include "cuUtils.h"
#include "box.h"

// чтобы не писать однотипные функции и вводить однотипные переменные для каждого рода статистик,
//  теперь все статистики сохраняются в один буфер, и управляются одной послдеовательностью 
//  а с хоста идет комманда начиная с какого индекса выполнять копирование и начиная с какого байта писать в буфер
__global__ void write_stat(int iStep, int ind0, int shift, cudaMD* md)
// save current statistics data to buffer on device
{
    int ind = threadIdx.x + ind0;       // global array index
    // define position in buffer to write data
    int sh = /*md->stat_size * md->stat_count*/shift + md->stat_shifts[ind];
    void* addr = (md->stat_buf + sh);

    if (md->stat_types[ind] == 0)   // int type
    {
        *(int*)addr = *(int*)md->stat_addr[ind];
        //printf("stat int:%d = %d, addr_in_buffer:%p shift=%d shift[]=%d\n", ind, *(int*)md->stat_addr[ind], addr, shift, md->stat_shifts[ind]);
    }
    else  // float type
    {
        *(float*)addr = *(float*)md->stat_addr[ind];
        //printf("stat float:%d = %f  from addr=%p addr_bnd=%f\n", ind, *(float*)md->stat_addr[ind], md->stat_addr[ind], &(md->bondTypes[1].ltMean));
    }
}

//! теперь эта функция общая для всех статистик
// копирует статистику в буфер (size - размер копируемого куска, nstep - число временных шагов для распарисвания
// ndata - число данных для распарсивания, dstep, dtime - шаги по числу шагов и по времени, step0, time0 - начальные шаги
void copy_stat(cudaMD* hmd, hostManagMD *man, statStruct *stat, double dt)
// copy statistics from device to host and save it to file (hmd must be keeped on host)
{
    int i, j;
    int step = stat->step0;
    double time = stat->step0 * dt;
    char* addr = man->stat_buffer;

    //printf("pos: %d, size: %d, buf_addr: %p, our_buf: %p\n", stat->position, stat->count * stat->size, hmd->stat_buf, hmd->stat_buf + stat->position);
    cudaMemcpy(man->stat_buffer, hmd->stat_buf + stat->position, stat->count * stat->size, cudaMemcpyDeviceToHost);
    for (i = 0; i < stat->count; i++)
    {
        fprintf(stat->out_file, "%f\t%d", time, step);
        for (j = 0 ; j < stat->ndata; j++)
        {
            if (man->stat_types[j + stat->typeind] == 0)  // int type
            {
                fprintf(stat->out_file, "\t%d", *(int*)addr);
                addr += int_size;
            }
            else  // float type
            {
                fprintf(stat->out_file, "\t%f", *(float*)addr);
                addr += sizeof(float);
            }
        }
        fprintf(stat->out_file, "\n");
        step += stat->dstep;
        time += stat->dtime;
    }
    //fflush(stat->out_file);   //! error
}

void add_stat(int ndata, int size, /*int nstep,*/ int dstep, double dt, int &tot_ndata, int &host_bufsize, int &dev_bufsize, statStruct *stat)
{
    //int buf_size = size * nstep;
    int buf_size = size * stat->nstep;

    stat->ndata = ndata;
    //stat->nstep = nstep;    //! это количество "шагов", которые хранятся в буфере на девайсе, теперь считывается из cuda.txt
    stat->size = size;
    stat->dstep = dstep;
    stat->dtime = dstep * dt;
    stat->count = 0;
    stat->step0 = 0;

    stat->typeind = tot_ndata;
    tot_ndata += ndata;

    stat->position = dev_bufsize;
    dev_bufsize += buf_size;
    if (host_bufsize < buf_size)
        host_bufsize = buf_size;
}

// для каждой статистики надо задать ndata и nstep, а также dstep и dtime = (dstep * timestep)
// кроме того, надо выбрать наибольший размер статистики, чтобы задать буфер, а на девайсе разместить суммарный буфер и определить свдиги для каждой статистики
void init_cuda_stat(cudaMD* hmd, hostManagMD* man, Sim* sim, Field *fld)
// prepare data for statistics output on host side
{
    int i, j;
    int host_buf_size = 0, dev_buf_size = 0;
    int tot_ndata = 0;
    int ndata, size;
    int nfirst_float;


    //FIRST PART. to add parameter you need call function add_stat with number of new parameters and their size
    //! функция add_stat должна вызываться единожды для каждой из статистик
    int nparam = 12;
    int params_size = nparam * float_size;
    if (fld->nBdata) // add bond energy to stat:
    {
        nparam++;
        params_size += float_size;
    }
    if (fld->nAdata) // add angle energy to stat:
    {
        nparam++;
        params_size += float_size;
    }
    nfirst_float = nparam;
    nparam += sim->nVarSpec;
    params_size += sim->nVarSpec * int_size;
    add_stat(nparam, params_size, /*10,*/ sim->stat, sim->tSt, tot_ndata, host_buf_size, dev_buf_size, &(man->stat));
//    nstat = nparam;

    if (fld->nBdata)    // на самом деле это нужно, не всегда, когда есть связи, а когда они переменные
    {
        ndata = 1 + (fld->nBdata - 1) * 3;   // totalcount, count, length, lifetime
        size = int_size + (int_size + 2 * sizeof(float)) * (fld->nBdata - 1);   // -1 as [0] is deleted bond
        add_stat(ndata, size, /*10,*/ sim->stat, sim->tSt, tot_ndata, host_buf_size, dev_buf_size, &(man->sbnd));
//        nbnd = ndata;
    }

    // msd statistics
    ndata = fld->nSpec * 6;
    size = ndata * int_size;
    add_stat(ndata, size, /*10,*/ sim->stat, sim->tSt, tot_ndata, host_buf_size, dev_buf_size, &(man->smsd));
//    nmsd = ndata;

    // e-jump statistics
///*
    if (sim->ejtype)
    {
        ndata = 3;
        size = ndata * int_size;
        add_stat(ndata, size, sim->stat, sim->tSt, tot_ndata, host_buf_size, dev_buf_size, &(man->sjmp));
    }
//*/
    //! короче, это все нужно переделать, что типа вводишь строчки вида "0011000" и это сразу преобразуется и в то, какие типы есть и в то, сколько места занимает

    man->stat_types = (int*)malloc(tot_ndata * int_size);

    //SECOND PART. to add parameter you need define type of new parameter: 0 - integer, 1 - float
    j = 0;
    while (j < nfirst_float) 
    {
        man->stat_types[j] = 1;     // float parameters
        j++;
    }
    while (j < man->stat.ndata) // rest parameters ar related with variable spec count
    {
        man->stat_types[j] = 0;
        j++;
    }

    if (fld->nBdata)    // на самом деле это нужно, не всегда, когда есть связи, а когда они переменные
    {
        man->stat_types[j] = 0;     // total bonds count
        j++;
        for (i = 1; i < fld->nBdata; i++)   // [0] is reserved for deleted bonds
        {
            man->stat_types[j] = 0;     // count
            man->stat_types[j+1] = 1;     // length
            man->stat_types[j+2] = 1;     // lifetime
            j += 3;
        }
    }

    // msd statistics
    for (i = 0; i < fld->nSpec * 6; i++)
    {
        man->stat_types[j] = 0;
        j++;
    }

///*
    // e-jump statistics
    if (sim->ejtype)
        for (i = 0; i < 3; i++)
        {
            man->stat_types[j] = 0;
            j++;
        }
//*/

    man->stat_buffer = (char*)malloc(host_buf_size);
    man->tot_ndata = tot_ndata;

    // hmd
    //hmd->stat_size = stat_size;
    cudaMalloc((void**)&(hmd->stat_buf), dev_buf_size);
    //printf("tot_data=%d: stat=%d bnd=%d msd=%d ejmp=3 stat_buf=%p dev_buf_size=%d\n", tot_ndata, nstat, nbnd, nmsd, hmd->stat_buf, dev_buf_size);
    cudaMalloc((void**)&(hmd->stat_addr), tot_ndata * pointer_size);
    cudaMalloc((void**)&(hmd->stat_shifts), tot_ndata * int_size);
    cudaMalloc((void**)&(hmd->stat_types), tot_ndata * int_size);
    cudaMemcpy(hmd->stat_types, man->stat_types, tot_ndata * int_size, cudaMemcpyHostToDevice);
}

__device__ void add_stat(int &ind, void *addr, int type, int &shift, cudaMD *md)
{

    int size;
    if (type == 0)
        size = int_size;
    else
        size = sizeof(float);


    md->stat_addr[ind] = addr;
    md->stat_shifts[ind] = shift;
    shift += size;
    //md->stat_types[ind] = type;
    ind++;
}

__global__ void prepare_stat_addr(cudaMD* md)
// nVarSpec is the number of species with variable quantity
{
    int i;
    int shift = 0;
    int index = 0;

    //general statistics
    //to add parameter for output just call function add_stat with address of parameter and type (for each type of statistics)
    add_stat(index, &(md->engTot), 1, shift, md);
    add_stat(index, &(md->engKin), 1, shift, md);
    add_stat(index, &(md->engVdW), 1, shift, md);
    add_stat(index, &(md->engCoul1), 1, shift, md);
    add_stat(index, &(md->engCoul2), 1, shift, md);
    if (md->use_bnd)
      add_stat(index, &(md->engBond), 1, shift, md);
    if (md->use_angl)
        add_stat(index, &(md->engAngl), 1, shift, md);
    add_stat(index, &(md->posMom.x), 1, shift, md);
    add_stat(index, &(md->negMom.x), 1, shift, md);
    add_stat(index, &(md->posMom.y), 1, shift, md);
    add_stat(index, &(md->negMom.y), 1, shift, md);
    add_stat(index, &(md->posMom.z), 1, shift, md);
    add_stat(index, &(md->negMom.z), 1, shift, md);
    add_stat(index, &(md->pressure), 1, shift, md);
    for (i = 0; i < md->nSpec; i++)
        if (md->specs[i].varNumber)
            add_stat(index, &(md->specs[i].number), 0, shift, md);


    // bond statistics
    // сдвиг надо сбрасывать для каждлой статистики
    shift = 0;
    if (md->use_bnd)
    {
        add_stat(index, &(md->nBond), 0, shift, md);
        for (i = 1; i < md->nBndTypes; i++) // starting from 1 as [0] reserved as 'no bond'
        {
            add_stat(index, &(md->bondTypes[i].count), 0, shift, md);
            add_stat(index, &(md->bondTypes[i].rMean), 1, shift, md);
            add_stat(index, &(md->bondTypes[i].ltMean), 1, shift, md);
        }
    }

    // MSD statistics
    shift = 0;
    for (i = 0; i < md->nSpec; i++)
    {
        add_stat(index, &(md->specAcBoxPos[i].x), 0, shift, md);
        add_stat(index, &(md->specAcBoxNeg[i].x), 0, shift, md);
        add_stat(index, &(md->specAcBoxPos[i].y), 0, shift, md);
        add_stat(index, &(md->specAcBoxNeg[i].y), 0, shift, md);
        add_stat(index, &(md->specAcBoxPos[i].z), 0, shift, md);
        add_stat(index, &(md->specAcBoxNeg[i].z), 0, shift, md);
    }

    // e-jump statistics
    if (md->use_ejump)
    {
        shift = 0;
        add_stat(index, &(md->nJump), 0, shift, md);
        add_stat(index, &(md->posBxJump.x), 0, shift, md);
        add_stat(index, &(md->negBxJump.x), 0, shift, md);
    }
}

void start_stat(hostManagMD* man, Field *fld, Sim *sim)
// open statistics file, reset counters
{
    int i;
    char* s1, * s2;

    man->stat.out_file = fopen("stat.dat", "w");
    // header (first line):
    fprintf(man->stat.out_file, "time\tstep\tengTot\tengKin\tengVdW\tengCoul1\tengCoul2");
    if (fld->nBdata)
        fprintf(man->stat.out_file, "\tengBnd");
    if (fld->nAdata)
        fprintf(man->stat.out_file, "\tengAngle");
    fprintf(man->stat.out_file, "\tmomPx\tmomNx\tmomPy\tmomNy\tmomPz\tmomNz\tpress");
    for (i = 0; i < sim->nVarSpec; i++)
        fprintf(man->stat.out_file, "\t%s", fld->snames[sim->varSpecs[i]]);

    // header (second line: +units)
    fprintf(man->stat.out_file, "\ntime, ps\tstep, n\tengTot, eV\tengKin, eV\tengVdW, eV\tengCoul1, eV\tengCoul2, eV");
    if (fld->nBdata)
        fprintf(man->stat.out_file, "\tengBnd, eV");
    if (fld->nAdata)
        fprintf(man->stat.out_file, "\tengAngle, eV");
    fprintf(man->stat.out_file, "\tmomPx, eVps/A\tmomNx, eVps/A\tmomPy, eVps/A\tmomNy, eVps/A\tmomPz, eVps/A\tmomNz, eVps/A\tpress, atm");
    for (i = 0; i < sim->nVarSpec; i++)
        fprintf(man->stat.out_file, "\t%s", fld->snames[sim->varSpecs[i]]);
    fprintf(man->stat.out_file, "\n");

    if (fld->nBdata)    // на самом деле это нужно, не всегда, когда есть связи, а когда они переменные
    {
        man->sbnd.out_file = fopen("stat_bnd.dat", "w");
        fprintf(man->sbnd.out_file, "time\tstep\ttot_bnd");
        for (i = 1; i < fld->nBdata; i++)   // from 1 as 0 is reserved for deleted bond
        {
            s1 = fld->snames[fld->bdata[i].spec1];
            s2 = fld->snames[fld->bdata[i].spec2];
            fprintf(man->sbnd.out_file, "\tcnt%s-%s\tleng%s-%s\ttime%s-%s", s1, s2, s1, s2, s1, s2);
        }
        fprintf(man->sbnd.out_file, "\n");
    }

    // MSD statistics
    man->smsd.out_file = fopen("msd.dat", "w");
    fprintf(man->smsd.out_file, "time\tstep");
    for (i = 0; i < fld->nSpec; i++)
        fprintf(man->smsd.out_file, "\t%s_px\tnx\tpy\tny\tpz\tnz", fld->snames[i]);
    fprintf(man->smsd.out_file, "\n");

    // E-JUMP
    if (sim->ejtype)
    {
        man->sjmp.out_file = fopen("jumps.dat", "w");
        fprintf(man->sjmp.out_file, "time\tstep\tnTot\tpos\tneg\n");
    }
}

void end_stat(hostManagMD *man, Field *fld, Sim *sim, cudaMD *hmd, double dt)
// close stat files
{
    if (man->stat.count)
        copy_stat(hmd, man, &(man->stat), dt);
    fclose(man->stat.out_file);

    if (fld->nBdata)    // на самом деле это нужно, не всегда, когда есть связи, а когда они переменные
    {
        if (man->sbnd.count)
            copy_stat(hmd, man, &(man->sbnd), dt);
        fclose(man->sbnd.out_file);
    }

    if (man->smsd.count)
        copy_stat(hmd, man, &(man->smsd), dt);
    fclose(man->smsd.out_file);

    // E-JUMP
    if (sim->ejtype)
    {
        if (man->sjmp.count)
            copy_stat(hmd, man, &(man->sjmp), dt);
        fclose(man->sjmp.out_file);
    }
}

void stat_iter(int step, hostManagMD *man, statStruct *stat, cudaMD *dmd, cudaMD *hmd, double dt)
// dmd and hmd - device and host exemplar of MD data struct
{
    int shift;

    if (step % stat->dstep == 0)
    {
        shift = stat->position + stat->count * stat->size;
        write_stat << <1, stat->ndata >> > (step, stat->typeind, shift, dmd);
        stat->count++;
        if (stat->count >= stat->nstep)
        {
            copy_stat(hmd, man, stat, dt);
            stat->step0 = step + stat->dstep;
            stat->count = 0;
        }
    }
}

void free_cuda_stat(cudaMD* hmd, hostManagMD* man)
// free device arrays for statistics and arrays in manager
{
    cudaFree(hmd->stat_buf);
    cudaFree(hmd->stat_addr);
    cudaFree(hmd->stat_shifts);
    cudaFree(hmd->stat_types);

    delete[]  man->stat_types;
    delete[] man->stat_buffer;
}

void init_cuda_rdf(Field *fld, Sim *sim, hostManagMD *man, cudaMD *hmd)
{
    int i;
    int n = sim->nRDF * fld->nPair;
    man->rdf_size = n * float_size;
    man->rdf_buffer = (float*)malloc(man->rdf_size);
    for (i = 0; i < n; i++)
        man->rdf_buffer[i] = 0.f;
    data_to_device((void**)&(hmd->rdf), man->rdf_buffer, man->rdf_size);
    man->rdf_count = 0;
}

void free_cuda_rdf(hostManagMD* man, cudaMD* hmd)
{
    cudaFree(hmd->rdf);
    delete[] man->rdf_buffer;
}

__global__ void brute_rdf(int nSpec, int nPair, float idRDF, float r2max, cudaMD *md)
// take rdf data by naive all_pair algorithm
{
    int i, j, iR, iPair, mn, mx;
    int step = blockDim.x * gridDim.x;
    int ex = 0;  // exit flag
    int nAt = md->nAt;
    float dx, dy, dz, r2;
    int m = nSpec - 1;
    float C2;


    //degub
    //int count = 0;

    i = 0;
    j = blockIdx.x * blockDim.x + threadIdx.x + 1;     // first pair is 0-1
    //printf("th(%d,%d) j=%d nAt=%d\n", blockIdx.x, threadIdx.x, j, nAt);
    while (1)
    {
        while (j >= nAt)
        {
            i++;
            if (i >= nAt - 1)
            {
                ex = 1;
                break;
            }
            j = i + 1 + j - nAt;
        }
        if (ex) break;

        // rdf calculation
        dx = md->xyz[i].x - md->xyz[j].x;
        dy = md->xyz[i].y - md->xyz[j].y;
        dz = md->xyz[i].z - md->xyz[j].z;
        delta_periodic(dx, dy, dz, md);
        r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < r2max)
        {
            iR = sqrt((double)r2) * idRDF;                  // define id of RDF bin . Convert to double to exclude than roundation error gives iR out of range
            // define pair index:
            mn = md->types[i];
            mx = md->types[j];
            if (mn > mx)
            {
                mn = md->types[j];
                mx = md->types[i];
            }
            iPair = mn * m + mn * (1 - mn) / 2 + mx;
#ifdef DEBUG_MODE
            if ((iR * nPair + iPair) >= nPair * 500)
                printf("RDF out of range: iR=%d, iPair=%d (idRD=%f, r2=%f)\n", iR, iPair, idRDF, r2);
#endif
            
            // protection from buffer values overfilling: to do this normalization here, not in output
            //C2 = 1.f / (3.f * iR * (iR + 1.f) + 1.f);
            C2 = 1.f;

            //! rdf_buffer indexed as follow: at first 0th indexes of bins from all pairs, then 1th indexes of bins from all pairs, etc...
            //atomicAdd(&(md->rdf[iR * nPair + iPair]), 2.f * md->volume / (float)md->specs[mn].number / (float)md->specs[mx].number);
            atomicAdd(&(md->rdf[iR * nPair + iPair]), C2 * md->volume / (float)md->specs[mn].number / (float)md->specs[mx].number);
        }

        j = j + step;
    }

#ifdef DEBUG_MODE
    if (threadIdx.x == 0)
        for (i = 1; i < md->nBndTypes; i++)
            if ((md->bondTypes[i].spec1 < 0) || (md->bondTypes[i].spec2 < 0) || (md->bondTypes[i].spec1 >= MX_SPEC) || (md->bondTypes[i].spec2 >= MX_SPEC))
            {
                printf("aft brute_rdf: bl[%d] step %d bnd[%d] spec1=%d spec2=%d\n", blockIdx.x, step, i, md->bondTypes[i].spec1, md->bondTypes[i].spec2);
                md->xyz[9999999999].x = 15.f;  // crash cuda
            }
#endif
}

void copy_rdf(Field* fld, Sim* sim, hostManagMD* man, cudaMD* hmd, char *fname)
// copy rdf from device to host
{
    int i, j, k, l;
    FILE* f;
    double C1 = 2 / (sphera * sim->dRDF * sim->dRDF * sim->dRDF * man->rdf_count);    // factor 2.f was removed from rdf calculations at device
    //float C1 = 1.f / ((float)sphera * (float)sim->dRDF * (float)sim->dRDF * (float)sim->dRDF);
    float C2, C3;
    float* addr = man->rdf_buffer;


    cudaMemcpy(man->rdf_buffer, hmd->rdf, man->rdf_size, cudaMemcpyDeviceToHost);
    f = fopen(fname, "w");
    fprintf(f, "r");
    for (i = 0; i < fld->nSpec; i++)
        for (j = i; j < fld->nSpec; j++)
            fprintf(f, "\t%s-%s", fld->snames[i], fld->snames[j]);
    fprintf(f, "\n");


    for (i = 0; i < sim->nRDF; i++)
    {
        fprintf(f, "%f", (i + 0.5) * sim->dRDF);
        C2 = 1.f / (3.f * i * (i + 1.f) + 1.f);

        k = 0; l = 0;
        for (j = 0; j < fld->nPair; j++)
        {
            // accounting that for pair of different atoms can be divided to 2
            if (k == l)
                C3 = 1.f;
            else
                C3 = 0.5f;

            fprintf(f, "\t%f", (*addr) * C1 * C2 * C3);
            addr++;

            l++;
            if (l == fld->nSpec)
            {
                k++;
                l = k;
            }
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void rdf_iter(int step, Field *fld, Sim *sim, hostManagMD *man, cudaMD *hmd, cudaMD *dmd)
{
    int i, n;
    char rdfname[32];     // for output of intermediate configurations

    if (step % sim->frRDF == 0)
    {
        brute_rdf << <man->nMultProc, man->nSingProc >> > (fld->nSpec, fld->nPair, (float)sim->idRDF, (float)sim->r2RDF, dmd);
        cudaThreadSynchronize();
        man->rdf_count++;
        //if (man->rdf_count >= 500)   //! temp
        if (sim->frRDFout)      // if 0 - no intermediate rdf-output
            if (step % sim->frRDFout == 0)
            {
                sprintf(rdfname, "rdf%d.dat\0", step);
                copy_rdf(fld, sim, man, hmd, rdfname);
            }

        //protection from buffer overfull: clear buffer
        if (man->rdf_count > 1000)
        {
            n = sim->nRDF * fld->nPair;
            for (i = 0; i < n; i++)
                man->rdf_buffer[i] = 0.f;
            cudaMemcpy((void**)&(hmd->rdf), man->rdf_buffer, man->rdf_size, cudaMemcpyHostToDevice);
            man->rdf_count = 0;
        }


    }
}


void init_cuda_nrdf(Field* fld, Sim* sim, hostManagMD* man, cudaMD* hmd)
{
    int i;
    int n = sim->nRDF * (npairs(fld->nNucl) + fld->nNucl);
    man->nrdf_size = n * float_size;
    man->nrdf_buffer = (float*)malloc(man->nrdf_size);
    for (i = 0; i < n; i++)
        man->nrdf_buffer[i] = 0.f;
    data_to_device((void**)&(hmd->nrdf), man->nrdf_buffer, man->nrdf_size);
    //man->rdf_count = 0;   //! буду использовать общий с обычной rdf count - нет, вообще буду вызывать вместо brute_rdf - brute_nrdf
}

void free_cuda_nrdf(hostManagMD* man, cudaMD* hmd)
{
    cudaFree(hmd->nrdf);
    delete[] man->nrdf_buffer;
}

__global__ void brute_nrdf(int nSpec, int nNucl, int nPair, int n_nPair, float idRDF, float r2max, cudaMD* md)
// take rdf data by naive all_pair algorithm
{
    int i, j, iR, iPair, n_iPair, mn, mx, n_mn, n_mx;
    int step = blockDim.x * gridDim.x;
    int ex = 0;  // exit flag
    int nAt = md->nAt;
    float dx, dy, dz, r2;
    int m = nSpec - 1;
    int n_m = nNucl - 1;


    //degub
    //int count = 0;

    i = 0;
    j = blockIdx.x * blockDim.x + threadIdx.x + 1;     // first pair is 0-1
    //printf("th(%d,%d) j=%d nAt=%d\n", blockIdx.x, threadIdx.x, j, nAt);
    while (1)
    {
        while (j >= nAt)
        {
            i++;
            if (i >= nAt - 1)
            {
                ex = 1;
                break;
            }
            j = i + 1 + j - nAt;
        }
        if (ex) break;

        // rdf calculation
        dx = md->xyz[i].x - md->xyz[j].x;
        dy = md->xyz[i].y - md->xyz[j].y;
        dz = md->xyz[i].z - md->xyz[j].z;
        delta_periodic(dx, dy, dz, md);
        r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < r2max)
        {
            iR = sqrt((double)r2) * idRDF;                  // define id of RDF bin . Convert to double to exclude than roundation error gives iR out of range
            // define pair index:
            mn = md->types[i];
            mx = md->types[j];
            n_mn = md->specs[mn].nuclei;
            n_mx = md->specs[mx].nuclei;
            if (n_mn > n_mx)
            {
                n_mn = md->specs[mx].nuclei;
                n_mx = md->specs[mn].nuclei;
            }
            if (mn > mx)
            {
                mn = md->types[j];
                mx = md->types[i];
            }
            iPair = mn * m + mn * (1 - mn) / 2 + mx;
            n_iPair = n_mn * n_m + n_mn * (1 - n_mn) / 2 + n_mx;
#ifdef DEBUG_MODE
            if ((iR * nPair + iPair) >= nPair * 500)
                printf("RDF out of range: iR=%d, iPair=%d (idRD=%f, r2=%f)\n", iR, iPair, idRDF, r2);
#endif
            //! сделаем так, что сначала идут данные для одной iR и разных пар, это удобнее распарсивать и выводить
            atomicAdd(&(md->rdf[iR * nPair + iPair]), 2.f * md->volume / md->specs[mn].number / md->specs[mx].number);
            atomicAdd(&(md->nrdf[iR * n_nPair + n_iPair]), 2.f * md->volume / md->nnumbers[n_mn] / md->nnumbers[n_mx]);
            //printf("add rdf:%f\n", 2.f * md->volume / md->specs[mn].number / md->specs[mx].number);
        }

        j = j + step;
    }

#ifdef DEBUG_MODE
    if (threadIdx.x == 0)
        for (i = 1; i < md->nBndTypes; i++)
            if ((md->bondTypes[i].spec1 < 0) || (md->bondTypes[i].spec2 < 0) || (md->bondTypes[i].spec1 >= MX_SPEC) || (md->bondTypes[i].spec2 >= MX_SPEC))
            {
                printf("aft brute_rdf: bl[%d] step %d bnd[%d] spec1=%d spec2=%d\n", blockIdx.x, step, i, md->bondTypes[i].spec1, md->bondTypes[i].spec2);
                md->xyz[9999999999].x = 15.f;  // crash cuda
            }
#endif
}

void copy_nrdf(Field* fld, Sim* sim, hostManagMD* man, cudaMD* hmd, char* fname, char* nfname)
// copy rdf from device to host
{
    int i, j, k, l;
    FILE* f, *nf;
    double C1 = 1.0 / (sphera * sim->dRDF * sim->dRDF * sim->dRDF * man->rdf_count);
    double C2, C3;
    float* addr = man->rdf_buffer;
    float* naddr = man->nrdf_buffer;


    cudaMemcpy(man->rdf_buffer, hmd->rdf, man->rdf_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(man->nrdf_buffer, hmd->nrdf, man->nrdf_size, cudaMemcpyDeviceToHost);
    f = fopen(fname, "w");
    fprintf(f, "r");
    for (i = 0; i < fld->nSpec; i++)
        for (j = i; j < fld->nSpec; j++)
            fprintf(f, "\t%s-%s", fld->snames[i], fld->snames[j]);
    fprintf(f, "\n");
    nf = fopen(nfname, "w");
    fprintf(nf, "r");
    for (i = 0; i < fld->nNucl; i++)
        for (j = i; j < fld->nNucl; j++)
            fprintf(nf, "\t%s-%s", fld->nnames[i], fld->nnames[j]);
    fprintf(nf, "\n");


    for (i = 0; i < sim->nRDF; i++)
    {
        fprintf(f, "%f", (i + 0.5) * sim->dRDF);
        fprintf(nf, "%f", (i + 0.5) * sim->dRDF);
        C2 = 1.0 / (3.0 * i * (i + 1.0) + 1.0);
        //printf("C1=%f C2=%f\n", C1, C2);

        k = 0; l = 0;
        for (j = 0; j < fld->nPair; j++)
        {
            // accounting that for pair of different atoms can be divided to 2
            if (k == l)
                C3 = 1.0;
            else
                C3 = 0.5;

            fprintf(f, "\t%f", (*addr) * C1 * C2 * C3);
            addr++;

            l++;
            if (l == fld->nSpec)
            {
                k++;
                l = k;
            }

        }
        fprintf(f, "\n");

        // nucleus output
        k = 0; l = 0;
        for (j = 0; j < (npairs(fld->nNucl) + fld->nNucl); j++)
        {
            // accounting that for pair of different atoms can be divided to 2
            if (k == l)
                C3 = 1.0;
            else
                C3 = 0.5;

            fprintf(nf, "\t%f", (*naddr) * C1 * C2 * C3);
            naddr++;

            l++;
            if (l == fld->nNucl)
            {
                k++;
                l = k;
            }

        }
        fprintf(nf, "\n");
    }
    fclose(f);
    fclose(nf);
}

void nrdf_iter(int step, Field* fld, Sim* sim, hostManagMD* man, cudaMD* hmd, cudaMD* dmd)
{
    char rdfname[32];     // for output of intermediate configurations
    char nrdfname[32];     // for output of intermediate configurations

    if (step % sim->frRDF == 0)
    {
        brute_nrdf << <man->nMultProc, man->nSingProc >> > (fld->nSpec, fld->nNucl, fld->nPair, (npairs(fld->nNucl) + fld->nNucl), (float)sim->idRDF, (float)sim->r2RDF, dmd);
        man->rdf_count++;
        //if (man->rdf_count >= 500)   //! temp
        if (sim->frRDFout)      // if 0 - no intermediate rdf-output
            if (step % sim->frRDFout == 0)
            {
                sprintf(rdfname, "rdf%d.dat\0", step);
                sprintf(nrdfname, "rdf_n%d.dat\0", step);
                copy_nrdf(fld, sim, man, hmd, rdfname, nrdfname);
            }
    }
}

void init_cuda_trajs(Atoms *atm, cudaMD* hmd, hostManagMD* man)
// alloc arrays for trajectories output on both device and host sides
{
    int nparam = 5;     // number of parameters to output: x, y, z, type, ptype
    int size = atm->nAt * nparam * man->nstep_traj * float_size;     // every atom has 3 coordinates, nstep - number of steps, which can be stored on device
    man->traj_buffer = (float*)malloc(size);
    cudaMalloc((void**)&(hmd->traj_buf), size);
}

__global__ void write_traj(int iStep, int shift, int atPerBlock, int atPerThread, cudaMD* md)
// version for sorted atoms!
// save current coordinates to trajectories buffer on device (shift - shift to current trajectory step in positions, not in bytes)
{
    float* addr = (md->traj_buf + shift);
    //float* addr = (md->traj_buf);
    int nparam = 5;     // number of parameters to output: x, y, z, type, ptype
    int p;      // parent

    int i;
    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);
    for (i = id0; i < N; i++)
    {
        addr[i * nparam] = md->xyz[md->sort_trajs[i]].x;
        addr[i * nparam + 1] = md->xyz[md->sort_trajs[i]].y;
        addr[i * nparam + 2] = md->xyz[md->sort_trajs[i]].z;

        // for type and ptype output:
        addr[i * nparam + 3] = md->types[md->sort_trajs[i]];
        p = md->parents[md->sort_trajs[i]];
        if (p > 0)  //! почему 0? должно быть -1, но с 0 правильно работает а с -1 - неет
            addr[i * nparam + 4] = md->types[p];
        else
            addr[i * nparam + 4] = -1.f;
    }
}

void copy_traj(cudaMD* hmd, hostManagMD* man, double dt, int nAt)
// copy statistics from device to host and save it to file (hmd must be keeped on host)
{
    int i, j;
    int step = man->traj_step0;
    double time = step * dt;
    int nparam = 5;     // number of parameters to output x, y, z, type, ptype

    cudaMemcpy(man->traj_buffer, hmd->traj_buf, man->traj_count * man->traj_size, cudaMemcpyDeviceToHost);
    float* addr = man->traj_buffer;

    for (i = 0; i < man->traj_count; i++)
    {
        fprintf(man->traj_file, "\n%f\t%d", time, step);
        for (j = 0; j < nAt * nparam; j++)
        {
            fprintf(man->traj_file, "\t%f", *addr);
            addr++;
        }
        //fprintf(man->traj_file, "\n");
        step += man->traj_dstep;
        time += man->traj_dtime;
    }
}

void start_traj(Atoms *atm, hostManagMD* man, Field* fld, Sim *sim)
// open file, reset counters
{
    int i;
    int nparam = 5;     // number of parameters to output x, y, z, type, ptype

    man->traj_file = fopen("traj.dat", "w");
    // header (first line):
    fprintf(man->traj_file, "time\tstep");
    for (i = 0; i < atm->nAt; i++)
    {
        fprintf(man->traj_file, "\t%sx\ty\tz", fld->snames[atm->types[i]]);
        if (nparam > 3)
            fprintf(man->traj_file, "\ttype\tptype");
    }

    //! может это в init ?
    man->traj_count = 0;
    man->traj_size = atm->nAt * nparam * float_size;
    man->traj_dstep = sim->frTraj;      //! эти свойства друг друга копируют, можно удалить одну из сущностей
    man->traj_dtime = man->traj_dstep * sim->tSt;
    man->traj_step0 = sim->stTraj;
}

void traj_iter(int step, hostManagMD* man, cudaMD* dmd, cudaMD* hmd, Sim *sim, Atoms *atm)
// dmd and hmd - device and host exemplar of MD data struct
{
    int shift;
    int nparam = 5;     // number of parameters to output x, y, z, type, ptype

    if (step >= sim->stTraj)
        if (step % sim->frTraj == 0)
        {
            shift = man->traj_count * atm->nAt * nparam;
            write_traj << < man->nAtBlock, man->nAtThread >> > (step, shift, man->atPerBlock, man->atPerThread, dmd);
            man->traj_count++;
            if (man->traj_count >= man->nstep_traj)
            {
                copy_traj(hmd, man, sim->tSt, atm->nAt);
                man->traj_step0 = step + man->traj_dstep;
                man->traj_count = 0;
            }
        }
}

void end_traj(hostManagMD* man, cudaMD* hmd, Sim* sim, Atoms* atm)
// close traj file
{
    if (man->traj_count)
        copy_traj(hmd, man, sim->tSt, atm->nAt);
    fclose(man->traj_file);
}

void free_cuda_trajs(cudaMD* hmd, hostManagMD* man)
// free memory on both device and host side
{
    cudaFree(hmd->traj_buf);
    delete[] man->traj_buffer;
}

void init_cuda_bindtrajs(Sim* sim, cudaMD* hmd, hostManagMD* man)
// alloc arrays for bind trajectories output on both device and host sides
{
    // {x,y,z} type, nbond, parentType, parent{x,y,z}
    int size = sim->nBindTrajAtoms * (6 * float_size + 3 * int_size) * man->nstep_bindtraj;

    data_to_device((void**)&(hmd->bindtraj_atoms), sim->bindTrajAtoms, sim->nBindTrajAtoms * int_size);
    hmd->nBindTrajAtm = sim->nBindTrajAtoms;
    //! кстати, тут можно и высвободить sim->bindTrajAtoms, он вроде нигде не нужен уже

    cudaMalloc((void**)&(hmd->bindtraj_buf), size);
    man->bindtraj_buffer = (char*)malloc(size);

    man->bindTrajPerBlock = man->nBindTrajThread * man->bindTrajPerThread;
    man->nBindTrajBlock = ceil((double)sim->nBindTrajAtoms / man->bindTrajPerBlock);
}

__global__ void write_bindtraj(int iStep, int shift, int bindTrajPerBlock, int bindTrajPerThread, cudaMD* md)
// version for sorted atoms!
// save current coordinates to trajectories buffer on device (here shift in bytes)
{
    int bytesPerAtom = 6 * float_size + 3 * int_size;
    char* addr = (md->bindtraj_buf + shift);
    //float* addr = (md->traj_buf);

    int i, j, p;
    int id0 = blockIdx.x * bindTrajPerBlock + threadIdx.x * bindTrajPerThread;
    int N = min(id0 + bindTrajPerThread, md->nBindTrajAtm);
    addr += id0 * bytesPerAtom;
    for (i = id0; i < N; i++)
    {
        j = md->sort_trajs[md->bindtraj_atoms[i]];
        //j = 1;
        *(float*)addr = md->xyz[j].x;
        addr += float_size;
        *(float*)addr = md->xyz[j].y;
        addr += float_size;
        *(float*)addr = md->xyz[j].z;
        addr += float_size;
        *(int*)addr = md->types[j];
        addr += int_size;
        *(int*)addr = md->nbonds[j];
        addr += int_size;
        if (md->parents[j] > 0)     //! я не понимаю почему у атомов без связей parent стоит не -1, а 0
        {
            p = md->parents[j];
            *(int*)addr = md->types[p];
            addr += int_size;
            *(float*)addr = md->xyz[p].x;
            addr += float_size;
            *(float*)addr = md->xyz[p].y;
            addr += float_size;
            *(float*)addr = md->xyz[p].z;
            addr += float_size;
        }
        else // there is no parent
        {
            *(int*)addr = -1;
            addr += int_size;
            *(float*)addr = 0.f;
            addr += float_size;
            *(float*)addr = 0.f;
            addr += float_size;
            *(float*)addr = 0.f;
            addr += float_size;
        }
    }
}

void copy_bindtraj(Sim *sim, Box *bx, cudaMD* hmd, hostManagMD* man)
// copy statistics from device to host and save it to file (hmd must be keeped on host)
{
    int i, j;
    int step = man->bindtraj_step0;
    double time = step * sim->tSt;
    double x, y, z, px, py, pz, r;
    int tp, nbnd, ptp;

    cudaMemcpy(man->bindtraj_buffer, hmd->bindtraj_buf, man->bindtraj_count * man->bindtraj_size, cudaMemcpyDeviceToHost);
    char* addr = man->bindtraj_buffer;

    for (i = 0; i < man->bindtraj_count; i++)
    {
        fprintf(man->bindtraj_file, "\n%f\t%d", time, step);
        for (j = 0; j < sim->nBindTrajAtoms; j++)
        {
            // atom coordinates
            x = *(float*)addr;
            addr += float_size;
            y = *(float*)addr;
            addr += float_size;
            z = *(float*)addr;
            addr += float_size;

            // atom type, number of bonds, parent type
            tp = *(int*)addr;
            addr += int_size;
            nbnd = *(int*)addr;
            addr += int_size;
            ptp = *(int*)addr;
            addr += int_size;

            // parent coordinates
            px = *(float*)addr;
            addr += float_size;
            py = *(float*)addr;
            addr += float_size;
            pz = *(float*)addr;
            addr += float_size;

            if (ptp == -1)
                fprintf(man->bindtraj_file, "\t%f\t%f\t%f\t%d\t%d\t-1\t\t\t\t", x, y, z, tp, nbnd);
            else
            {
                r = distance_by_coord(x, y, z, px, py, pz, bx);
                fprintf(man->bindtraj_file, "\t%f\t%f\t%f\t%d\t%d\t%d\t%f\t%f\t%f\t%f", x, y, z, tp, nbnd, ptp, px, py, pz, r);
            }
        }
        step += man->bindtraj_dstep;
        time += man->bindtraj_dtime;
    }
}

void start_bindtraj(hostManagMD* man, Field* fld, Sim* sim)
// open file, reset counters
{
    int i;

    man->bindtraj_file = fopen("traj_bnd.dat", "w");
    // header (first line):
    fprintf(man->bindtraj_file, "time\tstep");
    for (i = 0; i < sim->nBindTrajAtoms; i++)
        fprintf(man->bindtraj_file, "\tx\ty\tz\ttype\tnbnd\tptype\tpx\tpy\tpz\tr");

    //! может это в init ?
    man->bindtraj_count = 0;
    man->bindtraj_size = sim->nBindTrajAtoms * (6 * sizeof(float) + 3 * int_size);
    man->bindtraj_dstep = sim->bindTrajFreq;      //! эти свойства друг друга копируют, можно удалить одну из сущностей
    man->bindtraj_dtime = man->traj_dstep * sim->tSt;
    man->bindtraj_step0 = sim->bindTrajStart;
}

void bindtraj_iter(int step, hostManagMD* man, cudaMD* dmd, cudaMD* hmd, Sim* sim, Box *bx)
// dmd and hmd - device and host exemplar of MD data struct
{
    int shift;

    if (step >= sim->bindTrajStart)
        if (step % sim->bindTrajFreq == 0)
        {
            shift = man->bindtraj_count * sim->nBindTrajAtoms * (6 * sizeof(float) + 3 * int_size);     // in bytes
            write_bindtraj << < man->nBindTrajBlock, man->nBindTrajThread >> > (step, shift, man->bindTrajPerBlock, man->bindTrajPerThread, dmd);
            man->bindtraj_count++;
            if (man->bindtraj_count >= man->nstep_bindtraj)
            {
                copy_bindtraj(sim, bx, hmd, man);
                man->bindtraj_step0 = step + man->bindtraj_dstep;
                man->bindtraj_count = 0;
            }
        }
}

void end_bindtraj(hostManagMD* man, cudaMD* hmd, Sim* sim, Box* bx)
// close traj file
{
    if (man->bindtraj_count)
        copy_bindtraj(sim, bx, hmd, man);
    fclose(man->bindtraj_file);
}

void free_cuda_bindtrajs(cudaMD* hmd, hostManagMD* man)
// free memory on both device and host side
{
    cudaFree(hmd->bindtraj_atoms);
    cudaFree(hmd->bindtraj_buf);
    delete[] man->bindtraj_buffer;
}