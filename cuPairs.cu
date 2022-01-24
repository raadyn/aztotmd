#include "defines.h"
#include "cuStruct.h"
#include "cuPairs.h"
#include "cuElec.h"
#include "cuBonds.h"
#include "cuMDfunc.h"
#include "cuUtils.h"
#include "cuSort.h"
#include "cuEjump.h"

__device__ float delta_and_r2(float3 xyz1, float3 xyz2, float3 &delta)
// return delta coordinates and r2 = dx2 + dy2 + dz2
{
    delta = make_float3(xyz1.x - xyz2.x, xyz1.y - xyz2.y, xyz1.z - xyz2.z);
    return delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
}

__device__ float delta_and_r2_shift(float3 xyz1, float3 xyz2, float3& delta, float3 shift)
// return delta coordinates and r2 = dx2 + dy2 + dz2 with additional delta (shift)
{
    delta = make_float3(xyz1.x - xyz2.x - shift.x, xyz1.y - xyz2.y - shift.y, xyz1.z - xyz2.z - shift.z);
    return delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
}

__device__ float delta_and_r2_byInd(int i, int j, cudaMD *md, float3 &delta)
{
    float dx = md->xyz[i].x - md->xyz[j].x;
    float dy = md->xyz[i].y - md->xyz[j].y;
    float dz = md->xyz[i].z - md->xyz[j].z;

    //! rectangular periodic
    if (dx > md->halfLeng.x)
        dx -= md->leng.x;
    else
        if (dx < -md->halfLeng.x)
            dx += md->leng.x;

    if (dy > md->halfLeng.y)
        dy -= md->leng.y;
    else
        if (dy < -md->halfLeng.y)
            dy += md->leng.y;

    if (dz > md->halfLeng.z)
        dz -= md->leng.z;
    else
        if (dz < -md->halfLeng.z)
            dz += md->leng.z;

    delta = make_float3(dx, dy, dz);
    return dx * dx + dy * dy + dz * dz;
}

__device__ void halfAtomicAddForces(float3* f1, float3* f2, float force, float3 delta)
// add (force * delta) to both atoms forces, only second atom by atomic operation
{
    f1->x += force * delta.x;
    atomicAdd(&(f2->x), -force * delta.x);

    f1->y += force * delta.y;
    atomicAdd(&(f2->y), -force * delta.y);

    f1->z += force * delta.z;
    atomicAdd(&(f2->z), -force * delta.z);
}

__device__ void save_coul_vdw(float coul, float vdw, float *shCoul, float *shVdw, cudaMD *md)
// save coul and vdw energies into shared variables and then into cudaMD struct
{
    atomicAdd(shCoul, coul);
    atomicAdd(shVdw, vdw);
    __syncthreads();    // wait all threads

    // from each block to global memory
    if (threadIdx.x == 0)
    {
        atomicAdd(&md->engCoul1, *shCoul);  // to global memory
        atomicAdd(&md->engVdW, *shVdw);
    }

}

__device__ void stupid_force_verif(char *prefix, float force, float r2, int type1, int type2)
// some simplest verification for obtained force (needed in DEBUG_MODE)
{
    if (isnan(force))
        printf("bl(%d) %s: f=nan: r2=%f types(%d-%d)\n", blockIdx.x, prefix, r2, type1, type2);
    if (force > 1e4)
        printf("bl(%d) %s: f=%f: r2=%f types(%d-%d)\n", blockIdx.x, prefix, force, r2, type1, type2);
    if (force < -1e6)
        printf("bl(%d) %s: f=%f: r2=%f types(%d-%d)\n", blockIdx.x, prefix, force, r2, type1, type2);
}

__device__ void force_autocap(char* prefix, float &force, float r2, int type1, int type2)
// automatically scale force if its too high
{
    if (isnan(force))
        printf("bl(%d) %s: f=nan: r2=%f types(%d-%d)\n", blockIdx.x, prefix, r2, type1, type2);
#ifdef AUTO_CAP
    float new_f;
    if (force > MX_FRC)
    {
        new_f = MX_FRC;
        printf("bl(%d) %s: f=%f: new_force=%f r2=%f types(%d-%d)\n", blockIdx.x, prefix, force, new_f, r2, type1, type2);
        force = new_f;
    }
    else 
        if (force < -MX_FRC)
        {
            new_f = -MX_FRC;
            printf("bl(%d) %s: f=%f: new_force=%f r2=%f types(%d-%d)\n", blockIdx.x, prefix, force, new_f, r2, type1, type2);
            force = new_f;
        }
#endif
}

__device__ void pair_1(int idA, int idB, float3 xyzA, float3* fxyzA, int typeA, float3 xyzB, float3* fxyzB, int typeB, float3 shift, cudaMD* md, float& engVdw, float& engCoul)
// universal variant of pair interaction
{
    float3 delta;
    float r2 = delta_and_r2_shift(xyzA, xyzB, delta, shift);    //* или без шифта, если в одной ячейке

    if (r2 <= md->r2Max)    //* если в одной ячейке, и разбиение типа sqrt(3) - можно опустить
    {
        float f = 0.f;
        float r = 0.f;      //* если есть Кулон

        // electrostatic contribution
        if (md->use_coul)
            f += md->funcCoul(r2, r, md->chProd[typeA][typeB], md, engCoul);    //* если есть Кулон

        // van der Waals contribution
        cudaVdW* vdw = md->vdws[typeA][typeB];
        if (vdw != NULL)
            if (r2 <= vdw->r2cut)   //* если в одной ячейке, и разбиение типа sqrt(3) - можно опустить
            {
                if (!vdw->use_radii)
                {
                    if (md->use_coul)
                        f += vdw->feng_r(r2, r, vdw, engVdw); //* либо версия только с силами, без энергий, если noCoul, то ещё и без r - хотя все это не даёт никакой оптимизации
                    else
                        f += vdw->feng(r2, vdw, engVdw); //* либо версия только с силами, без энергий, если noCoul, то ещё и без r - хотя все это не даёт никакой оптимизации
                }
                else // radii- (and, =>, temperature-) dependent pair potential
                    f += vdw->radi_func(r2, md->radii[idA], md->radii[idB], vdw, engVdw);

                //printf("vdw(at r=%f)=%f xA=%f xB=%f shift=%f fA + %f\n", sqrt(r2), f, xyzA.x, xyzB.x, shift.x, f * delta.x);
#ifdef DEBUG_MODE
                atomicAdd(&(md->nVdWcall), 1);
#endif
            }

        if (md->use_bnd == 2)
            try_to_bind(r2, idA, idB, typeA, typeB, md);
        if (md->use_ejump)
            try_to_jump(r2, idA, idB, typeA, typeB, md);
#ifdef AUTO_CAP
        force_autocap("betw_cell with Ind ", f, r2, typeA, typeB);
#else
        //stupid_force_verif("betw_cell with Ind ", f, r2, typeA, typeB);
#endif
        halfAtomicAddForces(fxyzA, fxyzB, f, delta);  //* half/full
#ifdef DEBUG_MODE
        atomicAdd(&(md->nFCall), 1);
#endif
    }
}

__device__ void atomicAddForces(float3* f1, float3* f2, float force, float3 delta)
// add (force * delta) to both atoms forces, full atomic operation
{
    atomicAdd(&(f1->x), force * delta.x);
    atomicAdd(&(f2->x), -force * delta.x);
    atomicAdd(&(f1->y), force * delta.y);
    atomicAdd(&(f2->y), -force * delta.y);
    atomicAdd(&(f1->z), force * delta.z);
    atomicAdd(&(f2->z), -force * delta.z);
}

__device__ void global_pair(int i, int j, int iStep, cudaMD* md, float &engCoul, float &engVdw)
{
    float3 delta;
    float r2 = delta_and_r2_byInd(i, j, md, delta);

    if (r2 <= md->r2Elec)    //! cuttoff в ряде случаев можно опустить эту проверку
    {
        float r = 0.f;
        float f = 0.f;
        int ti = md->types[i];
        int tj = md->types[j];

        // electrostatic contribution
        f += md->funcCoul(r2, r, md->chProd[ti][tj], md, engCoul);    //! если частицы заряжены

        // van der Waals contribution
        cudaVdW* vdw = md->vdws[ti][tj];
        if (vdw != NULL)
            if (r2 <= vdw->r2cut)   //! внутри ячейки эту проверку можно опустить
            {
                f += vdw->feng_r(r2, r, vdw, engVdw); //! либо версия только с силами, без энергий
#ifdef DEBUG_MODE
                atomicAdd(&(md->nVdWcall), 1);
#endif
            }
        if (md->use_bnd == 2)   // variable bonds
            try_to_bind(r2, i, j, ti, tj, md);
#ifdef AUTO_CAP
        force_autocap("global pair ", f, r2, ti, tj);
#else
        stupid_force_verif("global pair ", f, r2, ti, tj);
#endif
        atomicAddForces(&(md->frs[i]), &(md->frs[j]), f, delta);
#ifdef DEBUG_MODE
        atomicAdd(&(md->nFCall), 1);
        atomicAdd(&(md->sqrCoul), engCoul * engCoul);
#endif
    }
}
                                                                                                                         
__global__ void all_pair(int iStep, cudaMD* md)
{
    int i, j;
    int step = blockDim.x * gridDim.x;
    //    printf("%d\n", step);
    int ex = 0;  // exit flag
    int nAt = md->nAt;

    float engCoul = 0.f;
    float engVdw = 0.f;

    //degub
    int count = 0;

    __shared__ float shEngCoul;
    __shared__ float shEngVdw;

    if (threadIdx.x == 0)
    {
        shEngCoul = 0.f;
        shEngVdw = 0.f;
    }
    __syncthreads();

      
    i = 0;
    j = blockIdx.x * blockDim.x + threadIdx.x + 1;     // first pair is 0-1
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
        global_pair(i, j, iStep, md, engCoul, engVdw);
        count++;
        j = j + step;
    }

    save_coul_vdw(engCoul, engVdw, &shEngCoul, &shEngVdw, md);
#ifdef DEBUG_MODE
    atomicAdd(&(md->nAllPair), count);
#endif
}


//! идём чуть дальше, если потоков сильно много это по прежнему будет неэффективно, нужно сделать, чтобы одна пара обрабатывалась несколькими нитями
//! у нас все время идёт повышение иерархии: сначала нить на ячейку с соседями, затем нить на пару ячеек, теперь несколько нитей на пару ячеек
//! для этой реализации список атомов внутри ячейки должен быть двумерным (это увеличивает размер, но должно и увеличить скорость обращения)
//! и наконец то здесь будет использоваться shared memory и блоки. Тут блок идёт на пару ячеек, поэтому ячейки не могут быть произвольно большими
//!, а ограничены сколько частиц влезает в шаред мемори от двух ячекк

//! ну и раз уж мы здесь загрузили координаты, выполним циклы внутри ячеек. Запустим отдельный цикл по неперекрывающимся парам
//! здесь один блок - это одна пара ячеек! запускается на все первичные пары (0-1, 2-3, 4-5 и т.д.)
//! половина потоков будет копировать в ячейку A, половина - в ячейку B. Надо, чтобы размер блока был четным
__global__ void cell_list2a(cudaMD* md)
// ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ SHARED MEMORY
{

    int iC, iCA, iCB, step, step1;          // index of pair and cells
    int i, j, id0, id1, N, N1, deltId;      // counters and limits
    int id;                                 // atom index
    int nA, nB, nOwn;   // the number of atoms in cells A and B
    int ex = 0;  // exit flag

    //energy accumulators per block:
    __shared__ float engVdW, engCoul;
    //energy accumulators in thread:
    float eVdW = 0.0;
    float eCoul = 0.0;

    extern __shared__ int shMem[];      // declaration of dynamically allocated shared memory
    int* sAids = shMem;
    int* sAtypes = (int*)&sAids[md->maxAtPerCell];
    float3* sAxyz = (float3*)&sAtypes[md->maxAtPerCell];
    float3* sAfrs = (float3*)&sAxyz[md->maxAtPerCell];
    int* sBids = (int*)&sAfrs[md->maxAtPerCell];
    int* sBtypes = (int*)&sBids[md->maxAtPerCell];
    float3* sBxyz = (float3*)&sBtypes[md->maxAtPerCell];
    float3* sBfrs = (float3*)&sBxyz[md->maxAtPerCell];

    float3* sXyz;
    float3* sFrs;  // A or B depends on thread.id
    int* sTypes;
    int* sIds;

    //one block - one cell pair
    iCA = md->cellPairs[blockIdx.x].x;  // index of the first cell in the cell pair
    iCB = md->cellPairs[blockIdx.x].y;  // index of the second cell in the cell pair
    //float3 shift = get_shift(md->cellPairs[blockIdx.x].w, md);

    if (threadIdx.x == 0)
    {
        engVdW = 0.0;
        engCoul = 0.0;
#ifdef DEBUG_MODE
        md->nPairCult[blockIdx.x] += 1;
        md->nCelCult[iCA] += 1;
        md->nCelCult[iCB] += 1;
#endif
    }

    //! сперва потоки копируют данные в shared_memory
    if (threadIdx.x < blockDim.x / 2) //! половина потоков будет копировать в ячейку A, половина - в ячейку B. Надо, чтобы размер блока был четным
    {
        iC = iCA;
        sXyz = sAxyz;
        sFrs = sAfrs;
        sTypes = sAtypes;
        sIds = sAids;
        deltId = 0;
    }
    else
    {
        iC = iCB;
        sXyz = sBxyz;
        sFrs = sBfrs;
        sTypes = sBtypes;
        sIds = sBids;
        deltId = blockDim.x / 2;
    }
    nA = md->cells[iCA][0];
    nB = md->cells[iCB][0];
    nOwn = md->cells[iC][0]; // nA or nB

    //number of atoms per thread (factor 2 as two cells)
    step = ceilf(2 * (double)nOwn / (double)blockDim.x);

    //take into account that the second half of threads deal with the B cell
    id0 = (threadIdx.x - deltId) * step;
    N = min(id0 + step, nOwn);
    for (i = id0; i < N; i++)
    {
        id = md->cells[iC][i + 1];
        sIds[i] = id;     // save atom index
        sXyz[i] = md->xyz[id];    // copy atom coordinates
        sFrs[i] = make_float3(0.f, 0.f, 0.f);    // set zero forces
        sTypes[i] = md->types[id];
    }
    __syncthreads();

    //! interactions inside of cells
    //! алгоритм такой: в отличие от пред. раз, где мы весь интервал разбивали на равные отрезки и их обрабатывали нитями
    //! здесь мы раздадим каждой нити начальную точку, а инкеремент будет равен кол-ву нитей
    float3 zero_shift = make_float3(0.f, 0.f, 0.f);
    i = 0;
    j = threadIdx.x - deltId + 1;     // first pair is 0-1
    step1 = blockDim.x / 2;
    while (1)
    {
        while (j >= nOwn)
        {
            i++;
            if (i >= nOwn - 1)
            {
                ex = 1;
                break;
            }
            j = i + 1 + j - nOwn;
        }
        if (ex) break;
        pair_1(sIds[i], sIds[j], sXyz[i], &(sFrs[i]), sTypes[i], sXyz[j], &(sFrs[j]), sTypes[j], zero_shift, md, eVdW, eCoul);
        j = j + step1;
    }
    __syncthreads();

    //! вычисляем взаимодействия между атомами в разных ячейках

    step1 = ceilf((double)nA / (double)blockDim.x);
    id1 = threadIdx.x * step1;
    N1 = min(id1 + step1, nA);
    for (i = id1; i < N1; i++)
        for (j = 0; j < nB; j++)
        {
            pair_1(sAids[i], sBids[j], sAxyz[i], &sAfrs[i], sAtypes[i], sBxyz[j], &sBfrs[j], sBtypes[j], /*shift get_shift(md->cellPairs[blockIdx.x].w, md)*/ md->cellShifts[blockIdx.x], md, eVdW, eCoul);
        }
    __syncthreads();

    //! copy forces to global memory
    //! the same id0 and N as for global->shared copy
    for (i = id0; i < N; i++)
    {
        inc_float3((&md->frs[sIds[i]]), sFrs[i]);
    }

    save_coul_vdw(eCoul, eVdW, &engCoul, &engVdW, md); // to shared then to global memory
}
//end 'cell_list2a' function

//! вторая итерация, здесь уже будут все оставшиеся пары. Внутри ячеек все посчитано, поэтому только между ячейками
__global__ void cell_list2b(cudaMD* md)
{
    int iPair, iC, iCA, iCB, step; // index of pair and cells
    int i, j, id0, id1, N, N1, deltId;    // counters and limits
    int id;     // atom index
    int nA, nB, nOwn;  // the number of atoms in cells A and B

    //energy accumulators per block:
    __shared__ float engVdW, engCoul;
    //energy accumulators in thread:
    float eVdW = 0.f;
    float eCoul = 0.f;

    extern __shared__ int shMem[];      // declaration of shared memory
    int* sAids = shMem;
    int* sAtypes = (int*)&sAids[md->maxAtPerCell];
    float3* sAxyz = (float3*)&sAtypes[md->maxAtPerCell];
    float3* sAfrs = (float3*)&sAxyz[md->maxAtPerCell];
    int* sBids = (int*)&sAfrs[md->maxAtPerCell];
    int* sBtypes = (int*)&sBids[md->maxAtPerCell];
    float3* sBxyz = (float3*)&sBtypes[md->maxAtPerCell];
    float3* sBfrs = (float3*)&sBxyz[md->maxAtPerCell];

    // A or B depends on thread.id
    float3* sXyz;
    float3* sFrs;
    int* sTypes;
    int* sIds;

    iPair = blockIdx.x + md->nPair1;
    if (threadIdx.x == 0)
    {
        engVdW = 0.f;
        engCoul = 0.f;
#ifdef DEBUG_MODE
        atomicAdd(&(md->nPairCult[iPair]), 1);
        //md->nPairCult[iPair] += 1;
#endif
    }
    //__syncthreads();

    //one block - one cell pair
    iCA = md->cellPairs[iPair].x;  // index of the first cell in the cell pair
    iCB = md->cellPairs[iPair].y;  // index of the second cell in the cell pair
    //float3 shift = get_shift(md->cellPairs[iPair].w, md);

    // get the number of atoms
    nA = md->cells[iCA][0];
    nB = md->cells[iCB][0];

    // skip if no atoms in one of the cell
    if (!nA)
        return;
    if (!nB)
        return;

    //! сперва потоки копируют данные в shared_memory
    if (threadIdx.x < blockDim.x / 2) //! половина потоков будет копировать в ячейку A, половина - в ячейку B. Надо, чтобы размер блока был четным
    {
        iC = iCA;
        sXyz = sAxyz;
        sFrs = sAfrs;
        sTypes = sAtypes;
        sIds = sAids;
        deltId = 0;
    }
    else
    {
        iC = iCB;
        sXyz = sBxyz;
        sFrs = sBfrs;
        sTypes = sBtypes;
        sIds = sBids;
        deltId = blockDim.x / 2;
    }
    nOwn = md->cells[iC][0]; // nA or nB

    //number of atoms per thread (factor 2 as two cells)
    step = ceilf(2 * (double)nOwn / (double)blockDim.x);


    //take into account that the second half of threads deal with the B cell
    id0 = (threadIdx.x - deltId) * step;
    N = min(id0 + step, nOwn);
    for (i = id0; i < N; i++)
    {
        id = md->cells[iC][i + 1];
        sIds[i] = id;     // save atom index
        sXyz[i] = md->xyz[id];    // copy atom coordinates
        sFrs[i] = make_float3(0.f, 0.f, 0.f);    // set zero forces
        sTypes[i] = md->types[id];
    }
    __syncthreads();

    //! вычисляем взаимодействия между атомами в разных ячейках
    step = ceilf((double)nA / (double)blockDim.x);
    id1 = threadIdx.x * step;
    N1 = min(id1 + step, nA);

    for (i = id1; i < N1; i++)
        for (j = 0; j < nB; j++)
        {
            pair_1(sAids[i], sBids[j], sAxyz[i], &sAfrs[i], sAtypes[i], sBxyz[j], &sBfrs[j], sBtypes[j], /*shift*/ md->cellShifts[iPair], md, eVdW, eCoul);
        }
    __syncthreads();

    //! copy forces to global memory
    for (i = id0; i < N; i++)
    {
        inc_float3((&md->frs[sIds[i]]), sFrs[i]);
    }
    save_coul_vdw(eCoul, eVdW, &engCoul, &engVdW, md); // to shared then to global memory
}
//end 'cell_list2b' function

// Заметим, что для одного из моих конкретных вычислений ячейки получились слишком маленькие от 2 до 7 атомов
// так что большая часть потоков простаивала. Можно увеличить размер ячейки, 
// но тогда прийдется добавлять обязательным условие проверки расстояния между частицами, может это и не скажется на производительности
// а может мы немного перепишем код, чтобы блок обрабатывал сразу несколько пар
// заодно можем попробовать задействовать дополнительные измерения в блоке
__global__ void cell_list3a(int iStep, int pairStep, cudaMD* md)
// pairStep - step by pair numbers
{

    int iPair, iCell;   // indexes of pair and cell in global memory
    int intCell;        // internal cell number
    int step, sh;
    int i, j, id0, N, N1;    // counters and limits
    int id;     // atom index
    int ex = 0;  // exit flag

    //energy accumulators per block:
    __shared__ float engVdW, engCoul;
    //energy accumulators in thread:
    float eVdW = 0.f;
    float eCoul = 0.f;

    __shared__ int totAtm;      // total number of atoms

    // shared data arrays:
    extern __shared__ int shMem[];      // dynamically allocated shared memory
    int* id0s = shMem;         // first indexes of of shared arrays corresponding to a given cell
    int* nAts = (int*)&id0s[pairStep * 2];  // number of atoms in a given cell
    int* ids = (int*)&nAts[pairStep * 2];     // index is an index of a cell in block
    int* types = (int*)&ids[md->maxAtPerBlock];
    float3* xyz = (float3*)&types[md->maxAtPerBlock];
    float3* frs = (float3*)&xyz[md->maxAtPerBlock];
    //! теперь массивы общие для всех ячеек, надо лишь указать диапазоны каждому потоку

    //! НУЖНО РАЗДЕЛЯТЬ НОМЕР ЯЧЕЙКИ ФАКТИЧЕСКИЙ И НОМЕР ЯЧЕЙКИ в массиве ids0, который делит шаред массивы на куски согласно внутреннему номеру ячейки
    int xThreadPerPair = blockDim.x / pairStep;  // число thread в измерении x на пару (совпадает с числом потоков на ячейку)
    iPair = blockIdx.x * pairStep + threadIdx.x / xThreadPerPair;
    // exit from function if out of range
    if (iPair >= md->nPair1)
    {
        return;
    }

    //! нужно пробежаться по всем ячейкам всех пар, чтобы запомнить кол-ва атомов в каждой и начальные индексы
    if (threadIdx.x == 0)
        if (threadIdx.y == 0)
        {
            engVdW = 0.f;
            engCoul = 0.f;


            // define the first indexes in shared arrays corresponding to cell
            totAtm = 0;
            id0 = blockIdx.x * pairStep;
            N = min(id0 + pairStep, md->nPair1);
            j = 0;
            for (i = id0; i < N; i++)
            {
                id0s[j] = totAtm;
                nAts[j] = md->cells[md->cellPairs[i].x][0];
                totAtm += nAts[j];
                id0s[j + 1] = totAtm;
                nAts[j + 1] = md->cells[md->cellPairs[i].y][0];
                totAtm += nAts[j + 1];
                j += 2;
            }
            //if (totAtm > md->maxAtPerBlock)
              //  printf("%d 3a: in block %d maxAtm=%d, more than max = %d\n", iStep, blockIdx.x, totAtm, md->maxAtPerBlock);

        }
    __syncthreads();

    if (threadIdx.y == 0)
        iCell = md->cellPairs[iPair].x;
    else // threadIdx.y == 1
        iCell = md->cellPairs[iPair].y;

    // an interanl index of cell in block (not in global memory!)
    intCell = threadIdx.x / xThreadPerPair * 2 + threadIdx.y;

    // не забываем, что щас кол-во thread - двумерно, поэтому индексы потоков: [0..31][0..1]. y = 0 обрабатывает 1ую ячейку в паре, y=1 - вторую
    // thread index (the same for threadIdx.y = 0 and 1) in one cell/pair
    int iThread = threadIdx.x % xThreadPerPair;

    //! сперва потоки копируют данные в shared_memory

    step = ceil((double)nAts[intCell] / (double)xThreadPerPair);    // количество атомов, обрабатываемых данным потоком
    sh = id0s[intCell];    // индекс в shared memory, в который надо начинать копировать 
    id0 = iThread * step;
    N = min(id0 + step, nAts[intCell]);
    for (i = id0; i < N; i++)
    {
        id = md->cells[iCell][i + 1];
        ids[sh + i] = id;     // save atom index
        xyz[sh + i] = md->xyz[id];    // copy atom coordinates
        frs[sh + i] = make_float3(0.f, 0.f, 0.f);    // set zero forces
        types[sh + i] = md->types[id];
    }
    __syncthreads();

    //! interactions inside of cells

    //! алгоритм такой: в отличие от пред. раз, где мы весь интервал разбивали на равные отрезки и их обрабатывали нитями
    //! здесь мы раздадим каждой нити начальную точку, а инкеремент будет равен кол-ву нитей

    i = 0;
    j = iThread + 1;     // first pair is 0-1
    step = xThreadPerPair;
    float3 force = make_float3(0.f, 0.f, 0.f);   // keep force for i-th atom to decrease the number atomic operations
    float3 zero_shift = make_float3(0.f, 0.f, 0.f);
    while (1)
    {
        while (j >= nAts[intCell])
        {
            // we finish this current value of i, copy force to shared mem and refresh its variable
            atomic_incFloat3(&(frs[i + sh]), force);
            force = make_float3(0.f, 0.f, 0.f);
            i++;
            if (i >= nAts[intCell] - 1)
            {
                ex = 1;
                break;
            }
            j = i + 1 + j - nAts[intCell];
        }
        if (ex) break;
        pair_1(ids[i + sh], ids[j + sh], xyz[i + sh], &(force), types[i + sh], xyz[j + sh], &(frs[j + sh]), types[j + sh], zero_shift, md, eVdW, eCoul);
        j = j + step;
    }
    __syncthreads();

/*
    // новый способ обхода пар
    int n1 = nAts[intCell];
    int np = n1 * (n1 - 1) / 2;
    step = ceil((double)np / xThreadPerPair);
    int k, k0, nk;
    k0 = iThread * step;
    nk = min(k0 + step, np);
    i = 0;
    j = k0 + 1;
    while (j >= n1)
    {
        j = j - n1 + i + 2;
        i++;
    }
    float3 force = make_float3(0.f, 0.f, 0.f);   // keep force for i-th atom to decrease the number atomic operations
    for (k = k0; k < nk; k++)
    {
        pair_in_cell_wInd_ha(ids[i + sh], ids[j + sh], xyz[i + sh], &(force), types[i + sh], xyz[j + sh], &(frs[j + sh]), types[j + sh], md, eVdW, eCoul);
        j++;
        if (j >= n1)
        {
            // we finish this current value of i, copy force to shared mem and refresh its variable
            atomic_incFloat3(&(frs[i + sh]), force);
            i++;
            j = i + 1;
            force = make_float3(0.f, 0.f, 0.f);
        }

    }
    __syncthreads();
*/


    //! вычисляем взаимодействия между атомами в разных ячейках

    //! тут получается в 2 раза больше потоков, посколкьу на пару
    //! выбираем ячейку где больше атомов и от неёё пляшем
    //! а нет, переменная cellShift чувствительна к перестановке i-j, поэтому сохраняем стандатый порядок

    intCell = intCell - threadIdx.y;    // получаем внутр индекс первой ячейки в паре


    step = ceilf((double)nAts[intCell]/*nAt*/ / (double)(xThreadPerPair * 2));

    iThread = iThread * 2 + threadIdx.y;    // теперь уже у нас в два раза больше потоков
    id0 = id0s[intCell] + iThread * step;
    N = min(id0 + step, id0s[intCell] + nAts[intCell]/*nAt*/);
    N1 = id0s[intCell + 1] + nAts[intCell + 1]/*md->cells[md->cellPairs[iPair].y][0]*/;

    //if (blockIdx.x == 0)
      //  printf("th[%d,%d] cell %d i=%d..%d(nAt=%d) j=%d..%d\n", threadIdx.x, threadIdx.y, intCell, id0, N-1, nAt, id0s[intCell + 1], N1-1);

    for (i = id0; i < N; i++)
    {
        force = make_float3(0.f, 0.f, 0.f);
        for (j = id0s[intCell + 1]; j < N1; j++)
        {
            pair_1(ids[i], ids[j], xyz[i], &force, types[i], xyz[j], &frs[j], types[j], md->cellShifts[iPair], md, eVdW, eCoul);
        }
        inc_float3(&(frs[i]), force);
    }
    __syncthreads();


    //! copy forces to global memory, use all threads for whole shared array range
    step = ceilf((double)totAtm / (double)(blockDim.x * blockDim.y));
    iThread = threadIdx.x * 2 + threadIdx.y;
    id0 = iThread * step;
    N = min(id0 + step, totAtm);
    //    if (blockIdx.x == 0)
      //      printf("i=%d..<%d\n", id0, N);
    for (i = id0; i < N; i++)
    {
        //md->frs[ids[i]] = frs[i];
        //! прийдется делать так, поскольку силы добавляются уже в apply_bonds перед этим
        //! нет, не обязательно атомик, главное +=
        inc_float3(&(md->frs[ids[i]]), frs[i]);
    }

    //! unite energy from each thread
    // from threads inside one block to shared memory
    atomicAdd(&engCoul, eCoul);
    atomicAdd(&engVdW, eVdW);
    __syncthreads();

    // form each block to global memory
    if (threadIdx.x == 0)
        if (threadIdx.y == 0)
        {
            atomicAdd(&md->engCoul1, engCoul);  // to global memory
            atomicAdd(&md->engVdW, engVdW);
        }
}
// end 'cell_list3a' function

//! вторая итерация, здесь уже будут все оставшиеся пары. Внутри ячеек все посчитано, поэтому только между ячейками
__global__ void cell_list3b(int iStep, int pairStep, cudaMD* md)
// pairStep - step by pair numbers
{
    int iPair, iCell;   // indexes of pair and cell in global memory
    int intCell;        // internal cell number
    int step, sh;
    int i, j, id0, N, N1;    // counters and limits
    int id;     // atom index

    //energy accumulators per block:
    __shared__ float engVdW, engCoul;
    //energy accumulators in thread:
    float eVdW = 0.f;
    float eCoul = 0.f;

    __shared__ int totAtm;      // total number of atoms

    // shared data arrays:
    extern __shared__ int shMem[];      // dynamically allocated shared memory
    int* id0s = shMem;         // first indexes of of shared arrays corresponding to a given cell
    int* nAts = (int*)&id0s[pairStep * 2];  // number of atoms in a given cell
    int* ids = (int*)&nAts[pairStep * 2];     // index is an index of a cell in block
    int* types = (int*)&ids[md->maxAtPerBlock];
    float3* xyz = (float3*)&types[md->maxAtPerBlock];
    float3* frs = (float3*)&xyz[md->maxAtPerBlock];
    //! теперь массивы общие для всех ячеек, надо лишь указать диапазоны каждому потоку

    //! НУЖНО РАЗДЕЛЯТЬ НОМЕР ЯЧЕЙКИ ФАКТИЧЕСКИЙ И НОМЕР ЯЧЕЙКИ в массиве ids0, который делит шаред массивы на куски согласно внутреннему номеру ячейки
    int xThreadPerPair = blockDim.x / pairStep;  // число thread в измерении x на пару (совпадает с числом потоков на ячейку)

    iPair = md->nPair1 + blockIdx.x * pairStep + threadIdx.x / xThreadPerPair;
    // exit from function if out of range
    if (iPair >= md->nPair)
    {
        return;
    }

    //! нужно пробежаться по всем ячейкам всех пар, чтобы запомнить кол-ва атомов в каждой и начальные индексы
    if (threadIdx.x == 0)
        if (threadIdx.y == 0)
        {
            engVdW = 0.f;
            engCoul = 0.f;

            // define the first indexes in shared arrays corresponding to cell
            totAtm = 0;
            id0 = md->nPair1 + blockIdx.x * pairStep;
            N = min(id0 + pairStep, md->nPair);
            j = 0;
            for (i = id0; i < N; i++)
            {
                id0s[j] = totAtm;
                nAts[j] = md->cells[md->cellPairs[i].x][0];
                totAtm += nAts[j];
                id0s[j + 1] = totAtm;
                nAts[j + 1] = md->cells[md->cellPairs[i].y][0];
                totAtm += nAts[j + 1];
                j += 2;
            }
            if (totAtm > md->maxAtPerBlock)
                printf("%d: th(%d,%d) in block %d (pair: %d) maxAtm=%d, more than max = %d nPair1=%d\n", iStep, threadIdx.x, threadIdx.y, blockIdx.x, iPair, totAtm, md->maxAtPerBlock, md->nPair1);


        }
    __syncthreads();


    if (threadIdx.y == 0)
        iCell = md->cellPairs[iPair].x;
    else // threadIdx.y == 1
        iCell = md->cellPairs[iPair].y;

    // an interanl index of cell in block (not in global memory!)
    intCell = threadIdx.x / xThreadPerPair * 2 + threadIdx.y;

    // не забываем, что щас кол-во thread - двумерно, поэтому индексы потоков: [0..31][0..1]. y = 0 обрабатывает 1ую ячейку в паре, y=1 - вторую
    // thread index (the same for threadIdx.y = 0 and 1) in one cell/pair
    int iThread = threadIdx.x % xThreadPerPair;

    //! сперва потоки копируют данные в shared_memory

    step = ceil((double)nAts[intCell] / (double)xThreadPerPair);    // количество атомов, обрабатываемых данным потоком
    sh = id0s[intCell];    // индекс в shared memory, в который надо начинать копировать 
    id0 = iThread * step;
    N = min(id0 + step, nAts[intCell]);
    //if (blockIdx.x == 220)
      //  printf("thread[%d,%d]: iPair=%d iCell=%d intCell=%d iThread=%d i=%d..<%d -> %d..<%d shFind=%d nAt=%d\n", threadIdx.x, threadIdx.y, iPair, iCell, intCell, iThread, id0, N, sh+id0, sh+N, sh, nAts[intCell]);
    for (i = id0; i < N; i++)
    {
        //if (blockIdx.x == 220)
          //  printf("thread[%d,%d]: cell[%d][%d](%d) -> sh[%d] val=%d  | ids0=%d step=%d iThread=%d shFind=%d(+%d=%d)\n", threadIdx.x, threadIdx.y, iCell, i, md->cells[iCell][0], sh + i, md->cells[iCell][i + 1], id0s[intCell], step, iThread, sh, i, sh + i);
        id = md->cells[iCell][i + 1];
        ids[sh + i] = id;     // save atom index
        xyz[sh + i] = md->xyz[id];    // copy atom coordinates
        frs[sh + i] = make_float3(0.f, 0.f, 0.f);    // set zero forces
        types[sh + i] = md->types[id];
    }
    __syncthreads();


    //! вычисляем взаимодействия между атомами в разных ячейках
    intCell = intCell - threadIdx.y;    // получаем внутр индекс первой ячейки в паре
    step = ceilf((double)nAts[intCell] / (double)(xThreadPerPair * 2));
    iThread = iThread * 2 + threadIdx.y;    // теперь уже у нас в два раза больше потоков
    id0 = id0s[intCell] + iThread * step;
    N = min(id0 + step, id0s[intCell] + nAts[intCell]);
    N1 = id0s[intCell + 1] + nAts[intCell + 1];
    //    if (blockIdx.x == 220)
      //      printf("th[%d,%d] cell %d i=%d..%d(nAt=%d) j=%d..%d\n", threadIdx.x, threadIdx.y, intCell, id0, N-1, nAt, id0s[intCell + 1], N1-1);


    float3 force;
    for (i = id0; i < N; i++)
    {
        force = make_float3(0.f, 0.f, 0.f);
        for (j = id0s[intCell + 1]; j < N1; j++)
        {
           pair_1(ids[i], ids[j], xyz[i], &force, types[i], xyz[j], &frs[j], types[j], md->cellShifts[iPair], md, eVdW, eCoul);
        }
        inc_float3(&(frs[i]), force);
    }
    __syncthreads();

    //! copy forces to global memory, use all threads for whole shared array range
    step = ceilf((double)totAtm / (double)(blockDim.x * blockDim.y));
    iThread = threadIdx.x * 2 + threadIdx.y;
    id0 = iThread * step;
    N = min(id0 + step, totAtm);
    for (i = id0; i < N; i++)
    {
        atomic_incFloat3(&(md->frs[ids[i]]), frs[i]);
    }

    //! unite energy from each thread
    // from threads inside one block to shared memory and...
    atomicAdd(&engCoul, eCoul);
    atomicAdd(&engVdW, eVdW);
    __syncthreads();

    //... from each block to global memory
    if (threadIdx.x == 0)
        if (threadIdx.y == 0)
        {
            atomicAdd(&md->engCoul1, engCoul);
            atomicAdd(&md->engVdW, engVdW);
        }
}
// end 'cell_list3b' function

__global__ void cell_list3b_noshared(int iStep, cudaMD* md)
// without shared memory, so one block - one cell pair
{
    int i, j;    // counters and limits
    int ia, ja;     // indexes of atoms

    //energy accumulators per block:
    __shared__ float engVdW, engCoul;
    //energy accumulators in thread:
    float eVdW = 0.f;
    float eCoul = 0.f;

    __shared__ int cell1, cell2;    // cell ids
    __shared__ int na1, na2;        // number of atoms in cell


    int iPair = md->nPair1 + blockIdx.x;

    if (threadIdx.x == 0)
    {
        engVdW = 0.f;
        engCoul = 0.f;
        cell1 = md->cellPairs[iPair].x;
        na1 = md->cells[cell1][0];
        cell2 = md->cellPairs[iPair].y;
        na2 = md->cells[cell2][0];
    }
    __syncthreads();

    // skip, if no atomis in one of cell
    if (na1 == 0)
        return;
    if (na2 == 0)
        return;

    int step = ceil((double)na1 / blockDim.x);    // количество атомов, обрабатываемых данным потоком
    int id0 = threadIdx.x * step;
    int N = min(id0 + step, na1);

    float3 force;
    for (i = id0; i < N; i++)
    {
        ia = md->cells[cell1][i + 1];
        force = make_float3(0.f, 0.f, 0.f);
        for (j = 0; j < na2; j++)
        {
            ja = md->cells[cell2][j + 1];
            pair_1(ia, ja, md->xyz[ia], &force, md->types[ia], md->xyz[ja], &(md->frs[ja]), md->types[ja], md->cellShifts[iPair], md, eVdW, eCoul);
        }
        atomic_incFloat3(&(md->frs[ia]), force);
    }

    save_coul_vdw(eCoul, eVdW, &engCoul, &engVdW, md); // to shared then to global memory
}
// end 'cell_list3b_noshared' function

__global__ void cell_list_inSameCell(cudaMD* md)
// one block - one cell
{
    int id1, id2;
    int i, j;    
    int ex = 0;  // exit flag

    //energy accumulators per block:
    __shared__ float engVdW, engCoul;
    //energy accumulators in thread:
    float eVdW = 0.f;
    float eCoul = 0.f;

    if (threadIdx.x == 0)
    {
        engVdW = 0.f;
        engCoul = 0.f;
#ifdef DEBUG_MODE
        //md->nPairCult[blockIdx.x] += 1;
        //md->nCelCult[iCA] += 1;
        //md->nCelCult[iCB] += 1;
#endif
    }

    int nAt = md->cells[blockIdx.x][0];
    if (nAt < 2)   // skip empty cells or cell with one atom (no pairs)
        return;

    //! interactions inside of cells
    //! алгоритм такой: в отличие от пред. раз, где мы весь интервал разбивали на равные отрезки и их обрабатывали нитями
    //! здесь мы раздадим каждой нити начальную точку, а инкеремент будет равен кол-ву нитей
    float3 zero_shift = make_float3(0.f, 0.f, 0.f);
    i = 0;
    j = threadIdx.x + 1;     // first pair is 0-1
    int step = blockDim.x;
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
        id1 = md->cells[blockIdx.x][i + 1];
        id2 = md->cells[blockIdx.x][j + 1];
        pair_1(id1, id2, md->xyz[id1], &(md->frs[id1]), md->types[id1], md->xyz[id2], &(md->frs[id2]), md->types[id2], zero_shift, md, eVdW, eCoul);
        j = j + step;
    }

    save_coul_vdw(eCoul, eVdW, &engCoul, &engVdW, md); // to shared then to global memory
}
//end 'cell_list_inSameCell' function

__global__ void cell_list_allCellPair(cudaMD* md)
{
    int iCA, iCB, step; // index of pair and cells
    int i, j, id1, N1;    // counters and limits
    //int id;     // atom index
    int nA, nB;  // the number of atoms in cells A and B
    float3 force;
    int a1, a2; // real indexes of atoms

    //energy accumulators per block:
    __shared__ float engVdW, engCoul;
    //energy accumulators in thread:
    float eVdW = 0.f;
    float eCoul = 0.f;

    void (*funcTreat)(int idA, int idB, float3 xyzA, float3 * fxyzA, int typeA, float3 xyzB, float3 * fxyzB, int typeB, float3 shift, cudaMD * md, float& engVdw, float& engCoul);

    if (threadIdx.x == 0)
    {
        engVdW = 0.f;
        engCoul = 0.f;
#ifdef DEBUG_MODE
        //atomicAdd(&(md->nPairCult[iPair]), 1);
        //md->nPairCult[iPair] += 1;
#endif
    }
    __syncthreads();

    //one block - one cell pair
    iCA = md->cellPairs[blockIdx.x].x;  // index of the first cell in the cell pair
    iCB = md->cellPairs[blockIdx.x].y;  // index of the second cell in the cell pair
    //float3 shift = get_shift(md->cellPairs[iPair].w, md);

    // get the number of atoms
    nA = md->cells[iCA][0];
    nB = md->cells[iCB][0];

    // skip if no atoms in one of the cell
    if (!nA)
        return;
    if (!nB)
        return;

    step = ceilf((double)nA / (double)blockDim.x);
    id1 = threadIdx.x * step;
    N1 = min(id1 + step, nA);

    for (i = id1; i < N1; i++)
    {
        a1 = md->cells[iCA][i + 1];
        force = make_float3(0.f, 0.f, 0.f);
        for (j = 0; j < nB; j++)
        {
            a2 = md->cells[iCB][j + 1];
            pair_1(a1, a2, md->xyz[a1], &force, md->types[a1], md->xyz[a2], &(md->frs[a2]), md->types[a2], /*shift*/ md->cellShifts[blockIdx.x], md, eVdW, eCoul);
        }
        // add force to global memory
        atomic_incFloat3(&(md->frs[a1]), force);
    }

    save_coul_vdw(eCoul, eVdW, &engCoul, &engVdW, md); // to shared then to global memory
}
//end 'cell_list_allCellPair' function


void iter_cellList(int iStep, int nB1, int nB2, dim3 dim, hostManagMD *man, cudaMD *devMD)
//! перенести определение nB1, nB2 и dim в init_cuda
{
#ifdef USE_CELL2
    //cell_list2a << < man->nPair1Block, man->nSingProc, man->memPerPairBlock >> > (devMD);
    cell_list_inSameCell << <1332, 4 >> > (devMD); // 4|16 - 123  8/16 - 124 с
#endif
#ifdef USE_CELL3
    cell_list3a << < nB1, dim, man->memPerPairsBlock >> > (iStep, man->pairPerBlock, devMD);
#endif
#ifndef USE_ALLPAIR
    cudaThreadSynchronize();
#endif
#ifdef USE_CELL2
    //cell_list2b << < man->nPair2Block, man->nSingProc, man->memPerPairBlock >> > (devMD);
    cell_list_allCellPair << <167706, 10 >> > (devMD); // was 4 131 s 8 - 124  16 - 123  4/10 -124s
#endif
#ifdef USE_CELL3
    cell_list3b << < nB2, dim, man->memPerPairsBlock >> > (iStep, man->pairPerBlock, devMD);
    //cell_list3b_noshared << < man->nPair2Block, 16 >> > (iStep, devMD);
#endif
#ifdef USE_ALLPAIR
    all_pair << < man->nMultProc, man->nSingProc >> > (iStep, devMD);
#endif
    cudaThreadSynchronize();
}

__device__ int get_internal_id(int id, int tot_id, int divider, int& divider_id, int& nids)
// допустим надо разделить равномерно N потоков на M чего-то там, а на цело они не делятся, эта функция как раз и возвращает:
//  номер потока внутри чего-то там (ячейки например), число потоков в этой ячейке nids и номер ячейки divider_id
{
    if (divider == 0)
        printf("div by zero!!!\n");

    int d = tot_id / divider;
    int m = tot_id % divider;   //! maybe there is one function for both calc

    int first = m * (d + 1);
    if (id < first)
    {
        divider_id = id / (d + 1);
        nids = d + 1;
        return id % (d + 1);
    }
    else
    {
        divider_id = (id - first) / d + m;
        nids = d;
        return (id - first) % d;
    }
}

//! эта функция обрабатывает пары 0-1 2-3 4-5 т.е. без пересечения по атомам. + вычисляет взаимодействия в самих ячейках, а потом ещё и пересекающиеся (елси несколько пар)
// каждый блок считывает несколько пар, все блоки считывают полностью все ячейки
__global__ void cell_list4a(int cellPerBlock, cudaMD* md)
// pairStep - step by pair numbers
//! ограничение: функция должна вызываться так, чтобы число потоков нацело делилось на число ячеек
{

    int i, j;
    int sh1, sh2, step;
    int ex = 0;     // exit flag
    //if (threadIdx.x == 0)
      //  printf("start 4a(%d,%d)\n", blockIdx.x, threadIdx.x);

    //energy accumulators per block:
    __shared__ float shVdWEng, shCoulEng;
    //energy accumulators in thread:
    float eVdW = 0.f;
    float eCoul = 0.f;

    if (threadIdx.x == 0)
    {
        shVdWEng = 0.f;
        shCoulEng = 0.f;
    }

    // shared data arrays:
    extern __shared__ int shMem[];      // dynamically allocated shared memory
    //! теперь массивы общие для всех ячеек, надо лишь указать диапазоны каждому потоку

    //! имеет ли смысл эти вещи тоже вынести в shared?
    // first cell index and last cell index + 1
    int cell0 = blockIdx.x * cellPerBlock;
    int cellN = min(cell0 + cellPerBlock, md->nCell);
    int nCell = cellN - cell0;
    // first atom index for block
    int at0 = md->firstAtomInCell[cell0];
    // number of atoms in block:
    //int nat = md->firstAtomInCell[cellN] - at0 + 1; // this variant crush for the last block (cellN out of range)
    int nat = md->firstAtomInCell[cellN - 1] - at0 + md->nAtInCell[cellN - 1]; // this variant crush for the last block (cellN out of range)
    int atPerThread = ceil((double)nat / blockDim.x);       // define number of copied atoms per thread
    __syncthreads();


    int* nAts = shMem;                  // number of atoms in a given cell
    int* fAts = (int*)&nAts[nCell];     // first index of atom for a given cell (in internal numeration)
    int* types = (int*)&fAts[nCell];
    float3* xyz = (float3*)&types[nat];
    float3* frs = (float3*)&xyz[nat];

    // copy number of atoms and first index
    //! число потоков должно быть больше числа ячеек
    if (threadIdx.x < nCell)
    {
        nAts[threadIdx.x] = md->nAtInCell[cell0 + threadIdx.x];
        fAts[threadIdx.x] = md->firstAtomInCell[cell0 + threadIdx.x] - at0; // internal index
    }

    // define atom range for current thread:
    int ia0 = atPerThread * threadIdx.x;
    int Na = min(ia0 + atPerThread, nat);
    // copy atoms to shared memory
    for (i = ia0; i < Na; i++)
    {
        //id = md->cells[iCell][i + 1];
        //ids[sh + i] = id;     // save atom index
        xyz[i] = md->xyz[i + at0];    // copy atom coordinates
        types[i] = md->types[i + at0];
        frs[i] = make_float3(0.f, 0.f, 0.f);    // set zero forces
    }
    __syncthreads();

    //INTERACTION INSIDE SAME CELLS
    //! алгоритм такой: в отличие от пред. раз, где мы весь интервал разбивали на равные отрезки и их обрабатывали нитями
    //! здесь мы раздадим каждой нити начальную точку, а инкеремент будет равен кол-ву нитей

    float3 zero_shift = make_float3(0.f, 0.f, 0.f);
    int iCell;
    int iThread = get_internal_id(threadIdx.x, blockDim.x, nCell, iCell, step); // сразу получим и номер потока внутри ячейки и номер ячейки и число потоков на ячейку (step)

    int cur_nat = nAts[iCell];

    if (cur_nat > 1)
    {
        sh1 = fAts[iCell];
        i = 0;
        j = iThread + 1;     // first pair is 0-1
        while (1)
        {
            while (j >= cur_nat)
            {
                i++;
                if (i >= cur_nat - 1)
                {
                    ex = 1;
                    break;
                }
                j = i + 1 + j - cur_nat;
            }
            if (ex) break;
            pair_1(i + sh1 + at0, j + sh1 + at0, xyz[i + sh1], &(frs[i + sh1]), types[i + sh1], xyz[j + sh1], &(frs[j + sh1]), types[j + sh1], zero_shift, md, eVdW, eCoul);
            j = j + step;
        }
    }
    __syncthreads();

    //INTERACTION WITH DIFFERENT CELLS
    // допущение: потоков все равно >= пар ячеек. Для 8 ячеек максимум 4*7=28 пар, потоков должно быть больше (дефолтно 64)
    int iPair, cell1, cell2;
    if (md->nFirstPairs[blockIdx.x] != 0)  // sometimes block of cells may not have pairs (for example, if only one cell in block)
    {
        iThread = get_internal_id(threadIdx.x, blockDim.x, md->nFirstPairs[blockIdx.x], iPair, step); // сразу получим и номер потока внутри праы и номер пары и число потоков на пару (step)
        // а теперь у нас есть несколько потоков на пару. Каждый поток пусть возьмёт диапазон атомов первыой ячейки пары и все атомы второй ячейки:
        // read cell numbers from pairs and convert into internal number

        cell1 = md->firstPairs[blockIdx.x][iPair].x - cell0;
        cell2 = md->firstPairs[blockIdx.x][iPair].y - cell0;
        sh1 = fAts[cell1];
        sh2 = fAts[cell2];

        atPerThread = ceil((double)nAts[cell1] / step);
        int id0 = iThread * atPerThread;
        int N = min(id0 + atPerThread, nAts[cell1]);

        float3 force;
        for (i = id0; i < N; i++)
        {
            force = make_float3(0.f, 0.f, 0.f);
            for (j = 0; j < nAts[cell2]; j++)
                pair_1(i + sh1 + at0, j + sh2 + at0, xyz[i + sh1], &force, types[i + sh1], xyz[j + sh2], &frs[j + sh2], types[j + sh2], md->firstShifts[blockIdx.x][iPair], md, eVdW, eCoul);
            atomic_incFloat3(&(frs[i + sh1]), force);
        }

    }
    __syncthreads();

    //! copy forces to global memory, use all threads for whole shared array range
    for (i = ia0; i < Na; i++)
        inc_float3(&(md->frs[i + at0]), frs[i]);

    save_coul_vdw(eCoul, eVdW, &shCoulEng, &shVdWEng, md); // to shared then to global memory
}
// end 'cell_list4a' function


//! блоки получают на вход: индекс первой ячейки, индекс второй ячейки, кол-во ячеек после неё 
__global__ void cell_list4b(cudaMD* md)
{
    int i, j, k;
    //int sh1, sh2, step;

    //energy accumulators per block:
    __shared__ float shVdWEng, shCoulEng;
    //energy accumulators in thread:
    float eVdW = 0.f;
    float eCoul = 0.f;

    if (threadIdx.x == 0)
    {
        shVdWEng = 0.f;
        shCoulEng = 0.f;
    }

    // shared data arrays:
    extern __shared__ int shMem[];      // dynamically allocated shared memory
    //! теперь массивы общие для всех ячеек, надо лишь указать диапазоны каждому потоку

    int cell0 = md->cellBlocks[blockIdx.x].x;
    int cell1 = md->cellBlocks[blockIdx.x].y;
    int nCell = md->cellBlocks[blockIdx.x].z;
    //int nPair = md->cellBlocks[blockIdx.x].w;

    if (md->nAtInCell[cell0] == 0)  // no atoms in the first cell, escape block
        return;

    // first atom index for block
    int at0 = md->firstAtomInCell[cell0];
    int at0n = md->firstAtomInCell[cell1];
    // number of atoms in block:
    //int nat = md->firstAtomInCell[cellN] - at0 + 1; // this variant crush for the last block (cellN out of range)
    int nat = md->nAtInCell[cell0] + md->firstAtomInCell[cell1 + nCell - 1] - at0n + md->nAtInCell[cell1 + nCell - 1];

    if (nat == md->nAtInCell[cell0])    // atoms only in the first cell, escap
        return;

    int nat1 = nat - md->nAtInCell[cell0];

    int atPerThread = ceil((double)nat / blockDim.x);       // define number of copied atoms per thread
    __syncthreads();

    int* nAts = shMem;                  // number of atoms in a given cell
    int* fAts = (int*)&nAts[nCell + 1];     // first index of atom for a given cell
    int* types = (int*)&fAts[nCell + 1];
    int* shIds = (int*)&types[nat];     // ids in secShifts array
    float3* xyz = (float3*)&shIds[nat];
    float3* frs = (float3*)&xyz[nat];

    // copy number of atoms and first index
    if (threadIdx.x == 0)
    {
        nAts[0] = md->nAtInCell[cell0];
        fAts[0] = 0;
    }
    else
        if (threadIdx.x <= nCell)
        {
            nAts[threadIdx.x] = md->nAtInCell[cell1 + threadIdx.x - 1];
            fAts[threadIdx.x] = md->firstAtomInCell[cell1 + threadIdx.x - 1] - md->firstAtomInCell[cell1] + md->nAtInCell[cell0];
        }

    // define which atom belong to which cell(and corresponding cell shift)
    if (threadIdx.x == 0)
    {
        j = nCell;
        for (i = nat - 1; i >= nAts[0]; i--)
        {
            while (i < fAts[j])
                j--;
            shIds[i] = j - 1;   // pair starts from 0-1, so the second cell coresponds to index = 0
        }

    }
    __syncthreads();


    // define atom range for current thread:
    int ia0 = atPerThread * threadIdx.x;
    int Na = min(ia0 + atPerThread, nat);
    // copy atoms to shared memory
    for (i = ia0; i < Na; i++)
    {
        //id = md->cells[iCell][i + 1];
        //ids[sh + i] = id;     // save atom index
        if (i < nAts[0])    // copy atoms of the first cell
        {
            xyz[i] = md->xyz[i + at0];
            types[i] = md->types[i + at0];
        }
        else  // copy atoms of the rest cells
        {
            j = i - nAts[0] + at0n;
            xyz[i] = md->xyz[j];
            types[i] = md->types[j];
        }
        frs[i] = make_float3(0.f, 0.f, 0.f);
    }
    __syncthreads();

    //INTERACTION WITH DIFFERENT CELLS
    // разделяем число потоков поровну по атомам второй и последующей ячейки, каждый поток обрабатывает взаимодействие "своего" диапазона с атомами первой ячейки
    float3 shift = make_float3(0.f, 0.f, 0.f);
    atPerThread = ceil((double)nat1 / blockDim.x);
    int id0 = threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, nat1);
    float3 force;
    for (i = id0; i < N; i++)
    {
        force = make_float3(0.f, 0.f, 0.f);
        for (j = 0; j < nAts[0]; j++)
        {
            k = i + fAts[1];
            pair_1(j + at0, i + at0n, xyz[j], &force, types[j], xyz[k], &frs[k], types[k], md->secShifts[blockIdx.x][shIds[k]], md, eVdW, eCoul);
        }
        atomic_incFloat3(&frs[j], force);
    }
    __syncthreads();

    //! copy forces to global memory, use all threads for whole shared array range
    for (i = ia0; i < Na; i++)
    {
        if (i < nAts[0])    // copy atoms of the first cell
        {
            atomicAdd(&(md->frs[i + at0].x), frs[i].x);
            atomicAdd(&(md->frs[i + at0].y), frs[i].y);
            atomicAdd(&(md->frs[i + at0].z), frs[i].z);
        }
        else  // copy atoms of the rest cells
        {
            j = i - nAts[0] + at0n;
            atomicAdd(&(md->frs[j].x), frs[i].x);
            atomicAdd(&(md->frs[j].y), frs[i].y);
            atomicAdd(&(md->frs[j].z), frs[i].z);
        }
    }

    save_coul_vdw(eCoul, eVdW, &shCoulEng, &shVdWEng, md); // to shared then to global memory
}
// end 'cell_list4b' function

__global__ void cell_list4b_noshared(cudaMD* md)
// cell list, part b (between different cells): version without shared memory
// the algorithm requires sorted atom arrays
// called 'no shared' but use shared memory for storing of 'shifts' (coordinated differences for periodic accounintg)
{
    int i, j, k;
    int cell0, cell1, nCell, at0, at0n, n0, nat, nat1;

    cell0 = md->cellBlocks[blockIdx.x].x;       // index of the first cell
    n0 = md->nAtInCell[cell0];                  // number of atoms in the first cell
    if (n0 == 0)                                // no atoms in the first cell, escape block
        return;

    at0 = md->firstAtomInCell[cell0];   // the first atom index for block
    cell1 = md->cellBlocks[blockIdx.x].y;
    at0n = md->firstAtomInCell[cell1];
    nCell = md->cellBlocks[blockIdx.x].z;

    nat1 = md->firstAtomInCell[cell1 + nCell - 1] - at0n + md->nAtInCell[cell1 + nCell - 1];
    if (nat1 == 0)    // atoms only in the first cell, escape
        return;
    //nat1 = nat - n0;
    nat = nat1 + n0;

    //energy accumulators per block:
    __shared__ float shVdWEng, shCoulEng;
    //energy accumulators in thread:
    float eVdW = 0.f;
    float eCoul = 0.f;

    //__shared__ int cell0, cell1, nCell, at0, at0n, n0, nat, nat1;
    if (threadIdx.x == 0)
    {
        shVdWEng = 0.f;
        shCoulEng = 0.f;
    }
    __syncthreads();

    extern __shared__ int s[];      // dynamically allocated shared memory
    float3* shifts = (float3*)s;
    //! nthread must be greater than nCell!
    if (threadIdx.x < nCell)
    {
        float3 sh = md->secShifts[blockIdx.x][threadIdx.x];
        j = md->firstAtomInCell[cell1 + threadIdx.x];
        int j0 = md->firstAtomInCell[cell1];
        k = md->nAtInCell[cell1 + threadIdx.x];
        for (i = 0; i < k; i++)
                shifts[j - j0 + i] = sh;
    }
    __syncthreads();

    //INTERACTION WITH DIFFERENT CELLS
    // разделяем число потоков поровну по атомам второй и последующей ячейки, каждый поток обрабатывает взаимодействие "своего" диапазона с атомами первой ячейки
    int atPerThread = ceil((double)nat1 / blockDim.x);
    int id0 = threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, nat1);
    float3 force;
    for (i = id0; i < N; i++)
    {
        force = make_float3(0.f, 0.f, 0.f);
        for (j = 0; j < n0; j++)    // loop by atoms in the first cell
        {
            pair_1(i + at0n, j + at0, md->xyz[i + at0n], &force, md->types[i + at0n], md->xyz[j + at0], &(md->frs[j + at0]), md->types[j + at0], shifts[i], md, eVdW, eCoul);
        }
        atomic_incFloat3(&(md->frs[i + at0n]), force);
    }

    save_coul_vdw(eCoul, eVdW, &shCoulEng, &shVdWEng, md); // to shared then to global memory
}
// end 'cell_list4b_noshared' function


__device__ void distribute_trheads_pairs(int nCell, int* nats, int* nthrds, int* fthrds, int& nFillCell)
// distrubute threads proportional to number of pairs in cell
//   nats[] - number of atoms in cell, nthrds[] - number of threads in cell, fthrds - first id of thread in cell
{
    int i, x;
    int n = 0;
    int mxVal = 0;
    int mxInd;
    int nonzero = 0;        // number of cells with pairs
    // calculate number of pairs in each cell
    for (i = 0; i < nCell; i++)
    {
        nthrds[i] = dev_npair(nats[i]);
        if (nthrds[i])
        {
            nonzero++;
            n += nthrds[i];         // total pairs number
            if (nthrds[i] > mxVal)
            {
                mxVal = nthrds[i];
                mxInd = i;
            }
        }
    }
    nFillCell = nonzero;
    if (nonzero == 0)
        return;

    int rest = blockDim.x - nonzero;    // rest number of threads
    // distribute threads between cells
    int first = 0;  // index of first thread in cell
    for (i = 0; i < nCell; i++)
    {
        fthrds[i] = first;
        if (nthrds[i])  // we have stored number of pairs here (temporary) 
        {
            x = blockDim.x * nthrds[i] / n;
            if (x < 2)      // 0 or 1, in both cases take 1
            {
                nthrds[i] = 1;
                //first++;
            }
            else
                if ((x - 1) > rest) // too much
                {
                    nthrds[i] = rest + 1;  // because 1 thread per non-zero cell we has already reserved
                    //first += (rest + 1);
                    rest = 0;
                }
                else // common case
                {
                    nthrds[i] = x;
                    //first += x;
                    rest -= (x - 1);    // because 1 thread per non-zero cell we has already reserved
                }
            first += nthrds[i];
        }
    }

    // add rest thread to cell with maximal number of pairs:
    if (rest)
    {
        //printf("rest %d id=%d\n", rest, mxInd);
        nthrds[mxInd] += rest;
        for (i = mxInd + 1; i < nCell; i++)
            fthrds[i] += rest;
    }

    // verification
#ifdef DEBUG_MODE
    n = 0;
    first = 0;
    for (i = 0; i < nCell; i++)
    {
        if (first != fthrds[i])
            printf("[%d] wrong first! expected:%d fact:%d (rest=%d)\n", i, first, fthrds[i], rest);

        if (nats[i] < 2) // no pairs
        {
            if (nthrds[i] > 0)
                printf("[%d] to much threads!\n", i);
        }
        else // there are pairs
        {
            if (nthrds[i] == 0)
                printf("[%d] no threads!\n", i);
        }

        n += nthrds[i];
        first = n;
    }
    if (n != blockDim.x)
        printf("threads:%d places:%d\n", n, blockDim.x);
#endif
}


__device__ int internal_thread_id(int n, int* nThreads, int* firstThread, int& cell_id, int& nthread)
// threadIdx.x is used
// если у нас потоки разделены неравномерно по массиву, то данная функция вычисляет внутренний ид потока, кол-во потоков на ячейку массива, и кол-во потоков в ячейке
{
    int i;
    for (i = n - 1; i > 0; i--)
        if (threadIdx.x >= firstThread[i])
        {
            cell_id = i;
            nthread = nThreads[i];
            return threadIdx.x - firstThread[i];
        }

    // if the cycle has been finished, but we do not escape yet, then our cell is [0]
    cell_id = 0;
    nthread = nThreads[0];
    return threadIdx.x - firstThread[0];
}


__global__ void cell_list4a_1(int cellPerBlock, cudaMD* md)
// the same as 4a, but with smart distribution of threads
{

    int i, j;
    int sh1, sh2, step;
    int ex = 0;     // exit flag

    //energy accumulators per block:
    __shared__ float shVdWEng, shCoulEng;
    //energy accumulators in thread:
    float eVdW = 0.f;
    float eCoul = 0.f;

    if (threadIdx.x == 0)
    {
        shVdWEng = 0.f;
        shCoulEng = 0.f;
    }

    // shared data arrays:
    extern __shared__ int shMem[];      // dynamically allocated shared memory
    //! теперь массивы общие для всех ячеек, надо лишь указать диапазоны каждому потоку

    //! имеет ли смысл эти вещи тоже вынести в shared?
    // first cell index and last cell index + 1
    int cell0 = blockIdx.x * cellPerBlock;
    int cellN = min(cell0 + cellPerBlock, md->nCell);
    int nCell = cellN - cell0;
    // first atom index for block
    int at0 = md->firstAtomInCell[cell0];
    // number of atoms in block:
    //int nat = md->firstAtomInCell[cellN] - at0 + 1; // this variant crush for the last block (cellN out of range)
    int nat = md->firstAtomInCell[cellN - 1] - at0 + md->nAtInCell[cellN - 1]; // this variant crush for the last block (cellN out of range)
    int atPerThread = ceil((double)nat / blockDim.x);       // define number of copied atoms per thread
    __syncthreads();


    int* nAts = shMem;                  // number of atoms in a given cell
    int* fAts = (int*)&nAts[nCell];     // first index of atom for a given cell (in internal numeration)
    int* nThrds = (int*)&fAts[nCell];   // number of threads per cell
    int* fThrds = (int*)&nThrds[nCell]; // first index of thread for a cell
    int* types = (int*)&fThrds[nCell];
    float3* xyz = (float3*)&types[nat];
    float3* frs = (float3*)&xyz[nat];

    // copy number of atoms and first index
    //! число потоков должно быть больше числа ячеек
    if (threadIdx.x < nCell)
    {
        nAts[threadIdx.x] = md->nAtInCell[cell0 + threadIdx.x];
        fAts[threadIdx.x] = md->firstAtomInCell[cell0 + threadIdx.x] - at0; // internal index
    }
    __syncthreads();

    // define number of thread for each cell
    __shared__ int nFillCell;
    if (threadIdx.x == 0)
    {
        distribute_trheads_pairs(nCell, nAts, nThrds, fThrds, nFillCell);
    }

    // define atom range for current thread:
    int ia0 = atPerThread * threadIdx.x;
    int Na = min(ia0 + atPerThread, nat);
    // copy atoms to shared memory
    for (i = ia0; i < Na; i++)
    {
        //id = md->cells[iCell][i + 1];
        //ids[sh + i] = id;     // save atom index
        xyz[i] = md->xyz[i + at0];    // copy atom coordinates
        types[i] = md->types[i + at0];
        frs[i] = make_float3(0.f, 0.f, 0.f);    // set zero forces
    }
    __syncthreads();


    //INTERACTION INSIDE SAME CELLS
    //! алгоритм такой: в отличие от пред. раз, где мы весь интервал разбивали на равные отрезки и их обрабатывали нитями
    //! здесь мы раздадим каждой нити начальную точку, а инкеремент будет равен кол-ву нитей

    int iCell, iThread;
    float3 zero_shift = make_float3(0.f, 0.f, 0.f);
    //int iThread = get_internal_id(threadIdx.x, blockDim.x, nCell, iCell, step); // сразу получим и номер потока внутри ячейки и номер ячейки и число потоков на ячейку (step)
    if (nFillCell)
    {
        iThread = internal_thread_id(nCell, nThrds, fThrds, iCell, step);
        int cur_nat = nAts[iCell];

        //if (cur_nat > 1)  теперь эта проверка не нужна, потому что мы автоматически попадаем в непустую ячейку (хотя может быть все непустые?)
        //{
        sh1 = fAts[iCell];
        i = 0;
        j = iThread + 1;     // first pair is 0-1
        while (1/*i < cur_nat*/)
        {
            while (j >= cur_nat)
            {
                i++;
                if (i >= cur_nat - 1)
                {
                    ex = 1;
                    break;
                }
                j = i + 1 + j - cur_nat;
            }
            if (ex) break;
            pair_1(i + sh1 + at0, j + sh1 + at0, xyz[i + sh1], &(frs[i + sh1]), types[i + sh1], xyz[j + sh1], &(frs[j + sh1]), types[j + sh1], zero_shift, md, eVdW, eCoul);
            j = j + step;
        }
        //}
    }


    __syncthreads();

    //INTERACTION WITH DIFFERENT CELLS
    // допущение: потоков все равно >= пар ячеек. Для 8 ячеек максимум 4*7=28 пар, потоков должно быть больше (дефолтно 64)
    int iPair, cell1, cell2;
    if (md->nFirstPairs[blockIdx.x] != 0)  // sometimes block of cells may not have pairs (for example, if only one cell in block)
    {
        iThread = get_internal_id(threadIdx.x, blockDim.x, md->nFirstPairs[blockIdx.x], iPair, step); // сразу получим и номер потока внутри праы и номер пары и число потоков на пару (step)
        // а теперь у нас есть несколько потоков на пару. Каждый поток пусть возьмёт диапазон атомов первыой ячейки пары и все атомы второй ячейки:
        // read cell numbers from pairs and convert into internal number
        cell1 = md->firstPairs[blockIdx.x][iPair].x - cell0;
        cell2 = md->firstPairs[blockIdx.x][iPair].y - cell0;
        sh1 = fAts[cell1];
        sh2 = fAts[cell2];
        atPerThread = ceil((double)nAts[cell1] / step);
        int id0 = iThread * atPerThread;
        int N = min(id0 + atPerThread, nAts[cell1]);

        float3 force;
        for (i = id0; i < N; i++)
        {
            force = make_float3(0.f, 0.f, 0.f);
            for (j = 0; j < nAts[cell2]; j++)
            {
                pair_1(i + sh1 + at0, j + sh2 + at0, xyz[i + sh1], &force, types[i + sh1], xyz[j + sh2], &frs[j + sh2], types[j + sh2], md->firstShifts[blockIdx.x][iPair], md, eVdW, eCoul);
            }
            atomic_incFloat3(&frs[i + sh1], force);
        }
    }
    __syncthreads();

    //! copy forces to global memory, use all threads for whole shared array range
    for (i = ia0; i < Na; i++)
    {
        md->frs[i + at0].x += frs[i].x;
        md->frs[i + at0].y += frs[i].y;
        md->frs[i + at0].z += frs[i].z;
    }

    save_coul_vdw(eCoul, eVdW, &shCoulEng, &shVdWEng, md); // to shared then to global memory
}
// end 'cell_list4a_1' function

__global__ void cell_list4a_noshared(int cellPerBlock, cudaMD* md)
// version without copy data to shared mem
{

    int i, j;
    int sh1, sh2, step;
    int ex = 0;     // exit flag

    //energy accumulators per block:
    __shared__ float shVdWEng, shCoulEng;
    //energy accumulators in thread:
    float eVdW = 0.f;
    float eCoul = 0.f;

    if (threadIdx.x == 0)
    {
        shVdWEng = 0.f;
        shCoulEng = 0.f;
    }
    __syncthreads();


    //! имеет ли смысл эти вещи тоже вынести в shared?
    // first cell index and last cell index + 1
    int cell0 = blockIdx.x * cellPerBlock;
    int cellN = min(cell0 + cellPerBlock, md->nCell);
    int nCell = cellN - cell0;


    //INTERACTION INSIDE SAME CELLS
    //! алгоритм такой: в отличие от пред. раз, где мы весь интервал разбивали на равные отрезки и их обрабатывали нитями
    //! здесь мы раздадим каждой нити начальную точку, а инкеремент будет равен кол-ву нитей

    float3 zero_shift = make_float3(0.f, 0.f, 0.f);
    int iCell;
    int iThread = get_internal_id(threadIdx.x, blockDim.x, nCell, iCell, step); // сразу получим и номер потока внутри ячейки и номер ячейки и число потоков на ячейку (step)
    iCell = iCell + cell0;

    int cur_nat = md->nAtInCell[iCell];

    if (cur_nat > 1)
    {
        sh1 = md->firstAtomInCell[iCell];
        i = 0;
        j = iThread + 1;     // first pair is 0-1
        while (1/*i < cur_nat*/)
        {
            while (j >= cur_nat)
            {
                i++;
                if (i >= cur_nat - 1)
                {
                    ex = 1;
                    break;
                }
                j = i + 1 + j - cur_nat;
            }
            if (ex) break;
            pair_1(i + sh1, j + sh1, md->xyz[i + sh1], &(md->frs[i + sh1]), md->types[i + sh1], md->xyz[j + sh1], &(md->frs[j + sh1]), md->types[j + sh1], zero_shift, md, eVdW, eCoul);
            j = j + step;
        }
    }
    __syncthreads();


    //INTERACTION WITH DIFFERENT CELLS
    // допущение: потоков все равно >= пар ячеек. Для 8 ячеек максимум 4*7=28 пар, потоков должно быть больше (дефолтно 64)
    int iPair, cell1, cell2;
    if (md->nFirstPairs[blockIdx.x] > 0)  // sometimes block of cells may not have pairs (for example, if only one cell in block)
    {
        iThread = get_internal_id(threadIdx.x, blockDim.x, md->nFirstPairs[blockIdx.x], iPair, step); // сразу получим и номер потока внутри праы и номер пары и число потоков на пару (step)
        // а теперь у нас есть несколько потоков на пару. Каждый поток пусть возьмёт диапазон атомов первыой ячейки пары и все атомы второй ячейки:
        cell1 = md->firstPairs[blockIdx.x][iPair].x;
        cell2 = md->firstPairs[blockIdx.x][iPair].y;
        sh1 = md->firstAtomInCell[cell1];
        sh2 = md->firstAtomInCell[cell2];
        int atPerThread = ceil((double)md->nAtInCell[cell1] / step);
        int id0 = iThread * atPerThread;
        int N = min(id0 + atPerThread, md->nAtInCell[cell1]);

        float3 force;
        for (i = id0; i < N; i++)
        {
            force = make_float3(0.f, 0.f, 0.f);
            for (j = 0; j < md->nAtInCell[cell2]; j++)
            {
                pair_1(i + sh1, j + sh2, md->xyz[i + sh1], &(force), md->types[i + sh1], md->xyz[j + sh2], &(md->frs[j + sh2]), md->types[j + sh2], md->firstShifts[blockIdx.x][iPair], md, eVdW, eCoul);
            }
            atomic_incFloat3(&(md->frs[i + sh1]), force);
        }
    }

    save_coul_vdw(eCoul, eVdW, &shCoulEng, &shVdWEng, md); // to shared then to global memory
}
// end 'cell_list4a_noshared' function

__device__ void interaction_by_ind(int i, int j, cudaMD* md, float& engVdw, float& engCoul)
{
    float r, r2, f;
//    float dx, dy, dz;
    float3 dxyz;
    cudaVdW* vdw = NULL;

/*
    dx = md->xyz[i].x - md->xyz[j].x; // + shift
    dy = md->xyz[i].y - md->xyz[j].y;
    dz = md->xyz[i].z - md->xyz[j].z;
*/
    dxyz = float3_dif(md->xyz[i], md->xyz[j]);      // + shift
    r2 = float3_sqr(dxyz);  // dx* dx + dy * dy + dz * dz;

    //if (r2 <= sim->r2Max)    //! cuttoff в ряде случаев можно опустить эту проверку
    //{
    r = 0.f;
    f = 0.f;
    int typeA = md->types[i];
    int typeB = md->types[j];

    // electrostatic contribution
#ifdef TX_CHARGE
    f += md->funcCoul(r2, r, tex2D(qProd, typeA, typeB), engCoul);    //! если частицы заряжены
#else
    f += md->funcCoul(r2, r, md->chProd[typeA][typeB], md, engCoul);    //! если частицы заряжены
#endif

    // van der Waals contribution
    //! иногда можно не рассматривать этот блок, если vdw заведомо вне действия

    vdw = md->vdws[typeA][typeB];
    if (vdw != NULL)
        //if (r2 <= vdw->r2cut)   //! внутри ячейки эту проверку можно опустить
    {
        f += vdw->feng_r(r2, r, vdw, engVdw); //! либо версия только с силами, без энергий
#ifdef DEBUG_MODE
        atomicAdd(&(md->nVdWcall), 1);
#endif
    }
    if (md->use_bnd == 2)   // variable bonds
        try_to_bind(r2, i, j, typeA, typeB, md);

    if (isnan(f))
        printf("bl(%d) in_cell: nan: r2=%f xi=%f xj=%f\n", blockIdx.x, r2, md->xyz[i].x, md->xyz[j].x);

    /*
    md->frs[i].x += f * dx;
    md->frs[i].y += f * dy;
    md->frs[i].z += f * dz;
    */
    inc_float3_coef(&(md->frs[i]), dxyz, f);

#ifdef DEBUG_MODE
    atomicAdd(&(md->nFCall), 1);
#endif
    //    atomicAdd(&(md->sqrCoul), engCoul * engCoul);
}

__device__ void interaction_by_ind_wShift(int i, int j, int shift, int type, cudaMD* md, float& engVdw, float& engCoul)
{
    float r, r2, f;
    float dx, dy, dz;
    cudaVdW* vdw = NULL;
    float3 delta = get_shift(shift, md);

    dx = md->xyz[i].x - md->xyz[j].x - delta.x;
    dy = md->xyz[i].y - md->xyz[j].y - delta.y;
    dz = md->xyz[i].z - md->xyz[j].z - delta.z;
    r2 = dx * dx + dy * dy + dz * dz;

    if (r2 <= md->r2Max)    //! cuttoff в ряде случаев можно опустить эту проверку
    {
        r = 0.f;
        f = 0.f;
        int typeA = md->types[i];
        int typeB = md->types[j];

        // electrostatic contribution
#ifdef TX_CHARGE
        f += md->funcCoul(r2, r, tex2D(qProd, typeA, typeB), engCoul);    //! если частицы заряжены
#else
        f += md->funcCoul(r2, r, md->chProd[typeA][typeB], md, engCoul);    //! если частицы заряжены
#endif

    // van der Waals contribution
    //! иногда можно не рассматривать этот блок, если vdw заведомо вне действия

        vdw = md->vdws[typeA][typeB];
        if (vdw != NULL)
            if (r2 <= vdw->r2cut)   //! внутри ячейки эту проверку можно опустить
            {
                f += vdw->feng_r(r2, r, vdw, engVdw); //! либо версия только с силами, без энергий
#ifdef DEBUG_MODE
                atomicAdd(&(md->nVdWcall), 1);
#endif
            }
        if (md->use_bnd == 2)   // variable bonds
            try_to_bind(r2, i, j, typeA, typeB, md);

        if (isnan(f))
            printf("bl(%d) in_cell: nan: r2=%f (%f %f %f)\n", blockIdx.x, r2, dx, dy, dz);
        if (f > 1e6)
            printf("bl(%d) in_cell: f:%f r2=%f (%f,%f,%f)\n", blockIdx.x, f, r2, dx, dy, dz);
        if (f < -1e6)
            printf("bl(%d) in_cell: f:%f r2=%f (%f,%f,%f)\n", blockIdx.x, f, r2, dx, dy, dz);

        md->frs[i].x += f * dx;
        md->frs[i].y += f * dy;
        md->frs[i].z += f * dz;

#ifdef DEBUG_MODE
        atomicAdd(&(md->nFCall), 1);
#endif
    }
}

__global__ void oneAtom_celllist(int atPerBlock, int atPerThread, int iStep, cudaMD* md)
{
    int i, j;

    //energy accumulators per block:
    __shared__ float shVdWEng, shCoulEng;
    //energy accumulators in thread:
    float eVdW = 0.0f;
    float eCoul = 0.0f;
    if (threadIdx.x == 0)
    {
        shVdWEng = 0.f;
        shCoulEng = 0.f;
    }
    __syncthreads();

    //printf("!!! atPerBlock=%d, atPerThread=%d\n", atPerBlock, atPerThread);
    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);
    //printf("!!! atPerBlock=%d, atPerThread=%d id0=%d N=%d\n", atPerBlock, atPerThread, id0, N);
    int iat, nat, fat;
    int iCell, jCell, nCell, shType, iterType;

    for (iat = id0; iat < N; iat++)
    {
        //iCell = md->cellIndexes[iat];     //! так делать нельзя, потому что этот массив не был отсортирован, в отличие от остальных
        //! поэтому вычислим ячейку заново (хотя можно было отсортировать)
        iCell = floor((double)md->xyz[iat].x * (double)md->cRevSize.x) * md->cnYZ + floor((double)md->xyz[iat].y * (double)md->cRevSize.y) * md->cNumber.z + floor((double)md->xyz[iat].z * (double)md->cRevSize.z);


        // interaction inside the cell
        nat = md->nAtInCell[iCell];
        fat = md->firstAtomInCell[iCell];
        for (i = fat; i < nat + fat; i++)
            if (i != iat)
            {
                interaction_by_ind(iat, i, md, eVdW, eCoul);
            }

        // loop by other cells
        nCell = md->nNeighCell[iCell];
        for (j = 0; j < nCell; j++)
        {
            jCell = md->neighCells[iCell][j].x; // cell index is kept in .x
            shType = md->neighCells[iCell][j].y;
            iterType = md->neighCells[iCell][j].z;
            nat = md->nAtInCell[jCell];
            fat = md->firstAtomInCell[jCell];
            //printf("[%d,%d]iat(%d<%d) = %d jat=%d<%d\n", blockIdx.x, threadIdx.x, id0, N, iat, fat, fat + nat);
            for (i = fat; i < nat + fat; i++)
                //if (iat != i)
                interaction_by_ind_wShift(iat, i, shType, iterType, md, eVdW, eCoul);
            //else
              //  printf("[%d,%d]iat(%d<%d) = %d jat=%d<%d iCell=%d(fa=%d) jCell=%d(fa=%d)\n", blockIdx.x, threadIdx.x, id0, N, iat, fat, fat + nat, iCell, md->firstAtomInCell[iCell], jCell, md->firstAtomInCell[jCell]);
        }
    } // end loop by atoms

    //! unite energy from each thread
    // from threads inside one block to shared memory
    atomicAdd(&shCoulEng, eCoul);
    atomicAdd(&shVdWEng, eVdW);
    __syncthreads();

    // form each block to global memory
    if (threadIdx.x == 0)
    {
        atomicAdd(&md->engCoul1, shCoulEng * 0.5);  // because each iteration is calculated twice
        atomicAdd(&md->engVdW, shVdWEng * 0.5);
    }

}
// end 'oneAtom_celllist()' function

__global__ void cell_list5a(cudaMD* md)
// interactions inside a cell, requires sorting
// не меньше 4х потоков! - теперь нет такого ограничения
{
    int i, j;
    //int ex = 0;     // exit flag

    //energy accumulators per block:
    __shared__ float shVdWEng, shCoulEng;
    //__shared__ int nat, fat;
    //energy accumulators in thread:
    float eVdW = 0.f;
    float eCoul = 0.f;

    if (threadIdx.x == 0)
    {
        shVdWEng = 0.f;
        shCoulEng = 0.f;
    }
    int nat = md->nAtInCell[blockIdx.x];
    int fat = md->firstAtomInCell[blockIdx.x];

    __syncthreads();

    if (nat < 2)        // no pairs
        return;

    // тупой способ обхода, каждый поток получает свой i и перебирает j
    // есть недостаток - потоки загружены не равномерно
    int step = ceil((double)(nat - 1) / blockDim.x);
    int i0 = threadIdx.x * step;
    int maxi = min(i0 + step, nat - 1);
    float3 /*xyz,*/ force;
    //int type;
    float3 zero_shift = make_float3(0.f, 0.f, 0.f);
    for (i = i0; i < maxi; i++)
    {
        force = make_float3(0.f, 0.f, 0.f);
        //xyz = md->xyz[i + fat];
        //type = md->types[i + fat];
        for (j = i + 1; j < nat; j++)
            //pair_in_cell_wInd(i + fat, j + fat, xyz, &force, type, md->xyz[j + fat], &(md->frs[j + fat]), md->types[j + fat], md, eVdW, eCoul);
            pair_1(i + fat, j + fat, md->xyz[i + fat], &force, md->types[i + fat], md->xyz[j + fat], &(md->frs[j + fat]), md->types[j + fat], zero_shift, md, eVdW, eCoul);
        atomic_incFloat3(&(md->frs[i + fat]), force);
    }


    /*
            int step = blockDim.x;
            i = 0;
            j = threadIdx.x + 1;     // first pair is 0-1
            float3 force = make_float3(0.f, 0.f, 0.f);
            while (1)
            {
                while (j >= nat)
                {
                    atomic_incFloat3(&(md->frs[i + fat]), force);
                    force = make_float3(0.f, 0.f, 0.f);
                    i++;
                    if (i >= nat - 1)
                    {
                        ex = 1;
                        break;
                    }
                    j = i + 1 + j - nat;
                }
                if (ex) break;
        #ifdef USE_BINDING
                pair_in_cell_wInd(i + fat, j + fat, md->xyz[i + fat], &force, md->types[i + fat], md->xyz[j + fat], &(md->frs[j + fat]), md->types[j + fat], md, eVdW, eCoul);
        #else
                pair_in_cell(xyz[i + sh1], &(frs[i + sh1]), types[i + sh1], xyz[j + sh1], &(frs[j + sh1]), types[j + sh1], md, eVdW, eCoul);
        #endif
                j = j + step;
            }
    */

    /*
        // новый способ обхода пар
        int np = devNpairs(nat);
        //printf("npair = %d\n", np);
        int step = ceil((double)np / blockDim.x);

        int k, k0, nk;
        k0 = threadIdx.x * step;
        nk = min(k0 + step, np);
        i = 0;
        j = k0 + 1;
        while (j >= nat)
        {
            j = j - nat + i + 2;
            i++;
        }
        float3 force = make_float3(0.f, 0.f, 0.f);   // keep force for i-th atom to decrease the number atomic operations
        float3 xyz = md->xyz[i + fat];
        int type = md->types[i + fat];
        for (k = k0; k < nk; k++)
        {
            pair_in_cell_wInd(i + fat, j + fat, xyz, &force, type, md->xyz[j + fat], &(md->frs[j + fat]), md->types[j + fat], md, eVdW, eCoul);
            j++;
            if (j >= nat)
            {
                // we finish this current value of i, copy force to shared mem and refresh its variable
                atomic_incFloat3(&(md->frs[i + fat]), force);
                i++;
                j = i + 1;
                force = make_float3(0.f, 0.f, 0.f);
                xyz = md->xyz[i + fat];
                type = md->types[i + fat];
            }
        }
    */
    save_coul_vdw(eCoul, eVdW, &shCoulEng, &shVdWEng, md); // to shared then to global memory
}
// end 'cell_list5a' function

__global__ void cell_list5a_shared(cudaMD* md)
// interactions inside the cell
{
    int i, j;
    //int ex = 0;     // exit flag

    //energy accumulators per block:
    __shared__ float shVdWEng, shCoulEng;
    __shared__ int nat, fat;
    //energy accumulators in thread:
    float eVdW = 0.f;
    float eCoul = 0.f;

    if (threadIdx.x == 0)
    {
        shVdWEng = 0.f;
        shCoulEng = 0.f;
        nat = md->nAtInCell[blockIdx.x];
        fat = md->firstAtomInCell[blockIdx.x];
    }
    __syncthreads();

    extern __shared__ int shMem[];      // declaration of dynamically allocated shared memory
    int* types = shMem;
    float3* xyz = (float3*)&types[md->maxAtPerCell];
    float3* frs = (float3*)&xyz[md->maxAtPerCell];

    // copy data to shared memory
    int step0 = ceil((double)nat / blockDim.x);
    int id0 = threadIdx.x * step0;
    int N = min(id0 + step0, nat);
    for (i = id0; i < N; i++)
    {
        xyz[i] = md->xyz[i + fat];
        types[i] = md->types[i + fat];
        frs[i] = make_float3(0.f, 0.f, 0.f);    // set zero forces
    }
    __syncthreads();

    float3 zero_shift = make_float3(0.f, 0.f, 0.f);
    int step = ceil((double)(nat - 1) / blockDim.x);
    int i0 = threadIdx.x * step;
    int maxi = min(i0 + step, nat - 1);
    float3 force;
    //int type;
    for (i = i0; i < maxi; i++)
    {
        force = make_float3(0.f, 0.f, 0.f);
        for (j = i + 1; j < nat; j++)
            pair_1(i + fat, j + fat, xyz[i], &force, types[i], xyz[j], &(frs[j]), types[j], zero_shift, md, eVdW, eCoul);
        atomic_incFloat3(&(frs[i]), force);
    }
    __syncthreads();

    // copy data to global memory
    for (i = id0; i < N; i++)
        inc_float3(&(md->frs[i + fat]), frs[i]);

    save_coul_vdw(eCoul, eVdW, &shCoulEng, &shVdWEng, md); // to shared then to global memory
}

__global__ void cell_list5b(cudaMD* md)
{
    int i, j;

    //energy accumulators per block:
    __shared__ float shVdWEng, shCoulEng;
    //__shared__ int cell1, cell2, nat1, nat2, fat1, fat2;
    //__shared__ float3 shift;
    //energy accumulators in thread:
    float eVdW = 0.f;
    float eCoul = 0.f;

    if (threadIdx.x == 0)
    {
        shVdWEng = 0.f;
        shCoulEng = 0.f;
        /*
        cell1 = md->cellPairs[blockIdx.x].x;
        nat1 = md->nAtInCell[cell1];
        fat1 = md->firstAtomInCell[cell1];
        cell2 = md->cellPairs[blockIdx.x].y;
        nat2 = md->nAtInCell[cell2];
        fat2 = md->firstAtomInCell[cell2];
        shift = md->cellShifts[blockIdx.x];
        */
    }
    __syncthreads();
    int cell1 = md->cellPairs[blockIdx.x].x;
    int nat1 = md->nAtInCell[cell1];
    int fat1 = md->firstAtomInCell[cell1];
    int cell2 = md->cellPairs[blockIdx.x].y;
    int nat2 = md->nAtInCell[cell2];
    int fat2 = md->firstAtomInCell[cell2];
    float3 shift = md->cellShifts[blockIdx.x];

    if (nat1 == 0)
        return;
    if (nat2 == 0)
        return;

    int atPerThread = ceil((double)nat1 / blockDim.x);
    int id0 = threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, nat1);
    float3 /*xyz,*/ force;
    //int type;
    for (i = id0; i < N; i++)
    {
        //xyz = md->xyz[i + fat1];
        //type = md->types[i + fat1];
        force = make_float3(0.f, 0.f, 0.f);
        for (j = 0; j < nat2; j++)
        {
            pair_1(i + fat1, j + fat2, md->xyz[i + fat1], &force, md->types[i + fat1], md->xyz[j + fat2], &(md->frs[j + fat2]), md->types[j + fat2], shift, md, eVdW, eCoul);
            //pair_1(i + fat1, j + fat2, xyz, &force, type, md->xyz[j + fat2], &(md->frs[j + fat2]), md->types[j + fat2], shift, md, eVdW, eCoul);
        }
        atomic_incFloat3(&(md->frs[i + fat1]), force);
    }

    save_coul_vdw(eCoul, eVdW, &shCoulEng, &shVdWEng, md); // to shared then to global memory
}
// end 'cell_list5b' function

void iter_fastCellList(int iStep, Field* fld, cudaMD* dmd, hostManagMD* man)
{
    //int need_to_sync = 0;

    if (1) //!(need_sort) - you can't do in such manner, as cell_list4 and 5 (used furher) based on nAtInCell and firstAtInCell arrays, formed during sorting
    {
        calc_firstAtomInCell << <1, 1 >> > (dmd);
        cudaThreadSynchronize();

        sort_atoms << <man->nAtBlock, man->nAtThread/*man->nMultProc, man->nSingProc*/ >> > (fld->nBdata, fld->nAdata, man->atPerBlock, man->atPerThread, dmd);
        cudaThreadSynchronize();
        sort_dependent << <man->nAtBlock, man->nAtThread >> > (man->atPerBlock, man->atPerThread, dmd);
        if (fld->nBdata)   // insted 1 must be 'write_traj flag'
        {
            sort_bonds << <man->nMultProc, man->nSingProc >> > (man->bndPerBlock, man->bndPerThread, dmd);
            //need_to_sync = 1;
        }
        //else
          //  printf("(%d) no bond\n", iStep);
        if (fld->nAdata)
        {
            //printf("ndata = %d\n", fld->nAdata);
            sort_angles << <man->nMultProc, man->nSingProc >> > (man->angPerBlock, man->angPerThread, dmd);
            //need_to_sync = 1;
        }
        //else
          //  printf("(%d) no angles\n", iStep);
        //if (need_to_sync)
            cudaThreadSynchronize();
        refresh_arrays << <1, 1 >> > (fld->nBdata, fld->nAdata, dmd);
        cudaThreadSynchronize();
    }


    switch (man->bypass_type)
    {
    case 0:
        //init_bypass0(1, elec, fld, man, hmd);
        break;
    case 4:     // fast cell list
        //init_bypass4(9, atm->nAt, elec, fld, hmd, man);
        break;
    case 5:
        //init_bypass5(elec, fld, hmd, man);
        break;
    case 6:
        cell_list5a << < man->pairBlockA, man->pairThreadA >> > (dmd);
        cudaThreadSynchronize();
        cell_list4b_noshared << < man->pairBlockB, man->pairThreadB, man->pairMemB >> > (dmd);
        cudaThreadSynchronize();
        break;
    }
}