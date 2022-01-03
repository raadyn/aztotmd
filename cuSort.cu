#include "cuStruct.h"
#include "cuSort.h"
#include "utils.h"
#include "cuUtils.h"
//#include "cuTemp.h"

void alloc_sort(int nAt, int nCell, cudaMD* hmd)
{
    int flsize = sizeof(float) * nAt;
    int xyzsize = sizeof(float3) * nAt;
    int intsize = int_size * nAt;

    cudaMalloc((void**)&hmd->sort_xyz, xyzsize);
    cudaMalloc((void**)&hmd->sort_vls, xyzsize);
    cudaMalloc((void**)&hmd->sort_frs, xyzsize);        //! не всегда, возможно, что только если есть связи (т.е. есть что-то, что определяет силы перед сортировкой)
    cudaMalloc((void**)&hmd->sort_types, intsize);
    cudaMalloc((void**)&hmd->sort_ind, intsize);
    cudaMalloc((void**)&hmd->sort_parents, intsize);    //! не всегда
    cudaMalloc((void**)&hmd->sort_nbonds, intsize);     //! не всегда
    cudaMalloc((void**)&hmd->sort_nangles, intsize);    //! не всегда
    cudaMalloc((void**)&hmd->sort_oldTypes, intsize);   //! не всегда
    cudaMalloc((void**)&hmd->cellIndexes, intsize);     //? может это относится к cell list, а не к sort?
    cudaMalloc((void**)&hmd->insideCellIndex, intsize); //? может это относится к cell list, а не к sort?
    cudaMalloc((void**)&hmd->sort_masses, flsize);
    cudaMalloc((void**)&hmd->sort_rMasshdT, flsize);
    cudaMalloc((void**)(&hmd->sort_engs), flsize);          // for radiative thermostat only
    cudaMalloc((void**)(&hmd->sort_radii), flsize);         // for radiative thermostat only    //! not ONLY! if we used T-dependent pair pot we also need this array, even without radiative thermostat
    cudaMalloc((void**)(&hmd->sort_radstep), intsize);    // for radiative thermostat only

    // for trajectories output
    int* arr = (int*)malloc(intsize);
    //cudaMalloc((void**)(&hmd->sort_trajs), intsize);        
    int i;
    for (i = 0; i < nAt; i++)
        arr[i] = i;
    data_to_device((void**)(&hmd->sort_trajs), arr, intsize);

    //? может это относится к cell list, а не к sort?
    intsize = int_size * nCell;
    cudaMalloc((void**)&hmd->firstAtomInCell, intsize);
    cudaMalloc((void**)&hmd->nAtInCell, intsize);
}

void free_sort(cudaMD* hmd)
{
    cudaFree(hmd->sort_xyz);
    cudaFree(hmd->sort_vls);
    cudaFree(hmd->sort_frs);        //! не всегда, возможно, что только если есть связи (т.е. есть что-то, что определяет силы перед сортировкой)
    cudaFree(hmd->sort_types);
    cudaFree(hmd->sort_ind);
    cudaFree(hmd->sort_parents);    //! не всегда
    cudaFree(hmd->sort_nbonds);     //! не всегда
    cudaFree(hmd->sort_nangles);    //! не всегда    
    cudaFree(hmd->sort_oldTypes);   //! не всегда
    cudaFree(hmd->cellIndexes);     //? может это относится к cell list, а не к sort?
    cudaFree(hmd->insideCellIndex); //? может это относится к cell list, а не к sort?
    cudaFree(hmd->sort_masses);
    cudaFree(hmd->sort_rMasshdT);
    cudaFree(hmd->firstAtomInCell); //? может это относится к cell list, а не к sort?
    cudaFree(hmd->nAtInCell);       //? может это относится к cell list, а не к sort?
    cudaFree(hmd->sort_engs);    // for radiative thermostat only
    cudaFree(hmd->sort_radii);    // for radiative thermostat only
    cudaFree(hmd->sort_radstep);    // for radiative thermostat only
    cudaFree(hmd->sort_trajs);        // for trajectories output
}

__device__ void switch_pointers(void** p1, void** p2)
{
    void* ptr = *p1;
    *p1 = *p2;
    *p2 = ptr;
}

__global__ void refresh_arrays(int use_bnd, int use_ang, cudaMD* md)
{
    //printf("beging refresh arrays\n");

    switch_pointers((void**)&(md->xyz), (void**)&(md->sort_xyz));
    switch_pointers((void**)&(md->vls), (void**)&(md->sort_vls));
    switch_pointers((void**)&(md->frs), (void**)&(md->sort_frs));   //! не всегда, возможно, что только если есть связи (т.е. есть что-то, что определяет силы перед сортировкой)
    switch_pointers((void**)&(md->types), (void**)&(md->sort_types));
    if (use_bnd)
    {
        //printf("switch_bnd\n");
        switch_pointers((void**)&(md->parents), (void**)&(md->sort_parents));
        switch_pointers((void**)&(md->nbonds), (void**)&(md->sort_nbonds));
        switch_pointers((void**)&(md->oldTypes), (void**)&(md->sort_oldTypes));
    }
    if (use_ang)
    {
        switch_pointers((void**)&(md->nangles), (void**)&(md->sort_nangles));
    }
    switch_pointers((void**)&(md->masses), (void**)&(md->sort_masses));
    switch_pointers((void**)&(md->rMasshdT), (void**)&(md->sort_rMasshdT));
    switch_pointers((void**)&(md->engs), (void**)&(md->sort_engs));    // for radiative thermostat only
    switch_pointers((void**)&(md->radii), (void**)&(md->sort_radii));    // for radiative thermostat only //! not ONLY! for T-dependent FF also
    switch_pointers((void**)&(md->radstep), (void**)&(md->sort_radstep));    // for radiative thermostat only

    //printf("end refresh arrays\n");
}

/* уже есть в другом модуле
__global__ void clear_list(int cellPerBlock, int cellPerThread, cudaMD* md)
{
    int i;
    int id0 = blockIdx.x * cellPerBlock + threadIdx.x * cellPerThread;
    int N = min(id0 + cellPerThread, md->nCell);

    for (i = id0; i < N; i++)
        md->nAtInCell[i] = 0;
}
*/

__device__ void count_cell(int index, float3 xyz, cudaMD* md)
// save atom index with coordinates xyz in the cell list
{
    int c, j;

    c = floor((double)xyz.x * (double)md->cRevSize.x) * md->cnYZ + floor((double)xyz.y * (double)md->cRevSize.y) * md->cNumber.z + floor((double)xyz.z * (double)md->cRevSize.z);
    if (c >= md->nCell)
        printf("count cell: xyz=(%f; %f; %f)revsizes:[%f %f %f] c = %d\n", xyz.x, xyz.y, xyz.z, md->cRevSize.x, md->cRevSize.y, md->cRevSize.z, c);
    if (c < 0)
        printf("count cell: xyz=(%f; %f; %f) c = %d\n", xyz.x, xyz.y, xyz.z, c);

    md->cellIndexes[index] = c;
    j = atomicAdd(&(md->nAtInCell[c]), 1);    // increase the number of particles in cell[c] (it keeps in the 0th element of cell[cell_index] array)
    md->insideCellIndex[index] = j;
}

__global__ void calc_firstAtomInCell(cudaMD* md)
// define first index of atom in ordered array corresponding to each cell
//! этот код по сути серийный
{
    int i;
    int cnt = 0;
    //printf("start calc first atom\n");
    for (i = 0; i < md->nCell; i++)
    {
        md->firstAtomInCell[i] = cnt;
        cnt += md->nAtInCell[i];
    }
    //printf("end calc first atom\n");
}

__global__ void sort_atoms(int use_bnd, int use_ang, int atPerBlock, int atPerThread, cudaMD* md)
// sort atoms according to the cells belonging
{
    //printf("BEGIN SORT ATOMS(%d, %d)\n", blockIdx.x, threadIdx.x);
    int i, j;
    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);
    for (i = id0; i < N; i++)
    {
        //if (i == 0)
          //  printf("begin sort atoms\n");

        // define new index:
        j = md->firstAtomInCell[md->cellIndexes[i]] + md->insideCellIndex[i];
        // copy data to ordered arrays:
        md->sort_xyz[j] = md->xyz[i];
        md->sort_vls[j] = md->vls[i];
        md->sort_frs[j] = md->frs[i];   //! не всегда, возможно, что только если есть связи (т.е. есть что-то, что определяет силы перед сортировкой)
        md->sort_types[j] = md->types[i];
        md->sort_masses[j] = md->masses[i];
        md->sort_rMasshdT[j] = md->rMasshdT[i];
        if (md->tstat == 2) // radiative thermostat /! тут должна быть константа ctTermRadi
        {
            md->sort_engs[j] = md->engs[i];    // for radiative thermostat only
            md->sort_radstep[j] = md->radstep[i];    // for radiative thermostat only
        }
        if (md->tstat == 2 || md->tdep_force)
            md->sort_radii[j] = md->radii[i];    //! for radiative thermostat and T-dependent force field
    //printf("SORT ATOMS bef use_bnd(%d, %d)\n", blockIdx.x, threadIdx.x);
        if (use_bnd)
        {
            // сортировку родитиелей нужно делать в два действия, сначала перемещаем значение родителя на новое место
            // а поскольку значение родителя все ещё в старой терминологии, отдельным кернелом пересчитываем его
            md->sort_parents[j] = md->parents[i];
            md->sort_nbonds[j] = md->nbonds[i];
        }
        //printf("SORT ATOMS bef use_ang(%d, %d)\n", blockIdx.x, threadIdx.x);
        if (use_ang)
        {
            md->sort_nangles[j] = md->nangles[i];
        }
        //printf("SORT ATOMS bef use_ang || use_bnd(%d, %d)\n", blockIdx.x, threadIdx.x);
        //! или если используется вывод траекторий!
        md->sort_ind[i] = j;
        if (use_bnd || use_ang)
        {
            md->sort_oldTypes[j] = md->oldTypes[i];
            //md->sort_trajs[j] = i;
        }
        //printf("SORT ATOMS aft use_ang || use_bnd(%d, %d)\n", blockIdx.x, threadIdx.x);

        //if (i == 0)
          //  printf("end sort atoms\n");
    }
}

__global__ void sort_dependent(int atPerBlock, int atPerThread, cudaMD* md)
// sort data which dependent on sorting index and can be arranged only after defined of sort_ind array
{
    int i;// , j;
    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);
    for (i = id0; i < N; i++)
    {
        if (md->use_bnd)
            //md->sort_parents[i] = md->sort_ind[md->sort_parents[i]]; //! I don't know how it was obtained and how does it work, the next line is correct:
            md->sort_parents[md->sort_ind[i]] = md->sort_ind[md->parents[i]];
        md->sort_trajs[i] = md->sort_ind[md->sort_trajs[i]];
    }
}


__global__ void sort_bonds(int bndPerBlock, int bndPerThread, cudaMD* md)
{
    int id0 = blockIdx.x * bndPerBlock + threadIdx.x * bndPerThread;
    int N = min(id0 + bndPerThread, md->nBond);
    int iBnd;
    for (iBnd = id0; iBnd < N; iBnd++)
    {
        md->bonds[iBnd].x = md->sort_ind[md->bonds[iBnd].x];
        md->bonds[iBnd].y = md->sort_ind[md->bonds[iBnd].y];
    }
}

__global__ void sort_angles(int angPerBlock, int angPerThread, cudaMD* md)
{
    int i;
    int id0 = blockIdx.x * angPerBlock + threadIdx.x * angPerThread;
    int N = min(id0 + angPerThread, md->nAngle);
    for (i = id0; i < N; i++)
    {
        md->angles[i].x = md->sort_ind[md->angles[i].x];
        md->angles[i].y = md->sort_ind[md->angles[i].y];
        md->angles[i].z = md->sort_ind[md->angles[i].z];
    }
}