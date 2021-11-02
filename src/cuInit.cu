// unit for preparation data for CUDA
//#include <math.h>

#include "defines.h"
#include "cuStruct.h"
#include "dataStruct.h" 
#include "sys_init.h"
#include "vdw.h"
#include "cuVdW.h"
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

int code_shift(int x, int y, int z)
// convert shifts for each direction if format -1 / 0 / 1 to one integer number
{
    return (x + 1) + 3 * (y + 1) + 9 * (z + 1);
}

int cell_dist(int xi, int xj, int mx, float length, float csize, float rskip, float& rmin, float& rmax, float& shift, int &int_shift)
// âû÷èñëÿåì ìèíèìàëüíîå è ìàêñèìàëüíîå ðàññòîÿíèå (â êîë-âå ÿ÷ååê) ìåæäó ÷àñòèöàìè èç 2õ ðàçíûõ ÿ÷ååê, à òàêæå ñäâèã, èñïîëüçóåìûé ïðè ó÷åòå ïåðèîäè÷åñêèõ óñëîâèé
// xi, xj - êîîðäèíàòû 2õ ÿ÷ååê ïî 1ìó èçìåðåíèþ, mx - ìàêñèìàëüíîå êîë-âî ÿ÷ååê â ýòîì èçìåðåíèè, length - äëèíà áîêñà â ýòîì èçìåðåíèè csize - ðàçìåð ÿ÷åéêè
// âîçâðàùàåì 1 åñëè ÿ÷åéêè ñëèøêîì äàëåêî (áîëüøå rskip) èíà÷å 0
{

    shift = 0.0;
    int_shift = 0;
    int delt = abs(xi - xj);
    if (delt != 0)
    {
        if (delt > (double)(mx / 2))
        {
            delt = mx - delt;
            if (xi > xj)
                //shift = length;
                int_shift = 1;
            else
                //shift = -length;
                int_shift = -1;

            shift = length * int_shift;
        }
        rmin = (delt - 1) * csize - 0.0 * csize; //! -1.0 csize - temporary addtion!
        //! temp!
        if (rmin < 0)
            rmin = 0.0f;
        rmax = (delt + 1) * csize;
        if (rmin > rskip)
            return 1;
        else
            return 0;
    }
    else
    {
        rmin = 0.0;
        rmax = csize;
        return 0;
    }

}

int cell_dist_int(int xi, int xj, int mx, float length, float csize, float rskip, float& rmin, float& rmax, int& shift)
// version with integer shift (-1 / 0 / 1)
{

    shift = 0;
    int delt = abs(xi - xj);
    if (delt != 0)
    {
        if (delt > (double)(mx / 2))
        {
            delt = mx - delt;
            if (xi > xj)
                shift = 1;
            else
                shift = -1;
        }
        rmin = (delt - 1) * csize - 0.0 * csize; //! -1.0 csize - temporary addtion!
        //! temp!
        if (rmin < 0)
            rmin = 0.0f;
        rmax = (delt + 1) * csize;
        if (rmin > rskip)
            return 1;
        else
            return 0;
    }
    else
    {
        rmin = 0.0;
        rmax = csize;
        return 0;
    }

}

void bonds_to_device(Atoms *atm, Field *fld, Sim *sim, cudaMD *hmd, hostManagMD *man)
// copy bonds data to device, hmd - host exemplar of cudaMD struct
{
    int i, j;
    int nsize = atm->nAt * int_size;

    hmd->mxBond = fld->mxBonds;
    hmd->nBond = fld->nBonds;

    man->bndPerBlock = ceil((double)fld->mxBonds / (double)man->nMultProc);
    man->bndPerThread = ceil((double)man->bndPerBlock / (double)man->nSingProc);
    if (man->bndPerBlock < (man->bndPerThread * man->nSingProc))
        man->bndPerBlock = man->bndPerThread * man->nSingProc;    // but not less 


    cudaBond* bndTypes = (cudaBond*)malloc(fld->nBdata * sizeof(cudaBond));
    hmd->nBndTypes = fld->nBdata;
    for (i = 1; i < fld->nBdata; i++)   // i = 0 reserved for empty(deleted) bond
    {
        bndTypes[i].type = fld->bdata[i].type;
        bndTypes[i].spec1 = fld->bdata[i].spec1;
        bndTypes[i].spec2 = fld->bdata[i].spec2;
        bndTypes[i].mxEx = fld->bdata[i].mxEx;
        bndTypes[i].mnEx = fld->bdata[i].mnEx;
        bndTypes[i].new_type[0] = fld->bdata[i].new_type[0];
        bndTypes[i].new_type[1] = fld->bdata[i].new_type[1];

        if (bndTypes[i].mnEx)
        {
            j = bndTypes[i].new_type[0];
            if (j < 0)  // invert species
            {
                bndTypes[i].new_spec1[0] = fld->bdata[-j].spec2;
                bndTypes[i].new_spec2[0] = fld->bdata[-j].spec1;
            }
            else
            {
                bndTypes[i].new_spec1[0] = fld->bdata[j].spec1;
                bndTypes[i].new_spec2[0] = fld->bdata[j].spec2;
            }
        }

        if ((bndTypes[i].mxEx) && (bndTypes[i].new_type[1] != 0))   // íå óäàëÿåì
        {
            j = bndTypes[i].new_type[1];
            if (j < 0)  // invert species
            {
                bndTypes[i].new_spec1[1] = fld->bdata[-j].spec2;
                bndTypes[i].new_spec2[1] = fld->bdata[-j].spec1;
            }
            else
            {
                bndTypes[i].new_spec1[1] = fld->bdata[j].spec1;
                bndTypes[i].new_spec2[1] = fld->bdata[j].spec2;
            }
        }


        bndTypes[i].new_spec1[0] = fld->bdata[i].new_spec1[0];
        bndTypes[i].new_spec1[1] = fld->bdata[i].new_spec1[1];
        bndTypes[i].new_spec2[0] = fld->bdata[i].new_spec2[0];
        bndTypes[i].new_spec2[1] = fld->bdata[i].new_spec2[1];
        bndTypes[i].hatom = fld->bdata[i].hatom;
        bndTypes[i].evol = fld->bdata[i].evol;

        //! ïåðåîïðåäåëèòü spec2 äëÿ ìèí è ìàêñ, åñëè ìàêñ íå óäàëèòü

        bndTypes[i].p0 = (float)fld->bdata[i].p0;
        bndTypes[i].p1 = (float)fld->bdata[i].p1;
        bndTypes[i].p2 = (float)fld->bdata[i].p2;
        bndTypes[i].p3 = (float)fld->bdata[i].p3;
        bndTypes[i].p4 = (float)fld->bdata[i].p4;
        bndTypes[i].r2min = (float)fld->bdata[i].r2min;
        bndTypes[i].r2max = (float)fld->bdata[i].r2max;
        bndTypes[i].count = fld->bdata[i].number;
        bndTypes[i].rSumm = 0.0f;
        bndTypes[i].rCount = 0;
        bndTypes[i].ltSumm = 0;
        bndTypes[i].ltCount = 0;
    }
    data_to_device((void**)&(hmd->bondTypes), bndTypes, fld->nBdata * sizeof(cudaBond));
    free(bndTypes);

    int4* bnds = (int4*)malloc(fld->nBonds * int4_size);
    for (i = 0; i < fld->nBonds; i++)
    {
        bnds[i] = make_int4(fld->at1[i], fld->at2[i], fld->bTypes[i], 0);
    }
    cudaMalloc((void**)&(hmd->bonds), fld->mxBonds * int4_size);
    cudaMemcpy(hmd->bonds, bnds, fld->nBonds * int4_size, cudaMemcpyHostToDevice);
    free(bnds);

    data_to_device((void**)&(hmd->nbonds), atm->nBonds, nsize);

    //int* int_array;
    int** int_int_array;
    float** fl_fl_array;
    int_int_array = (int**)malloc(fld->nSpec * pointer_size);
    for (i = 0; i < fld->nSpec; i++)
    {
        data_to_device((void**)&(int_int_array[i]), fld->bond_matrix[i], fld->nSpec * int_size);
    }
    data_to_device((void**)&(hmd->def_bonds), int_int_array, fld->nSpec * pointer_size);
    free(int_int_array);

    // parents are -1?
    /*
    for (i = 0; i < atm->nAt; i++)
        if (atm->parents[i] != -1)
            printf("parents[%d]=%d\n", i, atm->parents[i]);

    for (i = 0; i < atm->nAt; i++)
        if (atm->parents[i] == -1)
            printf("parents[%d]=-1\n", i);
     */

    data_to_device((void**)&(hmd->parents), atm->parents, nsize);

    if (sim->use_bnd == 2)  // binding
    {
        cudaMalloc((void**)&(hmd->neighToBind), nsize);
        cudaMalloc((void**)&(hmd->canBind), nsize);
        cudaMalloc((void**)&(hmd->r2Min), nsize);


        int_int_array = (int**)malloc(fld->nSpec * pointer_size);
        fl_fl_array = (float**)malloc(fld->nSpec * pointer_size);
        float* fl_array = (float*)malloc(fld->nSpec * sizeof(float));
        for (i = 0; i < fld->nSpec; i++)
        {
            data_to_device((void**)&(int_int_array[i]), fld->bonding_matr[i], fld->nSpec * int_size);
            for (j = 0; j < fld->nSpec; j++)
                fl_array[j] = (float)fld->bindR2matrix[i][j];
            data_to_device((void**)&(fl_fl_array[i]), fl_array, fld->nSpec * float_size);
        }
        data_to_device((void**)&(hmd->bindBonds), int_int_array, fld->nSpec * pointer_size);
        data_to_device((void**)&(hmd->bindR2), fl_fl_array, fld->nSpec * pointer_size);

        free(int_int_array);
        free(fl_fl_array);
        free(fl_array);
    }
}

__global__ void save_ptrs(int **ptr_arr, cudaMD *dmd)
// dmd - pointer device exemplar of cudaMD struct
{
    ptr_arr[0] = &(dmd->nAt);
    ptr_arr[1] = &(dmd->nBond);
    ptr_arr[2] = &(dmd->nAngle);
}

int save_cellpair(int3* cells, int ci, int cj, int4* pairs, float3* shifts, float rmax, float mxr2vdw, cudaMD* md, Sim* sim, int& index)
// save cell pairs ci-cj in pairs array and increase current index
// return 1 if out of range otherwise return 0
{
    float dxmin, dxmax, dymin, dymax, dzmin, dzmax, dr2max;
    float3 shift = make_float3(0.0f, 0.0f, 0.0f);
    int shx, shy, shz;  // periodic shifts
    //float rmax2 = rmax * rmax;

    // x dimension:
    if (cell_dist(cells[ci].x, cells[cj].x, md->cNumber.x, md->leng.x, md->cSize.x, rmax, dxmin, dxmax, shift.x, shx))
        return 1;

    // y dimension:
    if (cell_dist(cells[ci].y, cells[cj].y, md->cNumber.y, md->leng.y, md->cSize.y, rmax, dymin, dymax, shift.y, shy))
        return 1;

    // z dimension:
    if (cell_dist(cells[ci].z, cells[cj].z, md->cNumber.z, md->leng.z, md->cSize.z, rmax, dzmin, dzmax, shift.z, shz))
        return 1;


    float dr2min = dxmin * dxmin + dymin * dymin + dzmin * dzmin;
    if (dr2min > sim->r2Max)
        return 1;

    //âñå íåâçàèìîäåéñòâóþùèå ÿ÷åéêè óæå îòáðîøåíû, èä¸ì äàëåå

    int coul, vdw;
    dr2max = dxmax * dxmax + dymax * dymax + dzmax * dzmax;
    if (dr2max < sim->r2Elec)
        coul = 1;   // ãàðàíòèðîâàíî äîòÿãèâàåòñÿ Êóëîíîâñêîå âçàèìîäåéñòâèå
    else
        coul = 0;   // ìîæåò äîòÿãèâàåòñÿ, à ìîæåò è íåò

    if (dr2min > mxr2vdw)
        vdw = 0;    // ãàðàíòèðîâàíî íå äîñòàåò ÂäÂ
    else
        vdw = 1;    // ìîæåò è äîñòàåò

    pairs[index].x = ci;
    pairs[index].y = cj;
    pairs[index].z = coul * 2 + vdw;
    pairs[index].w = code_shift(shx, shy, shz);
    shifts[index] = shift;
    index++;
    return 0;
}

int is_cellneighbors(int x0, int y0, int z0, int x, int y, int z, float rmax, float mxr2vdw, cudaMD* md, Sim* sim, int& shift_type, int& inter_type)
// define that cells[x0;y0;z0] and [x;y;z] are neighbors (return 1 if so), save shift type and interaction type
{
    float dxmin, dxmax, dymin, dymax, dzmin, dzmax, dr2max;
    int shx, shy, shz;
    //float rmax2 = rmax * rmax;

    // x dimension:
    if (cell_dist_int(x0, x, md->cNumber.x, md->leng.x, md->cSize.x, rmax, dxmin, dxmax, shx))
        return 0;

    // y dimension:
    if (cell_dist_int(y0, y, md->cNumber.y, md->leng.y, md->cSize.y, rmax, dymin, dymax, shy))
        return 0;

    // z dimension:
    if (cell_dist_int(z0, z, md->cNumber.z, md->leng.z, md->cSize.z, rmax, dzmin, dzmax, shz))
        return 0;

    float dr2min = dxmin * dxmin + dymin * dymin + dzmin * dzmin;
    if (dr2min > sim->r2Max)
        return 0;

    //âñå íåâçàèìîäåéñòâóþùèå ÿ÷åéêè óæå îòáðîøåíû, èä¸ì äàëåå

    int coul, vdw;
    dr2max = dxmax * dxmax + dymax * dymax + dzmax * dzmax;
    if (dr2max < sim->r2Elec)
        coul = 1;   // ãàðàíòèðîâàíî äîòÿãèâàåòñÿ Êóëîíîâñêîå âçàèìîäåéñòâèå
    else
        coul = 0;   // ìîæåò äîòÿãèâàåòñÿ, à ìîæåò è íåò

    if (dr2min > mxr2vdw)
        vdw = 0;    // ãàðàíòèðîâàíî íå äîñòàåò ÂäÂ
    else
        vdw = 1;    // ìîæåò è äîñòàåò

    inter_type = coul * 2 + vdw;
    shift_type = code_shift(shx, shy, shz);
    return 1;
}

int pair_exists(int3* cells, int ci, int cj, float3& shift, float rmax, float mxr2vdw, cudaMD* md, Sim* sim)
// if pair in range return 1, otherwise - 0
{
    float dxmin, dxmax, dymin, dymax, dzmin, dzmax;
    int shx, shy, shz;
    //float dr2max;
    //float rmax2 = rmax * rmax;

    // x dimension:
    if (cell_dist(cells[ci].x, cells[cj].x, md->cNumber.x, md->leng.x, md->cSize.x, rmax, dxmin, dxmax, shift.x, shx))
        return 0;

    // y dimension:
    if (cell_dist(cells[ci].y, cells[cj].y, md->cNumber.y, md->leng.y, md->cSize.y, rmax, dymin, dymax, shift.y, shy))
        return 0;

    // z dimension:
    if (cell_dist(cells[ci].z, cells[cj].z, md->cNumber.z, md->leng.z, md->cSize.z, rmax, dzmin, dzmax, shift.z, shz))
        return 0;


    float dr2min = dxmin * dxmin + dymin * dymin + dzmin * dzmin;
    if (dr2min > sim->r2Max)
        return 0;

    //int coul, vdw;
    //dr2max = dxmax * dxmax + dymax * dymax + dzmax * dzmax;
    /*
    if (dr2max < sim->r2Elec)
        coul = 1;   // ãàðàíòèðîâàíî äîòÿãèâàåòñÿ Êóëîíîâñêîå âçàèìîäåéñòâèå
    else
        coul = 0;   // ìîæåò äîòÿãèâàåòñÿ, à ìîæåò è íåò

    if (dr2min > mxr2vdw)
        vdw = 0;    // ãàðàíòèðîâàíî íå äîñòàåò ÂäÂ
    else
        vdw = 1;    // ìîæåò è äîñòàåò

    pairs[index].x = ci;
    pairs[index].y = cj;
    pairs[index].z = coul * 2 + vdw;
    shifts[index] = shift;
    index++;
    */
    return 1;
}

int ncell(float minR, int add_to_even, cudaMD* hmd)
// calculate cell size and count and return number of cells
{
    hmd->cNumber = make_int3(ceil(hmd->leng.x / minR), ceil(hmd->leng.y / minR), ceil(hmd->leng.z / minR));
    //! òóò ïîëó÷àåòñÿ, ÷òî ÿ÷åéêè íå îáÿçàòåëüíî êóáè÷åñêîé ôîðìû. Íàäî ïîäóìàòü, êðèòè÷íî ýòî èëè íåò:
    hmd->cSize = make_float3(hmd->leng.x / hmd->cNumber.x, hmd->leng.y / hmd->cNumber.y, hmd->leng.z / hmd->cNumber.z);
    hmd->cRevSize = make_float3(hmd->cNumber.x / hmd->leng.x, hmd->cNumber.y / hmd->leng.y, hmd->cNumber.z / hmd->leng.z);
    hmd->cnYZ = hmd->cNumber.y * hmd->cNumber.z;
    //printf("minimall cell size: %f  md->cSize.x=%f\n", minR, hmd->cSize.x);
    //printf("cRev=(%f %f %f)\n", hmd->cRevSize.x, hmd->cRevSize.y, hmd->cRevSize.z);

    hmd->nCell = hmd->cNumber.x * hmd->cNumber.y * hmd->cNumber.z;
    if (add_to_even)
        if ((hmd->nCell % 2) != 0)
            hmd->nCell++;

    return hmd->nCell;
}

void init_cellList(float minRad, float maxR2, int nAt, Elec *elec, Sim *sim, cudaMD *hmd, hostManagMD *man)
//cell list
//! ùàñ ÿ ïðîáóþ ðàçáèåíèå, êîãäà äèàãîíàëü ÿ÷åéêè íå ïðåâûøàåò ìèíèìàëüíûé êóòîôô (ò.å. ÷àñòèöû â îäíîé ÿ÷åéêå ãàðàíòèðîâàííî â ðàäèóñå äåéñòâèÿ äðóã äðóãà)
{
    int i, j, k;
    float minR = minRad / (float)sqrt(3);

    int nCell = ncell(minR, 1, hmd);    // äëÿ îáùíîñòè àëãîðèòìà äîâåäåì ÷èñëî ÿ÷ååê äî ÷åòíîãî

    hmd->maxAtPerCell = nAt / nCell * 3 + 8;          // * 3 + 4 for excess
    hmd->maxAtPerBlock = 2 * 16 * (nAt / nCell) * 3 + 90;  // factor 3 for excess

    int3* cxyz = (int3*)malloc(nCell * sizeof(int3));
    int l = 0;
    for (i = 0; i < hmd->cNumber.x; i++)
        for (j = 0; j < hmd->cNumber.y; j++)
            for (k = 0; k < hmd->cNumber.z; k++)
            {
                cxyz[l] = make_int3(i, j, k);
                l++;
            }
    //! äîáàâèòü â îáðàáîòêó èñêóñòâåííóþ ÿ÷åéêó, åñëè èõ ÷èñëî íå÷åòíîå


    //! ïåðâûå ïàðû (0-1, 2-3, 4-5 è ò.ä.). ÎÍÈ ÍÅ ÏÅÐÅÑÅÊÀÞÒÑß ÏÎ ÄÀÍÍÛÌ
    hmd->nPair1 = nCell / 2;
    int nPair = nCell * (nCell - 1) / 2;
    int4* pairs1 = (int4*)malloc(nPair * sizeof(int4));
    float3* shifts1 = (float3*)malloc(nPair * sizeof(float3));

    k = 0;
    // pairs 0-1, 2-3, 4-5 and etc
    for (i = 0; i < hmd->nPair1; i++)
    {
        //if (save_cellpair(cxyz, i*2, i*2+1, pairs1, shifts1, rmax, maxR2, hmd, sim, k))
          //  continue;
        save_cellpair(cxyz, i * 2, i * 2 + 1, pairs1, shifts1, sim->rMax, sim->r2Max, hmd, sim, k);
    }
    if (k != hmd->nPair1)
    {
        printf("k=%d nPair1=%d", k, hmd->nPair1);
        hmd->nPair1 = k;
    }
    // rest pairs
    for (i = 0; i < nCell - 1; i++)
    {
        l = 2 - (i % 2); // ó÷èòûâàåì, ÷òî 0-1, 2-3, 4-5 ïàðû ìû óæå îòîáðàëè
        for (j = i + l; j < nCell; j++)
        {
            //if (save_cellpair(cxyz, i, j, pairs1, shifts1, rmax, maxR2, hmd, sim, k))
              //  continue;
            save_cellpair(cxyz, i, j, pairs1, shifts1, sim->rMax, sim->r2Max, hmd, sim, k);
        }
    }
    hmd->nPair = k;

    //for (i = 256; i < 266; i++)
      //  printf("cell[%d]=(%d, %d, %d)\n", i, cxyz[i].x, cxyz[i].y, cxyz[i].z);

    //pair verifiyng:
    for (i = 0; i < hmd->nPair; i++)
    {
        if (pairs1[i].x >= pairs1[i].y)
        {
            printf("pair %d: %d-%d\n", i, pairs1[i].x, pairs1[i].y);
        }
        /*
        for (j = i + 1; j < hmd->nPair; j++)
            if (pairs1[i].x == pairs1[j].x)
                if (pairs1[i].y == pairs1[j].y)
                {
                    printf("pairs %d and %d are the same!\n", i, j);
                }
        */
    }


    /*
    j = 0;
    k = 1;
    int c1, c2;
    for (i = 0; i < hmd->nPair; i++)
        if ((pairs1[i].x == k) || (pairs1[i].y == k))
        {
            c1 = pairs1[i].x;   // cell 1
            c2 = pairs1[i].y;   // cell 2
            //printf("pair %d: %d[%d %d %d]-%d[%d %d %d] (type=%d). shift (%f %f %f)\n", i, c1, cxyz[c1].x, cxyz[c1].y, cxyz[c1].z, c2, cxyz[c2].x, cxyz[c2].y, cxyz[c2].z, pairs1[i].z, shifts1[i].x, shifts1[i].y, shifts1[i].z);
            j++;
        }
    //printf("nPair with cell[%d] (%d, %d, %d) = %d\n", k, cxyz[c1].x, cxyz[c1].y, cxyz[c1].z, j);
    //printf("nPair1 = %d, nPair=%d, delt=%d\n", hmd->nPair1, hmd->nPair, hmd->nPair - hmd->nPair1);
    */
    printf("old cell list: nCell=%d nPair1=%d nPair2=%d totPair=%d\n", nCell, hmd->nPair1, hmd->nPair - hmd->nPair1, hmd->nPair);

    man->nPair1Block = hmd->nPair1;
    man->nPair2Block = hmd->nPair - hmd->nPair1;
    man->pairPerBlock = 16;      // çàäàåì îïûòíûì ïóòåì (÷èñëî ìóëüòèïðîöåññîðîâ â ÿäðå äîëæíî áûòü êðàòíî åìó)
    man->memPerPairBlock = hmd->maxAtPerCell * 4 * (sizeof(int) + sizeof(float3));
    man->memPerPairsBlock = hmd->maxAtPerBlock * 2 * (sizeof(int) + sizeof(float3)) + man->pairPerBlock * 4 * sizeof(int); // 4 = 2 * 2 ïîñêîëüêó 2 ÿ÷åéêå â ïàðå è íóæíî çàïîìèíàòü íà÷àëüíûé èíäåêñ è êîë-âî àòîìîâ


    data_to_device((void**)&(hmd->cellPairs), pairs1, hmd->nPair * sizeof(int4));
    data_to_device((void**)&(hmd->cellShifts), shifts1, hmd->nPair * sizeof(float3));

    free(cxyz);
    free(pairs1);
    free(shifts1);

    int** d_cells;
    int* cells_i;
    int** h_cells;

    h_cells = (int**)malloc(nCell * sizeof(int*));
    cudaMalloc((void**)&d_cells, nCell * sizeof(int*));
    for (i = 0; i < nCell; i++)
    {
        cudaMalloc((void**)&cells_i, (hmd->maxAtPerCell + 1) * sizeof(int));
        h_cells[i] = cells_i;
    }
    cudaMemcpy(d_cells, h_cells, nCell * sizeof(int*), cudaMemcpyHostToDevice);
    hmd->cells = d_cells;
    free(h_cells);
}
// end 'init_cellList' function'

int cell_id(int x, int y, int z, int nyz, int nz)
// return cell id by its coordinates (nyz = nz * ny, where nz and ny - the cell numbers in corresp directions
// automatically apply periodic boundaries 
{
    return x * nyz + y * nz + z;
}

int periodic_coord(int x, int max)
{
    if (x < 0)
        return x + max;
    else
        if (x >= max)
            return x - max;
        else
            return x;
}

int cell_id_periodic(int x, int y, int z, int nx, int ny, int nz)
// return cell id by its coordinates (nyz = nz * ny, where nz and ny - the cell numbers in corresp directions
// automatically apply periodic boundaries 
{
    int nyz = ny * nz;
    int x1 = periodic_coord(x, nx);
    int y1 = periodic_coord(y, ny);
    int z1 = periodic_coord(z, nz);

    return x1 * nyz + y1 * nz + z1;
}


void cell_xyz(int id, int nyz, int nz, int &x, int &y, int &z)
// return coordinates (as parameters) of cell with id = id
{
    int rest;
    x = del_and_rest(id, nyz, rest);
    y = del_and_rest(rest, nz, z);
}

void init_singleAtomCellList(float minRad, float maxR2, int nAt, Elec* elec, Sim* sim, cudaMD* hmd, hostManagMD* man)
// âåðñèÿ cell list, ñ áëîêàìè
{
#ifdef USE_FASTLIST
    int i, j, k, l, n;
    int x, y, z;
    int x1, y1, z1;     // coordinates of neighbour cell
    int shift_type, inter_type;
    float minR = minRad / (float)sqrt(3);
    //int totPairs = 0;       // total number of pairs (for verification)

    int nCell = ncell(minR, 0, hmd);

    // define maximal number of neighboring cell in the each direction
    int nxmax = ceil(sim->rMax / hmd->cSize.x);
    int nymax = ceil(sim->rMax / hmd->cSize.y);
    int nzmax = ceil(sim->rMax / hmd->cSize.z);

    int* nNeigh = (int*)malloc(nCell * int_size);   // for keeping neighbours count
    int3** neighList = (int3**)malloc(nCell * pointer_size);    // for keeping neighbours list
    int3* list = (int3*)malloc((2 * nxmax + 1) * (2 * nymax + 1) * (2 * nzmax + 1) * sizeof(int3)); // list of neigbors
    for (i = 0; i < nCell; i++)
    {
        cell_xyz(i, hmd->cnYZ, hmd->cNumber.z, x, y, z);
        n = 0;
        for (j = x - nxmax; j < x + nxmax + 1; j++)
            for (k = y - nymax; k < y + nymax + 1; k++)
                for (l = z - nzmax; l < z + nzmax + 1; l++)
                    if ((j != x) || (k != y) || (l != z))
                    {
                        x1 = periodic_coord(j, hmd->cNumber.x);
                        y1 = periodic_coord(k, hmd->cNumber.y);
                        z1 = periodic_coord(l, hmd->cNumber.z);
                        if (is_cellneighbors(x, y, z, x1, y1, z1, sim->rMax, sim->r2Max/*rmax, maxR2*/, hmd, sim, shift_type, inter_type))
                        {
                            list[n].x = cell_id(x1, y1, z1, hmd->cnYZ, hmd->cNumber.z);
                            list[n].y = shift_type;
                            list[n].z = inter_type;
                            n++;
                        }
                    }
        nNeigh[i] = n;
        data_to_device((void**)&(neighList[i]), list, n * sizeof(int3));
    }
    data_to_device((void**)&(hmd->neighCells), neighList, nCell * pointer_size);
    data_to_device((void**)&(hmd->nNeighCell), nNeigh, nCell * int_size);

    free(nNeigh);
    free(neighList);
    free(list);

    printf("single atom cell list. Last n = %d\n", n);
#endif
}
// end 'init_singleAtomCellList' function'

void free_singleAtomCellList(cudaMD* hmd)
{
#ifdef USE_FASTLIST
    cudaFree(hmd->nNeighCell);
    cuda2DFree((void**)hmd->neighCells, hmd->nCell);
#endif
}

int read_cuda(Field *fld, cudaMD *hmd, hostManagMD *man)
// read cuda settings from "cuda.txt"
{
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    man->nMultProc = devProp.multiProcessorCount;
    man->nSingProc = 64;   //! ïîõîæå èõ ìîæíî óçíàòü òîëüêî ïî äîêóìåíòàöèè ñàìîé âèäåîêàðòû //! ïîòîì áóäåì ñ÷èòûâàòü èç ôàéëà
    man->totCores = man->nMultProc * man->nSingProc;
    
    FILE *f = fopen("cuda.txt", "r");
    if (f == NULL)
    {
        printf("ERROR[a001]! Fatal Error. Can't open cuda.txt file\n");
        return 0;
    }

    //! ïåðåíåñòè ýòî â cuStat.cu
    if (!find_int_def(f, " nstep stat %d", man->stat.nstep, 10))
    {
        printf("WARNING[a003]: 'nstep stat' directive is not specified in cuda.txt, default value of 10 is used\n");
    }
    //! äîáàâèòü ïðîâåðêó, ÷òî òðàåêòîðèè íóæíû
    if (!find_int_def(f, " nstep traj %d", man->nstep_traj, 10))
    {
        printf("WARNING[b007]: 'nstep traj' directive is not specified in cuda.txt, default value of 10 is used\n");
    }

    //! äîáàâèòü ïðîâåðêó, ÷òî ñâÿçàííûå òðàåêòîðèè íóæíû
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


    fclose(f);
    return 1;
}

cudaMD* init_cudaMD(Atoms* atm, Field* fld, Sim* sim, TStat* tstat, Box* bx, Elec* elec, hostManagMD* man, cudaMD *h_md)
{
    int i, j, k;
    int xyzsize = atm->nAt * float3_size;
    int nsize = atm->nAt * int_size;
    int flsize = atm->nAt * float_size;

    // CUDA SETTINGS
    if (!read_cuda(fld, h_md, man))
        return NULL;

    // ATOMS DATA
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


    cudaSpec* h_specs = (cudaSpec*)malloc(fld->nSpec * sizeof(cudaSpec));
    cudaVdW* h_ppots = (cudaVdW*)malloc(fld->nVdW * sizeof(cudaVdW));
    cudaVdW*** h_vdw = (cudaVdW***)malloc(fld->nSpec * sizeof(void*));  // 2d array to pointer to cudaVdW


    //! ÿ èíèöèàëèçóðóþ ýòîò ìàññèâ çäåñü, ïîñêîëüêó åãî àäðåñà ìíå óæå íóæíû, ÷òîáû ññûëàòüñÿ íà íèõ èç ìàññèâà vdw
    cudaVdW* d_ppots;
    cudaMalloc((void**)&d_ppots, fld->nVdW * sizeof(cudaVdW));

    cudaVdW*** d_vdw;
    cudaVdW** vdw_i;
    cudaMalloc((void**)&d_vdw, fld->nSpec * sizeof(cudaVdW**));
#ifdef TX_CHARGE
    float* chprods = (float*)malloc(sizeof(float) * fld->nSpec * fld->nSpec);
#endif

    //float* qiqj = (float*)malloc(fld->nSpec * sizeof(float));
    float** h_chProd = (float**)malloc(fld->nSpec * pointer_size);
    float** d_chProd;
    float* chProd_i;
    cudaMalloc((void**)&d_chProd, fld->nSpec * pointer_size);

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
                    h_vdw[i][j] = &d_ppots[k];
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
    cudaMemcpy(d_vdw, h_vdw, fld->nSpec * pointer_size, cudaMemcpyHostToDevice);

    //van der Waals
    h_md->pairpots = d_ppots;
    h_md->vdws = d_vdw;

    data_to_device((void**)&(h_md->specs), h_specs, fld->nSpec * sizeof(cudaSpec));
    h_md->chProd = d_chProd;

    data_to_device((void**)&(h_md->nnumbers), fld->nnumbers, fld->nNucl * int_size);

    // ÏÎÏÐÎÁÓÅÌ ÒÀÊÆÅ ÏÐÎÈÇÂÅÄÅÍÈÅ ÇÀÐßÄÎÂ ÑÂßÇÀÒÜ ÒÅÊÑÒÓÐÍÎÉ ÏÀÌßÒÜÞ
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
    // ÊÎÍÅÖ ÒÅÊÑÒÓÐÍÎÃÎ ÊÓÑÊÀ

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

    h_md->nAt = atm->nAt;

    h_md->engKin = 0.f;
    h_md->engTot = 0.f;
    h_md->engTemp = 0.f;    // radiative thermostat
    h_md->engElecField = 0.f;
    h_md->engCoul1 = 0.f;
    h_md->engCoul2 = 0.f;
    h_md->engCoul3 = 0.f;
    h_md->engVdW = 0.f;

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

    //thermostat and temperature data
    init_cuda_tstat(atm->nAt, tstat, h_md, man);


    // BOX
    //! only for rectangular geometry!
    h_md->leng = make_float3((float)bx->la, (float)bx->lb, (float)bx->lc);
    h_md->halfLeng = make_float3(0.5f * h_md->leng.x, 0.5f * h_md->leng.y, 0.5f * h_md->leng.z);   //! âèäèìî íå âû÷èñëÿþòñÿ ýòè øòóêè ïðè ïîäãîòîâêå sim è box
    h_md->revLeng = make_float3(1.f / h_md->leng.x, 1.f / h_md->leng.y, 1.f / h_md->leng.z);       //! âèäèìî íå âû÷èñëÿþòñÿ ýòè øòóêè ïðè ïîäãîòîâêå sim è box
    h_md->edgeArea = make_float3(h_md->leng.y * h_md->leng.z, h_md->leng.x * h_md->leng.z, h_md->leng.x * h_md->leng.y);
    //! ìîæåò áûòü êàê ðàç ïðÿìûå ïëîùàäè è íå íóæíû, à òîëüêî îáðàòíûå?
    h_md->revEdgeArea = make_float3(1.f / h_md->edgeArea.x, 1.f / h_md->edgeArea.y, 1.f / h_md->edgeArea.z);
    h_md->volume = h_md->leng.x * h_md->leng.y * h_md->leng.z;

    h_md->elecField = make_float3((float)sim->Ux, (float)sim->Uy, (float)sim->Uz);

    h_md->tSt = (float)sim->tSt;
#ifdef USE_CONST
    cudaMemcpyToSymbol(&tStep, &(h_md->tSt), sizeof(float), 0, cudaMemcpyHostToDevice);
#endif

    // ELEC
    h_md->use_coul = elec->type;
    h_md->alpha = (float)elec->alpha;    // in Ewald summation
    h_md->daipi2 = (float)elec->daipi2;
    h_md->elC1 = (float)elec->scale;
    h_md->elC2 = (float)elec->scale2;
    h_md->rElec = (float)(elec->rReal);
    h_md->r2Elec = (float)(elec->r2Real);
    h_md->r2Max = sim->r2Max;


    // DEFINTION OF hostManagMD variables
    man->atStep = ceil((double)atm->nAt / man->totCores);   // >= 1
    int mxAtPerBlock = man->atStep * man->nSingProc;
    man->nAtBlock = ceil((double)atm->nAt / mxAtPerBlock);
    man->nAtThread = man->nSingProc;

    //!  çàãðóçèòü âñå ÌÏ ïîðîâíó, 1áëîê = 1ÌÏ, îïóñòèì ñèòóàöèþ, êîãäà ÷èñëî ÌÏ ìåíüøå êîë-âà àòîìîâ, âåäü äëÿ ìîäåëèðîâàíèÿ íàì íóæíû òûñÿ÷è àòîìîâ
    man->atPerBlock = ceil((double)atm->nAt / man->nMultProc); // ÷èñëî àòîìîâ íà ÌÏ
    man->atPerThread = ceil((double)man->atPerBlock / man->nSingProc);    //! ÷èñëî àòîìîâ íà ïîòîê
    if (man->atPerBlock < (man->atPerThread * man->nSingProc))
        man->atPerBlock = man->atPerThread * man->nSingProc;    // íî íå ìåíüøå, ÷åì ÷èñëî àòîìîâ âî âñåõ ïîòîêàõ áëîêà

    //cell list
#ifdef USE_FASTLIST
    //alloc_sort(atm->nAt, h_md->nCell, h_md);
    //init_singleAtomCellList(minR, maxR2, atm->nAt, elec, sim, h_md, man);

    init_cellList(1, 1, 6, sim->desired_cell_size, atm, fld, elec, h_md, man);
    //init_cellList(0, 1, 4, 0.0, atm, fld, elec, h_md, man);
#else
    init_cellList(minR, maxR2, atm->nAt, elec, sim, h_md, man);
#endif

    //! âîîáùå èõ íóæíî çàïîëíèòü íóëÿìè, íî ìîæåò ýòî äåôîëòíî òàê è äåëàåòñÿ?
    cudaMalloc((void**)&(h_md->specAcBoxPos), fld->nSpec * int3_size);
    cudaMalloc((void**)&(h_md->specAcBoxNeg), fld->nSpec * int3_size);

    // for debugging
#ifdef DEBUG_MODE
    cudaMalloc((void**)&(h_md->nCult), sizeof(int) * atm->nAt);
    cudaMalloc((void**)&(h_md->nPairCult), sizeof(int) * h_md->nPair);
    cudaMalloc((void**)&(h_md->nCelCult), sizeof(int) * h_md->nCell);
#endif

    // EWALD
#ifdef USE_EWALD
    h_md->nk = make_int3(elec->kx, elec->ky, elec->kz);

    // define rKcut2 as maximal kvector
    h_md->rKcut2 = elec->kx * h_md->revLeng.x;
    if (h_md->rKcut2 < elec->ky * h_md->revLeng.y)
        h_md->rKcut2 = elec->ky * h_md->revLeng.y;
    if (h_md->rKcut2 < elec->kz * h_md->revLeng.z)
        h_md->rKcut2 = elec->kz * h_md->revLeng.z;

    h_md->rKcut2 *= twopi * 1.05; // according to DL_POLY source
    h_md->rKcut2 *= h_md->rKcut2;

    float rvol = h_md->revLeng.x * h_md->revLeng.y * h_md->revLeng.z;
    h_md->ewEscale = (float)(twopi * rvol * Fcoul_scale / elec->eps);
    h_md->ewFscale = (float)(2 * twopi * rvol * Fcoul_scale / elec->eps);

    float* exprk2 = (float*)malloc(sizeof(float) * NTOTKVEC);
    float3* rk = (float3*)malloc(sizeof(float3) * NTOTKVEC);

    // define some ewald arrays
    float rkx, rky, rkz, rk2;
    int mmin = 0; int nmin = 1;
    int m, n;
    int ik = 0;
    float c = -0.25 / h_md->alpha / h_md->alpha;
    int l;
    for (l = 0; l < h_md->nk.x; l++)
    {
        rkx = (float)(l * twopi * h_md->revLeng.x); // only for rect geometry!
        for (m = mmin; m < h_md->nk.y; m++)
        {
            rky = (float)(m * twopi * h_md->revLeng.y);
            for (n = nmin; n < h_md->nk.z; n++)
            {
                rkz = n * twopi * h_md->revLeng.z;
                rk2 = rkx * rkx + rky * rky + rkz * rkz;
                if (rk2 < h_md->rKcut2) // cutoff
                {
                    rk[ik].x = rkx;
                    rk[ik].y = rky;
                    rk[ik].z = rkz;
                    exprk2[ik] = exp(c * rk2) / rk2;
                    ik++;
                }
            } // end n-loop (over kz-vectors)
            nmin = 1 - elec->kz;
        } // end m-loop (over ky-vectors)
        mmin = 1 - elec->ky;
    }  // end l-loop (over kx-vectors)

    h_md->nKvec = ik;
    man->memRecEwald = ik * sizeof(float2);

    data_to_device((void**)&(h_md->rk), rk, ik * float3_size);
    data_to_device((void**)&(h_md->exprk2), exprk2, ik * float_size);
    cudaMalloc((void**)&(h_md->qDens), ik * sizeof(float2));

    delete[] rk;
    delete[] exprk2;

    float2** qiexp = (float2**)malloc(atm->nAt * pointer_size);
    for (i = 0; i < atm->nAt; i++)
        cudaMalloc((void**)&(qiexp[i]), ik * sizeof(float2));
    data_to_device((void**)&(h_md->qiexp), qiexp, atm->nAt * pointer_size);
    delete[] qiexp;

    init_realEwald_tex(h_md, elec->rReal, elec->alpha);
#endif

    // statistics
    init_cuda_stat(h_md, man, sim, fld, tstat);
    init_cuda_rdf(fld, sim, man, h_md);
    //! íàôèã, óæå âíóòðè init_cuda_rdf ðåøàåì äåëàòü n_ èëè îáû÷íûé
    if (sim->nuclei_rdf)
        init_cuda_nrdf(fld, sim, man, h_md);


    h_md->use_angl = sim->use_angl;
    h_md->use_bnd = sim->use_bnd;
    if (fld->nBdata)
        bonds_to_device(atm, fld, sim, h_md, man);

    int* int_array;
    if (sim->use_angl || sim->use_bnd)
    {
        // create oldTypes array
        int_array = (int*)malloc(nsize);
        for (i = 0; i < atm->nAt; i++)
        {
            int_array[i] = -1;  // oldTypes[i] = -1
        }
        data_to_device((void**)&(h_md->oldTypes), int_array, nsize);
    }

    // angles:
    if (fld->nAdata)
    {
        h_md->nAngle = fld->nAngles;
        h_md->mxAngle = fld->mxAngles;

        man->angPerBlock = ceil((double)fld->mxAngles / (double)man->nMultProc); // ÷èñëî àòîìîâ íà ÌÏ
        man->angPerThread = ceil((double)man->angPerBlock / (double)man->nSingProc);    //! ÷èñëî àòîìîâ íà ïîòîê
        if (man->angPerBlock < (man->angPerThread * man->nSingProc))
            man->angPerBlock = man->angPerThread * man->nSingProc;    // íî íå ìåíüøå, ÷åì ÷èñëî àòîìîâ âî âñåõ ïîòîêàõ áëîêà


        int4* int4_array = (int4*)malloc(fld->mxAngles * sizeof(int4));
        for (i = 0; i < fld->nAngles; i++)
        {
            int4_array[i] = make_int4(fld->centrs[i], fld->lig1[i], fld->lig2[i], fld->angTypes[i]);
        }
        cudaMalloc((void**)&(h_md->angles), fld->mxAngles * sizeof(int4));
        cudaMemcpy(h_md->angles, int4_array, fld->mxAngles * sizeof(int4), cudaMemcpyHostToDevice);
        free(int4_array);

        cudaAngle* tang = (cudaAngle*)malloc(sizeof(cudaAngle) * fld->nAdata);
        for (i = 1; i < fld->nAdata; i++)   // i = 0 not interesting as just reffered to no angle
        {
            // + 1 as 0 reserved for deleted angle
            tang[i].type = fld->adata[i].type;   
            tang[i].p0 = (float)fld->adata[i].p0;
            tang[i].p1 = (float)fld->adata[i].p1;
            tang[i].p2 = (float)fld->adata[i].p2;
        }
        data_to_device((void**)&(h_md->angleTypes), tang, sizeof(cudaAngle) * fld->nAdata);
        free(tang);

        if (sim->use_angl)
        {
            //int* int_array2 = (int*)malloc(nsize);  // nangles
            // ïàìÿòü âûäåëåíà ðàíüøå ïðè ñîçäàíèè oldTypes
            for (i = 0; i < atm->nAt; i++)
            {
                //int_array[i] = -1;  // oldTypes[i] = -1
                //int_array2[i] = 0;
                int_array[i] = 0;   
            }
            // calculate number of angles:
            for (i = 0; i < fld->nAngles; i++)
                int_array/*2*/[fld->centrs[i]]++;

            data_to_device((void**)&(h_md->nangles), int_array/*2*/, nsize);
            //data_to_device((void**)&(h_md->oldTypes), int_array, nsize);
            free(int_array);
            //delete[] int_array2;

            int_array = (int*)malloc(fld->nSpec * int_size);
            for (i = 0; i < fld->nSpec; i++)
                int_array[i] = fld->species[i].angleType;
            cudaMalloc((void**)&(h_md->specAngles), fld->nSpec * int_size);
            cudaMemcpy(h_md->specAngles, int_array, fld->nSpec * int_size, cudaMemcpyHostToDevice);
            //delete[] int_array;
        }
    }
    // end angles

    if (sim->use_angl || sim->use_bnd)
    {
        free(int_array);
    }

    // electron jumps
    init_cuda_ejump(sim, atm, h_md);

    // trajectories
    if (sim->frTraj)
        init_cuda_trajs(atm, h_md, man);

    // bind trajectories
    if (sim->nBindTrajAtoms)
        init_cuda_bindtrajs(sim, h_md, man);

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

    free(h_specs);
    free(h_ppots);
    for (i = 0; i < fld->nSpec; i++)
    {
        //delete[] h_chProd[i];
        //delete[] h_vdw[i];
    }
    free(h_chProd);
    free(h_vdw);

    return d_md;
}
// end 'init_cudaMD' function

/*
void nxyz_to_host(float3 *buffer, Atoms* atm, cudaMD* hmd, hostManagMD* man)
// copy the number of atoms and coordinates array {xyz} from device to host
{
    cudaMemcpy(&(atm->nAt), man->nAtPtr, int_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(buffer, hmd->xyz, atm->nAt * sizeof(int3), cudaMemcpyDeviceToHost);
}
*/


void md_to_host(Atoms* atm, Field* fld, cudaMD *hmd, cudaMD *dmd, hostManagMD* man)
// copy md results from device to host (hmd - host exemplar of cudaMD)
{
    int i;

    //!!!!!!!!!
    // ó íàñ ÑÎÐÒÈÐÎÂÊÀ! ñëåäîâàòåëüíî ìàññèâû ÷åðåäóþòñÿ ìåñòàìè ñîðòèðîâàííûé è íåò! ò.å. ñ õîñòà ìû ìîæåì ññûëàòüñÿ íå íà òîò ìàññèâ äåâàéñà
    // ! îáíîâëÿåì ñòðóêòóðó
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
    bonds_to_host(int4_arr, hmd, fld, man);
    angles_to_host(int4_arr, hmd, fld, man);
    free(int4_arr);
}

void free_device_md(cudaMD* dmd, hostManagMD* man, Sim* sim, Field* fld, TStat *tstat, cudaMD *hmd)
// free all md-arrays on device
// hmd - host exemplar of cudaMD, dmd - on device
{
    cudaMemcpy(hmd, dmd, sizeof(cudaMD), cudaMemcpyDeviceToHost);

    //! âíèìåíèå, âåçäå, ãäå èñïîëüçóåòñÿ nCell è nAt â áóäóùåì íàäî ïåðåçàãðóçèòü ýòè ïîëÿ

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

    free_cuda_tstat(tstat, hmd);

#ifdef TX_CHARGE
    cudaUnbindTexture(&qProd);
    cudaFreeArray(hmd.texChProd);
#endif

#ifdef USE_EWALD
    cudaFree(hmd->rk);
    cudaFree(hmd->exprk2);
    cudaFree(hmd->qDens);
    cuda2DFree((void**)hmd->qiexp, hmd->nAt);
    free_realEwald_tex(hmd);
#endif
    
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
        free_cuda_nrdf;

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
