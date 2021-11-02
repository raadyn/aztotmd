#include "dataStruct.h"
#include "cuStruct.h"
#include "cuCellList.h"
#include "cuSort.h"
#include "utils.h"
#include "cuUtils.h"


int split_cells(int div_type, float r, int add_to_even, int nAt, cudaMD* hmd)
// div_type îïðåäåëÿåò òèï ðàçáèåíèÿ: 0 - ðåáðî ÿ÷åéêè  íå ïðåâûøàåò r, 1 - ðåáðî íå ìåíüøå r
// split box into cells
// calculate cell size and count and return number of cells
// nAt is the number of atoms
{
    if (div_type == 0)
        hmd->cNumber = make_int3(ceil(hmd->leng.x / r), ceil(hmd->leng.y / r), ceil(hmd->leng.z / r));
    else
        hmd->cNumber = make_int3(floor(hmd->leng.x / r), floor(hmd->leng.y / r), floor(hmd->leng.z / r));

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

    hmd->maxAtPerCell = (double)(nAt * 3) / hmd->nCell;          //! ÿ íå çíàþ òî÷íî, êàê ëó÷øå îïðåäåëèòü ýòî êîë-âî
    return hmd->nCell;
}

/*
int n_pairs(int n)
// return number of pairs for n element
{
    return n * (n - 1) / 2;
}
*/

int3 xyz_of_cell(int id, int nyz, int nz)
// return coordinates of cell with id = id
{
    int x, y, z, rest;
    x = del_and_rest(id, nyz, rest);
    y = del_and_rest(rest, nz, z);
    int3 xyz = make_int3(x, y, z);
    return xyz;
}

/*
int cell_dist(int xi, int xj, int mx, float length, float csize, float rskip, float& rmin, float& rmax, float& shift, int& int_shift)
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
*/

int cell_dist_int(int xi, int xj, int mx, float csize, float rskip, float& rmin, float& rmax, int& shift)
// âû÷èñëÿåì ìèíèìàëüíîå è ìàêñèìàëüíîå ðàññòîÿíèå (â êîë-âå ÿ÷ååê) ìåæäó ÷àñòèöàìè èç 2õ ðàçíûõ ÿ÷ååê, à òàêæå ñäâèã, èñïîëüçóåìûé ïðè ó÷åòå ïåðèîäè÷åñêèõ óñëîâèé
// xi, xj - êîîðäèíàòû 2õ ÿ÷ååê â îäíîì èç èçìåðåíèé, mx - ìàêñèìàëüíîå êîë-âî ÿ÷ååê â ýòîì èçìåðåíèè,  csize - ðàçìåð ÿ÷åéêè
// âîçâðàùàåì 1 åñëè ÿ÷åéêè ñëèøêîì äàëåêî (áîëüøå rskip) èíà÷å 0
// version with integer shift (-1 / 0 / 1)
// for full periodic rectangular conditions
{

    shift = 0;
    int delt = abs(xi - xj);
    if (delt != 0)
    {
        if (delt > (double)(mx / 2))    // periodic conditions
        {
            delt = mx - delt;
            if (xi > xj)
                shift = 1;
            else
                shift = -1;
        }
        rmin = (delt - 1) * csize;
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

void add_cell_pairs(int cell1, int cell2, int4 *pairs, float3 *shifts, int &index, Elec *elec, Field *fld, int in_any_case, cudaMD *hmd)
// verify that cells cell1 and cell2 are 'in range' and add it to pairs array with saving of shift and increase index
// 'in_any_case' parameter means that the pair is added in any case but with .w = -1 (it needed in 2a function)
{
    int out_of_range = 0;
    float dxmin, dymin, dzmin, dxmax, dymax, dzmax;
    int shx, shy, shz;
    int3 xyz1 = xyz_of_cell(cell1, hmd->cnYZ, hmd->cNumber.z);
    int3 xyz2 = xyz_of_cell(cell2, hmd->cnYZ, hmd->cNumber.z);
    float rmax = max(elec->rReal, fld->maxRvdw);

    // x dimension:
    if (cell_dist_int(xyz1.x, xyz2.x, hmd->cNumber.x, hmd->cSize.x, rmax, dxmin, dxmax, shx))
    {
        if (in_any_case)
            out_of_range = 1;
        else
            return;     // no need to continue 
    }

    // y dimension:
    if (!out_of_range)  // otherwise no need to calculate
        if (cell_dist_int(xyz1.y, xyz2.y, hmd->cNumber.y, hmd->cSize.y, rmax, dymin, dymax, shy))
        {
            if (in_any_case)
                out_of_range = 1;
            else
                return;     // no need to continue 
        }

    // z dimension:
    if (!out_of_range)  // otherwise no need to calculate
        if (cell_dist_int(xyz1.z, xyz2.z, hmd->cNumber.z, hmd->cSize.z, rmax, dzmin, dzmax, shz))
        {
            if (in_any_case)
                out_of_range = 1;
            else
                return;     // no need to continue 
        }

    if (!out_of_range)  // otherwise no need to calculate
    {
        float dr2min = sqr_sum(dxmin, dymin, dzmin);
        if (dr2min > (rmax * rmax))
        {
            if (in_any_case)
                out_of_range = 1;
            else
                return;     // no need to continue 
        }
    }

    if (!out_of_range)  // otherwise no need to calculate
    {
        int coul, vdw;
        float dr2max = sqr_sum(dxmax, dymax, dzmax);
        float dr2min = sqr_sum(dxmin, dymin, dzmin);
        if (dr2max < elec->r2Real)
            coul = 1;   // ãàðàíòèðîâàíî äîòÿãèâàåòñÿ Êóëîíîâñêîå âçàèìîäåéñòâèå
        else
            coul = 0;   // ìîæåò äîòÿãèâàåòñÿ, à ìîæåò è íåò

        if (dr2min > (fld->maxRvdw * fld->maxRvdw))
            vdw = 0;    // ãàðàíòèðîâàíî íå äîñòàåò ÂäÂ
        else
            vdw = 1;    // ìîæåò è äîñòàåò

        pairs[index].z = coul * 2 + vdw;
    }
    else
        pairs[index].z = -1;

    pairs[index].x = cell1;
    pairs[index].y = cell2;
    //pairs[index].w = code_shift(shx, shy, shz);
    shifts[index] = make_float3(shx * hmd->leng.x, shy * hmd->leng.y, shz * hmd->leng.z);
    index++;
}

int pair_exists_shift(int cell1, int cell2, float3 &shift, Elec* elec, Field* fld, cudaMD* hmd)
// similar to the previous function, but return 1 if cells 'are in range' and save shift as a parameter 
{
    float dxmin, dymin, dzmin, dxmax, dymax, dzmax;
    int shx, shy, shz;
    int3 xyz1 = xyz_of_cell(cell1, hmd->cnYZ, hmd->cNumber.z);
    int3 xyz2 = xyz_of_cell(cell2, hmd->cnYZ, hmd->cNumber.z);
    float rmax = max(elec->rReal, fld->maxRvdw);

    // x dimension:
    if (cell_dist_int(xyz1.x, xyz2.x, hmd->cNumber.x, hmd->cSize.x, rmax, dxmin, dxmax, shx))
        return 0;

    // y dimension:
    if (cell_dist_int(xyz1.y, xyz2.y, hmd->cNumber.y, hmd->cSize.y, rmax, dymin, dymax, shy))
        return 0;

    // z dimension:
    if (cell_dist_int(xyz1.z, xyz2.z, hmd->cNumber.z, hmd->cSize.z, rmax, dzmin, dzmax, shz))
        return 0;

    float dr2min = sqr_sum(dxmin, dymin, dzmin);
    if (dr2min > (rmax * rmax))
        return 0;

    shift = make_float3(shx * hmd->leng.x, shy * hmd->leng.y, shz * hmd->leng.z);
    return 1;
}

void init_bypass0(int pairPerBlock, Elec *elec, Field *fld, hostManagMD *man, cudaMD* hmd)
// îáõîä ðåàëèçîâàííûé â àëãîðèòìàõ (2a, 2b) è (3a, 3b): 2a è 2b ïîëó÷àþòñÿ ïðè pairPerBlock == 1
//  1àÿ ÷àñòü - ïàðû âèäà (i)-(i+1) è èõ ñäâèãè, 2àÿ ÷àñòü - îñòàëüíûå ïàðû è èõ ñäâèãè - îòäåëüíî. 
//  Îáå ÷àñòè õðàíÿòñÿ â îäíèõ ìàññèâàõ md->pairs è md->shifts
//  Âñå ïàðû â ôîðìàòå int4 (.x, .y - àáñîëþòíûå èíäåêñû ÿ÷ååê, îñòàëüíîå çàðåçåðâèðàâàíî ïîä òèï ñäâèãà è âçàèìîäåéñòâèÿ)
//  + íàñòðàèâàåò âñå íåîáõîäèìûå ïàðàìåòðû â man äëÿ âûçîâà ôóíêöèé (ïàìÿòü è ÷èñëî áëîêîâ/íèòåé)
{
    int i, j, k, index = 0;
    int art_cell = 0;
    int npair = npairs(hmd->nCell);
    // initialize arrays with excess
    int4 *pairs = (int4*)malloc(int4_size * npair);
    float3* shifts = (float3*)malloc(float3_size * npair);

    //! íóæíî åù¸ ó÷åñòü, ÷òî îäíà ÿ÷åéêà ìîæåò áûòü èñêóñòâåííàÿ è å¸ íóæíî óáðàòü èç ïàð!
    if (hmd->nCell != (hmd->cNumber.x * hmd->cNumber.y * hmd->cNumber.z))
        art_cell = 1;

    //! ïåðâûå ïàðû (0-1, 2-3, 4-5 è ò.ä.). ÎÍÈ ÍÅ ÏÅÐÅÑÅÊÀÞÒÑß ÏÎ ÄÀÍÍÛÌ
    // øòóêà â òîì, ÷òî â ïåðâîé ÷àñòè ïàðû ìîæåò è íå áûòü, íî çàãðóæàòü âñå ðàâíî íàäî, ïîñêîëüêó òàì îáðàáàòûâàþñÿ ïàðû âíóòðè ÿ÷åéêè!
    hmd->nPair1 = hmd->nCell / 2;
    for (i = 0; i < hmd->nPair1 - art_cell; i++) // íå çàáûòü ïðî èñêóñòâåííóþ ÿ÷åéêó
        add_cell_pairs(i * 2, i * 2 + 1, pairs, shifts, index, elec, fld, 1, hmd);    // verify, that cells in range and add in any case

    if (art_cell)
    {
        pairs[index] = make_int4(hmd->nCell - 2, hmd->nCell - 1, -1, 0);
        index++;
    }

    // one stupid verification
    if (index != hmd->nPair1)
    {
        printf("index=%d nPair1=%d", index, hmd->nPair1);
        hmd->nPair1 = index;
    }

    // rest pairs
    for (i = 0; i < hmd->nCell - 1 - art_cell; i++)
    {
        k = 2 - (i % 2); // ó÷èòûâàåì, ÷òî 0-1, 2-3, 4-5 ïàðû ìû óæå îòîáðàëè
        for (j = i + k; j < hmd->nCell - art_cell; j++)
            add_cell_pairs(i, j, pairs, shifts, index, elec, fld, 0, hmd);    // verify, that cells in range and add if it's so
    }
    hmd->nPair = index;

    //pair verifiyng:
    for (i = 0; i < hmd->nPair; i++)
    {
        if (pairs[i].x >= pairs[i].y)
        {
            printf("pair %d: %d-%d\n", i, pairs[i].x, pairs[i].y);
        }
    }

    data_to_device((void**)&(hmd->cellPairs), pairs, hmd->nPair * int4_size);
    data_to_device((void**)&(hmd->cellShifts), shifts, hmd->nPair * float3_size);
    free(pairs);
    free(shifts);

    // define variables for calling pair bypass functions
    man->pairPerBlock = pairPerBlock;
    int maxAtPerBlock = 2 * 16 * ((double)hmd->nAt / hmd->nCell) * 3 + 90;  // factor 3 for excess
    man->pairBlockA = ceil((double)hmd->nPair1 / pairPerBlock);
    man->pairBlockB = ceil((double)(hmd->nPair - hmd->nPair1) / pairPerBlock);
    man->pairThreadA = 16;      // çàäàåì îïûòíûì ïóòåì (÷èñëî ìóëüòèïðîöåññîðîâ â ÿäðå äîëæíî áûòü êðàòíî åìó)
    man->pairThreadB = 16;      // çàäàåì îïûòíûì ïóòåì (÷èñëî ìóëüòèïðîöåññîðîâ â ÿäðå äîëæíî áûòü êðàòíî åìó)

    //! íóæíî ïîïðàâèòü ýòè ïàðàìåòðû íà pairPerBlock
    man->pairMemA = hmd->maxAtPerCell * 4 * (int_size + float3_size);
    man->pairMemB = maxAtPerBlock * 2 * (int_size + float3_size) + man->pairPerBlock * 4 * int_size; // 4 = 2 * 2 ïîñêîëüêó 2 ÿ÷åéêå â ïàðå è íóæíî çàïîìèíàòü íà÷àëüíûé èíäåêñ è êîë-âî àòîìîâ
}

void free_bypass0(cudaMD* hmd)
{
    cudaFree(hmd->cellPairs);
    cudaFree(hmd->cellShifts);
}

void close_block(int4* cellBlocks, float3* shifts, float3** secShifts, int& blockIndex, int &open, int &totPairs)
// int open - flag: the block is still open or not
{
    open = 0;
    data_to_device((void**)&(secShifts[blockIndex]), shifts, cellBlocks[blockIndex].z * float3_size);
    totPairs += cellBlocks[blockIndex].z;
    blockIndex++;
}

void continue_block(int4 *cellBlocks, float3 *shifts, float3** secShifts, int &blockIndex, int maxBlocks, float3 shift, int& open, int &totPairs)
// int open - flag: the block is still open or not
{
    cellBlocks[blockIndex].z++;
    shifts[cellBlocks[blockIndex].z - 1] = make_float3(-shift.x, -shift.y, -shift.z);

    // apply limit on cell in block
    if (cellBlocks[blockIndex].z == maxBlocks)
        close_block(cellBlocks, shifts, secShifts, blockIndex, open, totPairs);
}

void start_block(int cell1, int cell2, int4* cellBlocks, float3* shifts, int& blockIndex, float3 shift, int& open)
// cell1 and cell2 are cell indexes
{
    open = 1;
    cellBlocks[blockIndex] = make_int4(cell1, cell2, 1, 0);  // last field is not used
    shifts[0] = make_float3(-shift.x, -shift.y, -shift.z);
}

void block_loop(int cell1, int j0, int jmax, int4* cellBlocks, float3* shifts, float3** secShifts, int& blockIndex, int maxBlocks, int& totPairs, Elec *elec, Field *fld, cudaMD *hmd)
{
    float3 shift;
    int open = 0;
    int j = j0;
    while (j < jmax)
    {
        if (pair_exists_shift(cell1, j, shift, elec, fld, hmd))
        {
            if (open)
            {
                continue_block(cellBlocks, shifts, secShifts, blockIndex, maxBlocks, shift, open, totPairs);
            }
            else
            {
                start_block(cell1, j, cellBlocks, shifts, blockIndex, shift, open);
            }
        }
        else
        {
            if (open)
            {
                close_block(cellBlocks, shifts, secShifts, blockIndex, open, totPairs);
            }
        }
        j++;
    }
    if (open)
    {
        close_block(cellBlocks, shifts, secShifts, blockIndex, open, totPairs);
    }
}

void init_bypass4(int cellInBlock, int nAt, Elec* elec, Field *fld, cudaMD* hmd, hostManagMD* man)
// analogous to previous init_fastCellList
{
    int i, j, k, index;
    int totPairs = 0;       // total number of pairs (for verification)
    int i0, N;

    j = npairs(cellInBlock);
    int4* pairs = (int4*)malloc(j * int4_size);
    float3* shifts = (float3*)malloc(j * float3_size);
    int* nPair = (int*)malloc(hmd->nCell * int_size);

    // òóò áóäóò õðàíèòñÿ óêàçàòåëè íà ñîîòâåòñòâóþùèå ìàññèâû, à ïîòîì çàôèãà÷èì èõ íà äåâàéñ
    int4** pairs_arr = (int4**)malloc(hmd->nCell * pointer_size);
    float3** shifts_arr = (float3**)malloc(hmd->nCell * pointer_size);

    //! ïåðâûå ïàðû ÿ÷ååê (0-1, 2-3, 4-5 è ò.ä.). ÎÍÈ ÍÅ ÏÅÐÅÑÅÊÀÞÒÑß ÏÎ ÄÀÍÍÛÌ
    int nBlock = ceil((double)hmd->nCell / cellInBlock);
    //printf("nCell=%d nBlock=%d cellInBlock=%d\n", hmd->nCell, nBlock, cellInBlock);
    for (i = 0; i < nBlock; i++)
    {
        i0 = i * cellInBlock;
        N = min(i0 + cellInBlock, hmd->nCell);
        index = 0;
        for (j = i0; j < N - 1; j++)
            for (k = j + 1; k < N; k++)
                add_cell_pairs(j, k, pairs, shifts, index, elec, fld, 0, hmd);    // verify, that cells in range and add if it's so
        nPair[i] = index;
        totPairs += index;
        data_to_device((void**)&(pairs_arr[i]), pairs, index * int4_size);
        data_to_device((void**)&(shifts_arr[i]), shifts, index * float3_size);
    }
    data_to_device((void**)&(hmd->nFirstPairs), nPair, hmd->nCell * int_size);
    data_to_device((void**)&(hmd->firstPairs), pairs_arr, hmd->nCell * pointer_size);
    data_to_device((void**)&(hmd->firstShifts), shifts_arr, hmd->nCell * pointer_size);
    free(nPair);
    free(pairs_arr);
    free(shifts_arr);

    // maximal possible number of rest pairs:
    int maxPairs = npairs(hmd->nCell) - totPairs;

    int4* cellBlocks = (int4*)malloc(maxPairs * int4_size);
    float3** secShifts = (float3**)malloc(maxPairs * pointer_size);
    k = 0;      // index of cellBlock
    // ëó÷øå ïîäåëèòü ïàðû (ñì. ðàññóæäåíèÿ íà õàáðå)
    int nFirstCell = (int)(sqrt(0.5) * hmd->nCell);
    for (i = 0; i < nFirstCell; i++)
    {
        block_loop(i, (i / cellInBlock + 1) * cellInBlock, nFirstCell, cellBlocks, shifts, secShifts, k, cellInBlock, totPairs, elec, fld, hmd);
    }

    // îñòàâøèåÿÿ áëîêè
    for (i = nFirstCell; i < hmd->nCell; i++)
    {
        block_loop(i, 0, (i / cellInBlock) * cellInBlock, cellBlocks, shifts, secShifts, k, cellInBlock, totPairs, elec, fld, hmd);
    }

    data_to_device((void**)&(hmd->cellBlocks), cellBlocks, k * sizeof(int4));
    data_to_device((void**)&(hmd->secShifts), secShifts, k * pointer_size);
    free(cellBlocks);
    free(secShifts);
    free(pairs);
    free(shifts);

    man->pairBlockA = nBlock;
    man->pairBlockB = k;
    man->cellPerBlockA = cellInBlock;
    //man->pairThreadA = cellInBlock * 4;   // äîëæíî áûòü êðàòíî ÷èñëó ÿ÷ååê â áëîêå
    man->pairThreadA = 64;   // äîëæíî áûòü êðàòíî ÷èñëó ÿ÷ååê â áëîêå
    man->pairThreadB = 32;

    int mxAtPerBlock = hmd->maxAtPerCell * cellInBlock + 20;       // with excess

    //! âîîáùå ýòî çàâèñèò îò òîãî, þçàåì ìû ðàçäåëÿåìóþ ïàìÿòü èëè íåò
    man->pairMemA = cellInBlock * int_size * 4 + mxAtPerBlock * (2 * float_size + int_size);    // cellInBlock * int * 4 - for 4a_1, for 4a int*2 is enough
    man->pairMemB = cellInBlock * int_size * 2 + mxAtPerBlock * (2 * float_size + 2 * int_size);

    printf("nPair: %d. ", totPairs);
}

void free_bypass4(cudaMD* hmd, hostManagMD* man)
{
    cudaFree(hmd->nFirstPairs);
    cudaFree(hmd->cellBlocks);
    cuda2DFree((void**)hmd->firstPairs, man->pairBlockA);
    cuda2DFree((void**)hmd->firstShifts, man->pairBlockA);
    cuda2DFree((void**)hmd->secShifts, man->pairBlockB);
}

void init_bypass5(Elec *elec, Field *fld, cudaMD *hmd, hostManagMD* man)
// part A : interaction inside cells. Part B: between cells. Used sorted arrays
{
    int i, j, index = 0;
    int n = npairs(hmd->nCell);
    int4* pairs = (int4*)malloc(int4_size * n);
    float3* shifts = (float3*)malloc(float3_size * n);

    for (i = 0; i < hmd->nCell - 1; i++)
        for (j = i + 1; j < hmd->nCell; j++)
            add_cell_pairs(i, j, pairs, shifts, index, elec, fld, 0, hmd);    // verify, that cells in range and add if it's so
    hmd->nPair = index;

    data_to_device((void**)&(hmd->cellPairs), pairs, hmd->nPair * int4_size);
    data_to_device((void**)&(hmd->cellShifts), shifts, hmd->nPair * float3_size);

    free(pairs);
    free(shifts);

    man->pairBlockA = hmd->nCell;
    man->pairThreadA = 16;
    man->pairBlockB = hmd->nPair;
    man->pairThreadB = 32;
    man->pairMemA = hmd->maxAtPerCell * (2 * float_size + int_size);

    printf("nPair: %d. ", hmd->nPair);
}

void free_bypass5(cudaMD* hmd)
{
    cudaFree(hmd->cellPairs);
    cudaFree(hmd->cellShifts);
}

void init_bypass6(int cellInBlock, int nAt, Elec* elec, Field* fld, cudaMD* hmd, hostManagMD* man)
// áóäåò îáðàáàòûâàòü ÿ÷åéêè îòäåëüíî, êàê bypass5 - íå òðåáóåò ñïåöèàëüíûõ ñòðóêòóð, à ïàðû - âñå êàê bypass4, íî âñå ïàðû âîîáùå
{
    int i, k, index;
    int totPairs = 0;       // total number of pairs (for verification)
    int i0;

    int4* pairs = (int4*)malloc(cellInBlock * int4_size);
    float3* shifts = (float3*)malloc(cellInBlock * float3_size);

    int maxPairs = npairs(hmd->nCell);
    int4* cellBlocks = (int4*)malloc(maxPairs * int4_size);
    float3** secShifts = (float3**)malloc(maxPairs * pointer_size);
    k = 0;      // index of cellBlock
    // ëó÷øå ïîäåëèòü ïàðû (ñì. ðàññóæäåíèÿ íà õàáðå)
    int nFirstCell = (int)(sqrt(0.5) * hmd->nCell);
    for (i = 0; i < nFirstCell; i++)
    {
        block_loop(i, i + 1, nFirstCell, cellBlocks, shifts, secShifts, k, cellInBlock, totPairs, elec, fld, hmd);
    }

    // îñòàâøèåÿÿ áëîêè
    for (i = nFirstCell; i < hmd->nCell; i++)
    {
        block_loop(i, 0, i, cellBlocks, shifts, secShifts, k, cellInBlock, totPairs, elec, fld, hmd);
    }

    data_to_device((void**)&(hmd->cellBlocks), cellBlocks, k * int4_size);
    data_to_device((void**)&(hmd->secShifts), secShifts, k * pointer_size);
    free(cellBlocks);
    free(secShifts);
    free(pairs);
    free(shifts);

    man->pairBlockA = hmd->nCell;
    man->pairBlockB = k;
    //man->cellPerBlockA = cellInBlock;
    
    //! âîîáùå ýòè øòóêè ñ÷èòàëèñü èç ôàéëà cuda.txt, äëÿ íåêîòîðûõ ñëó÷àåâ íà íèõ íàêëàäûâàþòñÿ íåêîòîðûå îãðàíè÷åíèÿ
    //man->pairThreadA = 16;
    //man->pairThreadB = 32;

    int mxAtPerBlock = hmd->maxAtPerCell * cellInBlock + 20;       // with excess

    //! âîîáùå ýòî çàâèñèò îò òîãî, þçàåì ìû ðàçäåëÿåìóþ ïàìÿòü èëè íåò
    man->pairMemA = cellInBlock * int_size * 4 + mxAtPerBlock * (2 * float_size + int_size);    // cellInBlock * int * 4 - for 4a_1, for 4a int*2 is enough
    man->pairMemB = cellInBlock * int_size * 2 + mxAtPerBlock * (2 * float_size + 2 * int_size);

    printf("bp6: nPair: %d. nThreads=(%d, %d)", totPairs, man->pairThreadA, man->pairThreadB);
}

void free_bypass6(cudaMD* hmd, hostManagMD* man)
{
    cudaFree(hmd->cellBlocks);
    cuda2DFree((void**)hmd->secShifts, man->pairBlockB);
}

void alloc_2dlist(int nCell, cudaMD *hmd)
// allocate 2d-array for cell list keeping
{
    int i;

    int** cells = (int**)malloc(nCell * pointer_size);
    for (i = 0; i < nCell; i++)
    {
        cudaMalloc((void**)&(cells[i]), (hmd->maxAtPerCell + 1) * int_size);
    }
    data_to_device((void**)&(hmd->cells), cells, nCell * pointer_size);
    free(cells);
}

void free_2dlist(cudaMD *hmd)
// free 2d-array for cell list keeping
{
    cuda2DFree((void**)&(hmd->cells), hmd->nCell);
}

void init_cellList(int div_type, int list_type, int bypass_type, float size, Atoms* atm, Field* fld, Elec *elec, cudaMD* hmd, hostManagMD *man)
// ôóíêöèÿ ïîäãîòàâëèâàåò âñå ïåðåìåííûå, èíèöèàëèçèðóåò è çàïîëíÿåò ìàññèâû äëÿ öåëë ëèñòà
//  ïðè íåîáõîäèìîñòè âûäåëÿåò ïàìÿòü äëÿ ðåàëèçàöèè ñîðòèðîâêè
// div_type îïðåäåëÿåò òèï ðàçáèåíèÿ: 0 - ÿ÷åéêà òàêàÿ, ÷òîáû äèàãîíðàëü íå ïðåâûøàëà min(Rvdw), 1 - ðåáðî íå ìåíüøå ìàêñèìàëüíîãî ðàäèóñà âçàèìîäåéñòâèÿ
// size - æåëàåìûé ðàçìåð ÿ÷åéêè, íî íå îáÿçàòåëüíî, ÷òî ïðîãðàììà ðàçîáü¸ò ïî íåìó
// list_type = 0 - 2d-array, 1 - based on sort
// bypass type - ñïîñîáû îáõîäà cell list
{
    man->div_type = div_type;
    man->list_type = list_type;
    man->bypass_type = bypass_type;
    
    int add_to_even = 0;    // íóæíî ëè äîâîäèòü ÷èñëî ÿ÷ååê äî ÷åòíîãî?
    if (bypass_type == 0)
        add_to_even = 1;

    int nCell;
    float r;
    //int i;

    // ùàñ âèäèìî óæå íåâàæíî, ïðîñòî ïèøè r = size
    if (div_type == 0)
    {
        r = fld->minRvdw / sqrt(3);
    }
    else
        r = size;   //! temp çäåñü äîëæíî áûòü ñðàâíåíèå ñ Êóëîíîì è ìàêñèìàëüíûì ÂäÂ

    nCell = split_cells(div_type, r, add_to_even, atm->nAt, hmd);

    if (list_type == 1)     // list based on sorted arrays
        alloc_sort(atm->nAt, nCell, hmd);
    else      // list as 2-d array
        alloc_2dlist(nCell, hmd);

    //! ôóíêöèè òèïà init_bypass îïðåäåëÿþò ìàññèâû äëÿ îáõîäà ïàð è ïàðàìåòðû ê íèì,òàêèå êàê ÷èñëî áëîêîâ, ÷èñëî ïîòîêîâ è ðàçìð ðàçäåëÿåìîåé ïàìÿòè
    switch (bypass_type)
    {
     case 0:     // for usage of cell_list2a and 2b or 3a and 3b functions (for 3a,3b first parameter must be >1)
         init_bypass0(1, elec, fld, man, hmd);
         break;
     case 4:     // fast cell list
         init_bypass4(9, atm->nAt, elec, fld, hmd, man);
         break;
     case 5:     
         init_bypass5(elec, fld, hmd, man);
         break;
     case 6:
         init_bypass6(10, atm->nAt, elec, fld, hmd, man);
         break;
    }

    printf("Used subdivision with cell=[%f, %f, %f] (desired=%f). nCell=%d A(%d, %d) B(%d, %d)\n", hmd->cSize.x, hmd->cSize.y, hmd->cSize.z, size, hmd->nCell, man->pairBlockA, man->pairThreadA, man->pairBlockB, man->pairThreadB);
}

void free_cellList(cudaMD* hmd, hostManagMD* man)
{
    if (man->list_type == 1)
        free_sort(hmd);
    else
        free_2dlist(hmd);

    // free arrays for bypass of pairs
    switch (man->bypass_type)
    {
    case 0:     // for usage of cell_list2a and 2b or 3a and 3b functions
        free_bypass0(hmd);
        break;
    case 4:     // fast cell list
        free_bypass4(hmd, man);
        break;
    case 5:  
        free_bypass5(hmd);
        break;
    case 6:     // fast cell list
        free_bypass6(hmd, man);
        break;
    }

}
