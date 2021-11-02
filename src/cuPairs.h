// unit for calculation of pair interactions
#ifndef CUPAIRS_H
#define CUPAIRS_H
#include "dataStruct.h"


void iter_cellList(int iStep, int nB1, int nB2, dim3 dim, hostManagMD* man, cudaMD* devMD);
void iter_fastCellList(int iStep, Field* fld, cudaMD* dmd, hostManagMD* man);


#endif  // CUPAIRS_H
