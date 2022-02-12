#ifndef CUCELLLIST_H
#define CUCELLLIST_H

void init_cellList(int div_type, int list_type, int bypass_type, float size, Atoms* atm, Field* fld, Elec* elec, cudaMD* hmd, hostManagMD* man, int box_type);
void free_cellList(cudaMD* hmd, hostManagMD* man);


#endif // CUCELLLIST_H
