#ifndef BONDS_H
#define BONDS_H

//int init_bonds(Sim *sim, int **nbonds, int ***bonds, int ***neigs/*, Sim *sim, Box *box*/, int def_nbonds);

void alloc_bonds(int n, Field *field);
int read_bondlist(Atoms *atm, Field *field, Sim *sim);

void save_bondlist(char *fname, Field *field);

int read_bond(int id, FILE* f, Field* field, Sim* sim);


//int autobonding(Atoms *atm, Field *field, Sim *sim, Box *bx, int Naut, double rc, FILE *f, int mxbond);

int read_linkage(FILE *f, int Nlnk, Field *field, int nBonds);
double bond_iter(double r2, Bond *bnd, double &eng);
double bond_iter_r(double r2, double &r, Bond *bnd, double &eng);

int create_bond(int at1, int at2, int type, Atoms *atm, Field *field);

void exec_bondlist(Atoms *atm, Field *field, Sim *sim, Box *bx);
double bond_eng_change(int iat, int jat, int i2type, int j2type, Atoms* atm, Field *field, Box *bx);
void change_bonds(int iat, int jat, int i2type, int j2type, Atoms* atm, Field *field);
void bond_out(Atoms *atm, Field *field, Box *bx, char *fname);

//void bonding(Atoms *atm, Spec *sp, VdW ***vdws, Sim *sim, Box *bx, Bond *btps, Field *field);


int free_bonds(Sim *sim, int **nbonds, int ***bonds, int ***neigs/*, Sim *sim, Box *box*/, int nAtm);

#endif // BONDS_H
