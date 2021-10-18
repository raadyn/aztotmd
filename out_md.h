#ifndef OUT_MD_H
#define OUT_MD_H

void history_header(FILE *f);
void info_header(FILE *f, Atoms *at, Spec *sp);
void msd_header(FILE *f, Sim *sim, Field *field);
void stat_header(FILE *f, Sim *sm, Spec *sp);

void out_stat(FILE *f, double tm, int step, Sim *sm, Spec *sp);

int out_atoms(Atoms *atm, int N, Spec *spec, Box *bx, char *fname);
void out_info(FILE *f, double tm, int step, Atoms *at, Spec *sp);
int out_msd(FILE *f, Atoms *atm, int N, Spec *spec, int NSp, Box *bx, double tm, int tst);
int out_ncn(Atoms* atm, Field* field, Box* bx, Sim* sim, char* fname);
int out_cn(Atoms *atm, Field *field, Box *bx, Sim *sim, char *fname);
int out_velocities(Atoms *atm, Field *field, char *fname);
void traj_header(FILE *f, Atoms *at, Spec *sp, Sim *sim);
void traj_info(FILE *f, double tm, int step, Atoms *at, Spec *sp, Sim *sim);

void vaf_init(Atoms *atm);
void vaf_header(FILE *f, Field *field, Sim *sim);
void vaf_info(FILE *f, double tm, int step, Atoms *atm, Field *field, Sim *sim);


#endif // OUT_MD_H
