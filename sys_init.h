#ifndef SYS_INIT_H
#define SYS_INIT_H

#include "temperature.h"

int spec_by_name(Field* field, char* name, int& id);
//  searching of two species indexes by their names
int twospec_by_name(Field *field, char *name1, char *name2, int &id1, int &id2);
int nucl_by_name(Field* field, char* name, int& id);
//  system initialization
int init_md(Atoms *atm, Field *field, Sim *sim, Elec *elec, TStat *tstat, Box *bx);
int init_serial(Atoms* atm, Field* field, Sim* sim, Elec* elec, TStat* tstat, Box* bx);
// return information about simulation setup
void info_md(Atoms *atm, Sim *sim);
// deallocate arrays
void free_serial(Atoms* atm, Field* field, Elec* elec, Sim* sim);
void free_md(Atoms* atm, Field* field, TStat* tstat);

#endif  /* SYS_INIT_H */
