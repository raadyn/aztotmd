#ifndef INTEGRATORS_H
#define INTEGRATORS_H

void clear_force(Atoms *atm, Spec *sp, Sim *sim, Box *bx);
void reset_chars(Sim *sim);
void calc_chars(Sim *sim, double &sim_time);

// pair interactions:
void pair_inter(int i, int j, Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim);
void pair_inter_lst(int i, int j, Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim);

// ways of pair processing
void cell_list(Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim);
void all_pairs(Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim);

void integrate1(Atoms *atm, Spec *spec, Sim *sim, Box *box, TStat *tstat);
void integrate1_clst(Atoms *atm, Spec *spec, Sim *sim, Box *box, TStat *tstat);
void integrate2(Atoms *atm, Spec *sp, Sim* sim, int tScale, TStat *tstat);

#endif // INTEGRATORS_H
