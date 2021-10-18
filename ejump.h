#ifndef EJUMP_H
#define EJUMP_H

// electron jump types
const int nJumpTypes    = 4;    // number of electron jump types
const int tpJumpNone    = 0;    // no jumps
const int tpJumpEq      = 1;    // equality (Frank-Condon principle)
const int tpJumpMin     = 2;    // minimization
const int tpJumpMetr    = 3;    // Metropolis scheme (comparision with kB*T)

// names of electron jump types
const char nmsJumpType[nJumpTypes][30] = {"none", "equality(Frank-Condon)", "minimal", "Metropolis"};

// donor and acceptor binary flags:
const int bfDonor =    0;
const int bfAcceptor = 1;

// ejump output:
void ejump_header(FILE *f, Field *field);
void ejump_out(FILE *f, double tm, int step, Field *field, Sim *sim);

// ejump allocation/deallocation:
void init_ejump(Atoms *atm, Field *field, Sim *sim);
void free_ejump(Sim *sim, Field *field);

// do electron jumps (return their numbers):
int ejump(Atoms *atm, Field *field, Sim *sim, Box *bx);         // with dE = 0
int ejump_min(Atoms *atm, Field *field, Sim *sim, Box *bx);     // with dE < 0
int ejump_metr(Atoms *atm, Field *field, Sim *sim, Box *bx);    // according to the Metropolis scheme
typedef int (*funcEjump)(Atoms*,  Field*, Sim*, Box*);
const funcEjump jumps[nJumpTypes] = {NULL, ejump, ejump_min, ejump_metr};

// superstructure of electron jumping
void jmp_rare(Atoms *atm, Field *field, Sim *sim, Box *bx, int step);   // do once every Nth timestep
void jmp_oft(Atoms *atm, Field *field, Sim *sim, Box *bx, int step);    // do N times per timestep
void jmp_none(Atoms *atm, Field *field, Sim *sim, Box *bx, int step);   // no jumps

#endif  /* EJUMP_H */
