// elec.h
//  заголовочный файл модул€, работающего с алгоритмом Ёвальда
#ifndef ELEC_H
#define ELEC_H

// types for electrostatic calculations
const int nElecTypes = 4;
const int tpElecNone    = 0;
const int tpElecDir = 1;		// direct (naive sheme by Coulomb law with Elec cutoff)
const int tpElecEwald   = 2;
const int tpElecFennel = 3;		// Fennel and Gezelter http://dx.doi.org/10.1063/1.2206581

// FUNCTIONS of electrostatic calulations:
// electrostatic part embedded in pair loop
typedef void (*funcPairElec)(Spec *sp, int it, int jt, double r2, double r, Elec *elec, Sim *sim, double &force);
// electrostatic part calculated separately (for example ewald summation in recipropal space)
typedef void (*funcAddElec)(Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim);


void init_elec(Elec *elec, Box *bx, Sim *sim, Atoms *atm);
// read setting for electrostatic calculations (return success or not)
int read_elec(FILE *f, Elec *elec, Field *fld);
double ewald_const(Atoms *atm, Spec *sp, Elec *elec, Box *box);
// return constant part of Columbic potential energy via Ewald method
//   (!) need to be recalculated only then volume or summ(q) are changed

void ewald_rec(Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim);
// calculate reciprocal part of Ewald summ and corresponding forces
void no_elec(Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim);
// empty electrostatic calculations

double coul_iter(double r2, double &r, double chprd, double alpha, double &eng);
// real part of Ewald for each pair

// функции дл€ обхода кулоновсокго взаимодействи€ в реальном пространстве
void none_elec(Spec* sp, int it, int jt, double r2, double r, Elec* elec, Sim* sim, double& force);		// no electrostatic
void direct_coul(Spec* sp, int it, int jt, double r2, double r, Elec* elec, Sim* sim, double& force);	// brute, by Coulomb law
void direct_ewald(Spec* sp, int it, int jt, double r2, double r, Elec* elec, Sim* sim, double& force);	// Ewald
void fennel(Spec* sp, int it, int jt, double r2, double r, Elec* elec, Sim* sim, double& force);	// Fennel and Gezelter http://dx.doi.org/10.1063/1.2206581

void prepare_elec(Atoms *atm, Field *field, Elec *elec, Sim *sim, Box *bx);
void free_elec(Elec *elec, Atoms *atm);

// constant function arrays
const funcPairElec pair_elecs[nElecTypes] = {none_elec, direct_coul, direct_ewald, fennel};
const funcAddElec  add_elecs[nElecTypes] = {no_elec, no_elec, ewald_rec, no_elec};


#endif  /* ELEC_H */
