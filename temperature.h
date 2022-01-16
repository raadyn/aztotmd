// temperature.h
#ifndef TEMPERATURE_H
#define TEMPERATURE_H

#include <stdio.h>   // *FILE

#include "dataStruct.h"

//  thermostat types:
int const tpTermNone    = 0;    // no thermostat (NVE ensemble)
int const tpTermNose    = 1;    // Nose-Hoover
int const tpTermRadi    = 2;    // radiative 

// thermostats structure
struct TStat
{
    int type;
    double Temp;    //target temperature
    double tKin;    // (dp) target kinetic energy, according to temperature (sigma in DL_POLY source)  (dp) - derived parameter
    double tau;    // termostat relaxation time
    int step;       // number of steps between applying thermostat (for some types of tstat)

    //for Nose-Hoover
    //double nu = 0.005;
    //double l0 = 0.05;
    double chit;
    double conint;
    double qMass;       // qMass = 2*tKin*tauT^2;
    double rQmass;      // 1/Qmass = 1/(2*tKin*tauT^2)
    double qMassTau2;   // qmass/tauT^2 = 2*tKin

    // for radiative thermostat
    double mxEng;       // maximal energy of photon
    double* photons;    // array of photon energies
    // random vector of unit length, their projections
    double* randVx;   
    double* randVy;
    double* randVz;

    // unit vectors
    double* uvectX;
    double* uvectY;
    double* uvectZ;
};

int read_tstat(FILE* f, TStat* tstat, int nAt);
void gauss_temp(Atoms* atm, Spec* spec, TStat* tstat, Sim* sim); // set velocities according to gaussian
double tstat_nose(Atoms *atm, Sim *sim, TStat *tstat);  // apply Nose-Hoover tstat
void free_tstat(TStat* tstat);


// for radiationess thermostat


#endif // TEMPERATURE_H
