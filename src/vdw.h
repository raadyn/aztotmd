#ifndef VDW_H
#define VDW_H

/*
//constants:
//  names for user (in species.txt)
const char lj_name[5] = "lnjs";  // lennard-jones
const char bh_name[5] = "bkhm";  // buckingham          U = A exp(-r/ro) - C/r^6
const char CuCl_name[5] = "p746";  // potential 7-4-6   U = A/r^7 - B/r^4 - C/r^6
const char BHM_name[5] = "bmhs"; // Born-Mayer-Huggins U = Aexp[B(s-r)] - C/r^6 - D/r^8
*/

// identifers
const int lj_type = 1;  // U = 4e[(s/r)^12 - (s/r)^6]
const int bh_type = 2;  // U = A exp(-r/ro) - C/r^6
const int CuCl_type = 3;   // U = A/r^7 - B/r^4 - C/r^6
const int BHM_type = 4;    // U = Aexp[B(s-r)] - C/r^6 - D/r^8
const int elin_type = 5;	// U = A*exp(-r/ro) + c*x
const int einv_type = 6;	// U = A*exp(-r/ro) - c/x
const int surk_type = 7;	// potential derived by Platon Surkov

// names[type]
const char vdw_names[8][20] = {"", "Lenard-Jones", "Buckingham", "CuCl(7-4-6)", "Born-Mayer-Huggins", "exp+line", "exp-inv", "Platon Surkov"};


int read_vdw(int id, FILE *f, Field *field, Sim *sim);

//double vdw_iter(double r2, VdW *vdw, double &eng);
// calculate energy and return force of vdw iteraction
//  r2 - square of distance

// pair potential functions:
double fer_lj(double r2, double &r, VdW *vdw, double &eng);
double fe_lj(double r2, VdW *vdw, double &eng);
double e_lj(double r2, VdW *vdw);
double er_lj(double r2, double r, VdW *vdw);
double fer_buckingham(double r2, double &r, VdW *vdw, double &eng);
double fe_buckingham(double r2, VdW *vdw, double &eng);
double e_buckingham(double r2, VdW *vdw);
double er_buckingham(double r2, double r, VdW *vdw);
double fer_bhm(double r2, double &r, VdW *vdw, double &eng);
double fe_bhm(double r2, VdW *vdw, double &eng);
double e_bhm(double r2, VdW *vdw);
double er_bhm(double r2, double r, VdW *vdw);
double fer_746(double r2, double &r, VdW *vdw, double &eng);
double fe_746(double r2, VdW *vdw, double &eng);
double e_746(double r2, VdW *vdw);
double er_746(double r2, double r, VdW *vdw);

#endif /* VDW_H */
