//MODULE vdw.cpp
//with HEADER vdw.h CONTAINS CONST FOR VDW READING AND APPLYING
#include <math.h>
#include <string.h>
#include <stdio.h>   //! temp, for loggin and debuging


#include "const.h"
#include "dataStruct.h"  // Sim, Box, Atoms ....
#include "vdw.h"
#include "sys_init.h"   // twospec_by_name
#include "utils.h"  // min, max

// Here and next fer_, fe_ and e_ functions must contain the same code
//   prefexis fer_, fe_ and e_ means force and energy calculation by r also, force and energy and energy
double fer_lj(double r2, double &r, VdW *vdw, double &eng)
// calculate force and energy by Lennard-Jones pair potential: U = 4e[(s/r)^12 - (s/r)^6]
//  r2 - square of distance, vdw - parameters (the same for next PP functions)
{
    double r2i = 1.0 / r2;
    double sr2 = vdw->p1 * r2i;
    double sr6 = sr2 * sr2 * sr2;

    eng += vdw->p0 * sr6 * (sr6 - 1.0);
    return vdw->p2 * r2i * sr6 * (2.0 * sr6 - 1.0);
}

double fe_lj(double r2, VdW *vdw, double &eng)
// calculate force and energy by Lennard-Jones pair potential: U = 4e[(s/r)^12 - (s/r)^6]
//  r2 - square of distance, vdw - parameters (the same for next PP functions)
{
    double r2i = 1.0 / r2;
    double sr2 = vdw->p1 * r2i;
    double sr6 = sr2 * sr2 * sr2;

    eng += vdw->p0 * sr6 * (sr6 - 1.0);
    return vdw->p2 * r2i * sr6 * (2.0 * sr6 - 1.0);
}

double e_lj(double r2, VdW *vdw)
// calculate energy by Lennard-Jones pair potential
{
    double r2i = 1.0 / r2;
    double sr2 = vdw->p1 * r2i;
    double sr6 = sr2 * sr2 * sr2;

    return vdw->p0 * sr6 * (sr6 - 1.0);
}

double er_lj(double r2, double r, VdW *vdw)
// calculate energy by Lennard-Jones pair potential
{
    double r2i = 1.0 / r2;
    double sr2 = vdw->p1 * r2i;
    double sr6 = sr2 * sr2 * sr2;

    return vdw->p0 * sr6 * (sr6 - 1.0);
}

double fer_buckingham(double r2, double &r, VdW *vdw, double &eng)
// calculate force and energy (with r) by Buckingham pair potential: U = A exp(-r/ro) - C/r^6
{
   double r2i = 1.0 / r2;
   double r4i = r2i * r2i;
   if (r == 0.0)    // calculate if unkonwn (zero) and use otherwise
     r = sqrt(r2);

   eng += vdw->p0 * exp(-r/vdw->p1) - vdw->p2 * r4i * r2i;
   return vdw->p0 * exp(-r/vdw->p1) / r / vdw->p1 - 6.0 * vdw->p2 * r4i * r4i;
}

double fe_buckingham(double r2, VdW *vdw, double &eng)
// calculate force and energy by Buckingham pair potential: U = A exp(-r/ro) - C/r^6
{
   double r2i = 1.0 / r2;
   double r  = sqrt(r2);
   double r4i = r2i * r2i;

   eng += vdw->p0 * exp(-r/vdw->p1) - vdw->p2 * r4i * r2i;
   return vdw->p0 * exp(-r/vdw->p1) / r / vdw->p1 - 6.0 * vdw->p2 * r4i * r4i;
}

double e_buckingham(double r2, VdW *vdw)
// calculate energy by Buckingham pair potential: U = A exp(-r/ro) - C/r^6
{
   double r2i = 1.0 / r2;
   double r  = sqrt(r2);
   double r4i = r2i * r2i;

   return vdw->p0 * exp(-r/vdw->p1) - vdw->p2 * r4i * r2i;
}

double er_buckingham(double r2, double r, VdW *vdw)
// calculate energy by Buckingham pair potential: U = A exp(-r/ro) - C/r^6
{
   double r2i = 1.0 / r2;
   double r4i = r2i * r2i;

   return vdw->p0 * exp(-r/vdw->p1) - vdw->p2 * r4i * r2i;
}

double fer_bhm(double r2, double &r, VdW *vdw, double &eng)
// calculate force and energy (with r) by Born-Huggins-Maier pair potential: U = Aexp[B(s-r)] - C/r^6 - D/r^8
{
   double r2i = 1.0 / r2;
   double r4i = r2i * r2i;
   if (r == 0.0)    // calculate if unkonwn (zero) and use otherwise
     r = sqrt(r2);

   eng += vdw->p0 * exp(vdw->p1*(vdw->p2 - r)) - vdw->p3 * r4i * r2i - vdw->p4 * r4i * r4i;
   return vdw->p0 * vdw->p1 * exp(vdw->p1*(vdw->p2 - r)) / r  - 6.0 * vdw->p3 * r4i * r4i - 8.0 * vdw->p4 * r4i * r4i * r2i;
}

double fe_bhm(double r2, VdW *vdw, double &eng)
// calculate force and energy by Born-Huggins-Maier pair potential: U = Aexp[B(s-r)] - C/r^6 - D/r^8
{
   double r2i = 1.0 / r2;
   double r  = sqrt(r2);
   double r4i = r2i * r2i;

   eng += vdw->p0 * exp(vdw->p1*(vdw->p2 - r)) - vdw->p3 * r4i * r2i - vdw->p4 * r4i * r4i;
   return vdw->p0 * vdw->p1 * exp(vdw->p1*(vdw->p2 - r)) / r  - 6.0 * vdw->p3 * r4i * r4i - 8.0 * vdw->p4 * r4i * r4i * r2i;
}

double e_bhm(double r2, VdW *vdw)
// calculate energy by Born-Huggins-Maier pair potential: U = Aexp[B(s-r)] - C/r^6 - D/r^8
{
   double r2i = 1.0 / r2;
   double r  = sqrt(r2);
   double r4i = r2i * r2i;

   return vdw->p0 * exp(vdw->p1*(vdw->p2 - r)) - vdw->p3 * r4i * r2i - vdw->p4 * r4i * r4i;
}

double er_bhm(double r2, double r, VdW *vdw)
// calculate energy by Born-Huggins-Maier pair potential: U = Aexp[B(s-r)] - C/r^6 - D/r^8
{
   double r2i = 1.0 / r2;
   double r4i = r2i * r2i;

   return vdw->p0 * exp(vdw->p1*(vdw->p2 - r)) - vdw->p3 * r4i * r2i - vdw->p4 * r4i * r4i;
}

double fer_746(double r2, double &r, VdW *vdw, double &eng)
// calculate force and energy (with r) by "746" pair potential (Staddford et.al.[]): U = A/r^7 - B/r^4 - C/r^6
{
   double r2i = 1.0 / r2;
   double r4i = r2i * r2i;
   double ri;
   if (r == 0.0)    // calculate from (r2i) if r unkonwn (zero) and calculate from r otherwise
     ri = sqrt(r2i);
   else
     ri = 1.0 / r;

   eng += r4i * (vdw->p0 * r2i * ri - vdw->p1 - vdw->p2 * r2i);
   return r4i * r2i * (7.0 * vdw->p0 * r2i * ri - 4.0 * vdw->p1 - 6.0 * vdw->p2 * r2i);
}


double fe_746(double r2, VdW *vdw, double &eng)
// calculate force and energy by "746" pair potential (Staddford et.al.[]): U = A/r^7 - B/r^4 - C/r^6
{
   double r2i = 1.0 / r2;
   double ri  = sqrt(r2i);
   double r4i = r2i * r2i;

   eng += r4i * (vdw->p0 * r2i * ri - vdw->p1 - vdw->p2 * r2i);
   return r4i * r2i * (7.0 * vdw->p0 * r2i * ri - 4.0 * vdw->p1 - 6.0 * vdw->p2 * r2i);
}

double e_746(double r2, VdW *vdw)
// calculate energy by "746" pair potential (Staddford et.al.[]): U = A/r^7 - B/r^4 - C/r^6
{
   double r2i = 1.0 / r2;
   double ri  = sqrt(r2i);
   double r4i = r2i * r2i;

   return r4i * (vdw->p0 * r2i * ri - vdw->p1 - vdw->p2 * r2i);
}

double er_746(double r2, double r, VdW *vdw)
// calculate energy by "746" pair potential (Staddford et.al.[]): U = A/r^7 - B/r^4 - C/r^6
{
   double r2i = 1.0 / r2;
   double ri  = 1.0 / r;
   double r4i = r2i * r2i;

   return r4i * (vdw->p0 * r2i * ri - vdw->p1 - vdw->p2 * r2i);
}
// END PP functions definition

const int nVdWType = 7;
const char vdw_abbr[nVdWType][5] = {"lnjs", "buck", "p746", "bmhs", "elin", "einv", "surk"};
// number of parameters for potential (0th element is reserved)
const int vdw_nparam[nVdWType + 1] = {0, 2, 3, 3, 5, 3, 3, 4};
//                                  none LJ  buck   746 BHM eline   einv surk

//define function types:
typedef double(*feng_r)(double r2, double &r, VdW *vdw, double &eng);
typedef double(*feng)(double r2, VdW *vdw, double &eng);
typedef double(*eng_r)(double r2, double r, VdW *vdw);
typedef double(*eng)(double r2, VdW *vdw);

const feng_r vdw_fer[nVdWType + 1]= {NULL, fer_lj, fer_buckingham, fer_746, fer_bhm, NULL};
const feng vdw_fe[nVdWType + 1] = {NULL, fe_lj, fe_buckingham, fe_746, fe_bhm, NULL };
const eng_r vdw_er[nVdWType + 1] = {NULL, er_lj, er_buckingham, er_746, er_bhm, NULL };
const eng vdw_e[nVdWType + 1] = {NULL, e_lj, e_buckingham, e_746, e_bhm, NULL };

// scale parameters constants
const double r4scale = r_scale * r_scale * r_scale * r_scale;
const double r6scale = r4scale * r_scale * r_scale;
const double r8scale = r4scale * r4scale;
// массивы для каждого параметра в парном потенциале. Элемент массива - тип потенциала
const double vdw_scale0[nVdWType + 1] = {0, 4*E_scale, E_scale/*A[eV]*/, E_scale * r_scale * r6scale/*A[eV*A^7]*/, E_scale/*A[eV]*/, E_scale/*A[eV]*/, E_scale/*A[eV]*/, E_scale * r_scale};
                             // for LJ p0 =  4*epsilon //(?)
const double vdw_scale1[nVdWType + 1] = {0, r_scale, r_scale, E_scale*r4scale/*B[eVA^4]*/, 1.0/r_scale/*B[1/A]*/, r_scale, r_scale, E_scale * r4scale * r_scale};
const double vdw_scale2[nVdWType + 1] = {0, 0.0, r6scale*E_scale/*C[eV*A^6]*/, E_scale*r6scale/*C[eV*A^6]*/, r_scale/*s[1/A]*/, E_scale / r_scale, E_scale * r_scale, 1.0};
const double vdw_scale3[nVdWType + 1] = {0, 0.0, 0.0, 0.0, E_scale*r6scale/*C[eV*A^6]*/, 0, 0, 1.0};
const double vdw_scale4[nVdWType + 1] = {0, 0.0, 0.0, 0.0, E_scale*r8scale/*D[eV*A^8]*/, 0, 0, 0};
                //                      none LJ  buck   746 BHM eline   einv

int vdwtype_by_name(char *name)
{
   int i;
   for (i = 0; i < nVdWType; i++)
     if (strcmp(name, vdw_abbr[i]) == 0)
       {
          return i + 1;
       }

   return 0;
}

int read_vdw(int id, FILE *f, Field *field, Sim *sim)
{
   int i, at1, at2, type;
   VdW pp;// , app;
   char aname[8], bname[8], cname[8];
   double* point_to_params[5];
   point_to_params[2] = &pp.p2;
   point_to_params[3] = &pp.p3;
   point_to_params[4] = &pp.p4;

   fscanf(f, " %8s %8s %8s %lf %lf %lf ", aname, bname, cname, &pp.r2cut, &pp.p0, &pp.p1);  
   type = vdwtype_by_name(cname);   // return id + 1
   if (!type)
   {
      printf("ERROR[006]! Unknown potential type (%s) in %d vdw-line\n", cname, id + 1);
      return 0;
   }

   for (i = 2; i < vdw_nparam[type]; i++)
     fscanf(f, " %lf", point_to_params[i]);

   if (!twospec_by_name(field, aname, bname, at1, at2))
   {
      printf("ERROR[005]! Unknown atom type in vdw-line: %s   %s   %s\n", aname, bname, cname);
      return 0;
   }

   pp.r2cut *= r_scale; // convert external length units to MD units
   field->minRvdw = min(field->minRvdw, pp.r2cut);
   field->maxRvdw = max(field->maxRvdw, pp.r2cut);

   // PREPARATION OF PAIR POTENTIAl
   // user enter radii, but prog need sqaure of radii
   pp.r2cut = pp.r2cut * pp.r2cut;
   pp.type = type;

   // set functions
   pp.eng = vdw_e[type];
   pp.eng_r = vdw_er[type];
   pp.feng = vdw_fe[type];
   pp.feng_r = vdw_fer[type];

   //all parameters need to be converted into MD units (and some similar preparation)
   pp.p0 *= vdw_scale0[type];
   pp.p1 *= vdw_scale1[type];
   pp.p2 *= vdw_scale2[type];
   pp.p3 *= vdw_scale3[type];
   pp.p4 *= vdw_scale4[type];

   if (type == lj_type)
   {
         pp.p1 = pp.p1 * pp.p1;  // sigma^2  in L-J pair potential
         pp.p2 = 6 * pp.p0;    // 24*epsilon for force calculation
         //p0 = 4e;  p1 = s^2;  p2 = 24e;
   }
   if (type == surk_type)
   {
       pp.use_radii = 1;

       // this potential is assymetric relatively particles swiching
       //app = pp;
       //app.p2 = pp.p3;
       //app.p3 = pp.p2;
   }
   else
       pp.use_radii = 0;

   // SAVE PAIR POTENTIAL
   field->pairpots[id] = pp;
   if (field->vdws[at1][at2] != NULL)
       printf("WARNING[002]: Pair potential between %s and %s redeclarated\n", aname, bname);
   field->vdws[at1][at2] = &field->pairpots[id];
   if (type != surk_type) // this potential is assymetric relatively particles swiching
     field->vdws[at2][at1] = &field->pairpots[id];
}

/*
double vdw_iter(double r2, VdW *vdw, double &eng)
// calculate energy and return force of vdw iteraction  (   Fx/dx = -(1/r)*dU(r)/dr   )
//  r2 - square of distance
{
   double r2i, sr2, sr6;
   double ri, r4i;
   double r;

   switch (vdw->type)
    {
       case lj_type:   // U = 4e[(s/r)^12 - (s/r)^6]
         r2i = 1.0 / r2;
         sr2 = vdw->p1 * r2i;
         sr6 = sr2 * sr2 * sr2;

         eng += vdw->p0 * sr6 * (sr6 - 1.0);
         return vdw->p2 * r2i * sr6 * (2.0 * sr6 - 1.0);
         //break; /// break после return не имеет смысла

       case bh_type:    // U = A exp(-r/ro) - C/r^6
         r2i = 1.0 / r2;
         r  = sqrt(r2);
         r4i = r2i * r2i;

         eng += vdw->p0 * exp(-r/vdw->p1) - vdw->p2 * r4i * r2i;
         return vdw->p0 * exp(-r/vdw->p1) / r / vdw->p1 - 6.0 * vdw->p2 * r4i * r4i;

       case BHM_type:    // U = Aexp[B(s-r)] - C/r^6 - D/r^8
         r2i = 1.0 / r2;
         r  = sqrt(r2);
         r4i = r2i * r2i;

         eng += vdw->p0 * exp(vdw->p1*(vdw->p2 - r)) - vdw->p3 * r4i * r2i - vdw->p4 * r4i * r4i;
         return vdw->p0 * vdw->p1 * exp(vdw->p1*(vdw->p2 - r)) / r  - 6.0 * vdw->p3 * r4i * r4i - 8.0 * vdw->p4 * r4i * r4i * r2i;


       case CuCl_type:   // U = A/r^7 - B/r^4 - C/r^6
         r2i = 1.0 / r2;
         ri  = sqrt(r2i);
         r4i = r2i * r2i;
         //double sr2 = vdw->p1 * r2i;
         //double sr6 = sr2 * sr2 * sr2;

         eng += r4i * (vdw->p0 * r2i * ri - vdw->p1 - vdw->p2 * r2i);
         return r4i * r2i * (7.0 * vdw->p0 * r2i * ri - 4.0 * vdw->p1 - 6.0 * vdw->p2 * r2i);
         //break; /// break после return не имеет смысла
    }
}
// end 'vdw_iter' function
*/

