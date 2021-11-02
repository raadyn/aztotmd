// UNIT FOR ELECTROSTATIC CALCULATIONS
//   ...also SINCOS function is also described here
//#include <stdio.h>   // FILE, fprintf, scanf
#include <string.h>  //strcmp
#include <stdlib.h>  // malloc, alloc, rand, NULL
#include <math.h>   // log, sqrt

#include "const.h"
#include "dataStruct.h"  // Sim, Box, Atoms ....
#include "utils.h"  // int_size, pointer_size, etc...
#include "elec.h"

// read setting for electrostatic calculations (return success or not)
int read_elec(FILE *f, Elec *elec, Field *fld)
{
  char s[5];
  int res = 1;

  if (find_str(f, " elec %s", s))
  {
     if (strcmp(s, "none") == 0) // no electrostatic
     {
        elec->type = tpElecNone;
        elec->rReal = 0.0;
        if (fld->charged_spec)
            printf("WARNING[b003] The species have charges, but electrostatic directive is none. Charges will be ignored!\n");
     }
     else if (strcmp(s, "dir") == 0) // direct Coulomb
     {
         elec->type = tpElecDir;
         fscanf(f, " %lf ", &elec->rReal);
         elec->rReal *= r_scale;
     }
     else if (strcmp(s, "pme") == 0) // particle mesh ewald
     {
        elec->type = tpElecEwald;
        fscanf(f, " %lf %lf %d %d %d", &elec->rReal, &elec->alpha, &elec->kx, &elec->ky, &elec->kz);
        elec->rReal *= r_scale;
     }
     else if (strcmp(s, "fenn") == 0) // Fennel
     {
         elec->type = tpElecFennel;
         fscanf(f, " %lf %lf", &elec->rReal, &elec->alpha);
         elec->rReal *= r_scale;
     }
     else
     {
        printf("ERROR[404]: Unknown type of electrostatic calculations: %s.\n", s);
        res = 0;
     }

     if (!fld->charged_spec && elec->type)
     {
         printf("WARNING[b004] The species do not have charges, but electrostatic directive is %s. The electrostatic type is switched to none!\n", s);
         elec->type = tpElecNone;
     }
     elec->r2Real = elec->rReal * elec->rReal;

  }
  else
  {
      printf("ERROR[401]: electrostatic calculations are not specified in control.txt. Use 'elec' directive!\n");
      res = 0;
  }

  return res;
}

void init_ewald(Elec *elec, Box *bx, Sim *sim, Atoms *atm)
// create arrays for ewald recipoal space calculation:
//   el - exp(i 2pi x/a * l); em - exp (i 2 pi y/b * m); en - exp (i 2pi z/c * n)
//   -c - cosinus (or real) part; -s - sinus (or imaginary) part
//   lm = el * em; ck = q * el * em * en

{
   int i;
   //double **arr, **arr1;
   int Nat = atm->nAt;
   //int kx = elec->kx;
   int ky = elec->ky;
   int kz = elec->kz;

   // create elc and els arrays: only 2 elements as we will fill [0..N][0] elements by [0..N][k] elements
   elec->elc = (double**)malloc(Nat * pointer_size);
   elec->els = (double**)malloc(Nat * pointer_size);
   //arr = *elc;
   //arr1 = *els;
   for (i = 0; i < Nat; i++)
    {
      elec->elc[i] = (double*)malloc(2 * double_size);
      elec->els[i] = (double*)malloc(2 * double_size);
    }

   // create emc and ems arrays for every ky
   elec->emc = (double**)malloc(Nat * pointer_size);
   elec->ems = (double**)malloc(Nat * pointer_size);
   //arr = *emc;
   //arr1 = *ems;
   for (i = 0; i < Nat; i++)
    {
      elec->emc[i] = (double*)malloc(ky * double_size);
      elec->ems[i] = (double*)malloc(ky * double_size);

      // constant part:  exp(i * 2pi * 0 * m)
      elec->emc[i][0] = 1.0;
      elec->ems[i][0] = 0.0;
    }

   // create enc and ens arrays for every kz
   elec->enc = (double**)malloc(Nat * pointer_size);
   elec->ens = (double**)malloc(Nat * pointer_size);
   //arr = *enc;
   //arr1 = *ens;
   for (i = 0; i < Nat; i++)
    {
      elec->enc[i] = (double*)malloc(kz * double_size);
      elec->ens[i] = (double*)malloc(kz * double_size);

      // constant part:  exp(i * 2pi * 0 * n)
      elec->enc[i][0] = 1.0;
      elec->ens[i][0] = 0.0;
    }

   // create lmc, lms, ckc, cks buffer-arrays:
   elec->lmc = (double*)malloc(Nat * double_size);
   elec->lms = (double*)malloc(Nat * double_size);
   elec->ckc = (double*)malloc(Nat * double_size);
   elec->cks = (double*)malloc(Nat * double_size);
}
// end 'init_ewald' function

void init_elec(Elec *elec, Box *bx, Sim *sim, Atoms *atm)
// allocate elec arrays according to selected type of electrostatic calculations
{
    switch (elec->type)
    {
       case tpElecEwald:
         init_ewald(elec, bx, sim, atm);
         break;
    }
}
// end 'init_elec' function

double ewald_const(Atoms *atm, Spec *sp, Elec *elec, Box *box)
// return constant part of Columbic potential energy via Ewald method
//   (!) need to be recalculated only then volume or summ(q) are changed
{
   int i;
   double q;
   double sq = 0.0;
   double eng = 0.0;

   for (i = 0; i < atm->nAt; i++)
     {
        q = sp[atm->types[i]].charge;
        sq += q;
        eng += q*q;
     }

   eng *= (-1.0) * elec->alpha / sqrtpi;  // constant part of Ewald summation
   q = -0.5 * pi * (sq * sq / elec->alpha / elec->alpha) * box->rvol;

   return Fcoul_scale * (eng + q) / elec->eps;
}
// end 'ewald_const' function

void ewald_rec(Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim)
// calculate reciprocal part of Ewald summ and corresponding forces
{
   int i, l, m, n;
   int mmin = 0;
   int nmin = 1;
   double x, ch; // temporary variables (for complex *=,  for charge)
   double rkx, rky, rkz, rk2, akk, eng, sumC, sumS;
   int Nat = atm->nAt;
   int kx = elec->kx;
   int ky = elec->ky;
   int kz = elec->kz;
   Spec *sp = field->species;
   // double rkcut2 = sim->rkcut2;  //! Â DL_POLY ýòî âû÷èñëÿåìàÿ âåëè÷èíà
   //printf("ewald_rec Nat=%d kx=%d ky=%d kz=%d  rkut2=%f\n", Nat, kx, ky, kz, ew->rkcut2);

   double **elc = elec->elc;
   double **els = elec->els;
   double **emc = elec->emc;
   double **ems = elec->ems;
   double **enc = elec->enc;
   double **ens = elec->ens;

   double *lmc = elec->lmc;
   double *lms = elec->lms;
   double *ckc = elec->ckc;
   double *cks = elec->cks;

   eng = 0.0;
   //! òóò âñ¸ âåðíî äëÿ ïðÿìîóãîëüíîé ãåîìåòðèè. Åñëè ÿ÷åéêà áóäåò êðèâàÿ, êîä íóæíî ïðàâèòü
   for (i = 0; i < Nat; i++)
     {
        // exp (i 2pi * 0 * l) for em- and en- arrays this step omitted as they set in 'init_ewald'
        //   el- arrays need to refresh (according cycle by l)
        elc[i][0] = 1.0;
        els[i][0] = 0.0;

        // exp (ikr)
        sincos(twopi * atm->xs[i] * bx->ra, els[i][1], elc[i][1]);
        sincos(twopi * atm->ys[i] * bx->rb, ems[i][1], emc[i][1]);
        sincos(twopi * atm->zs[i] * bx->rc, ens[i][1], enc[i][1]);
        //printf("a=%f  sin:%f  cos: %f\n", twopi * atm[i].x * bx->ra, els[i][1], elc[i][1]);
     }

    // fil exp(iky) array by complex multiplication
    for (l = 2; l < ky; l++)
      for (i = 0; i < Nat; i++)
        {
           emc[i][l] = emc[i][l-1] * emc[i][1] - ems[i][l-1] * ems[i][1];
           ems[i][l] = ems[i][l-1] * emc[i][1] + emc[i][l-1] * ems[i][1];
        }

    // fil exp(ikz) array by complex multiplication
    for (l = 2; l < kz; l++)
      for (i = 0; i < Nat; i++)
        {
           enc[i][l] = enc[i][l-1] * enc[i][1] - ens[i][l-1] * ens[i][1];
           ens[i][l] = ens[i][l-1] * enc[i][1] + enc[i][l-1] * ens[i][1];
        }

    // MAIN LOOP OVER K-VECTORS:
    for (l = 0; l < kx; l++)
      {
         rkx = l * twopi * bx->ra; // only for rect geometry!
         // move exp(ikx[l]) to ikx[0] for memory saving (ikx[i>1] are not used)
         if (l == 1)
           for (i = 0; i < Nat; i++)
             {
                elc[i][0] = elc[i][1];
                els[i][0] = els[i][1];
             }
         else if (l > 1)
           for (i = 0; i < Nat; i++)
             {
                // exp(ikx[0]) = exp(ikx[0]) * exp(ikx[1])
                x = elc[i][0];
                elc[i][0] = x * elc[i][1] - els[i][0] * els[i][1];
                els[i][0] = els[i][0] * elc[i][1] + x * els[i][1];
             }

         //ky - loop:
         for (m = mmin; m < ky; m++)
           {
              rky = m * twopi * bx->rb;
              //fil temp arrays for keeping e^ikx * e^iky
              if (m >= 0)
                for (i = 0; i < Nat; i++)
                  {
                     lmc[i] = elc[i][0] * emc[i][m] - els[i][0] * ems[i][m];
                     lms[i] = els[i][0] * emc[i][m] + ems[i][m] * elc[i][0];
                  }
              else // for negative ky give complex adjustment to positive ky:
                for (i = 0; i < Nat; i++)
                  {
                     lmc[i] = elc[i][0] * emc[i][-m] + els[i][0] * ems[i][-m];
                     lms[i] = els[i][0] * emc[i][-m] - ems[i][-m] * elc[i][0];
                  }

              //kz - loop:
              for (n = nmin; n < kz; n++)
                {
                   rkz = n * twopi * bx->rc;
                   // rk2 = (2pi * l / a)^2 + (2pi * m / b)^2 + (2pi * n / c)^2   !only for rectangular geometry!
                   //rk2 = twopi2 * (l * l * bx->ra2 + m * m * bx->rb2 + n * n * bx->rc2);
                   rk2 = rkx * rkx + rky * rky + rkz * rkz;
                   //! ó íàñ cuttof è rk2 âîçìîæíî â ðàçíûõ åäèíèöàõ èçìåðåíèÿ, íàäî ýòî ïðîâåíòèëèðîâàòü
                   //printf("rk2 * rkcut2 :  %f  *  %f\n", rk2, ew->rkcut2);
                   if (rk2 < elec->rkcut2) // cutoff
                     {
                        // calculate summ(ikr*q[iAt])
                        sumC = 0; sumS = 0;
                        if (n >= 0)
                          for (i = 0; i < Nat; i++)
                            {
                               ch = sp[atm->types[i]].charge;

                               ckc[i] = ch * (lmc[i] * enc[i][n] - lms[i] * ens[i][n]);
                               cks[i] = ch * (lms[i] * enc[i][n] + lmc[i] * ens[i][n]);

                               sumC += ckc[i];
                               sumS += cks[i];
                            }
                        else // for negative kz give complex adjustment to positive kz:
                          for (i = 0; i < Nat; i++)
                            {
                               ch = sp[atm->types[i]].charge;

                               ckc[i] = ch * (lmc[i] * enc[i][-n] + lms[i] * ens[i][-n]);
                               cks[i] = ch * (lms[i] * enc[i][-n] - lmc[i] * ens[i][-n]);

                               sumC += ckc[i];
                               sumS += cks[i];
                            }

                        //energy and force calculation!
                        akk = exp(rk2*elec->mr4a2) / rk2;
                        eng += akk * (sumC * sumC + sumS * sumS);
                        //printf("akk=%f, sumC=%f,  sumS=%f\n", akk, sumC, sumS);

                        for (i = 0; i < Nat; i++)
                          {
                             //! rkx = 2pi * l / a - ïîñìîòðåòü ÷òî áûñòðåå, ââîäèòü rkx, rky è rkz èëè ñäåëàòü êàê ùàñ
                             x = akk * (cks[i] * sumC - ckc[i] * sumS);
                             // in DLPOLY there is a factor  4 * pi / V / (4piee0):
                             x *= elec->scale2;
                             atm->fxs[i] += rkx * x;
                             atm->fys[i] += rky * x;
                             atm->fzs[i] += rkz * x;
                          }

                        //printf("eng=%f, \n", eng);
                     }
                } // end n-loop (over kz-vectors)

              nmin = 1 - kz;

           } // end m-loop (over ky-vectors)

         mmin = 1 - ky;

      }  // end l-loop (over kx-vectors)


    //printf("rvol=%f, coul-scale=%f,  eng=%f\n", bx->rvol, Fcoul_scale, eng);

    //! íàäî åù¸ äîáàâèòü ýòè ïîñòîÿííûå ÷ëåíû
    //printf("ewald_rec=%f\n", scale * eng);
    sim->engElec2 = elec->scale * eng;
}
// end 'ewald_rec' function

void no_elec(Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim)
// empty function if no electorstatic
{

}

double coul_iter(double r2, double &r, double chprd, double alpha, double &eng)
// calculate energy and return force of real part Coulombic iteraction via Ewald procedure
//  r2 - square of distance, chi, chj - charges of i and j particles
//  chprd - production of charges
//  eng - for saving energy
//! òóò íàäî åù¸ ââåñòè ýïñèëîí â çàêîí êóëîíà
{
   //double r;
   double ar; //alpha * r
   double erfcar; // erfc(alpha * r);
   double kqq = chprd * Fcoul_scale; // q[i]*q[j]*1/4pie0;

   //brute force calc:
   if (r == 0)
     r = sqrt(r2);  // if r is unknown, calculate it
   //! íàäî ïðåäóñìîòðåòü âàðèàíò, êîãäà r2 íåèçâåñòíî
   ar = alpha * r;
   erfcar = erfc(ar);

   //printf("r2=%f  ar=%f  effcar=%f  E=%4.2E,    F=%4.2E\n", r2, ar, erfcar, kqq * erfcar /  r, kqq / r / r2 * (erfcar + 2 * ar / sqrtpi * exp(-ar*ar)));

   eng += kqq * erfcar /  r;
   //return 0;
   return kqq / r / r2 * (erfcar + 2 * ar / sqrtpi * exp(-ar*ar));
}
// end 'coul_iter' function

void prepare_elec(Atoms *atm, Field *field, Elec *elec, Sim *sim, Box *bx)
{
   int kx = elec->kx;
   int ky = elec->ky;
   int kz = elec->kz;

   if (elec->type == tpElecEwald)
   {
       //constants definition:
       elec->daipi2 = 2 * elec->alpha / sqrtpi;
       elec->scale = 2 * twopi * bx->rvol * Fcoul_scale / elec->eps;
       elec->scale2 = 2 * elec->scale;
       elec->mr4a2 = -0.25 / elec->alpha / elec->alpha; // -1/(4a^2)

       elec->rkcut = kx * bx->ip1;
       //! íàéòè å¸ îäèí ðàç è âïåð¸ä - íàèìåíüøàÿ ñðåäè ka * ipa
       if (elec->rkcut > ky * bx->ip2)
         elec->rkcut = ky * bx->ip2;
       //printf("ip2=%f  2pi=%f rkcut=%f\n", bx->ip2, twopi, ew->rkcut);
       if (elec->rkcut > kz * bx->ip3)
         elec->rkcut = kz * bx->ip3;
       //printf("ip3=%f  2pi=%f rkcut=%f\n", bx->ip3, twopi, ew->rkcut);
       elec->rkcut *= twopi * 1.05; // according to DL_POLY source
       elec->rkcut2 = elec->rkcut * elec->rkcut;
       //printf("ip1=%f  2pi=%f rkcut=%f rkcut2=%f\n", bx->ip1, twopi, ew->rkcut, ew->rkcut2);

       sim->engElec1 = ewald_const(atm, field->species, elec, bx); // constant part of Ewald
   }
   else if (elec->type == tpElecFennel)  //! âîîáùå òóò íàäî ïîñòàâèòü switch, à åù¸ âîîáùå - âûíåñòè â read
   {
       double aRc = elec->alpha * elec->rReal;
       elec->daipi2 = 2 * elec->alpha / sqrtpi;
       elec->scale = erfc(aRc) / elec->rReal;
       elec->scale2 = erfc(aRc) / elec->r2Real + elec->daipi2 * exp(-aRc*aRc) / elec->rReal;
   }
}

void direct_ewald(Spec *sp, int it, int jt, double r2, double r, Elec *elec, Sim *sim, double &force)
{
    if (sp[it].charged)  //! åù¸ îäíà âîçìîæíàÿ îïòèìèçàöèÿ, ñîñòàâèòü ìàññèâ ïðîèçâåäåíèé çàðÿäîâ [i][j]
      if (sp[jt].charged) //! õîòÿ íå ôàêò, ÷òî ïåðåìåùåíèå ïî äâóìåðíîìó ìàññèâó áóäåò áûñòðåå, ÷åì äâà IF
        force += coul_iter(r2, r, sp[it].charge * sp[jt].charge, elec->alpha, sim->engElec3);
}

void direct_coul(Spec* sp, int it, int jt, double r2, double r, Elec* elec, Sim* sim, double& force)
{
    if (sp[it].charged)  //! åù¸ îäíà âîçìîæíàÿ îïòèìèçàöèÿ, ñîñòàâèòü ìàññèâ ïðîèçâåäåíèé çàðÿäîâ [i][j]
        if (sp[jt].charged) //! õîòÿ íå ôàêò, ÷òî ïåðåìåùåíèå ïî äâóìåðíîìó ìàññèâó áóäåò áûñòðåå, ÷åì äâà IF
        {
            double kqq = sp[it].charge * sp[jt].charge * Fcoul_scale; // q[i]*q[j]*1/4pie0;
            //brute force calc:
            if (r == 0)
                r = sqrt(r2);  // if r is unknown, calculate it
            //! ïðåäóñìîòðåòü âàðèàíò, êîãäà íåèçâåñòåí r2?
            sim->engElec3 += kqq / r;       //! à ïî÷åìó Elec3 ???
            force += kqq / r / r2;
        }
}

void fennel(Spec* sp, int it, int jt, double r2, double r, Elec* elec, Sim* sim, double& force)
{
    if (sp[it].charged)  //! åù¸ îäíà âîçìîæíàÿ îïòèìèçàöèÿ, ñîñòàâèòü ìàññèâ ïðîèçâåäåíèé çàðÿäîâ [i][j]
        if (sp[jt].charged) //! õîòÿ íå ôàêò, ÷òî ïåðåìåùåíèå ïî äâóìåðíîìó ìàññèâó áóäåò áûñòðåå, ÷åì äâà IF
        {
            if (r == 0)
                r = sqrt(r2);  // if r is unknown, calculate it
            double ir = 1.0 / r;    // inverted r
            double kqq = sp[it].charge * sp[jt].charge * Fcoul_scale; // q[i]*q[j]*1/4pie0;
            double ar = elec->alpha * r; //alpha * r
            double erfcar = erfc(ar);
            sim->engElec3 += kqq * (erfcar * ir - elec->scale + elec->scale2 * (r - elec->rReal));        //! à ïî÷åìó Elec3 ???
            force += kqq * ir * ((erfcar / r2 + elec->daipi2 * exp(-ar * ar) * ir) - elec->scale2);
        }
}


void none_elec(Spec *sp, int it, int jt, double r2, double r, Elec *elec, Sim *sim, double &force)
{

}

void free_ewald(Elec *elec, Atoms *atm)
// free memory from ewald algorithm
//   el - exp(i 2pi x/a * l); em - exp (i 2 pi y/b * m); en - exp (i 2pi z/c * n)
//   -c - cosinus (or real) part; -s - sinus (or imaginary) part
//   lm = el * em; ck = q * el * em * en
{
   int i;
   //double **arr, **arr1;

   // free elc and els arrays: only 2 elements as we will fill [0..N][0] elements by [0..N][k] elements
   for (i = 0; i < atm->nAt; i++)
     {
        free(elec->elc[i]);
        free(elec->els[i]);
        free(elec->emc[i]);
        free(elec->ems[i]);
        free(elec->enc[i]);
        free(elec->ens[i]);
     }

   free(elec->elc);
   free(elec->els);
   free(elec->emc);
   free(elec->ems);
   free(elec->enc);
   free(elec->ens);

   // free lmc, lms, ckc, cks buffer-arrays:
   free(elec->lmc);
   free(elec->lms);
   free(elec->ckc);
   free(elec->cks);
}
// end 'free_ewald' function

void free_elec(Elec *elec, Atoms *atm)
// deallocate arrays for electrostatic calculations
{
    switch (elec->type)
    {
       case tpElecEwald:
         free_ewald(elec, atm);
         break;
    }
}
// end 'free_elec' function
