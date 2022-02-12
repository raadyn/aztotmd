// UNIT for temperature (thermostats and etc)
#include <math.h>       // log, sqrt, fabs
#include <string.h>     // strcmp
#include <stdlib.h>     // malloc

#include "temperature.h"
#include "dataStruct.h"
#include "md_utils.h"  // gauss(), rand01
#include "utils.h"      // find_double
#include "const.h"      // kB, light

double prob2(double x, double y, double theta)
{
    double ty = theta * y;
    return (1 - x) * exp(y * theta) - (0.5 * ty * ty + ty + 1);
}

const double r6 = 1.0 / 6.0;
const double r24 = 0.25 * r6;       // 1/24 = 1/6 * 1/4

double prob4(double x, double y, double theta)
{
    double ty = theta * y;
    double ty2 = ty * ty;
    return (1 - x) * exp(y * theta) - (r24 * ty2 * ty2 + r6 * ty2 * ty + 0.5 * ty * ty + ty + 1);
}

double photon_engs(int n, double* engs, double T)
// fill an array of photon energies according to distrubution function:
// P = 1/C e^4 exp(-e/kT)
// return maximal value of energy
{
    //double h = 1 / n;
    //double r = rand01();
    //double k = r / h;
    const double eps = 1e-3;
    double theta = 1.0 / (kB * T);
    const int limit = 20;

    int i, k;
    double x, y, r, ra, rb, a, b;
    double mx = 0.0;     // max value of energy
    double(*func)(double x, double y, double theta) = prob4;

    FILE* f = fopen("photon_engs.dat", "w");
    for (i = 0; i < n; i++)
    {
        a = 0.0;
        b = 1.0;
        // сделаем защиту от бесконечного дальнейшего цикла, произведение ra*rb должно быть меньше нуля
        do
        {
            x = rand01();
            ra = func(x, 0.0, theta);
            rb = func(x, 1.0, theta);
        } while (ra * rb > 0);

        y = 0.5;
        r = func(x, y, theta);

        k = 0;
        while ((r > eps) || (r < -eps)) // при этих условиях y - то что нам нужно
        {
            if ((r * ra) < 0)
            {
                b = y;
                y = 0.5 * (a + y);
            }
            else //! вообще тут нужно проверить r * rc < 0 - но я так понимаю, других вариантов нет
            {
                a = y;
                y = 0.5 * (y + b);
            }
            r = func(x, y, theta);
            
            // protection from endless loop
            k++;
            if (k >= limit)
            {
                y = engs[i - 1];    // take a previous value
                break;
            }
        }
        fprintf(f, "%f\n", y);
        if (isnan(y))
            printf("y is nan!\n");
        else
        {
            engs[i] = y;
            mx = max(mx, y);
        }
    }
    fclose(f);
    return mx;
}

int read_tstat(FILE *f, TStat *tstat, int mxAt)
// read thermostat parameters from file and return success or not
// mxAt = maximal number of atoms
{
   int i, j, k, res = 1;
   char str[5];

   if (find_double(f, " temperature %lf ", tstat->Temp))
   {
       // read thermostat type
       fscanf(f, "%s", str);
       if (strcmp(str, "none") == 0)
         tstat->type = tpTermNone;
       else if (strcmp(str, "nose") == 0)
       {
         tstat->type = tpTermNose;
         fscanf(f, " %lf ", &tstat->tau);

         // simple preparation
         tstat->chit = 0.0;
         tstat->conint = 0.0;
       }
       else if (strcmp(str, "radi") == 0)   // radiative thermostate
       {
           if (fscanf(f, "%d", &tstat->step) != 1)
           {
               printf("ERROR[a002]: there is no step parameter for radiative thermostat!\n");
               return 0;
           }
           tstat->type = tpTermRadi;

           // photon energies, corresponding to desired temperature
           tstat->photons = (double*)malloc(mxAt * double_size);
           tstat->mxEng = photon_engs(mxAt, tstat->photons, tstat->Temp);

           double phi, theta, cost;
           int nAt2 = mxAt / 2;          // only for even number of atoms
           double ransq, ran1, ran2, ranh;

           tstat->randVx = (double*)malloc(mxAt * double_size);
           tstat->randVy = (double*)malloc(mxAt * double_size);
           tstat->randVz = (double*)malloc(mxAt * double_size);
           for (i = 0; i < nAt2; i++)
           {
               // old variant
               /*
               phi = rand01() * twopi;
               theta = rand01() * pi;
               sincos(theta, tstat->randVz[i], cost);
               sincos(phi, tstat->randVy[i], tstat->randVx[i]);
               tstat->randVy[i] *= cost;
               tstat->randVx[i] *= cost;
               */

               // new variant from Frenkel, p. 578
               ransq = 2.0;
               while (ransq > 1.0)
               {
                   ran1 = 1.0 - 2.0 * rand01();
                   ran2 = 1.0 - 2.0 * rand01();
                   ransq = ran1 * ran1 + ran2 * ran2;
               }
               ranh = 2.0 * sqrt(1.0 - ransq);
               tstat->randVx[i] = ran1 * ranh;
               tstat->randVy[i] = ran2 * ranh;
               tstat->randVz[i] = 1.0 - 2.0 * ransq;

               // in opposite direction
               tstat->randVx[i + nAt2] = tstat->randVx[i];
               tstat->randVy[i + nAt2] = tstat->randVy[i];
               tstat->randVz[i + nAt2] = tstat->randVz[i];
           }

           // try predified vectors:
           const int nTh = 16;
           const int nPhi = 32;   // number of theta and phi angles
           int nVect = nTh * nPhi * 2 * 3;
           tstat->uvectX = (double*)malloc(nVect * double_size);
           tstat->uvectY = (double*)malloc(nVect * double_size);
           tstat->uvectZ = (double*)malloc(nVect * double_size);
           k = 0;
           for (i = 0; i < nPhi; i++)
           {
               phi = (double)i / nPhi *  twopi;
               for (j = 0; j < nTh; j++)
               {
                   theta = (double)j / nTh * pi;// -0.5 * pi;
                   sincos(theta, tstat->uvectZ[k], cost);
                   sincos(phi, tstat->uvectY[k], tstat->uvectX[k]);
                   tstat->uvectY[k] *= cost;
                   tstat->uvectX[k] *= cost;
                   tstat->uvectX[k + 1] = -tstat->uvectX[k];
                   tstat->uvectY[k + 1] = -tstat->uvectY[k];
                   tstat->uvectZ[k + 1] = -tstat->uvectZ[k];
                   k += 2;
                   //k++;
               }
           }
           // следующий варианты для симметрии
           for (i = 0; i < nPhi; i++)
           {
               phi = (double)i / nPhi * twopi;
               for (j = 0; j < nTh; j++)
               {
                   theta = (double)j / nTh * pi;// -0.5 * pi;
                   sincos(theta, tstat->uvectY[k], cost);
                   sincos(phi, tstat->uvectZ[k], tstat->uvectX[k]);
                   tstat->uvectX[k] *= cost;
                   tstat->uvectZ[k] *= cost;
                   tstat->uvectX[k + 1] = -tstat->uvectX[k];
                   tstat->uvectY[k + 1] = -tstat->uvectY[k];
                   tstat->uvectZ[k + 1] = -tstat->uvectZ[k];
                   k += 2;
                   //k++;
               }
           }
           for (i = 0; i < nPhi; i++)
           {
               phi = (double)i / nPhi * twopi;
               for (j = 0; j < nTh; j++)
               {
                   theta = (double)j / nTh * pi;// -0.5 * pi;
                   sincos(theta, tstat->uvectX[k], cost);
                   sincos(phi, tstat->uvectY[k], tstat->uvectZ[k]);
                   tstat->uvectY[k] *= cost;
                   tstat->uvectZ[k] *= cost;
                   tstat->uvectX[k + 1] = -tstat->uvectX[k];
                   tstat->uvectY[k + 1] = -tstat->uvectY[k];
                   tstat->uvectZ[k + 1] = -tstat->uvectZ[k];
                   k += 2;
                   //k++;
               }
           }

           // verify
           printf("tot number of unit vectors=%d\n", k);
           double sx = 0.0, sy = 0.0, sz = 0.0;
           double sx2 = 0.0, sy2 = 0.0, sz2 = 0.0; // сравним квадраты чтобы посмотреть преимущественность направлений
           double sqr;
           for (i = 0; i < k; i++)
           {
               sqr = tstat->uvectX[i] * tstat->uvectX[i] + tstat->uvectY[i] * tstat->uvectY[i] + tstat->uvectZ[i] * tstat->uvectZ[i];
               //if (sqr != 1.0)
                 //  printf("wrong vector[%d]: %f %f %f (length^2=%f)\n", i, tstat->uvectX[i], tstat->uvectY[i], tstat->uvectZ[i], sqr);
               sx += tstat->uvectX[i];
               sy += tstat->uvectY[i];
               sz += tstat->uvectZ[i];
               sx2 += tstat->uvectX[i] * tstat->uvectX[i];
               sy2 += tstat->uvectY[i] * tstat->uvectY[i];
               sz2 += tstat->uvectZ[i] * tstat->uvectZ[i];
           }
           if ((sx != 0.0) || (sy != 0.0) || (sz != 0))
               printf("non zero vectors summ: %f %f %f\n", sx, sy, sz);
           printf("sqr summ: (%f %f %f)\n", sx2, sy2, sz2);
       }
       else
       {
         printf("ERROR[405]: unknown thermostat type!\n");
         res = 0;
       }
   }
   else
   {
       printf("ERROR[404]: temperature is not defined in control.txt file!\n");
       res = 0;
   }

   return res;
}
// end 'read_tstat' function

void gauss_temp(Atoms *atm, Spec *spec, TStat *tstat, Sim *sim)
// set velocities according to Gauss distribution
{
   int i;
   double k, m;

   // centre of mass
   double cmx = 0.0;
   double cmy = 0.0;
   double cmz = 0.0;

   // centre of momentum
   double cpx = 0.0;
   double cpy = 0.0;
   double cpz = 0.0;

   double totMass = 0.0;    // total mass

   double mean = 0.0;   //  mean velocity must be zero
   double stdev = 0.5;  // used standart deviation
   double kE = 0.0;

   // set random velocities and summarize them, momentum and mass
   for (i = 0; i < atm->nAt; i++)
   {
        atm->vxs[i] = gauss(stdev, mean);
        atm->vys[i] = gauss(stdev, mean);
        atm->vzs[i] = gauss(stdev, mean);
        m = spec[atm->types[i]].mass;

        cmx += atm->xs[i] * m;
        cmy += atm->ys[i] * m;
        cmz += atm->zs[i] * m;

        cpx += atm->vxs[i] * m;
        cpy += atm->vys[i] * m;
        cpz += atm->vzs[i] * m;
        totMass += m;
   }

   // total momentum must be zero:
   cmx /= totMass;
   cmy /= totMass;
   cmz /= totMass;
   cpx /= totMass;
   cpy /= totMass;
   cpz /= totMass;


   // correct velocities, calculate kinetic energy
   for (i = 0; i < atm->nAt; i++)
     {
        atm->vxs[i] -= cpx;
        atm->vys[i] -= cpy;
        atm->vzs[i] -= cpz;

        kE += spec[atm->types[i]].mass * (atm->vxs[i] * atm->vxs[i] + atm->vys[i] * atm->vys[i] + atm->vzs[i] * atm->vzs[i]);
     }
   kE *= 0.5;   // Ekin = 1/2 mv2

   // normalization on desired temperature:
   // tKin is a target kinetic energy (calculated according to temparature)
   //! maybe replace tKin into tstat structure?
   k = sqrt(tstat->tKin / kE);
   for (i = 0; i < atm->nAt; i++)
   {
      atm->vxs[i] *= k;
      atm->vys[i] *= k;
      atm->vzs[i] *= k;
   }

   //if something is wrong - recalculate
   if (isnan(atm->vxs[0]))
     gauss_temp(atm, spec, tstat, sim);
}
// end 'gauss_temp' function

double tstat_nose(Atoms *atm, Sim *sim, TStat *tstat)
// apply Nose-Hoover thermostat, return new kinetic energy
{
   int i;
   double kinE;
   double scale;

   tstat->chit += sim->tSt * (sim->engKin - tstat->tKin) * tstat->rQmass;
   scale = 1 - sim->tSt * tstat->chit;
   for (i = 0; i < atm->nAt; i++)
   {
       atm->vxs[i] *= scale;
       atm->vys[i] *= scale;
       atm->vzs[i] *= scale;
   }
   kinE = sim->engKin * scale * scale;
   tstat->conint += sim->tSt * tstat->chit * tstat->qMassTau2;
   tstat->chit += sim->tSt * (kinE - tstat->tKin) * tstat->rQmass; // new kinetic energy (отличие от первого действия этой процедуры)

   return kinE;
}
// end 'tstat_nose' function


// --- NOT USED ------

/*
void term_andersen()
//applying Andersen termostat
{
  tempA /= (3 * N); // instaneous tempertaure
  for (i = 0; i < N; i++)
    if (rand01() < tStNu)
    {
      kinE -= (atm[i].vx * atm[i].vx + atm[i].vy * atm[i].vy + atm[i].vz * atm[i].vz) * sp[atm[i].type].mass / 0.5;
      atm[i].vx = gauss(sigma, l0);
      atm[i].vy = gauss(sigma, l0);
      atm[i].vz = gauss(sigma, l0);
      kinE += (atm[i].vx * atm[i].vx + atm[i].vy * atm[i].vy + atm[i].vz * atm[i].vz) * sp[atm[i].type].mass / 0.5;
    }
}
*/

void free_tstat(TStat* tstat)
{
    switch (tstat->type)
    {
        case tpTermRadi:
            free(tstat->photons);
            free(tstat->randVx);
            free(tstat->randVy);
            free(tstat->randVz);

            free(tstat->uvectX);
            free(tstat->uvectY);
            free(tstat->uvectZ);
            break;
    }
}

