// UNIT of integrators (and related functions)
// azTotMD (by Anton Raskovalov)
#include <math.h>       // log, sqrt, fabs
#include <stdio.h>      // FILE

#include "const.h"      // pi, 2pi, Boltzman constant, etc...
#include "dataStruct.h" // Sim, Box, Atoms ....
#include "box.h"
#include "vdw.h"        // van der Waals interatcions
#include "elec.h"       // electrostatic procedures
#include "bonds.h"
#include "temperature.h"// termostates etc
#include "cell_list.h"  // cell list method

#include <string.h>     //! temp, for strcmp

void clear_force(Atoms *atm, Spec *sp, Sim *sim, Box *bx)
// set forces to zero (or according to external electric fieled / shifting procedure), reset neighbors (if needed)
{
  int i;
  int N = atm->nAt;

  for (i = 0; i < N; i++)
  {
     //clear neighbors list
     if (sim->ejtype)  //! now neighbor list is used only for electron jumping
       sim->nNbors[i] = 0;

      //external constant electric field
      atm->fxs[i] = -sp[atm->types[i]].charge * sim->Ux;   // F = -q dU/dx (dU/dx = Ux)
      atm->fys[i] = 0.0;

      //! shifting procedure (for special purposes)
      if (atm->xs[i] > sim->shiftX)
        atm->fzs[i] = sim->shiftVal;
      else
        atm->fzs[i] = 0.0;
  }
}
// end 'clear_force' function

void reset_chars(Sim *sim)
// reset systems characteristics as energies
{
   // reset some quantities
   sim->engVdW = 0.0;
   sim->engElec3 = 0.0;     // part of electrostatic energy which is calculated by pair function
   sim->engElecField = 0.0; // energy of external electric field
   sim->engBond = 0.0;      // energy of covalent bonds
   sim->engAngle = 0.0;     // energy of valent angles

   // don't need to reset:
   //engTot   - calculated every step
   //engKin   - calculated in integrate2
   //engElec1  - constant part of Ewald summation (calculated once and after changes in charge and volume)
   //engElec2   - calculated every timestep
   //engElecTot - calculated every timestep
   //engOwn - calculated once and then after changes in atoms composition
   //Temp - calculated every timestep
}
// end 'reset_chars' function

void calc_chars(Sim *sim, double &sim_time)
//  update system characteristics (with simulation time)
{
   sim_time += sim->tSt;
   sim->Temp = 2.0 * sim->engKin * sim->revDegFree * rkB;

   // all contributions to the electrostatic energy
   sim->engElecTot = sim->engElec1 + sim->engElec2 + sim->engElec3;
   sim->engTot = sim->engElecField + sim->engVdW + sim->engElecTot + sim->engKin + sim->engBond + sim->engAngle + sim->engOwn;
}
// end 'calc_chars' function

void save_neigh(int iat, int jat, double r, double r2, Sim *sim, Field *fld)
// save a neighbour and its type into neighbors list
{
   int Ni, Nj;

   if ((sim->nNbors[iat] < sim->maxNbors) || (sim->nNbors[jat] < sim->maxNbors))
   {
      if (r == 0.0) // flag that r is unknown, so we need to calculate r
        r = sqrt(r2);

      Ni = sim->nNbors[iat];
      Nj = sim->nNbors[jat];

      sim->nbors[iat][Ni] = jat;
      sim->nbors[jat][Nj] = iat;
      sim->distances[iat][Ni] = r;
      sim->distances[jat][Nj] = r;

      // the neighbor in VdW range
      if (r2 <= fld->maxR2vdw/*sim->mxRvdw2*/)
      {
          sim->tnbors[iat][Ni] |= 1 << bfDistVdW;  // bit flag "in vdw range"
      }

      // the neighbor in eJump range
      if (r2 <= sim->r2Elec)
      {
          sim->tnbors[iat][Ni] |= 1 << bfDistEjump;  // bit flag "in ejump range"
      }

      // flags are simmetric
      sim->tnbors[jat][Nj] = sim->tnbors[iat][Ni];

      // increase neighbors number
      sim->nNbors[iat]++;
      sim->nNbors[jat]++;
   }
   else
     printf("WARNING[111]: the maximal number of neighbors is reached\n");
}
// end 'save_neigh' function

void bonding(int i, int j, double r2, Atoms *atm, Field *field, Sim *sim)
// try to bond ith and jth particles
{
   // exclude double bonding
   if (atm->parents[i] != j)
       return;
   if (atm->parents[j] != i)
       return;

   //! maybe pass it and jt as parameters of function?
   int it = atm->types[i];
   int jt = atm->types[j];

   int btype = field->bonding_matr[it][jt];
   if (btype != 0)
       if (r2 < field->bindR2matrix[it][it])
       {
           create_bond(i, j, btype, atm, field);
       }
}
// end 'bonding' function

void pair_inter(int i, int j, Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim)
// pair interaction between ith and jth particles
{
   double r, r2, f;//, eng_vdw, eng_real;
   double dx, dy, dz;
   int it, jt;// , bt;
   VdW *vdw;
   Spec* sp = field->species;

   r2 = sqr_distance_proj(i, j, atm, bx, dx, dy, dz);
   if (r2 <= sim->r2Max)    // cuttoff
   {
      it = atm->types[i];
      jt = atm->types[j];

      r = 0.0;
      f = 0.0;

      // electrostatic contribution
      sim->pair_elec(sp, it, jt, r2, r, elec, sim, f);

      // van der Waals contribution
      vdw = field->vdws[it][jt];
      if (vdw != NULL)
        if (r2 <= vdw->r2cut)
          f += vdw -> feng_r(r2, r, vdw, sim->engVdW);

      // try to bond atoms (if needed)
      bonding(i, j, r2, atm, field, sim);

      //! protection from useless work
      if (f*f > 1e10)
      {
         printf("ERROR[413] pair_inter: force between pair %s[%d] and %s[%d] (r2=%f) is colossal(%f)\n", sp[it].name, i, sp[jt].name, j, r2, f);
         return;
      }

      // forces update
      atm->fxs[i] += f * dx;
      atm->fxs[j] -= f * dx;
      atm->fys[i] += f * dy;
      atm->fys[j] -= f * dy;
      atm->fzs[i] += f * dz;
      atm->fzs[j] -= f * dz;
   }
}
// end 'pair_inter' function

void pair_inter_lst(int i, int j, Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim)
// similar to pair_inter but with neighbor saving
{
   double r, r2, f;//, eng_vdw, eng_real;
   double dx, dy, dz;
   int it, jt;// , bt;
   VdW *vdw;
   Spec* sp = field->species;

   r2 = sqr_distance_proj(i, j, atm, bx, dx, dy, dz);
   if (r2 <= sim->r2Max)    // cuttoff
   {
      it = atm->types[i];
      jt = atm->types[j];

      r = 0.0;
      f = 0.0;

      // electrostatic contribution
      sim->pair_elec(sp, it, jt, r2, r, elec, sim, f);

      // van der Waals contribution
      vdw = field->vdws[it][jt];
      if (vdw != NULL)
        if (r2 <= vdw->r2cut)
          f += vdw -> feng_r(r2, r, vdw, sim->engVdW);

      // try to bond atoms (if needed)
      bonding(i, j, r2, atm, field, sim);

      //! protection from useless work
      if (f*f > 1e10)
      {
         printf("ERROR[414] pair_inter_lst: force between pair %s[%d] and %s[%d] (r2=%f) is colossal(%f)\n", sp[it].name, i, sp[jt].name, j, r2, f);
         return;
      }

      //neighbors saving
      save_neigh(i, j, r, r2, sim, field);

      // forces update
      atm->fxs[i] += f * dx;
      atm->fxs[j] -= f * dx;
      atm->fys[i] += f * dy;
      atm->fys[j] -= f * dy;
      atm->fzs[i] += f * dz;
      atm->fzs[j] -= f * dz;
   }
}
// end 'pair_inter_lst' function

void cell_list(Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim)
// process pairs with cell_list algorithm
{
   int iC, jC; // cell indexes
   int i, j; // atom indexes

   for (iC = 0; iC < sim->nHead; iC++)
   {
      i = sim->chead[iC];

      // loop by all atoms in the cell
      while (i >= 0)
      {
         // atoms in the same cell
         j = sim->clist[i]; // the next atom index
         while (j >= 0)
         {
            sim->pair(i, j, atm, field, elec, bx, sim);
            j = sim->clist[j];  // the next atom index
         }

         // atoms in neighbors cells (loop by cell neighbors)
         for (jC = 0; jC < sim->nHeadNeig[iC]; jC++)
         {
            j = sim->chead[sim->lstHNeig[iC][jC]];
            while (j >= 0)
            {
               sim->pair(i, j, atm, field, elec, bx, sim);
               j = sim->clist[j];
            }
         }
         //! there is verification (j < i) in some books, to avoid repeated calculations,
         //! but i think it is redundat, because particles in different cells are different
         //! and j = clist[i] for the first pair excludes repeats in the same cell
         i = sim->clist[i];
      } // end loop by i atom
   }
}
// end 'cell_list' function

void all_pairs(Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim)
// process all pairs
{
   int i, j;
   int N = atm->nAt;

   for (i = 0; i < N-1; i++)
     for (j = i + 1; j < N; j++)
     {
        sim->pair(i, j, atm, field, elec, bx, sim);
     }
}
// end 'all_pair' function

void integrate1(Atoms *atm, Spec *spec, Sim *sim, Box *box, TStat *tstat)
// the first part of Velocity Verlet integrator: v += 1/2*F/m*dt; x += v * dt
{
  int i, t;
  double tSt = sim->tSt;
  double rMASSxTstHalf;
  double charge;

  // thermostat applying
  //! replace with function (to remove if condition)
  if (tstat->type == tpTermNose)
    tstat_nose(atm, sim, tstat);

  for (i = 0; i < atm->nAt; i++)
  {
     t = atm->types[i];
     rMASSxTstHalf = spec[t].rMass_hdt;
     charge = spec[t].charge;

     //the first stage of velocity update:
     // v = v + f/m * 0.5dt
     atm->vxs[i] += rMASSxTstHalf * atm->fxs[i];
     atm->vys[i] += rMASSxTstHalf * atm->fys[i];
     atm->vzs[i] += rMASSxTstHalf * atm->fzs[i];

     //update coordinates:
     atm->xs[i] += atm->vxs[i] * tSt;
     atm->ys[i] += atm->vys[i] * tSt;
     atm->zs[i] += atm->vzs[i] * tSt;

     //apply periodic boundaries
     put_periodic(atm, i, spec, box);

     //external fields energy
     sim->engElecField += charge * atm->xs[i] * sim->Ux; // Eng = q * x * dU/dx
  }
}
// end 'integrate1' function

void integrate1_clst(Atoms *atm, Spec *spec, Sim *sim, Box *box, TStat *tstat)
// similar to integrate1 but with cell list saving
{
  int i, t, c;
  double tSt = sim->tSt;
  double charge;
  double rMASSxTstHalf;

  // thermostat applying
  if (tstat->type == tpTermNose)
    tstat_nose(atm, sim, tstat);

  // clear cell_list
  for (i = 0; i < sim->nHead; i++)
    sim->chead[i] = -1;

  for (i = 0; i < atm->nAt; i++)
  {
     t = atm->types[i];
     rMASSxTstHalf = spec[t].rMass_hdt;
     charge = spec[t].charge;

     //the first stage of velocity update:
     //  v = v + f/m * 0.5 dt
     atm->vxs[i] += rMASSxTstHalf * atm->fxs[i];
     atm->vys[i] += rMASSxTstHalf * atm->fys[i];
     atm->vzs[i] += rMASSxTstHalf * atm->fzs[i];

     //update coordinates:
     // x = x + v * dt
     atm->xs[i] += atm->vxs[i] * tSt;
     atm->ys[i] += atm->vys[i] * tSt;
     atm->zs[i] += atm->vzs[i] * tSt;

     // apply periodic boundaries
     put_periodic(atm, i, spec, box);

     //save the atom in cell list
     c = cell_index_sim(atm->xs[i], atm->ys[i], atm->zs[i], sim);
     sim->clist[i] = sim->chead[c];
     sim->chead[c] = i;

     //external field
     sim->engElecField += charge * atm->xs[i] * sim->Ux; // Eng = q * x * dU/dx
  }
}
// end 'integrate1_lst' function

double vectorized_int2(Atoms *atm, Spec *sp, Sim *sim)
// vectorized version of integrate2 function
{
  int i, k, ind, ind1;
  int N = atm->nAt;
  double c[2], m[2];
  double *vx, *vy, *vz, *fx, *fy, *fz;
  double res = 0.0;

  k = 0;
  for (i = 0; i < N / 2; i++)
  {
    ind = atm->types[k];
    ind1 = atm->types[k+1];
    c[0] = sp[ind].rMass_hdt;
    c[1] = sp[ind1].rMass_hdt;
    m[0] = sp[ind].mass;
    m[1] = sp[ind1].mass;
    vx = &(atm->vxs[k]);
    fx = &(atm->fxs[k]);
    vy = &(atm->vys[k]);
    fy = &(atm->fys[k]);
    vz = &(atm->vzs[k]);
    fz = &(atm->fzs[k]);


/*
 __asm__ volatile
 (

  "movupd %[c], %%xmm0\n\t"
  "movupd %[fx], %%xmm1\n\t"
  "movupd %[fy], %%xmm2\n\t"
  "movupd %[fz], %%xmm3\n\t"
  "movupd %[vx], %%xmm4\n\t"
  "movupd %[vy], %%xmm5\n\t"
  "movupd %[vz], %%xmm6\n\t"
  "movupd %[m], %%xmm7\n\t"
  "mulpd %%xmm0, %%xmm1\n\t"	// xmm1 = xmm1 * xmm0
  "mulpd %%xmm0, %%xmm2\n\t"	//
  "mulpd %%xmm0, %%xmm3\n\t"	//
  "addpd %%xmm1, %%xmm4\n\t"	// перемножить пакеты плавающих точек: xmm4 = xmm4 * xmm1
  "addpd %%xmm2, %%xmm5\n\t"	// перемножить пакеты плавающих точек: xmm4 = xmm4 * xmm1
  "addpd %%xmm3, %%xmm6\n\t"	// перемножить пакеты плавающих точек: xmm4 = xmm4 * xmm1
  "movupd %%xmm4, %[vx]\n\t"	// выгрузить результаты из регистра xmm
  "movupd %%xmm5, %[vy]\n\t"	// выгрузить результаты из регистра xmm
  "movupd %%xmm6, %[vz]\n\t"	// выгрузить результаты из регистра xmm
  "mulpd %%xmm4, %%xmm4\n\t"	//
  "mulpd %%xmm5, %%xmm5\n\t"	//
  "mulpd %%xmm6, %%xmm6\n\t"	//
  "addpd %%xmm5, %%xmm4\n\t"	// перемножить пакеты плавающих точек: xmm4 = xmm4 + xmm5
  "addpd %%xmm6, %%xmm4\n\t"	// перемножить пакеты плавающих точек: xmm4 = xmm4 + xmm6
  "mulpd %%xmm4, %%xmm7\n\t"	// v2 = v2 * m
  "movupd %%xmm4, %[m]\n\t"	// выгрузить результаты из регистра xmm
  :
  : [c]"m"(*c), [fx]"m"(*fx), [fy]"m"(*fy), [fz]"m"(*fz), [vx]"m"(*vx), [vy]"m"(*vy), [vz]"m"(*vz), [m]"m"(*m)
  : "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
  );
*/

/*
 __asm__ volatile
 (

  "movupd %[c], %%xmm0\n\t"
  "movupd %[fx], %%xmm1\n\t"
  "mulpd %%xmm0, %%xmm1\n\t"	// xmm1 = xmm1 * xmm0
  "movupd %[fy], %%xmm2\n\t"
  "mulpd %%xmm0, %%xmm2\n\t"	//
  "movupd %[fz], %%xmm3\n\t"
  "mulpd %%xmm0, %%xmm3\n\t"	//
  "movupd %[vx], %%xmm4\n\t"
  "addpd %%xmm1, %%xmm4\n\t"	// перемножить пакеты плавающих точек: xmm4 = xmm4 * xmm1
  "movupd %[vy], %%xmm5\n\t"
  "addpd %%xmm2, %%xmm5\n\t"	// перемножить пакеты плавающих точек: xmm4 = xmm4 * xmm1
  "movupd %[vz], %%xmm6\n\t"
  "addpd %%xmm3, %%xmm6\n\t"	// перемножить пакеты плавающих точек: xmm4 = xmm4 * xmm1
  "movupd %[m], %%xmm7\n\t"
  "movupd %%xmm4, %[vx]\n\t"	// выгрузить результаты из регистра xmm
  "movupd %%xmm5, %[vy]\n\t"	// выгрузить результаты из регистра xmm
  "movupd %%xmm6, %[vz]\n\t"	// выгрузить результаты из регистра xmm
  "mulpd %%xmm4, %%xmm4\n\t"	//
  "mulpd %%xmm5, %%xmm5\n\t"	//
  "mulpd %%xmm6, %%xmm6\n\t"	//
  "addpd %%xmm5, %%xmm4\n\t"	// перемножить пакеты плавающих точек: xmm4 = xmm4 + xmm5
  "addpd %%xmm6, %%xmm4\n\t"	// перемножить пакеты плавающих точек: xmm4 = xmm4 + xmm6
  "mulpd %%xmm4, %%xmm7\n\t"	// v2 = v2 * m
  "movupd %%xmm4, %[m]\n\t"	// выгрузить результаты из регистра xmm
  :
  : [c]"m"(*c), [fx]"m"(*fx), [fy]"m"(*fy), [fz]"m"(*fz), [vx]"m"(*vx), [vy]"m"(*vy), [vz]"m"(*vz), [m]"m"(*m)
  : "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
  );
  */

    //if (i == 1)
    //{

    //printf("after:\n\n");
    //printf("c=(%f,%f), m=(%f, %f)\n vx=(%f,%f), vy=(%f,%f) vz=(%f,%f)\n fx=(%f,%f),fy=(%f,%f),fz=(%f,%f)\n", c[0],c[1],m[0],m[1],vx[0],vx[1],vy[0],vy[1],vz[0],vz[1],fx[0],fx[1],fy[0],fy[1],fz[0],fz[1]);
    //}

    res += m[0] + m[1];
    k += 2;
  }
  return res;
}

void integrate2(Atoms *atm, Spec *sp, Sim* sim, int tScale, TStat *tstat)
// the second part of verlet integrator (v = v + 0.5 f/m dt), save kinetic energy in sim
{
  double k;
  int i, ind;
  int N = atm->nAt;
  double tempA = 0.0;  // temperature
  double rMASSxTstHalf;

  //the second stage of the velocities update
  for (i = 0; i < N; i++)
  {
    ind = atm->types[i];

    rMASSxTstHalf = sp[ind].rMass_hdt;
    atm->vxs[i] += rMASSxTstHalf * atm->fxs[i];
    atm->vys[i] += rMASSxTstHalf * atm->fys[i];
    atm->vzs[i] += rMASSxTstHalf * atm->fzs[i];

    tempA += (atm->vxs[i] * atm->vxs[i] + atm->vys[i] * atm->vys[i] + atm->vzs[i] * atm->vzs[i]) * sp[ind].mass;
  }
  //! I've tried to use vectorized variant instead of abovementioned cycle, but there was no acceleration
  //tempA = vectorized_int2(atm, sp, sim);

  sim->engKin = 0.5 * tempA;

  // naive temperature scaling (if needed)
  if (tScale)
  {
     k = sqrt(tstat->tKin / sim->engKin);
     for (i = 0; i < N; i++)
     {
         atm->vxs[i] *= k;
         atm->vys[i] *= k;
         atm->vzs[i] *= k;
     }
     sim->engKin = tstat->tKin;
  }

  //applying thermostat
  if (tstat->type == tpTermNose)
  {
     sim->engKin = tstat_nose(atm, sim, tstat);
  }
}
// end 'integrate2' function
