// UNIT of electron hopping (jump)
// azTotMD (by Anton Raskovalov)
#include <stdlib.h>  // malloc, alloc, rand, NULL
#include <math.h>   // log, sqrt
#include <stdio.h>   // printf (temp)

#include "dataStruct.h"  // Sim, Box, Atoms ....
#include "utils.h"  // int_size, pointer_size, etc...
#include "const.h"  // sphera consts, kB
#include "box.h"  // rect_periodic
#include "vdw.h"  // vdw_iter
#include "bonds.h" // valence bonds
#include "ejump.h"

void ejump_header(FILE *f, Field *field)
// header for jumps.dat file
{
    int i, j;
    fprintf(f, "time\tstep\ttot\tpX\tnX\tp\tn\tpTot\tnTot");
    for (i = 0; i < field->nSpec; i++)
      if ((field->species[i].donacc >> 0) & 1)
        for (j = 0; j < field->nSpec; j++)
          if ((field->species[j].donacc >> 1) & 1)
            fprintf(f, "\t%s->%s", field->species[i].name, field->species[j].name);

    fprintf(f, "\n");
}
// end 'ejump_header' function

void ejump_out(FILE *f, double tm, int step, Field *field, Sim *sim)
// output ejumps data into a textfile
{
    int i, j;

    fprintf(f, "%f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d", tm, step, sim->nJump, sim->pBxEjump, sim->nBxEjump, sim->pEjump, sim->nEjump, sim->pTotJump, sim->nTotJump);
    for (i = 0; i < field->nSpec; i++)
        if ((field->species[i].donacc >> 0) & 1)
            for (j = 0; j < field->nSpec; j++)
                if ((field->species[j].donacc >> 1) & 1)
                    fprintf(f, "\t%d", sim->jumps[i][j]);

    fprintf(f, "\n");
}
// end 'ejump_out' function

void init_ejump(Atoms *atm, Field *field, Sim *sim)
// initialize arrays and vars for e-jump
{
   int i, j, k;
   Spec *sp = field->species;

   sim->nBxEjump = 0;
   sim->pBxEjump = 0;
   sim->nJump = 0;
   sim->jumps = (int**)malloc(field->nSpec * pointer_size);

   // old variants:
   //sim->nPairs = (int*)malloc(sim->nAt * int_size);
   //sim->totProbs = (int*)malloc(sim->nAt * int_size);
   //sim->acceptors = (int**)malloc(sim->nAt * pointer_size);
   //sim->dist = (double**)malloc(sim->nAt * pointer_size);
   //sim->probs = (int**)malloc(sim->nAt * pointer_size);
   //sim->EngDifs = (double*)malloc(sim->nJumpVar * double_size);

   // define electrons array and their positions
   k = 0;
   sim->electrons = (int*)malloc(sim->nFreeEl * int_size);
   for (i = 0; i < atm->nAt; i++)
     for (j = 0; j < sp[atm->types[i]].nFreeEl; j++)
     {
        sim->electrons[k] = i;
        k++;
     }

   // array for keeping electron jumps number between certain species
   for (i = 0; i < field->nSpec; i++)
   {
      sim->jumps[i] = (int*)malloc(field->nSpec * int_size);
      for (j = 0; j < field->nSpec; j++)
        sim->jumps[i][j] = 0;
   }

}
// end 'init_ejump' function

void free_ejump(Sim *sim, Field *field)
// deallocate arrays for e-jump
{

   int i;
   for (i = 0; i < field->nSpec; i++)
     {
       free(sim->jumps[i]);
     }

   free(sim->electrons);
}
// end 'free_ejump' function

void electron_move(int ind, int iat, int jat, int ti1, int ti2, int tj1, int tj2, int px, Field *field, Atoms *atm, Sim *sim, Box *bx)
// move electron [ind] from iat to jat
{
    Spec *sp = field->species;

    sim->electrons[ind] = jat;        // move electron to j-atom

    // if new electron localization center can't donor electron, we need to decrease free electrons
    //! now it is impossible
    /*
    if (!((sp[tj2].donacc >> 0) & 1))
    {
        sim->electrons[ind] = sim->electrons[sim->nFreeEl-1];
        //! in this case the last move will be skipped
        sim->nFreeEl--;
        printf("The free electron number is decreased!\n");
    }
    */

    atm->types[iat] = ti2;
    atm->types[jat] = tj2;
    sim->jumps[ti1][tj1]++;

    // change number of particles:
    sp[ti1].number--;
    sp[ti2].number++;
    sp[tj1].number--;
    sp[tj2].number++;

    // new bond types
    change_bonds(iat, jat, ti2, tj2, atm, field);

    if (px > 0) // px is flag of box edge crossing (-1, 0, 1)
    {
      sim->pBxEjump++;
      sim->pTotJump++;
    }
    else
      if (px < 0)
      {
         sim->nBxEjump++;
         sim->nTotJump++;
      }
      else // px == 0
      {
         if (atm->xs[jat] > atm->xs[iat])
           sim->pTotJump++;
         else if  (atm->xs[jat] < atm->xs[iat])
           sim->nTotJump++;
      }
    //! add other dimensions?

    // calculate the number of jumps through the mid section of the box
    if (atm->xs[iat] <= bx->ha)
    {
       if (atm->xs[jat] > bx->ha)
          if (atm->xs[iat] > (bx->ha - sim->rElec))
              sim->pEjump++;
    }
    else // atm->xs[iat] > box->ha
    {
         if (atm->xs[jat] <= bx->ha)
            if (atm->xs[iat] <= (bx->ha + sim->rElec))
                sim->nEjump++;
    }
}
// end 'electron_move' function

int ejump(Atoms *atm, Field *field, Sim *sim, Box *bx)
// do electron jumps (and return their number) according Frank-Condone rule (with energy conservation)
{
   Spec *sp = field->species;
   VdW ***vdws = field->vdws;

   int i, j, k;
   int iat, jat, kat, ktype;
   int tai1, taj1, tai2, taj2; //type of atom i/j  before jump(1) after jump(2)
   double U1, U2, dU; // energies before(1) and after jump(2), dU = U2 - U1
   VdW *vdw;
   double r2, rad;
   int px, py, pz;
   double kcharge;

   int result = 0;  // the number of electron jumps that we need to return
   for (i = 0; i < sim->nFreeEl; i++)
   {
      iat = sim->electrons[i]; //! probably, we need a verification that atom is still donor (it can be changed via bond forming, for example)
      tai1 = atm->types[iat];
      tai2 = sp[tai1].oxForm - 1;

      // seek neighbors-acceptors
      for (j = 0; j < sim->nNbors[iat]; j++)
        if ((sim->tnbors[iat][j] >> bfDistEjump) & 1) //distance verification
        {
           jat = sim->nbors[iat][j];
           taj1 = atm->types[jat];
           if ((sp[taj1].donacc >> bfAcceptor) & 1)  // verify, can the neighbor be an acceptor
           {
              taj2 = sp[taj1].redForm - 1;

              //ENERGY CALCULATION
              U1 = 0.0; U2 = 0.0; dU = 0.0;

              // loop by iat neighbors:
              //! maybe it's possble to combine this and the next cycles
              for (k = 0; k < sim->nNbors[iat]; k++)
              {
                 kat = sim->nbors[iat][k];
                 ktype = atm->types[kat];
                 kcharge = sp[ktype].charge;

                 //! we suppose that rad is filled because 'ejump' is called after 'forcefield'
                 rad = sim->distances[iat][k];
                 r2 = rad * rad;

                 // VdW energy before jump:
                 vdw = vdws[tai1][ktype];
                 if (vdw != NULL)
                   if (r2 <= vdw->r2cut)  //! verify, may be this condition is excess
                     U1 += vdw->eng_r(r2, rad, vdw);

                 //vdw energy after jump:
                 vdw = vdws[tai2][ktype];
                   if (vdw != NULL)
                     if (r2 <= vdw->r2cut) //! verify, may be this condition is excess
                       U2 += vdw->eng_r(r2, rad, vdw);

                 // Columbic energy after jump
                 //! change to Ewald or other technique? maybe screened Coulomb?
                 dU += Fcoul_scale * kcharge * (sp[tai2].charge - sp[tai1].charge) / rad;
              } // end loop by iat neigbors

              // loop by jat neighbors:
              for (k = 0; k < sim->nNbors[jat]; k++)
              {
                 kat = sim->nbors[jat][k];
                 if (kat == iat)  // this interaction was calculated in the previous cycle, skip
                   continue;

                 ktype = atm->types[kat];
                 kcharge = sp[ktype].charge;

                 //! we suppose that rad is filled because 'ejump' is called after 'forcefield'
                 rad = sim->distances[jat][k];
                 r2 = rad * rad;

                 // VdW energy before jump:
                 vdw = vdws[taj1][ktype];
                 if (vdw != NULL)
                   if (r2 <= vdw->r2cut)  //! verify, may be this condition is exces
                     U1 += vdw->eng_r(r2, rad, vdw);

                 //vdw energy after jump:
                 vdw = vdws[taj2][ktype];
                 if (vdw != NULL)
                   if (r2 <= vdw->r2cut)  //! verify, may be this condition is exces
                     U2 += vdw->eng_r(r2, rad, vdw);

                 // Columbic energy after jump
                 //! change to Ewald or other technique? maybe screened Coulomb?
                 dU += Fcoul_scale * kcharge * (sp[taj2].charge - sp[taj1].charge) / rad;
              }
              // end loop by jat neigbors

              dU += U2 - U1;
              // own energy difference
              dU += (sp[tai2].energy + sp[taj2].energy - sp[tai1].energy - sp[taj1].energy);
              //bond energy change
              dU += bond_eng_change(iat, jat, tai2, taj2, atm, field, bx);

              //external electric field addition:
              pass_periodic(jat, iat, atm, bx, px, py, pz); // find px, flag of box crossing
              dU += sim->Ux * (atm->xs[iat] * (sp[tai2].charge - sp[tai1].charge) + (atm->xs[jat] + px * bx->la) * (sp[taj2].charge - sp[taj1].charge));

              // energy equality condition
              if (dU < sim->dEjump)
                if (dU > -sim->dEjump)  // energy equality (the Frank-Condon principle), doing e-jump
                  {
                     result++;
                     electron_move(i, iat, jat, tai1, tai2, taj1, taj2, px, field, atm, sim, bx);
                     break; //! very small chance of the energy equality with several neighbors
                  }


           }  // end if jat is acceptor
        }  // end loop by neighbors of electron
   }  // end loop by electrons

   sim->nJump += result;
   return result;
}
// end 'ejump' function

int ejump_min(Atoms *atm, Field *field, Sim *sim, Box *bx)
// do electron jumps (and return their number) according energy minimization
{
   Spec *sp = field->species;
   VdW ***vdws = field->vdws;

   int i, j, k;
   int iat, jat, kat, ktype;
   int tai1, taj1, tai2, taj2; //type of atom i/j  before jump(1) after jump(2)
   double U1, U2, dU; // energies before(1) and after jump(2), dU = U2-U1
   VdW *vdw;
   double r2, rad;
   int px, py, pz;
   double kcharge;
   int indMin, pxMin;   // to choose the most advantageous variant
   double minE;

   int result = 0;  // the number of electron jumps that we need to return
   for (i = 0; i < sim->nFreeEl; i++)
   {
      iat = sim->electrons[i];  //! probably, we need a verification that atom is still donor (it can be changed via bond forming, for example)
      tai1 = atm->types[iat];
      tai2 = sp[tai1].oxForm - 1;

      minE = 0.0;
      indMin = 0;
      for (j = 0; j < sim->nNbors[iat]; j++)
        if ((sim->tnbors[iat][j] >> bfDistEjump) & 1) //distance type: enough for ejump
        {
           jat = sim->nbors[iat][j];
           taj1 = atm->types[jat];
           if ((sp[taj1].donacc >> bfAcceptor) & 1)  // verify, that neighbor is an acceptor
           {
              taj2 = sp[taj1].redForm - 1;

              //ENERGY CALCULATION
              U1 = 0.0; U2 = 0.0; dU = 0.0;

              // loop by iat neighbors:
              //! maybe its possible to combine this and the next cycle?
              for (k = 0; k < sim->nNbors[iat]; k++)
              {
                 kat = sim->nbors[iat][k];
                 ktype = atm->types[kat];
                 kcharge = sp[ktype].charge;

                 //! we suppose that rad is filled because 'ejump' is called after 'forcefield'
                 rad = sim->distances[iat][k];
                 r2 = rad * rad;

                 // VdW energy before jump:
                 vdw = vdws[tai1][ktype];
                 if (vdw != NULL)
                   if (r2 <= vdw->r2cut)  //! verify, may be this condition is exces
                     U1 += vdw->eng_r(r2, rad, vdw);

                 //vdw energy after jump:
                 vdw = vdws[tai2][ktype];
                   if (vdw != NULL)
                     if (r2 <= vdw->r2cut) //! verify, may be this condition is exces
                       U2 += vdw->eng_r(r2, rad, vdw);

                 // Columbic energy difference:
                 //! change to Ewald or other technique? maybe screened Coulomb?
                 dU += Fcoul_scale * kcharge * (sp[tai2].charge - sp[tai1].charge) / rad;
              } // end loop by iat neigbors

              // loop by jat neighbors:
              for (k = 0; k < sim->nNbors[jat]; k++)
              {
                 kat = sim->nbors[jat][k];
                 if (kat == iat)  // this interaction was calculated in previous cycle, skip
                   continue;

                 ktype = atm->types[kat];
                 kcharge = sp[ktype].charge;

                 //! we suppose that rad is filled because 'ejump' is called after 'forcefield'
                 rad = sim->distances[jat][k];
                 r2 = rad * rad;

                 // VdW energy before jump:
                 vdw = vdws[taj1][ktype];
                 if (vdw != NULL)
                   if (r2 <= vdw->r2cut)  //! verify, may be this condition is exces
                     U1 += vdw->eng_r(r2, rad, vdw);

                 //vdw energy after jump:
                 vdw = vdws[taj2][ktype];
                 if (vdw != NULL)
                   if (r2 <= vdw->r2cut) //! verify, may be this condition is exces
                     U2 += vdw->eng_r(r2, rad, vdw);

                 // Columbic energy difference
                 //! change to Ewald or other technique? maybe screened Coulomb?
                 dU += Fcoul_scale * kcharge * (sp[taj2].charge - sp[taj1].charge) / rad;
              }  // end loop by jat neigbors

              dU = dU + U2 - U1;
              // own energy
              dU += (sp[tai2].energy + sp[taj2].energy - sp[tai1].energy - sp[taj1].energy);
              //bond energy change
              dU += bond_eng_change(iat, jat, tai2, taj2, atm, field, bx);

              //external electric field addition:
              pass_periodic(jat, iat, atm, bx, px, py, pz);
              dU += sim->Ux * (atm->xs[iat] * (sp[tai2].charge - sp[tai1].charge) + (atm->xs[jat] + px * bx->la) * (sp[taj2].charge - sp[taj1].charge));
              //! simplified variant for single-electron changes in the charge:
              //dU += sim->Ux * (atm->xs[iat] - atm->xs[jat] - px * bx->la);

              // save variant
              if (dU < minE)
              {
                 minE = dU;
                 indMin = jat + 1; // 0 is reserved as empty
                 pxMin = px;
                 //! maybe break, if small number of variants, especially with dU < 0
              }
           }  // end if jat is acceptor
        }
      // end loop by neighbors of electron

      // select variant with minimal energy
      if (indMin)
      {
           result++;
           jat = indMin - 1;
           taj1 = atm->types[jat];
           taj2 = sp[taj1].redForm - 1;
           electron_move(i, iat, jat, tai1, tai2, taj1, taj2, px, field, atm, sim, bx);
      }

   }  // end loop by electrons

   sim->nJump += result;
   return result;
}
// end 'ejump_min' function

int ejump_metr(Atoms *atm, Field *field, Sim *sim, Box *bx)
// do electron jumps (return their numbers) according to the Metropolis scheme
{
   Spec *sp = field->species;
   VdW ***vdws = field->vdws;

   int i, j, k;
   int iat, jat, kat, ktype;
   int tai1, taj1, tai2, taj2; //type of atom i/j  before jump(1) after jump(2)
   double U1, U2, dU; // energies before(1) and after jump(2), dU = U2-U1
   VdW *vdw;
   double r2, rad;
   int px, py, pz;
   double kcharge;
   int indMin;
   int pxMin;
   double minE;
   double r;
   int succ; // flag of success move

   int result = 0;  // the number of electron jumps that we need to return
   for (i = 0; i < sim->nFreeEl; i++)
   {
      iat = sim->electrons[i]; //! probably, we need a verification that atom is still donor (it can be changed via bond forming, for example)
      tai1 = atm->types[iat];
      if (!((sp[tai1].donacc >> bfDonor) & 1))  //! this verification. Atom type can change via bond mutation
        continue;

      tai2 = sp[tai1].oxForm - 1;

      minE = 1e100; //! we expect that there will no such energy differences
      indMin = 0;
      for (j = 0; j < sim->nNbors[iat]; j++)
        if ((sim->tnbors[iat][j] >> bfDistEjump) & 1) // distance type: enough for ejump
        {
           //printf("ej dist=%f\n", sim->distances[iat][j]);
           jat = sim->nbors[iat][j];
           taj1 = atm->types[jat];
           if ((sp[taj1].donacc >> bfAcceptor) & 1)  // verify, that the neighbor is an acceptor
           {
              taj2 = sp[taj1].redForm - 1;

              //ENERGY CALCULATION
              U1 = 0.0; U2 = 0.0; dU = 0.0;

              // loop by iat neighbors:
              //! maybe combine this cycle with next one?
              for (k = 0; k < sim->nNbors[iat]; k++)
              {
                 kat = sim->nbors[iat][k];
                 ktype = atm->types[kat];
                 kcharge = sp[ktype].charge;

                 //! we suppose that rad is filled because 'ejump' is called after 'forcefield'
                 rad = sim->distances[iat][k];
                 r2 = rad * rad;

                 // VdW energy before jump:
                 vdw = vdws[tai1][ktype];
                 if (vdw != NULL)
                   if (r2 <= vdw->r2cut) //! verify, may be this condition is exces
                     U1 += vdw->eng_r(r2, rad, vdw);

                 //vdw energy after jump:
                 vdw = vdws[tai2][ktype];
                 if (vdw != NULL)
                   if (r2 <= vdw->r2cut) //! verify, may be this condition is exces
                     U2 += vdw->eng_r(r2, rad, vdw);

                 // Columbic energy difference:
                 //! change to Ewald or other technique? maybe screened Coulomb?
                 dU += Fcoul_scale * kcharge * (sp[tai2].charge - sp[tai1].charge) / rad;
              } // end loop by iat neigbors

              // loop by jat neighbors:
              for (k = 0; k < sim->nNbors[jat]; k++)
              {
                 kat = sim->nbors[jat][k];
                 if (kat == iat)  // this interaction was calculated in the previous cycle, skip
                   continue;

                 ktype = atm->types[kat];
                 kcharge = sp[ktype].charge;

                 //! we suppose that rad is filled because 'ejump' is called after 'forcefield'
                 rad = sim->distances[jat][k];
                 r2 = rad * rad;

                 // VdW energy before jump:
                 vdw = vdws[taj1][ktype];
                 if (vdw != NULL)
                   if (r2 <= vdw->r2cut)  //! verify, may be this condition is exces
                     U1 += vdw->eng_r(r2, rad, vdw);

                 //vdw energy after jump:
                 vdw = vdws[taj2][ktype];
                 if (vdw != NULL)
                   if (r2 <= vdw->r2cut)  //! verify, may be this condition is exces
                     U2 += vdw->eng_r(r2, rad, vdw);

                 // Columbic energy difference
                 //! change to Ewald or other technique? maybe screened Coulomb?
                 dU += Fcoul_scale * kcharge * (sp[taj2].charge - sp[taj1].charge) / rad;
              }  // end loop by jat neigbors

              dU = dU + U2 - U1;
              // own energy
              dU += (sp[tai2].energy + sp[taj2].energy - sp[tai1].energy - sp[taj1].energy);
              //bond energy change
              dU += bond_eng_change(iat, jat, tai2, taj2, atm, field, bx);

              //external electric field addition:
              pass_periodic(jat, iat, atm, bx, px, py, pz);
              dU += sim->Ux * (atm->xs[iat] * (sp[tai2].charge - sp[tai1].charge) + (atm->xs[jat] + px * bx->la) * (sp[taj2].charge - sp[taj1].charge));
              //! simplified variant for single-electron changes in the charge:
              //dU += sim->Ux * (atm->xs[iat] - atm->xs[jat] - px * bx->la);

              if (dU < minE) // more negative difference, rewrite old variant
              {
                 // save variant
                 indMin = jat + 1; // 0 is reserved as empty
                 minE = dU;
                 pxMin = px;
                 //! maybe break, if small number of variants, especially with dU < 0
              }
           }  // end if jat is acceptor
        }
      // end loop by neighbors of electron

      // accept electron move according to Metropolis scheme:
      succ = 0;
      if (indMin)   // only if there are acceptors near electrons[i]
      {
        if (minE < 0.0)
          succ = 1;
        else
        {
           r = rand01();
           if (r < exp(-rkB*minE/sim->tTemp)) //! maybe introduce 1/T parameter and use it (or 1/kT)
             succ = 1;
        }
      }

      // success, electron jumping
      if (succ)
      {
           result++;
           jat = indMin - 1;
           taj1 = atm->types[jat];
           taj2 = sp[taj1].redForm - 1;
           electron_move(i, iat, jat, tai1, tai2, taj1, taj2, px, field, atm, sim, bx);
      }

   }  // end loop by electrons

   sim->nJump += result;
   return result;
}
// end 'ejump_metr' function

void jmp_rare(Atoms *atm, Field *field, Sim *sim, Box *bx, int step)
// jumps every Nth timestep
{
    if (step % (-sim->eJump) == 0)
       sim->ejumper(atm, field, sim, bx);
}

void jmp_oft(Atoms *atm, Field *field, Sim *sim, Box *bx, int step)
// N jumps every timestep
{
   int i;
   //j = 0;
   for (i = 0; i < sim->eJump; i++)
   {
      if (!sim->ejumper(atm, field, sim, bx))
        break;
   }

/*
        //! counter (temp)
        if (j > mxJump)
            mxJump = j;

        tJump += j;
*/
}

void jmp_none(Atoms *atm, Field *field, Sim *sim, Box *bx, int step)
// no jumps
{
}
