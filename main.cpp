//!  to do! (or pay attention!)

// MAIN UNIT
// azToTMD 3.0  (by Raskovalov Anton)

#include <stdio.h>      // FILE
#include <stdlib.h>     // malloc, alloc, rand, NULL
#include <time.h>       // time
//#include <strings.h>  // strcpy, strcat
//#include <conio.h>

#include "dataStruct.h" // Sim, Box, Atoms ....
#include "sys_init.h"   // init_md, free_md
#include "out_md.h"     // md output
#include "rdf.h"        // function for work with rdf
#include "elec.h"       // electrostatic procedures
#include "box.h"        // rect_periodic, rect_box, box_prop
#include "ejump.h"      // electron transfer functions
#include "bonds.h"
#include "angles.h"
//#include "cell_list.h"    // cell list method
//#include "temperature.h"  // termostates etc
#include "integrators.h"

int main(int argc, char *argv[])
{
   //int mxJump = 0;      // (c) maximal number of eJump procedure per timestep
   //int tJump = 0;       // (c) the total number of eJump procedure
   int i;
   char revfname[32];     // for output of intermediate configurations
   char c;                // for press any key
   double revDeltaTime;   // (1/dt) for pressure calculation
   int start_time = time(NULL);

   printf("azTotMD by Raskovalov Anton\n");

   //SYSTEM INITIALISATION
   Elec *elec = (Elec*)malloc(sizeof(Elec));
   TStat *tstat = (TStat*)malloc(sizeof(TStat));
   Box *box = (Box*)malloc(sizeof(Box));
   Field *field = (Field*)malloc(sizeof(Field));
   Atoms *atoms = (Atoms*)malloc(sizeof(Atoms));
   Sim *sim = (Sim*)malloc(sizeof(Sim));
   if (!init_md(atoms, field, sim, elec, tstat, box))
   {
        printf("FATAL ERROR: SYSTEM CAN'T BE INITIALIZED\n");
        scanf("%c", &c);
        return 0;
   }
   if (!init_serial(atoms, field, sim, elec, tstat, box))
   {
       printf("FATAL ERROR: SYSTEM CAN'T BE INITIALIZED 2\n");
       scanf("%c", &c);
       return 0;
   }
   srand(time(NULL));   // ranomize for Metropolis scheme

   //OUTPUT FILES
   FILE *jf = NULL, *vaf_file = NULL, *traj_file = NULL;        // electron jump file, VAF file, trajectories file
   FILE *stat_file = fopen("stat.dat", "w");
   FILE *hf = fopen("hist.dat", "w");
   FILE *msd_file = fopen("msd.dat", "w");
   stat_header(stat_file, sim, field->species);
   history_header(hf);
   msd_header(msd_file, sim, field);
   if (sim->frTraj)
   {
     traj_file = fopen("traj.dat", "w");
     traj_header(traj_file, atoms, field->species, sim);
   }
   if (sim->vaf)
   {
     vaf_file = fopen("vaf.dat", "w");
     vaf_header(vaf_file, field, sim);
   }
   if (sim->eJump)
   {
     jf = fopen("jumps.dat", "w");
     ejump_header(jf, field);
   }

   // SOME STARTING INFO OUTPUT
   info_md(atoms, sim);

   // MAIN MD LOOP:
   //! DIVIDE INTO 2 LOOPS: EQULIBRATION and MAIN
   int iSt = 0;             // index of a timestep
   double tSim = 0.0;       // time of simulation
   while (iSt < sim->nSt)
   {
     iSt++;
     reset_chars(sim);

     // THE FIRST STAGE OF THE MOITION EQUATION INTEGRATION :  X += V*dt + F/m *dt^2/2;   V += F/m dt/2
     sim->integrator1(atoms, field->species, sim, box, tstat);

     // FORCE CALCULATION
     clear_force(atoms, field->species, sim, box);
     sim->add_elec(atoms, field, elec, box, sim);   // additional electrostatic (for exampe, recipropar part of Ewald)
     sim->forcefield(atoms, field, elec, box, sim);
     if (field->nBonds)
        exec_bondlist(atoms, field, sim, box);
     if (field->nAngles)
        exec_anglelist(atoms, field, sim, box);

     // perform electron hopping
     sim->do_jump(atoms, field, sim, box, iSt);

     // THE SECOND STAGE OF THE MOITION EQUATION INTEGRATION :   // V += F/m dt/2
     if (iSt > sim->nEq) // end equilibration, no temperature scaling
     {
          integrate2(atoms, field->species, sim, 0, tstat);
          if (iSt % sim->frRDF == 0)
            get_rdf(atoms, sim, field, box);
          if (sim->vaf)
            if (iSt % sim->vaf == 0)
              vaf_info(vaf_file, tSim, iSt, atoms, field, sim);
     }
     else             // equilibration period, T-Scale
     {
         if ((iSt % sim->freqEq) == 0)
           integrate2(atoms, field->species, sim, 1, tstat);
         else
           integrate2(atoms, field->species, sim, 0, tstat);

         //update initial coordinates (for MSD calculation) and VAF
         if (iSt == sim->nEq)
         {
           for (i = 0; i < atoms->nAt; i++)
             {
                atoms->x0s[i] = atoms->xs[i];
                atoms->y0s[i] = atoms->ys[i];
                atoms->z0s[i] = atoms->zs[i];
             }
           if (sim->vaf)
              vaf_init(atoms);
         }
     }
     // END THE SECOND STAGE OF THE INTEGRATOR

     // STATISTICS AND OUTPUT
     calc_chars(sim, tSim);   // calculate some quantites like total energy
     if (iSt % sim->stat == 0)
     {
         //pressure calculation
         revDeltaTime = 1.0 / (tSim - sim->prevTime);
         sim->presXn = 2.0 * 1.58e6 * box->revSOyz * (box->momXn - box->momXn0) * revDeltaTime;
         sim->presXp = 2.0 * 1.58e6 * box->revSOyz * (box->momXp - box->momXp0) * revDeltaTime;
         sim->presYn = 2.0 * 1.58e6 * box->revSOxz * (box->momYn - box->momYn0) * revDeltaTime;
         sim->presYp = 2.0 * 1.58e6 * box->revSOxz * (box->momYp - box->momYp0) * revDeltaTime;
         sim->presZn = 2.0 * 1.58e6 * box->revSOxy * (box->momZn - box->momZn0) * revDeltaTime;
         sim->presZp = 2.0 * 1.58e6 * box->revSOxy * (box->momZp - box->momZp0) * revDeltaTime;

         box->momXn0 = box->momXn;
         box->momXp0 = box->momXp;
         box->momYn0 = box->momYn;
         box->momYp0 = box->momYp;
         box->momZn0 = box->momZn;
         box->momZp0 = box->momZp;
         sim->prevTime = tSim;

         out_stat(stat_file, tSim, iSt, sim, field->species);
         out_msd(msd_file, atoms, atoms->nAt, field->species, field->nSpec, box, tSim, iSt);
         //out_info(info_file, tSim, iSt, atoms, field->species);
     }
     if (iSt % sim->hist == 0)
     {
       fprintf(hf, "%f %d %f %f %f %f %f %f %f %f %f %f %f\n", tSim, iSt, sim->engTot, sim->Temp, atoms->xs[0], atoms->ys[0], field->species[atoms->types[0]].charge, box->momXn, box->momXp, box->momYn, box->momYp, box->momZn, box->momZp);
       if (sim->eJump)
         ejump_out(jf, tSim, iSt, field, sim);
     }
     if (sim->revcon)
       if (iSt % sim->revcon == 0)
         {
             sprintf(revfname, "revcon%d.xyz\0", iSt);
             out_atoms(atoms, atoms->nAt, field->species, box, revfname);
         }
     if (sim->frTraj)
       if (iSt > sim->stTraj)
         if (iSt % sim->frTraj == 0)
           traj_info(traj_file, tSim, iSt, atoms, field->species, sim);
   }
   // END MAIN LOOP

   // FINAL OUTPUT
   /*
   out_atoms(atoms, atoms->nAt, field->species, box, "revcon.xyz");
   out_rdf(field, box, sim, "rdf.dat");
   out_velocities(atoms, field, "velocities.dat");
   if (field->nBonds)
   {
      save_bondlist("revbonds.txt", field);
      bond_out(atoms, field, box, "lengths.dat");
   }
   if (field->nAngles)
     save_anglelist("revangles.txt", field);
   if (sim->outCN)
     out_cn(atoms, field, box, sim, "cns.dat");
   */


   // CLOSE FILES
   if (hf != NULL)          fclose(hf);
   if (stat_file != NULL)   fclose(stat_file);
   if (msd_file != NULL)    fclose(msd_file);
   if (sim->frTraj)         fclose(traj_file);
   if (sim->vaf)            fclose(vaf_file);
   if (sim->ejtype)         fclose(jf);

   // FREE MEMORY
   free_serial(atoms, field, elec, sim);
   free_md(atoms, field);
   delete box;
   delete elec;
   delete atoms;
   delete field;
   delete tstat;
   delete sim;

   // SPENT TIME
   int end_time = time(NULL);
   int spent_time = end_time - start_time;
   int hours = spent_time / 3600;
   int minutes = spent_time - 3600 * hours;
   int secs = minutes % 60;
   minutes = minutes / 60;
   printf("The program's just finished correctly, the running time: %d s (%d h, %d min, %d sec)\n", spent_time, hours, minutes, secs);
   scanf("%c", &c);     // press any key
}
