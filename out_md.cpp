#include <stdio.h>   // FILE, fprintf, scanf
#include <stdlib.h>  // malloc, alloc, rand, NULL
#include <math.h>   // sqrt
#include <string.h>     // strcmp

#include "utils.h"      // int_size..
#include "dataStruct.h"  // Sim, Box, Atoms ....
#include "box.h"
#include "out_md.h"
#include "sys_init.h"

void history_header(FILE *f)
{
   //f = fopen("hist.dat", "w");
   fprintf(f, "time iStep totEn temp atm1x atm1y atm1ch momXn momXp momYn momYp momZn momZp\n");
   fprintf(f, "time,ps iStep totEn,eV temp,K atm[1].x,A atm[1].y,A atm1ch,e momXn momXp momYn momYp momZn momZp\n");
}

void msd_header(FILE *f, Sim *sim, Field *field)
{
   int i;

   fprintf(f, "Time\tStep");
   for (i = 0; i < field->nSpec; i++)
     {
        fprintf(f, "\t%s%s\t%s%s\t%s%s", field->species[i].name, "-msd", field->species[i].name, "-nOyz", field->species[i].name, "-pOyz");
     }
   fprintf(f, "\n");
}

void stat_header(FILE *f, Sim *sm, Spec *sp)
{
   int i;

   fprintf(f, "Time\tStep\tTemp\tpotE\tpotE1\tkinE\ttotE\tpresXn\tpresXp\tpresYn\tpresYp\tpresZn\tpresZp");
   // the number of species which can change
   for (i = 0; i < sm->nVarSpec; i++)
     fprintf(f, "\t%s", sp[sm->varSpecs[i]].name);
   fprintf(f, "\n");

   fprintf(f, "Time,ps\tStep\tTemp,K\tpotE,eV\tpotE1,eV\tkinE,eV\ttotE,eV\tpresXn, atm\tpresXp, atm\tpresYn, atm\tpresYp, atm\tpresZn, atm\tpresZp, atm");
   for (i = 0; i < sm->nVarSpec; i++)
     fprintf(f, "\t%s", sp[sm->varSpecs[i]].name);
   fprintf(f, "\n");
}

void out_stat(FILE *f, double tm, int step, Sim *sm, Spec *sp)
{
   int i;

   fprintf(f, "%f\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f", tm, step, sm->Temp, sm->engVdW + sm->engElec3, sm->engElec2, sm->engKin, sm->engTot, sm->presXn, sm->presXp, sm->presYn, sm->presYp, sm->presZn, sm->presZp);
   if (sm->nVarSpec)
     for (i = 0; i < sm->nVarSpec; i++)
       fprintf(f, "\t%d", sp[sm->varSpecs[i]].number);
   fprintf(f, "\n");
}

/*
void stat_out(FILE *f, )
{
   fprintf(f, "%f\t%d\t%f\t%f\t%f\t%f\t%f\n", tSim, iSt, Temp, potE, potE1, kinE, totE);
}
*/

int out_atoms(Atoms *atm, int N, Spec *spec, Box *bx, char *fname)
// write coordinates of atoms (.XYZ specification)
{
   int i;
   FILE *of;

   of = fopen(fname, "w");
   if (of == NULL)
     return 0; // error

   fprintf(of, "%d\n", N);

   //box parameters saving:
   save_box(of, bx);

   for (i = 0; i < N; i++)
     {
        fprintf(of, "%s\t%f\t%f\t%f\n", spec[atm->types[i]].name, atm->xs[i], atm->ys[i], atm->zs[i]);
     }
   fclose(of);
   return 1; // success
}
// end 'out_atoms' function

int out_msd(FILE *f, Atoms *atm, int N, Spec *spec, int NSp, Box *bx, double tm, int tst)
// write msd in open file f
{
   int i, j;
   double dx, dy, dz;


   for (i = 0; i < NSp; i++)
     {
        spec[i].number = 0; //! вообще нужно убрать это отсюда, а менять эти количества всегда, когда до этого доходит дело
        spec[i].displ = 0.0;
     }

   fprintf(f, "%f\t%d", tm, tst);
   for (i = 0; i < N; i++)
     {
        dx = atm -> xs[i] - atm -> x0s[i];
        dy = atm -> ys[i] - atm -> y0s[i];
        dz = atm -> zs[i] - atm -> z0s[i];
        delta_periodic(dx, dy, dz, bx);


        j = atm->types[i];

        spec[j].number++;
        spec[j].displ += dx * dx + dy * dy + dz * dz;

     }

   for (i = 0; i < NSp; i++)
     {
        fprintf(f, "\t%f\t%d\t%d", spec[i].displ / spec[i].number, spec[i].nOyz, spec[i].pOyz);
     }
   fprintf(f, "\n");
   return 1;
}

int out_velocities(Atoms *atm, Field *field, char *fname)
// write velocities in file
{
   int i, j, t, mx = 0;
   int  *curInds;    // currentIndex for saving
   FILE *f;

   // array of velocites and their components
   double **vels = (double**)malloc(field->nSpec * pointer_size);
   double **vxs = (double**)malloc(field->nSpec * pointer_size);
   double **vys = (double**)malloc(field->nSpec * pointer_size);
   double **vzs = (double**)malloc(field->nSpec * pointer_size);
   curInds = (int*)malloc(field->nSpec * int_size);
   for (i = 0; i < field->nSpec; i++)
   {
       vels[i] = (double*)malloc(field->species[i].number * double_size);
       vxs[i] = (double*)malloc(field->species[i].number * double_size);
       vys[i] = (double*)malloc(field->species[i].number * double_size);
       vzs[i] = (double*)malloc(field->species[i].number * double_size);
       curInds[i] = 0;
       mx = max(mx, field->species[i].number);
   }

   for (i = 0; i < atm->nAt; i++)
   {
      t = atm->types[i];
      vels[t][curInds[t]] = sqrt(atm->vxs[i] * atm->vxs[i] + atm->vys[i] * atm->vys[i] + atm->vzs[i] * atm->vzs[i]);
      vxs[t][curInds[t]] = atm->vxs[i];
      vys[t][curInds[t]] = atm->vys[i];
      vzs[t][curInds[t]] = atm->vzs[i];
      curInds[t]++;
   }

   f = fopen(fname, "w");
   if (f == NULL)
     return 0; // error

   fprintf(f, "No");
   for (i = 0; i < field->nSpec; i++)
     fprintf(f, "\t%s\tx\ty\tz", field->species[i].name);
   fprintf(f, "\n");

   for (i = 0; i < mx; i++)
   {
      fprintf(f, "%d", i + 1);
      for (j = 0; j < field->nSpec; j++)
        if (i < field->species[j].number)
          fprintf(f, "\t%f\t%f\t%f\t%f", vels[j][i], vxs[j][i], vys[j][i], vzs[j][i]);
        else
          fprintf(f, "\t\t\t\t");
      fprintf(f, "\n");
   }
   fclose(f);

   for (i = 0; i < field->nSpec; i++)
       if (field->species[i].number)
       {
           free(vels[i]);
           free(vxs[i]);
           free(vys[i]);
           free(vzs[i]);
       }
   free(vels);
   free(vxs);
   free(vys);
   free(vzs);
   free(curInds);
   return 1;
}

int out_ncn(Atoms* atm, Field* field, Box* bx, Sim* sim, char* fname)
// write nucleus coordination numbers in file (if directives in control file allow it)
// algorithm nCN
{
    int i, j, k, n, id1, id2;
    char nm1[8], nm2[8];
    char header[1024];
    //char *header = (char*)malloc(1024 * sizeof(char));
    double r;
    int res = 1;
    int mn, mx;

    // arrays:
    int* nLigands;
    int** ligOrders;
    int** pairInds;     // indexes of pairs to output
    double** ligRad2s;   // squares of CN cutoff
    int** flags;        // flags to proceed pair
    
    // read directives from nucleus CN output from control.txt
    FILE* f = fopen("control.txt", "r");
    if (f != NULL)
    {
        if (n = find_number(f, " ncn %d "))
        {
            sprintf(header, "CN");
            nLigands = (int*)malloc(field->nNucl * int_size);
            ligOrders = (int**)malloc(field->nNucl * pointer_size);
            pairInds = (int**)malloc(field->nNucl * pointer_size);
            flags = (int**)malloc(field->nNucl * pointer_size);
            ligRad2s = (double**)malloc(field->nNucl * pointer_size);
            for (i = 0; i < field->nNucl; i++)
            {
                nLigands[i] = 0;
                ligOrders[i] = (int*)malloc(field->nNucl * int_size);
                pairInds[i] = (int*)malloc(field->nNucl * int_size);
                flags[i] = (int*)malloc(field->nNucl * int_size);
                ligRad2s[i] = (double*)malloc(field->nNucl * double_size);
                for (j = 0; j < field->nNucl; j++)
                {
                    ligOrders[i][j] = 0;
                    flags[i][j] = 0;
                }
            }
            mx = 0;

            for (i = 0; i < n; i++)
            {
                fscanf(f, "%8s %8s %lf", nm1, nm2, &r);
                fscanf(f, "\n");
                if (!nucl_by_name(field, nm1, id1))
                {
                    printf("ERROR[b010] Unknown nuclei name(%s) in ncn section of control file! Line %d: %s %s %f\n", nm1, i+1, nm1, nm2, r);
                    res = 0;
                }
                if (!nucl_by_name(field, nm2, id2))
                {
                    printf("ERROR[b011] Unknown nuclei name(%s) in ncn section of control file! Line %d: %s %s %f\n", nm2, i + 1, nm1, nm2, r);
                    res = 0;
                }
                ligOrders[id1][id2] = nLigands[id1] + 1;        // as 0 is reserved for no pair
                // порядок пары в выводе
                pairInds[id1][nLigands[id1]] = i;
                ligRad2s[id1][id2] = r * r;
                nLigands[id1]++;
                mx = max(mx, nLigands[id1]);        // save maximal number of ligands for arrays initialization
                flags[id1][id2] = 1;
                flags[id2][id1] = 1;
                //sprintf(header, "\t%s-%s", nm1, nm2);
                sprintf(header, "%s\t%s-%s", header, nm1, nm2);
                //strcat(header, "\t");
                //header = strcat(header, "\t");
                //header = strcat(header, nm1);
                //header = strcat(header, "-");
                //header = strcat(header, nm2);
            }
            fclose(f);
            sprintf(header, "%s\n", header);
            //header = strcat(header, "\n");
        }
        else
        {
            fclose(f);
            return 1;
        }
    }
    else
    {
        printf("ERROR[b009] No control file!\n");
        return 0;
    }


    // main part, define atoms coordinations
    int** coords;
    coords = (int**)malloc(atm->nAt * pointer_size);
    for (i = 0; i < atm->nAt; i++)
    {
        coords[i] = (int*)malloc(mx * int_size);
        for (j = 0; j < mx; j++)
            coords[i][j] = 0;
    }

    // fill coords array (main loop)
    mn = 10; mx = 0;
    for (i = 0; i < (atm->nAt - 1); i++)
    {
        id1 = field->species[atm->types[i]].nuclei;
        for (j = i + 1; j < atm->nAt; j++)
        {
            id2 = field->species[atm->types[j]].nuclei;
            if (flags[id1][id2])
            {
                r = sqr_distance(i, j, atm, bx);

                // update the first atom in pair
                if (k = ligOrders[id1][id2])
                    if (r < ligRad2s[id1][id2])
                        coords[i][k - 1]++;

                // update the second atom in pair (all indexes are opposite)
                if (k = ligOrders[id2][id1])
                    if (r < ligRad2s[id2][id1])
                        coords[j][k - 1]++;

            }
        }

        // define min and max CN
        for (j = 0; j < nLigands[id1]; j++)
        {
            mn = min(mn, coords[i][j]);
            mx = max(mx, coords[i][j]);
        }
    }

    // final part, generation of output array and output
    int wdth = mx - mn + 1;
    int** out = (int**)malloc(n * pointer_size);
    for (i = 0; i < n; i++)
    {
        out[i] = (int*)malloc(wdth * int_size);
        for (j = 0; j < wdth; j++)
            out[i][j] = 0;
    }

    // convert coords array to out array
    for (i = 0; i < atm->nAt; i++)
    {
        id1 = field->species[atm->types[i]].nuclei;
        for (j = 0; j < nLigands[id1]; j++)
        {
            k = pairInds[id1][j];
            out[k][coords[i][j] - mn]++;
        }
    }

    // output
    f = fopen(fname, "w");
    fprintf(f, "%s", header);
    for (i = 0; i < wdth; i++)
    {
        fprintf(f, "%d", mn + i);
        for (j = 0; j < n; j++)
            fprintf(f, "\t%d", out[j][i]);
        fprintf(f, "\n");
    }
    fclose(f);

    // free memory
    for (i = 0; i < field->nNucl; i++)
    {
        free(ligOrders[i]);
        free(pairInds[i]);
        free(flags[i]);
        free(ligRad2s[i]);
    }
    free(ligOrders);
    free(pairInds);
    free(flags);
    free(ligRad2s);

    for (i = 0; i < atm->nAt; i++)
        free(coords[i]);
    free(coords);

    for (i = 0; i < n; i++)
        free(out[i]);
    free(out);

    free(nLigands);
}

int out_cn(Atoms *atm, Field *field, Box *bx, Sim *sim, char *fname)
// write coordination numbers in file
{
   int i, j, k, t, mx = 0;
   int nA;  // number of output atoms
   int nPair;   // number of CN pairs
   int  **coords;    // coordinations[atom_index][ligand_species_index]
   int *ligTypes;    // ligand types
   int *cenTypes;    // central atom types
   int **out;       // output array
   double r2;
   FILE *f;

   ligTypes = (int*)malloc((sim->nCountCN + 1) * int_size);
   cenTypes = (int*)malloc((sim->nCentrCN + 1) * int_size);

   nA = 0;
   for (i = 0; i < field->nSpec; i++)
   {
       if (field->species[i].idCentral)
       {
         nA += field->species[i].number;
         cenTypes[field->species[i].idCentral] = i;
       }
       if (field->species[i].idCounter)
         ligTypes[field->species[i].idCounter] = i;
   }
   coords = (int**)malloc(nA * pointer_size);

   //printf("atm filling\n");
   k = 0;
   for (i = 0; i < atm->nAt; i++)
   {
      if (field->species[atm->types[i]].idCentral)
      {
          coords[k] = (int*)malloc((sim->nCountCN + 1) * int_size);
          coords[k][0] = atm->types[i];     // first index is filled by type identificator (for fast access)
          for (j = 1; j <= sim->nCountCN; j++)
            coords[k][j] = 0;

          for (j = 0; j < atm->nAt; j++)
            if (field->species[atm->types[j]].idCounter)
            {
                r2 = sqr_distance(j, i, atm, bx);

                if (sim->r2CN >= r2)
                {
                   t = field->species[atm->types[j]].idCounter;
                   coords[k][t]++;
                }
            }

          // save maximal CN for output array generation
          for (j = 1; j <= sim->nCountCN; j++)
            if (coords[k][j] > mx)
              mx = coords[k][j];

          k++;
      }
   }

   nPair = sim->nCentrCN * sim->nCountCN;
   //printf("nPair=%d\n", nPair);
   out = (int**)malloc(nPair * pointer_size);
   for (i = 0; i < nPair; i++)
   {
      out[i] = (int*)malloc((mx + 1) * int_size);    // number of possible quantites of ligand
      for (j = 0; j <= mx; j++)
        out[i][j] = 0;
   }

   // fill output array
   //printf("output filling nA=%d\n", nA);
   for (i = 0; i < nA; i++)
   {
      // calculat index of pair (central atom part):
      t = (field->species[coords[i][0]].idCentral - 1) * sim->nCountCN;
      //printf("type=%d, central=%d, t=%d\n", coords[i][0], field->species[coords[i][0]].idCentral, t);
      for (j = 1; j <= sim->nCountCN; j++)
      {
         //printf("ind=%d\n", t);
         out[t][coords[i][j]]++;
         t++;
      }
   }

   f = fopen(fname, "w");
   if (f == NULL)
     return 0; // error

   // header
   fprintf(f, "CN");
   for (i = 1; i <= sim->nCentrCN; i++)
     for (j = 1; j <= sim->nCountCN; j++)
       fprintf(f, "\t%s-%s", field->snames[cenTypes[i]], field->snames[ligTypes[j]]);
   fprintf(f, "\n");

   for (i = 0; i <= mx; i++)
   {
      fprintf(f, "%d", i);
      for (j = 0; j < nPair; j++)
        fprintf(f, "\t%d", out[j][i]);
      fprintf(f, "\n");
   }
   fclose(f);

   for (i = 0; i < nPair; i++)
       free(out[i]);
   free(out);
   for (i = 0; i < nA; i++)
       free(coords[i]);
   free(coords);
   free(ligTypes);
   free(cenTypes);
   return 1;
}

void traj_header(FILE *f, Atoms *at, Spec *sp, Sim *sim)
{
//   fprintf(f, "time iStep x0 y%s0 x1 y%s1 x2 y%s2 x3 y%s3 x18 y%s18 x20 y%s20\n", s0, s1, s2, s3, s18, s20);
   int i;
   //char *s;

   fprintf(f, "time,ps\tiStep");
   for (i = sim->at1Traj; i < sim->at2Traj; i++)
   {
      //s = sp[at->types[i]].name;
      fprintf(f, "\tx%d,A\ty%s%d", i, sp[at->types[i]].name, i);
   }
   fprintf(f, "\n");
}

void traj_info(FILE *f, double tm, int step, Atoms *at, Spec *sp, Sim *sim)
{
   int i;
   //char *s;

   fprintf(f, "%f\t%d", tm, step);
   for (i = sim->at1Traj; i < sim->at2Traj; i++)
   {
      //s = sp[at->types[i]].name;
      fprintf(f, "\t%f\t%f", at->xs[i], at->ys[i]);
   }
   fprintf(f, "\n");
}

// VAF (velocity autocorellation function)
void vaf_init(Atoms *atm)
{
   int i;
   for (i = 0; i < atm->nAt; i++)
   {
       atm->vx0[i] = atm->vxs[i];
       atm->vy0[i] = atm->vys[i];
       atm->vz0[i] = atm->vzs[i];
   }
}

void vaf_header(FILE *f, Field *field, Sim *sim)
{
   int i;

   fprintf(f, "time,ps\tiStep");
   for (i = 0; i < field->nSpec; i++)
   {
      fprintf(f, "\t%s", field->snames[i]);
   }
   fprintf(f, "\n");
}

void vaf_info(FILE *f, double tm, int step, Atoms *atm, Field *field, Sim *sim)
{

   int i, t;
   for (i = 0; i < field->nSpec; i++)
   {
      field->species[i].vaf = 0.0;
   }

   for (i = 0; i < atm->nAt; i++)
   {
       t = atm->types[i];
       field->species[t].vaf += atm->vxs[i] * atm->vx0[i] + atm->vys[i] * atm->vy0[i] + atm->vzs[i] * atm->vz0[i];
   }

   fprintf(f, "%f\t%d", tm, step);
   for (i = 0; i < field->nSpec; i++)
   {
      if (field->species[i].number)
        field->species[i].vaf /= field->species[i].number;
      fprintf(f, "\t%f", field->species[i].vaf);
   }
   fprintf(f, "\n");
}

