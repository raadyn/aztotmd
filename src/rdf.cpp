// UNIT for RDF calculations (RDF variables are declared in struct Sim)

#include <stdlib.h>     // malloc, alloc, rand, NULL
#include <stdio.h>      // FILE
#include <math.h>       // log, sqrt
#include <string.h>     // strcmp

#include "dataStruct.h"  // Sim, Box, Atoms ....
#include "utils.h"      // int_size, pointer_size, etc...
#include "const.h"      // sphera consts
#include "box.h"        // rect_periodic
#include "rdf.h"

int read_rdf(FILE *f, Sim *sim)
{
    char str[235];
    int res = 1;

    if (find_double(f, " rdf %lf ", sim->rRDF))
    {
        fscanf(f, " %lf %d %d", &sim->dRDF, &sim->frRDF, &sim->frRDFout);
        sim->idRDF = 1.0 / sim->dRDF;
        sim->r2RDF = sim->rRDF * sim->rRDF;

        // use or not nuclei rdf
        fscanf(f, " %s", str);
        if (strcmp(str, "nucl") == 0)
            sim->nuclei_rdf = 1;
        else
            sim->nuclei_rdf = 0;
    }
    else
    {
        printf("ERROR[408] No rdf directive in control.txt file!\n");
        res = 0;
    }
    return res;
}

void init_rdf(Sim* sim, Box* box)
{
    /// íàäî ñäåëàòü, ÷òîáû ïðè ñ÷èòûâàíèè ïàðàìåòðîâ, âñå ïðîèçâîäíûå àâòîìàòè÷åñêè ðàñ÷èòûâàëèñü
    double minR = sim->rRDF;
    if (minR > box->la)  //! temp
        minR = box->la;
    sim->nRDF = minR * sim->idRDF;
}

int alloc_rdf(Sim *sim, Field *field, Box *box)
// create arrays and some values for RDF calculations
{
   int i, j;
   double minR = sim->rRDF;

   // counters and helper variables
   sim->nRDFout = 0;

   //! ÷àñòü ïîâòîðÿåòñÿ ñ init_rdf, ïîñìîòðåòü
   //seek min of sim.maxR or box.maxLength
   if (minR > box->maxLength)
     minR = box->maxLength;

   sim->nRDF = minR * sim->idRDF;
   if (!sim->nRDF)
     return 0;  // dR > R - no RDF points, go away

   // create arrays of nPair arrays
   sim->rdf = (double**)malloc(field->nPair * pointer_size);
   for (i = 0; i < field->nPair; i++)
     {
        sim->rdf[i] = (double*)malloc(sim->nRDF * double_size);
        for (j = 0; j < sim->nRDF; j++)
          sim->rdf[i][j] = 0.0;
     }

   return 1;
}
// end 'init_rdf' function

/*
void clear_rdf(Sim *sim, Field *field, int &nRDF)
// set RDF values to 0.0
{
   int i, j;

   for (i = 0; i < field->nPair; i++)
     for (j = 0; j < sim->nRDF; j++)
       sim->rdf[i][j] = 0.0;

   sim->nRDFout = 0;
}
// end 'clear_rdf' function
*/

void get_rdf(Atoms *atm, Sim *sim, Field *field, Box *box)
// single sampling for RDF calculation
{
   int i, j, iR;
   int iMin, iMax, iPair;
   double r2;
   int m = field->nSpec - 1;
   int N = atm->nAt;

   for(i = 0; i < N-1; i++)
     for(j = i + 1; j < N; j++)
     {
          r2 = sqr_distance(i, j, atm, box);

          //! accelerate determination of pair index
          if (r2 < sim->r2RDF)
          {
               iR = sqrt(r2) * sim->idRDF;
               iMin = atm->types[i];
               iMax = atm->types[j];
               if (iMin > iMax)
                 {
                    iMin = atm->types[j];
                    iMax = atm->types[i];
                 }
               iPair = iMin * m + iMin * (1 - iMin) / 2 + iMax;
               sim->rdf[iPair][iR] += 1.0;  //! maby add /nA/nB
          }
     }

   sim->nRDFout++;
}
// end 'get_rdf' function

int out_rdf(Field *field, Box *box, Sim *sim, char *fname)
// write radial distribution functions in file
{
   int i, j, k;
   int nP = field->nPair;
   double *nAnB = (double*)malloc(nP * double_size); // array for keeping production Na * Nb for every pair
   FILE *f;
   double dr3 = sim->dRDF * sim->dRDF * sim->dRDF;

   // constant factor for RDF calculation:
   double cnst = box->vol / sphera / dr3 / sim->nRDFout;

   f = fopen(fname, "w");
   if (f == NULL)
     return 0;

   // print header and calculate production of particle numbers
   k = 0;
   fprintf(f, "r ");
   for (i = 0; i < field->nSpec; i++)
     for (j = i; j < field->nSpec; j++)
     {
         fprintf(f, "%s-%s ", field->species[i].name, field->species[j].name);
         nAnB[k] = field->species[i].number * field->species[j].number;
         if (i == j)
           nAnB[k] *= 0.5; // äëÿ îäèíàêîâîé ïàðû ìû äîëæíû äîìíîæèòü ðåçóëüòàò íà 2 (èëè â äâà ðàçà óìåíüøèòü çíàìåíòàåëü)
         k++;
     }
   fprintf(f, "\n");

   // calculate RDF as: g(AB) = npair(AB) * V / v(ball_layer) / Na / Nb / N_sampling
   for (i = 0; i < sim->nRDF; i++)
   {
       fprintf(f, "%f ", (i + 0.5) * sim->dRDF);
       for (j = 0; j < nP; j++)
       {
            if (nAnB[j])
              sim->rdf[j][i] *= cnst / (3 * i * (i + 1) + 1) / nAnB[j];
            //! box volume and particle numbers also can vary, maybe add /nAnB[i] into get_rdf ? and keep the number of sampling for each pair

            fprintf(f, "%4.2E ", sim->rdf[j][i]);
       }
       fprintf(f, "\n");
   }

   fclose(f);
   free(nAnB);
   return 1; // success
}
// end 'out_rdf' function

void free_rdf(Sim *sim, Field *field)
// free RDF arrays
{
   int i;

   for (i = 0; i < field->nPair; i++)
       free(sim->rdf[i]);

   free(sim->rdf);
}
// end 'free_rdf' function
