//MODULE angles.cpp
#include <stdlib.h>  // malloc, alloc, rand, NULL
#include <stdio.h>   // *FILE
#include <string.h>  //strcmp

#include "utils.h"  // int_size, pointer_size
#include "dataStruct.h" // struct Bond, Atoms
#include "const.h" // r_scale
#include "box.h" // rect, box
#include "angles.h"
#include "sys_init.h"   // spec_by_name

void alloc_angles(int n, Field *field)
{
   field->mxAngles = n;
   field->centrs = (int*)malloc(field->mxAngles * int_size);
   field->lig1 = (int*)malloc(field->mxAngles * int_size);
   field->lig2 = (int*)malloc(field->mxAngles * int_size);
   field->angTypes = (int*)malloc(field->mxAngles * int_size);
}

void read_anglelist(Atoms *atm, Field *field, Sim *sim)
{
   int i, n, x;
   FILE *f = fopen("angles.txt", "r");
   if (f == NULL)
   {
       printf("WARNING[a002] angle list is used, but there is no such file. No anlges are downloaded\n");
       //! âðåìåííî! ýòà øòóêà è çàäàåò ìàêñèìàëüíîå êîëè÷åñòâî óãëîâ è àëëîöèðóåò ìåñòî. Íàäî ÷òîáû ìàêñèìàëüíîå ÷èñëî óãëîâ çàãðóæàëîñü èç äèðåêòèâû
       alloc_angles(6000, field);
       return;
   }

   fscanf(f, "%d", &n);
   field->nAngles = n;
   if (n)
       sim->use_angl = max(sim->use_angl, 1);       // can be already set as 2 by read_linkage routine
   if (n)
       alloc_angles(3000 + n, field);
   else 
       if (sim->use_angl == 2) // åñëè íå çàäàíî óãëîâ, íî ìû çíàåì, ÷òî îíè ìîãóò îáðàçîâûâàòüñÿ - âûäåëÿåì ïàìÿòü
           alloc_angles(4000, field);   //! temp value


   for (i = 0; i < n; i++)
   {
      fscanf(f, "%d %d %d %d", &field->centrs[i], &field->lig1[i], &field->lig2[i], &x);
      //barrs->types[i]--;
      //printf("%d %d %d\n", barrs->at1[i], barrs->at2[i], barrs->types[i]);
      if (x && (x < field->nAdata))
        field->angTypes[i] = x;
      else
        printf("ERROR[013] wrong atom type number in angles.txt\n");

      if (atm->types[field->centrs[i]] != field->adata[x].central)
        printf("ERROR[014] wrong central atom type (%s) in angle list (%d postion). Must be %s.\n", field->species[atm->types[field->centrs[i]]].name, i, field->species[field->adata[x].central].name);
   }
   //printf("angTypes=%d\n", field->angTypes);
   fclose(f);
}

void save_anglelist(char *fname, Field *field)
{
   int i, n;
   FILE *f = fopen(fname, "w");

   n = field->nAngles;
   fprintf(f, "%d\n", n);
   for (i = 0; i < n; i++)
   {
      fprintf(f, "%d %d %d %d\n", field->centrs[i], field->lig1[i], field->lig2[i], field->angTypes[i]);
   }
   fclose(f);
}

// read angle parameters from file and put them into structure
//   (id) - angle number
int read_angle(int id, FILE *f, Field *field)
{
   int n, i;
   double p0, p1/*, p2*/;
   char s1[8], key[8];
   int ind1;
   Angle *ang = &field->adata[id];

   fscanf(f, "%d %8s %8s %lf %lf", &n, s1, key, &p0, &p1);

   /*
   // define spec indexes
   ind1 = 0;
   for (i = 0; i < field->nSpec; i++)
    {
         if (strcmp(field->species[i].name, s1) == 0)
         {
            ind1 = i + 1;
            break;
         }
    }
   if (!ind1)
   {
      printf("ERROR[011]: Unknown species in angle declaration: %d %s %s", n, s1, key);
      return 0;
   }
   ang->central = ind1-1;
   */

   if (spec_by_name(field, s1, ind1))
       ang->central = ind1;
   else
       printf("ERROR[011]: Unknown species in angle declaration: %d %s %s", n, s1, key);

   //potential parameters
   if (strcmp(key, "hcos") == 0)  // U = 1/2 k (cos-cos_0)^2
     {
        ang -> type = 1;
        ang -> p0 = p0;     // k (eV)
        ang -> p0 *= E_scale;
        ang -> p1 = p1;     // cos0 (dimensionless)
        //ang -> p1 *= r_scale;
     }
   else
     {
        printf("ERROR[012]: Unknown potential type in angle declaration: %d %s", n, key);
        return 0;
     }

   return 1;
}

// delete angles near at1 and at2
void destroy_angles(int at1, int at2, Field *field)
{
   int i, j;
   int n = 0;

   //printf("call destroy angle: ");
   j = field->nAngles - 1;
   for (i = 0; i < field->nAngles; i++)
   {
      if ((at1 == field->centrs[i]) || (at2 == field->centrs[i]))
      {
         field->centrs[i] = field->centrs[j];
         field->lig1[i] = field->lig1[j];
         field->lig2[i] = field->lig2[j];
         field->angTypes[i] = field->angTypes[j];
         j--; //! âîîáùå òóò åñòü ïîòåíöèàëüíàÿ îïàñíîñòü, ÷òî ìû ïåðåçàïèøåì óäàëÿåìûé óãîë (åñëè íóæíî óäàëèòü îáà è ïîñëåäíèé - óäàëÿåìûé)
         n++;
         if (n > 1)
           break;
      }

   }
   //printf("%d items\n", n);
   field->nAngles -= n;
}

// try to create new angle in angle_list and return success or not
int create_angle(int c, int l1, int l2, int type, Field *field)
{
   //printf("create angle proc!\n");
   int n = field->nAngles;

   if (n < field->mxAngles)
   {
      field->centrs[n] = c;
      field->lig1[n] = l1;
      field->lig2[n] = l2;
      field->angTypes[n] = type;
      field->nAngles++;
      return 1;
   }
   else
   {
      printf("WARNING[116] maximal number of angles is reached!\n");
      return 0;
   }
}

void angle_iter(Atoms *atm, int c, int l1, int l2, Box *bx, Angle *ang, double &eng)
// calculate energy(&eng) and forces on atoms (c, l1 and l2 are atom indexes
{
   double k, cos0;
   //! ïîêà ó íàñ áóäåò òîëüêî îäíà ôóíêöèÿ, ãàðìîíè÷åñêèé êîñèíóñ U = k / 2 * (cos(th)-cos(th0))^
   if (ang->type == 1)
   {
      k = ang->p0;
      cos0 = ang->p1;
   }

   //! è òóò åù¸ ìîæíî ñõèòðèòü, ñðàçó âçÿòü ðàññòîÿíèÿ èç bonds, âåäü angle ìîæåò áûòü òîëüêî ìåæäó bonds

   // vector ij
   double xij = atm->xs[l1] - atm->xs[c];
   double yij = atm->ys[l1] - atm->ys[c];
   double zij = atm->zs[l1] - atm->zs[c];
   delta_periodic(xij, yij, zij, bx);
   double r2ij = xij*xij + yij*yij + zij*zij;
   double rij = sqrt(r2ij);

   // vector ik
   double xik = atm->xs[l2] - atm->xs[c];
   double yik = atm->ys[l2] - atm->ys[c];
   double zik = atm->zs[l2] - atm->zs[c];
   delta_periodic(xik, yik, zik, bx);
   double r2ik = xik*xik + yik*yik + zik*zik;
   double rik = sqrt(r2ik);

   double cos_th = (xij * xik + yij * yik + zij * zik) / rij / rik;
   double dCos = cos_th - cos0; // delta cosinus

   double c1 = -k * dCos;
   double c2 = 1.0 / rij / rik;

   atm->fxs[c] += -c1 * (xik*c2 + xij*c2 - cos_th * (xij/r2ij + xik/r2ik));
   atm->fys[c] += -c1 * (yik*c2 + yij*c2 - cos_th * (yij/r2ij + yik/r2ik));
   atm->fzs[c] += -c1 * (zik*c2 + zij*c2 - cos_th * (zij/r2ij + zik/r2ik));

   atm->fxs[l1] += c1 * (xik*c2 - cos_th*xij/r2ij);
   atm->fys[l1] += c1 * (yik*c2 - cos_th*yij/r2ij);
   atm->fzs[l1] += c1 * (zik*c2 - cos_th*zij/r2ij);

   atm->fxs[l2] += c1 * (xij*c2 - cos_th*xik/r2ik);
   atm->fys[l2] += c1 * (yij*c2 - cos_th*yik/r2ik);
   atm->fzs[l2] += c1 * (zij*c2 - cos_th*zik/r2ik);

   eng += 0.5 * k * dCos * dCos;
}

void exec_anglelist(Atoms *atm, Field *field, Sim *sim, Box *bx)
{
   int i;
   //double dx, dy, dz, r2, f;
   double eng = 0.0;

   for (i = 0; i < field->nAngles; i++)
   {
      angle_iter(atm, field->centrs[i], field->lig1[i], field->lig2[i], bx, &field->adata[field->angTypes[i]], eng);
   }

   sim->engAngle = eng;
   //printf("eng=%f f=%f r2=%f  r0^2=%f\n", eng, f, r2, btypes[barrs->types[i]].p1*btypes[barrs->types[i]].p1);
}

void free_angles(Field *field)
{
    if (field->nAdata)
    {
        if (field->nAngles)
        {
            free(field->centrs);
            free(field->lig1);
            free(field->lig2);
            free(field->angTypes);
        }

        free(field->adata);
    }
}

