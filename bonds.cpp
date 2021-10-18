//UNIT of covalent bonds (intramolecular interaction)
// azTotMD (by Anton Raskovalov)

#include <stdlib.h>  // malloc, alloc, rand, NULL
#include <stdio.h>   // *FILE
#include <string.h>  //strcmp

#include "utils.h"      // int_size, pointer_size
#include "dataStruct.h" // struct Bond, Atoms
#include "const.h"      // r_scale
#include "box.h"        // rect, box
#include "vdw.h"        // vdw_iter
#include "angles.h"
#include "sys_init.h"   // twospec_by_name

void alloc_bonds(int n, Field *field)
// allocate memory for bonds array
{
   field->mxBonds = n;
   field->at1 = (int*)malloc(field->mxBonds * int_size);
   field->at2 = (int*)malloc(field->mxBonds * int_size);
   field->bTypes = (int*)malloc(field->mxBonds * int_size);
}

int read_bondlist(Atoms *atm, Field *field, Sim *sim)
// read bonds from file
{
   int i, at1, at2, n, k;
   int res = 1;     // function result
   FILE *f = fopen("bonds.txt", "r");
   if (f == NULL)
   {
       printf("WARNING[a001] bond list is used, but there is no such file. No bonds are downloaded\n");
       //! временно! эта штука и задает максимальное количество связи и аллоцирует место. Надо чтобы максимальное число связей загружалось из директивы
       alloc_bonds(6000, field);
       return 0;
   }
   Bond *btypes = field->bdata;

   fscanf(f, "%d", &field->nBonds);
   if (field->nBonds)
       sim->use_bnd = max(sim->use_bnd, 1);     // maybe already set as 2 by read_bonds
   alloc_bonds(5000 + field->nBonds, field); //! allocation with excess

   for (i = 0; i < field -> nBonds; i++)
   {
      fscanf(f, "%d %d %d", &at1, &at2, &k);
      field->bTypes[i] = k;

      // turn bond (if its necessary) and verify its type
      if (btypes[k].spec1 == atm->types[at1])
      {
          if (btypes[k].spec2 != atm->types[at2])
          {
             printf("ERROR [121] incorrect type of 2th atom in bond (type: %d, line: %d)\n", k, i);
             res = 0;
          }
      }
      else
        if (btypes[k].spec1 == atm->types[at2])
        {
           if (btypes[k].spec2 == atm->types[at1])
           {
             n = at1;
             at1 = at2;
             at2 = n;
           }
           else
           {
               printf("ERROR [122] incorrect type of 1th atom in bond (type: %d, line: %d)\n", k, i);
               res = 0;
           }
        }
        else
        {
            printf("ERROR [123] incorrect type of atoms %s and %s for bond type(%d) in bond list, line: %d\n", field->snames[atm->types[at1]], field->snames[atm->types[at2]], k, i);
            res = 0;
        }

      btypes[k].number++;
      field->at1[i] = at1;
      field->at2[i] = at2;

      // H-bonds features:
      if (btypes[k].hatom == -1)    // usual bond
      {
          atm->nBonds[at1]++;
          atm->nBonds[at2]++;
          atm->parents[at1] = at2;
          atm->parents[at2] = at1;
      }
      else  // H-bond
      {
          if (atm->types[at1] == btypes[k].hatom)
              atm->parents[at1] = at2;
          else
              atm->parents[at2] = at1;
      }
   }

/*
   //autoparents verification:
   for (i = 0; i < atm->nAt; i++)
     if (atm->parents[i] == i)
       printf("autoparent %d\n", i);
*/
   fclose(f);
   return res;
}
// end 'read_bondlist' function

void save_bondlist(char *fname, Field *field)
{
   int i;
   FILE *f = fopen(fname, "w");

   fprintf(f, "%d\n", field->nBonds);
   for (i = 0; i < field->nBonds; i++)
   {
      fprintf(f, "%d %d %d\n", field->at1[i], field->at2[i], field->bTypes[i]);
   }
   fclose(f);
}

int read_bond(int id, FILE *f, Field *field, Sim *sim)
// read bond parameters from file and put them into bond structure
//   (id) - index of the bond in bdata array
{
   int n, br;
   double r, p0, p1, p2, p3, p4;
   char s1[8], s2[8], key[8];
   int ind1, ind2;
   Bond *bond = &field->bdata[id];

   fscanf(f, "%d %8s %8s %8s", &n, s1, s2, key);

   // define species indexes
   if (!twospec_by_name(field, s1, s2, ind1, ind2))
   {
      printf("ERROR[124]: Unknown species in bonds declaration: %d %s %s", n, s1, s2);
      return 0;
   }

   bond->hatom = -1;        // no hydrogen bond
   bond->evol = 0;
   bond->spec1 = ind1;
   bond->spec2 = ind2;
   bond->number = 0;

   //! save bond type as default between two species (later - to introduce flag, default bond)
   field->bond_matrix[ind1][ind2] = id;
   if (ind1 != ind2)
       field->bond_matrix[ind2][ind1] = -id; // negative index means that we need change atom places
   else
       field->bond_matrix[ind2][ind1] = id;

   if (strcmp(key, "harm") == 0)  // U = 1/2 k (r-r0)^2
     {
        bond -> type = 1;
        fscanf(f, "%lf %lf", &p0, &p1);
        bond -> p0 = p0;  // k (Eng / r^2)
        bond -> p0 *= E_scale;
        bond -> p0 /= (r_scale * r_scale);
        bond -> p1 = p1;  // r0 (r)
        bond -> p1 *= r_scale;
     }
   // Morse potential:
   else if (strcmp(key, "mors") == 0)  // U = D[1 - exp(-a(r-r0))]^2 - C
     {
        bond -> type = 2;
        fscanf(f, "%lf %lf %lf %lf", &p0, &p1, &p2, &p3);
        bond -> p0 = p0;            // D (Eng)
        bond -> p0 *= E_scale;

        p1 /= (r_scale * r_scale);  // a (1/r^2)
        bond -> p1 = p1;

        p2 *= r_scale;              // r0 (r)
        bond -> p2 = p2;

        p3 *= E_scale;              // C (ENg)
        bond -> p3 = p3;
     }
   // Pedone potential (Morse + C/r^12):
   else if (strcmp(key, "pdn") == 0)  // U = D[1 - exp(-a(r-r0))]^2 - C - E/r^12
     {
        bond -> type = 3;
        fscanf(f, "%lf %lf %lf %lf %lf", &p0, &p1, &p2, &p3, &p4);
        bond -> p0 = p0;            // D (Eng)
        bond -> p0 *= E_scale;

        p1 /= (r_scale * r_scale);  // a (1/r^2)
        bond -> p1 = p1;

        p2 *= r_scale;              // r0 (r)
        bond -> p2 = p2;

        p3 *= E_scale;              // C (Eng)
        bond -> p3 = p3;

        p4 *= E_scale;              // E (Eng)
        bond -> p4 = p4;
     }
   // Buckingham potential:
   else if (strcmp(key, "buck") == 0)  // U = A exp(-r/ro) - C/r^6
     {
        bond -> type = 4;
        fscanf(f, "%lf %lf %lf", &p0, &p1, &p2);
        bond -> p0 = p0;            // A (Eng)
        bond -> p0 *= E_scale;

        p1 *= r_scale;  //          ro (r)
        bond -> p1 = p1;

        p2 *= E_scale;              // C (Eng*r6)
        r = r_scale * r_scale * r_scale;
        r = r * r;
        p2 *= r;
        bond -> p2 = p2;
     }
   // exp-6-8-12 potential:
   else if (strcmp(key, "e612") == 0)  // U = A exp(-r/ro) - C/r^6 - D/r^8 - F/r^12
     {
        bond -> type = 5;
        fscanf(f, "%lf %lf %lf %lf %lf", &p0, &p1, &p2, &p3, &p4);
        bond -> p0 = p0;            // A (Eng)
        bond -> p0 *= E_scale;

        p1 *= r_scale;  //          ro (r)
        bond -> p1 = p1;

        p2 *= E_scale;              // C (Eng*r6)
        r = r_scale * r_scale * r_scale;
        r = r * r;
        p2 *= r;
        bond -> p2 = p2;

        p3 *= E_scale;              // D (Eng*r8)
        r = r_scale * r_scale;
        r = r * r; // r^4
        r = r * r;  // r^8
        p3 *= r;
        bond->p3 = p3;

        p4 *= E_scale;              // F (Eng*r12)
        r = r_scale * r_scale * r_scale;
        r = r * r; // r^6
        r = r * r; // r^12
        p4 *= r;
        bond->p4 = p4;
     }
   else
     {
        printf("ERROR[126]: Unknown potential type in bonds declaration: %d %s", n, key);
        return 0;
     }

     // read minimum limit
     fscanf(f, "%8s", key);
     if (strcmp(key, "con") == 0)  // constant
     {
         bond->mnEx = 0;
     }
     else if (strcmp(key, "mut") == 0)    // mutable bond
     {
         sim->use_bnd = 2;      // variable bonds
         bond->mnEx = 1;
         fscanf(f, "%lf %d", &(bond->r2min), &(bond->new_type[0]));
         bond->r2min *= bond->r2min;      // because single r in filed file

         // define that species consist bonds can change their quantity
         //! это нужно делать, после того, как загрузим все связи
     }
     else
     {
         printf("ERROR[501]: Unknown type of lower bond limit (%d): %s", n, key);
         return 0;
     }

     // read maximum limit
     fscanf(f, "%8s", key);
     if (strcmp(key, "con") == 0)  // constant
     {
         bond->mxEx = 0;
     }
     else if (strcmp(key, "mut") == 0)    // mutable bond
     {
         sim->use_bnd = 2;      // variable bonds
         bond->mxEx = 1;
         fscanf(f, "%lf %d", &(bond->r2max), &(bond->new_type[1]));
         bond->r2max *= bond->r2max;      // because single r in filed file

         // define that species consist bonds can change their quantity
         //! это нужно делать, после того, как загрузим все связи
     }
     else if (strcmp(key, "br") == 0)    // breakable  bond
     {
         sim->use_bnd = 2;      // variable bonds
         bond->mxEx = 1;
         fscanf(f, "%lf %8s %8s", &(bond->r2max), s1, s2);
         bond->new_type[1] = 0;
         bond->r2max *= bond->r2max;      // because single r in filed file
         if (!twospec_by_name(field, s1, s2, ind1, ind2))
         {
             printf("ERROR[503]: Unknown species in break bond %d declaration: %s %s", n, s1, s2);
             return 0;
         }
         bond->new_spec1[1] = ind1;
         bond->new_spec2[1] = ind2;
         if (bond->spec1 != ind1)
         {
             field->species[bond->spec1].varNumber = 1;
             field->species[ind1].varNumber = 1;
         }
         if (bond->spec2 != ind2)
         {
             field->species[bond->spec2].varNumber = 1;
             field->species[ind2].varNumber = 1;
         }
     }
     else
     {
         printf("ERROR[502]: Unknown type of upper bond limit (%d): %s", n, key);
         return 0;
     }

/*
   // read breaking bond parameters
   if (bond->breakable)
   {
       // set particiated species as with variable number, original species
       field->species[ind1].varNumber = 1;
       field->species[ind2].varNumber = 1;

       fscanf(f, "%lf %8s %8s", &r, s1, s2);
       if (!twospec_by_name(field, s1, s2, ind1, ind2))
       {
           printf("ERROR[125]: Unknown species in bonds declaration: %d %s %s", n, s1, s2);
           return 0;
       }
       bond->rMax2 = r * r;
       bond->spec1br = ind1;
       bond->spec2br = ind2;
       bond->energy = field->species[bond->spec1].energy + field->species[bond->spec2].energy - field->species[bond->spec1br].energy - field->species[bond->spec2br].energy;
       // set particiated species as with variable number, speiceis after breaking
       field->species[ind1].varNumber = 1;
       field->species[ind2].varNumber = 1;
   }

   // read  parameters for bond mutation (bond which can transform into another bond type)
   if (bond->mut)
   {
       // set particiated species as with variable number, speiceis after breaking
       field->species[ind1].varNumber = 1;
       field->species[ind2].varNumber = 1;

       fscanf(f, "%lf %d", &r, &n);
       bond->rMax2 = r * r;
       bond->newBond = n - 1;
       //! probably energy must be function of new bond type energy
   }
*/
   return 1;
}

// recalculate number of bonds of certain types (new type - with shift)
void replace_bondtype(int bond, int atom, int newAtomType, Atoms *atm, Field *field)
{
   int bt, left, new_type;

   field->bdata[field->bTypes[bond]].number--;

   //int at1 = atm->types[field->at1[bond]];
   //int at2 = atm->types[field->at2[bond]];

   if (field->at1[bond] == atom)
    {
        bt = atm->types[field->at2[bond]];
        new_type = field->bond_matrix[newAtomType][bt];
        //new_type = field->bond_matrix[t1aft][bt2];
    }
    else
    {
        bt = atm->types[field->at1[bond]];
        new_type = field->bond_matrix[bt][newAtomType];
        //new_type = field->bond_matrix[bt1][t1aft];
    }

    if (new_type)
    {
        if (new_type < 0)
        {
              field->bTypes[bond] = -new_type;
              //change atom places
              left = field->at1[bond];
              field->at1[bond] = field->at2[bond];
              field->at2[bond] = left;
              field->bdata[-new_type].number++;
        }
        else
        {
              field->bTypes[bond] = new_type; // as index=0 reserved
              field->bdata[new_type].number++;
        }
    }
}

// try to create new bond in bond_list and return success or not
int create_bond(int at1, int at2, int type, Atoms *atm, Field *field)
{
   int j, n, k, t1aft, t2aft, t1bef, t2bef;
   int p1, p2; // parent indexes
   int i = field->nBonds;

   field->bdata[type].number++;

   //printf("bond[%d]: (%s %s  -> %s %s for %s and %s\n", type, field->species[btypes[type].spec1].name, field->species[btypes[type].spec2].name, sp[btypes[type].spec1br].name, sp[btypes[type].spec2br].name, sp[atm->types[at1]].name, sp[atm->types[at2]].name);
   //printf("try to create #%d between %s and %s\n", type, field->species[atm->types[at1]].name, field->species[atm->types[at2]].name);

/* NOW CORRECT BOND TYPES VERIFIED NOT HERE
   // сразу развернем связь как надо и проверим, правильно ли связана
   if (field->bdata[type].spec1br == atm->types[at1])
      {
          if (field->bdata[type].spec2br != atm->types[at2])
            printf("(A)create wrong bond type[%d] between %s and %s\n", type+1, field->species[atm->types[at1]].name, field->species[atm->types[at2]].name);
      }
   else
      {
        //! тут ещё проверку правильный ли первый атом
        if (field->bdata[type].spec2br != atm->types[at1])
            printf("(B)create wrong bond type[%d] between %s and %s\n", type+1, field->species[atm->types[at1]].name, field->species[atm->types[at2]].name);

        if (field->bdata[type].spec1br == atm->types[at2])
        {
           n = at1;
           at1 = at2;
           at2 = n;
        }
        else
          printf("(C)create wrong bond type[%d] between %s and %s\n", type+1, field->species[atm->types[at1]].name, field->species[atm->types[at2]].name);
      }
*/

   //!!!проверка все ли связи нормальные
   /*
   int flag = 0;
   int obond = -1;
   //printf("1 before routine verif\n");
   for (j = 0; j < field->nBonds; j++)
      {
        if (field->at1[j] < 0)
          continue;
        bt = field->bTypes[j];
        if ((field->bdata[bt].spec1 != atm->types[field->at1[j]])||(field->bdata[bt].spec2 != atm->types[field->at2[j]]))
          {
             flag = 1;
             break;
          }
      }
   if (flag)
     printf("bad initial bonds\n");
   flag = 0;
   //printf("2 end pred verif\n");
   */

   /*
   for (j = 0; j < field->nBonds; j++)
     for (k = j + 1; k < field->nBonds - 1; k++)
        if (field->at1[j] > -1)
          if (((field->at1[j] == field->at1[k])&&(field->at2[j] == field->at2[k])) || ((field->at1[j] == field->at2[k])&&(field->at2[j] == field->at1[k])))
     {
            printf("(1)dublicate bond #%d = %d!\n", j, k);
            field->at1[k] = -1;
     }
   */


   if (i < field->mxBonds)
   {
      field->at1[i] = at1;
      field->at2[i] = at2;
      field->bTypes[i] = type;
      t1bef = atm->types[at1];
      t2bef = atm->types[at2];
      //printf("1\n");

      //change atoms and old bonds to him (+add valent angle! - this is in external place)
      t1aft = field->bdata[type].spec1;
      t2aft = field->bdata[type].spec2;

      //! сразу поменяем типы, чтобы не было противоречий
      atm->types[at1] = t1aft;
      atm->types[at2] = t2aft;

      // change types of existing bond:
      n = atm->nBonds[at1] + atm->nBonds[at2];

      k = 0;
      for (j = 0; j < field->nBonds; j++)
        if (field->at1[j] > -1)    // flag of brockeng bond!
        {

        if ((field->at1[j] == at1) || (field->at2[j] == at1))
        {
            replace_bondtype(j, at1, t1aft, atm, field);
            k++;
        }
        else
        if ((field->at1[j] == at2) || (field->at2[j] == at2))
        {
            replace_bondtype(j, at2, t2aft, atm, field);
            k++;
        }
        if (k == n)
          break;

        //VERIFICATION
        //bt = field->bTypes[j];
        //if ((field->bdata[bt].spec1 != atm->types[field->at1[j]])||(field->bdata[bt].spec2 != atm->types[field->at2[j]]))
        //  printf("error 3\n");

        }  // if (field->at1[j] != -1)
      //printf("3\n");

      //! temp: add angle!
      p1 = atm->parents[at1];
      p2 = atm->parents[at2];
      if (field->species[t1aft].angleType) // O(c)
      {
         //printf("new angle: %s[%d]: %s[%d] %s[%d]\n", field->species[t1aft].name, at1, field->species[t2aft].name, at2, field->species[atm->types[atm->parents[at1]]].name, atm->parents[at1]);
         create_angle(at1, at2, p1, field->species[t1aft].angleType, field);
      }
      if (field->species[t2aft].angleType)
      {
         //printf("new angle2: %s[%d]\n", field->species[t2aft].name, at2);
         create_angle(at2, at1, p2, field->species[t2aft].angleType, field);
      }
      //printf("end add angle\n");

      /*
      for (j = 0; j < field->nAngles; j++)
        for (k = j + 1; k < field->nAngles - 1; k++)
           if (field->centrs[j] == field->centrs[k])
              printf("dublicated angle[%d = %d]\n", j, k);
      */


      //! сделал это в начале процедуры
      //atm->types[at1] = t1aft;
      //atm->types[at2] = t2aft;
      field->species[t1bef].number--;
      field->species[t2bef].number--;
      field->species[t1aft].number++;
      field->species[t2aft].number++;
      atm->nBonds[at1]++;
      atm->nBonds[at2]++;
      if (atm->parents[at1] < 0)
        atm->parents[at1] = at2;
      if (atm->parents[at2] < 0)
        atm->parents[at2] = at1;
      field->nBonds++;
      //printf("4\n");
      //printf("Bond[%d] %s[%d]-%s[%d] between %s and %s is created!\n", type, sp[t1aft].name, at1, sp[t2aft].name, at2, sp[t1bef].name, sp[t2bef].name);

      /*
      //!!!проверка все ли связи нормальные
      //printf("after bonding verif\n");
      flag = 0;
      for (j = 0; j < field->nBonds; j++)
      {
        if (field->at1[j] < 0)
          continue;
        bt = field->bTypes[j];
        if ((field->bdata[bt].spec1 != atm->types[field->at1[j]])||(field->bdata[bt].spec2 != atm->types[field->at2[j]]))
        {
          printf("after creating wrong bond type[%d]=%d. Atoms:%d and %d\n", j, bt, field->at1[j], field->at2[j]);
          flag = 1;
        }
      }
      if (flag)
        printf("nBonds = %d\n", field->nBonds);
      //printf("end verif\n");
      */

      /*
      for (j = 0; j < field->nBonds; j++)
       for (k = j + 1; k < field->nBonds - 1; k++)
          if (field->at1[j] > -1)
            if (((field->at1[j] == field->at1[k])&&(field->at2[j] == field->at2[k])) || ((field->at1[j] == field->at2[k])&&(field->at2[j] == field->at1[k])))
              printf("dublicate bond after creating #%d = %d!\n", j, k);
      */

      return 1;
   }
   else
   {
      printf("WARNING[115] maximal number of bonds is reached!\n");
      return 0;
   }
}

// destroy bond with bnd index (via bond breaking)
void destroy_bond(int bnd, Atoms *atm, Field *field, Bond *bond)
{
   int a1, a2, p1, p2;
   int j, n, k, t1aft, t2aft, t1bef, t2bef;

   //printf("call destroy bond\n");
   a1 = field->at1[bnd];
   a2 = field->at2[bnd];

   t1bef = atm->types[a1];
   t2bef = atm->types[a2];

   t1aft = bond->new_spec1[1];
   t2aft = bond->new_spec2[1];

   bond->number--;

   //! теперь это в начале процедуры на всякий случай
   atm->types[a1] = t1aft;
   atm->types[a2] = t2aft;

/*
   //!Поиск дубликатных связей
   for (j = 0; j < field->nBonds; j++)
     for (k = j + 1; k < field->nBonds - 1; k++)
        if (field->at1[j] > -1)
          if (((field->at1[j] == field->at1[k])&&(field->at2[j] == field->at2[k])) || ((field->at1[j] == field->at2[k])&&(field->at2[j] == field->at1[k])))
            printf("dublicate before destroy #%d = %d!\n", j, k);
*/


   //change types of resting bonds
   k = 0;
   n = atm->nBonds[a1] + atm->nBonds[a2] - 1;
   p1 = -1; p2 = -1;
   for (j = 0; j < field->nBonds; j++)
     if (j != bnd)
     if (field->at1[j] > -1)  // skip broken bonds
     {
        //! temp (:???)
        if ((field->at1[j] == a1) || (field->at2[j] == a1)) // the bond connected with the first atom
        {

            //save possible parent:
            p1 = field->at1[j];
            if (p1 == a1)
              p1 = field->at2[j];

            replace_bondtype(j, a1, t1aft, atm, field);
            k++;
            if (k == n)
              break;
        }
        else
        if ((field->at1[j] == a2) || (field->at2[j] == a2)) // the bond connected with the second atom
        {
            //save possible parent:
            p2 = field->at1[j];
            if (p2 == a2)
              p2 = field->at2[j];

            replace_bondtype(j, a2, t2aft, atm, field);
            k++;
            if (k == n)
              break;
        }

     }
   // replace parents (if needed)
   if (atm->parents[a1] == a2)
      atm->parents[a1] = p1;
   if (atm->parents[a2] == a1)
      atm->parents[a2] = p2;

   //! поставил это в начале процедуры для избежания перезаписи
   //atm->types[a1] = t1aft;
   //atm->types[a2] = t2aft;

   field->species[t1bef].number--;
   field->species[t2bef].number--;
   field->species[t1aft].number++;
   field->species[t2aft].number++;
   atm->nBonds[a1]--;
   atm->nBonds[a2]--;
   field->at1[bnd] = -1;   // flag of destroying

/*
   //!Поиск дубликатных связей
   for (j = 0; j < field->nBonds; j++)
     for (k = j + 1; k < field->nBonds - 1; k++)
        if (field->at1[j] > -1)
          if (((field->at1[j] == field->at1[k])&&(field->at2[j] == field->at2[k])) || ((field->at1[j] == field->at2[k])&&(field->at2[j] == field->at1[k])))
            printf("dublicate after destroy #%d = %d!\n", j, k);
*/
   //printf("Bond[%d] between %s and %s is destroyed! Now they are: %s and %s\n", bnd, sp[t1bef].name, sp[t2bef].name, sp[t1aft].name, sp[t2aft].name);
}


// read ability of bond creating
//   nBonds = number of bond types
int read_linkage(FILE *f, int Nlnk, Field *field, int nBonds)
{
    int i, j, k, n, bond;
    int sp1, sp2; // indexeses of species
    char ion[8], ion2[8];
    int nSp = field->nSpec;
    double r;
    field->maxRbind = 0.0;


    for (i = 0; i < Nlnk; i++)
    {
        fscanf(f, "%8s %8s %lf %d", ion, ion2, &r, &k);     // spec1, spec2, binding disance, bond 
        if (!twospec_by_name(field, ion, ion2, sp1, sp2))
        {
            printf("ERROR[504]: Unknown species in linkage %d declaration: %s %s", i, ion, ion2);
            return 0;
        }
        field->bonding_matr[sp1][sp2] = k;
        field->bonding_matr[sp2][sp1] = -k;
        field->bindR2matrix[sp1][sp2] = r * r;
        field->bindR2matrix[sp2][sp1] = r * r;
        field->maxRbind = max(field->maxRbind, r);
    }

   return 1;
}

double bond_iter(double r2, Bond *bnd, double &eng)
// calculate energy and return force of bonded(intramolecular) iteraction  (   Fx/dx = -(1/r)*dU(r)/dr   )
//  r2 - square of distance
{
   double r, x, y, irn, ir2;

   switch (bnd->type)
    {
       case 1:   // U = 1/2 k (r-r0)^2   (k = p0, r0 = p1)
         r = sqrt(r2);
         x = r - bnd->p1; // r - r0

         eng += 0.5 * bnd->p0 * x * x;
         return -bnd->p0 / r * x;       // -dU/dr * (1/r)

       // Morse:
       case 2:   // U = D[1 - exp(-a(r-r0))]^2 - C   (D = p0, a = p1, r0 = p2, C = p3)
         r = sqrt(r2);
         x = r - bnd->p2; // r - r0
         x = exp(-bnd->p1 * x); // exp(-a(r-r0))
         y = 1 - x;

         eng += bnd->p0 * y * y - bnd->p3;
         return -2.0 * bnd->p0 * bnd->p1 * x * y / r;

       // Pedone (morse + E/r^12):
       case 3:   // U = D[1 - exp(-a(r-r0))]^2 - C - E/r^12  (D = p0, a = p1, r0 = p2, C = p3, E=p4)
         r = sqrt(r2);
         x = r - bnd->p2; // r - r0
         x = exp(-bnd->p1 * x); // exp(-a(r-r0))
         y = 1 - x;
         ir2 = 1.0 / r2; // 1/r^2
         irn = ir2 * ir2; // 1/r^4
         irn = irn * irn * irn; // 1/r^12

         eng += bnd->p0 * y * y - bnd->p3 - bnd->p4 * irn;
         return -2.0 * bnd->p0 * bnd->p1 * x * y / r - 12.0 * bnd->p4 * irn * ir2;   // -dU/dr * (1/r)

       // Buckingham:
       case 4:   // U = A exp(-r/ro) - C/r^6  (A = p0, ro = p1, C = p2)
         r = sqrt(r2);
         ir2 = 1.0 / r2;
         irn = ir2 * ir2;   // 1/r^4

         eng += bnd->p0 * exp(-r/bnd->p1) - bnd->p2 * irn * ir2;
         return bnd->p0 * exp(-r/bnd->p1) / r / bnd->p1 - 6.0 * bnd->p2 * irn * irn;

       // exp - 6 - 8 -12:
       case 5:   // U = A exp(-r/ro) - C/r^6 - D/r^8 - F/r^12  (A = p0, ro = p1, C = p2, D = p3, F = p4)
         r = sqrt(r2);
         ir2 = 1.0 / r2;
         irn = ir2 * ir2;   // 1/r^4

         eng += bnd->p0 * exp(-r/bnd->p1) - bnd->p2 * irn * ir2 - bnd->p3 * irn * irn - bnd->p4 * irn * irn * irn;
         return bnd->p0 * exp(-r/bnd->p1) / r / bnd->p1 - 6.0 * bnd->p2 * irn * irn - 8.0 * bnd->p3 * irn * irn * ir2 - 12.0 * bnd->p4 * irn * irn * irn * ir2;
    }
}

double bond_iter_r(double r2, double &r, Bond *bnd, double &eng)
// calculate energy and return force of bonded(intramolecular) iteraction  (   Fx/dx = -(1/r)*dU(r)/dr   )
//  with r version. r - distance (if known, else 0) r2 - square of distance
{
   double x, y, irn, ir2;

   if (r == 0.0)    // calculate if unkonwn (zero) and use otherwise
     r = sqrt(r2);

   switch (bnd->type)
    {
       case 1:   // U = 1/2 k (r-r0)^2   (k = p0, r0 = p1)
         x = r - bnd->p1; // r - r0

         eng += 0.5 * bnd->p0 * x * x;
         return -bnd->p0 / r * x;

       // Morse:
       case 2:   // U = D[1 - exp(-a(r-r0))]^2 - C   (D = p0, a = p1, r0 = p2, C = p3)
         x = r - bnd->p2; // r - r0
         x = exp(-bnd->p1 * x); // exp(-a(r-r0))
         y = 1 - x;

         eng += bnd->p0 * y * y - bnd->p3;
         return -2.0 * bnd->p0 * bnd->p1 * x * y / r;

       // Pedone (morse + E/r^12):
       case 3:   // U = D[1 - exp(-a(r-r0))]^2 - C - E/r^12  (D = p0, a = p1, r0 = p2, C = p3, E=p4)
         x = r - bnd->p2; // r - r0
         x = exp(-bnd->p1 * x); // exp(-a(r-r0))
         y = 1 - x;
         ir2 = 1.0 / r2; // 1/r^2
         irn = ir2 * ir2; // 1/r^4
         irn = irn * irn * irn; // /r^12

         eng += bnd->p0 * y * y - bnd->p3 - bnd->p4 * irn;
         return -2.0 * bnd->p0 * bnd->p1 * x * y / r - 12.0 * bnd->p4 * irn * ir2;   // -dU/dr * (1/r)

       // Buckingham:
       case 4:   // U = A exp(-r/ro) - C/r^6  (A = p0, ro = p1, C = p2)
         ir2 = 1.0 / r2;
         irn = ir2 * ir2;   // 1/r^4

         eng += bnd->p0 * exp(-r/bnd->p1) - bnd->p2 * irn * ir2;
         return bnd->p0 * exp(-r/bnd->p1) / r / bnd->p1 - 6.0 * bnd->p2 * irn * irn;

       // exp - 6 - 8 -12:
       case 5:   // U = A exp(-r/ro) - C/r^6 - D/r^8 - F/r^12  (A = p0, ro = p1, C = p2, D = p3, F = p4)
         ir2 = 1.0 / r2;
         irn = ir2 * ir2;   // 1/r^4

         eng += bnd->p0 * exp(-r/bnd->p1) - bnd->p2 * irn * ir2 - bnd->p3 * irn * irn - bnd->p4 * irn * irn * irn;
         return bnd->p0 * exp(-r/bnd->p1) / r / bnd->p1 - 6.0 * bnd->p2 * irn * irn - 8.0 * bnd->p3 * irn * irn * ir2 - 12.0 * bnd->p4 * irn * irn * irn * ir2;
    }
}

double bond_eng(double r, Bond *bnd)
// return energy of bond with separating distance r
{
   double x, y, ir2, irn;

   switch (bnd->type)
    {
       case 1:   // U = 1/2 k (r-r0)^2   (k = p0, r0 = p1)
         x = r - bnd->p1; // r - r0

         return 0.5 * bnd->p0 * x * x;

       // Morse:
       case 2:   // U = D[1 - exp(-a(r-r0))]^2 - C   (D = p0, a = p1, r0 = p2, C = p3)
         x = r - bnd->p2; // r - r0
         x = exp(-bnd->p1 * x); // exp(-a(r-r0))
         y = 1 - x;

         return bnd->p0 * y * y - bnd->p3;

       // Pedone (morse + E/r^12):
       case 3:   // U = D[1 - exp(-a(r-r0))]^2 - C - E/r^12  (D = p0, a = p1, r0 = p2, C = p3, E=p4)
         x = r - bnd->p2; // r - r0
         x = exp(-bnd->p1 * x); // exp(-a(r-r0))
         y = 1 - x;
         ir2 = 1.0 / (r * r); // 1/r^2
         irn = ir2 * ir2; // 1/r^4
         irn = irn * irn * irn; // /r^12

         return bnd->p0 * y * y - bnd->p3 - bnd->p4 * irn;

       // Buckingham:
       case 4:   // U = A exp(-r/ro) - C/r^6  (A = p0, ro = p1, C = p2)
         ir2 = 1.0 / (r * r);
         irn = ir2 * ir2;   // 1/r^4

         return bnd->p0 * exp(-r/bnd->p1) - bnd->p2 * irn * ir2;

       // exp - 6 - 8 -12:
       case 5:   // U = A exp(-r/ro) - C/r^6 - D/r^8 - F/r^12  (A = p0, ro = p1, C = p2, D = p3, F = p4)
         ir2 = 1.0 / (r * r);
         irn = ir2 * ir2;   // 1/r^4

         return bnd->p0 * exp(-r/bnd->p1) - bnd->p2 * irn * ir2 - bnd->p3 * irn * irn - bnd->p4 * irn * irn * irn;
    }

}

double bond_eng_change(int iat, int jat, int i2type, int j2type, Atoms* atm, Field *field, Box *bx)
// calculate energy change after altering types of i and j atoms(i2type - new type of i particle) j2type - new type of jparticle
{
    int i, ia, ja, bonded_type, new_bond, flag;
    int nb = atm->nBonds[iat] + atm->nBonds[jat];
    double r;   // interatomic distance
    double res = 0.0;
    Bond* bnd;

    for (i = 0; (i < field->nBonds) && nb; i++)
    {
       flag = 0;
       // the first atom
       if ((field->at1[i] == iat)||(field->at2[i] == iat))
       {
          ia = field->at1[i];
          ja = field->at2[i];

          // define index of an atom which is bonded with our atom
          if (ia == iat)
            bonded_type = atm->types[ja];
          else
            bonded_type = atm->types[ia];

          r = distance(ia, ja, atm, bx);

          bnd = &field->bdata[field->bTypes[i]];
          res -= bond_eng(r, bnd);  // initial state (current situation)

          // define new bond type (if any)
          new_bond = field->bond_matrix[i2type][bonded_type];
          if (new_bond > 0)
            {
               bnd = &field->bdata[new_bond - 1];
               res += bond_eng(r, bnd);
            }
          else if (new_bond < 0)
            {
               bnd = &field->bdata[-new_bond - 1];
               res += bond_eng(r, bnd);
            }
          nb--;
          flag = 1;
       }
       // the second atom
       if ((field->at1[i] == jat)||(field->at2[i] == jat))
       {
          if (flag)   // this bond is between i and j atoms and already has been evaluated
          {
             nb--;
             continue;
          }

          ia = field->at1[i];
          ja = field->at2[i];

          // define index of an atom which is bonded with our atom
          if (ia == jat)
            bonded_type = atm->types[ja];
          else
            bonded_type = atm->types[ia];

          r = distance(ia, ja, atm, bx);

          bnd = &field->bdata[field->bTypes[i]];
          res -= bond_eng(r, bnd);  // initial state (current situation)

          // define new bond type (if any)
          new_bond = field->bond_matrix[j2type][bonded_type];
          if (new_bond > 0)
            {
               bnd = &field->bdata[new_bond - 1];
               res += bond_eng(r, bnd);
            }
          else if (new_bond < 0)
            {
               bnd = &field->bdata[-new_bond - 1];
               res += bond_eng(r, bnd);
            }
          nb--;
       }
    } // end loop by bonds
    return res;
}

void change_bonds(int iat, int jat, int i2type, int j2type, Atoms* atm, Field *field)
// bonds change after altering types of i and j atoms(i2type - new type of i particle) j2type - new type of jparticle
{
    int i, ia, ja, k, bonded_type, new_bond, flag;
    int nb = atm->nBonds[iat] + atm->nBonds[jat];

    for (i = 0; (i < field->nBonds) && nb; i++)
    {
       flag = 0;
       // the first atom
       if ((field->at1[i] == iat)||(field->at2[i] == iat))
       {
          ia = field->at1[i];
          if (ia == -1)
            continue;   // skip broken bonds
          ja = field->at2[i];
          field->bdata[field->bTypes[i]].number--;

          // define index of an atom which is bonded with our atom
          if (ia == iat)
            bonded_type = atm->types[ja];
          else
            bonded_type = atm->types[ia];

          // define new bond type (if any)
          new_bond = field->bond_matrix[i2type][bonded_type];
          if (new_bond > 0)
            {
               field->bTypes[i] = new_bond - 1;
               field->bdata[new_bond - 1].number++;
            }
          else if (new_bond < 0)
            {
               field->bTypes[i] = -new_bond - 1;
               k = field->at1[i];
               field->at1[i] = field->at2[i];
               field->at2[i] = k;
               field->bdata[-new_bond-1].number++;
            }
          else // 0, no bond
            {
               field->at1[i] = -1;
            }
          nb--;
          flag = 1;
       }
       // the second atom
       if ((field->at1[i] == jat)||(field->at2[i] == jat))
       {
          if (flag)   // this bond is between i and j atoms and already has been evaluated
          {
             nb--;
             continue;
          }

          ia = field->at1[i];
          if (ia == -1)
            continue;   // skip broken bonds
          ja = field->at2[i];
          field->bdata[field->bTypes[i]].number--;

          // define index of an atom which is bonded with our atom
          if (ia == jat)
            bonded_type = atm->types[ja];
          else
            bonded_type = atm->types[ia];

          // define new bond type (if any)
          new_bond = field->bond_matrix[j2type][bonded_type];
          if (new_bond > 0)
            {
               field->bTypes[i] = new_bond - 1;
               field->bdata[new_bond - 1].number++;
            }
          else if (new_bond < 0)
            {
               field->bTypes[i] = -new_bond - 1;
               k = field->at1[i];
               field->at1[i] = field->at2[i];
               field->at2[i] = k;
               field->bdata[-new_bond-1].number++;
            }
          else // 0, no bond
            {
               field->at1[i] = -1;
            }
          nb--;
       }
    } // end loop by bonds
}

void exec_bondlist(Atoms *atm, Field *field, Sim *sim, Box *bx)
{
   int i, j, ia, ja, tp, k;
   int nbr = 0;  // the number of breaking bond
   int *bInds;   // indexes of breaking bond
   int mxBr = 20;   // maximum for breaking bond keeping
   double dx, dy, dz, r2, f;
   double eng = 0.0;
   Bond *btypes = field->bdata;
   Spec *sp = field->species;

   //! заменить на постоянный массив
   bInds = (int*)malloc(mxBr * int_size);

   //i = field->nBonds-1;
   //printf("exec_bondlist. Last bond: %d-%d %d\n", barrs->at1[i], barrs->at2[i], barrs->types[i]);

/*
   //! поиск дубликатных связей
   for (j = 0; j < field->nBonds; j++)
     for (k = j + 1; k < field->nBonds - 1; k++)
        if (field->at1[j] > -1)
          if (((field->at1[j] == field->at1[k])&&(field->at2[j] == field->at2[k])) || ((field->at1[j] == field->at2[k])&&(field->at2[j] == field->at1[k])))
     {
            printf("(2)exec dublicate bond #%d = %d!\n", j, k);
            //field->at1[k] = -1;
     }
 */


   //printf("bt=%d\n", field->bTypes[0]);
   k = 0;
   for (i = 0; i < field->nBonds; i++)
   {
      ia = field->at1[i];
      if (ia < 0)   //! flag of breaking bond  -- OLD FLAG del this
        continue;
      if (field->bTypes[i] == 0)    // new flag for breaking bond
          continue;
      ja = field->at2[i];

/*
      //! temp! bond type correction:
      t1 = atm->types[ia];
      t2 = atm->types[ja];
      barrs->types[i] = barrs->bond_matrix[t1][t2];
*/

      r2 = sqr_distance_proj(ia, ja, atm, bx, dx, dy, dz);
      tp = field->bTypes[i];
      //if (tp >= field->nBdata)
      //  printf("tp=%d\n", tp);

      if ((btypes[tp].spec1 != atm->types[ia])||(btypes[tp].spec2 != atm->types[ja]))
        printf("!!!unkn bond type[%d]=%d between %s[%d] and %s[%d]\n", i, tp, sp[atm->types[ia]].name, ia, sp[atm->types[ja]].name, ja);

      // minimal bond length
      if (btypes[tp].mnEx)  // if minimal bond lenght is defined
        if (r2 < btypes[tp].r2min)
        {
           // modify bond
            btypes[tp].number--;
            tp = btypes[tp].new_type[0];
            btypes[tp].number++;
            field->bTypes[i] = tp;
            field->species[atm->types[ia]].number--;
            field->species[atm->types[ja]].number--;
            atm->types[ia] = btypes[tp].spec1;
            atm->types[ja] = btypes[tp].spec2;
            field->species[atm->types[ia]].number++;
            field->species[atm->types[ja]].number++;
            change_bonds(ia, ja, atm->types[ia], atm->types[ja], atm, field);
        }

      // maximal bond length
      if (btypes[tp].mxEx)  // if maximal bond lenght is defined
          if (r2 > btypes[tp].r2max)
          {
              if (btypes[tp].new_type[1] == 0)  // delete bond
              {
                  nbr++;
                  destroy_bond(i, atm, field, &btypes[tp]);
                  destroy_angles(ia, ja, field);
                  if (k < mxBr)
                  {
                      bInds[k] = i; k++;
                  }
                  continue;
              }
              else  // modify bond
              {
                  btypes[tp].number--;
                  tp = btypes[tp].new_type[1];
                  btypes[tp].number++;
                  field->bTypes[i] = tp;
                  field->species[atm->types[ia]].number--;
                  field->species[atm->types[ja]].number--;
                  atm->types[ia] = btypes[tp].spec1;
                  atm->types[ja] = btypes[tp].spec2;
                  field->species[atm->types[ia]].number++;
                  field->species[atm->types[ja]].number++;
                  change_bonds(ia, ja, atm->types[ia], atm->types[ja], atm, field);
              }
          }

      f = bond_iter(r2, &btypes[tp], eng);

      atm->fxs[ia] += f * dx;
      atm->fxs[ja] -= f * dx;
      atm->fys[ia] += f * dy;
      atm->fys[ja] -= f * dy;
      atm->fzs[ia] += f * dz;
      atm->fzs[ja] -= f * dz;
   }

   //remove breaking bonds
   //! nbr и k почти одно и то же!
   if (nbr)
   {
      j = field->nBonds - 1;
      for (i = 0; i < k; i++)
      {

         while ((field->at1[j] < 0)&&(j > 0))
           j--;

         tp = bInds[i];
         //printf("%d, ", bInds[i]);
         if (j <= tp)
         {
            //nbr = i;
            j = j + 1;
            break;
         }
         field->at1[tp] = field->at1[j];
         field->at2[tp] = field->at2[j];
         field->bTypes[tp] = field->bTypes[j];
         field->at1[j] = -1;
         //j--;
      }
      //if (nbr > 0)
      //field->nBonds -= nbr;
      //printf("\nshortening bond list from %d to %d (%d)\n", field->nBonds, j, field->nBonds - j);
      field->nBonds = j;
   }

   delete[] bInds;
   //if (isnan(eng))
     // printf("bonds eng: NAN\n");
   sim->engBond = eng;

/*
   //! поиск дубликатных связей
   for (j = 0; j < field->nBonds; j++)
     for (k = j + 1; k < field->nBonds - 1; k++)
        if (field->at1[j] > -1)
          if (((field->at1[j] == field->at1[k])&&(field->at2[j] == field->at2[k])) || ((field->at1[j] == field->at2[k])&&(field->at2[j] == field->at1[k])))
     {
            printf("(3)aft exec dublicate bond #%d = %d!\n", j, k);
            field->at1[k] = -1;
     }
*/


   //printf("eng=%f f=%f r2=%f  r0^2=%f\n", eng, f, r2, btypes[barrs->types[i]].p1*btypes[barrs->types[i]].p1);
}

void bond_out(Atoms *atm, Field *field, Box *bx, char *fname)
{
   int i, j, ia, ja, mx, k;
   //double dx, dy, dz;

   // here some indexes are differs by 1 because field->bdata[0] is reserved as no bond
   double **r = (double**)malloc(pointer_size * (field->nBdata - 1));
   int* cur_index = (int*)malloc(int_size * (field->nBdata - 1));       // index of bond in define bond type

   // define maximal quantity of bonds
   mx = 0;
   for (i = 1; i < field->nBdata; i++)  // [0] - deleted bond
   {
       mx = max(mx, field->bdata[i].number);
       cur_index[i-1] = 0;
   }

   for (i = 1; i < field->nBdata; i++)
     r[i-1] = (double*)malloc(double_size * mx);

   for (i = 0; i < field->nBonds; i++)
    if (field->at1[i] > -1)         // old verifiying of deleted bond
        if (field->bTypes[i])       // new verifiying of deleted bond
        {
            k = field->bTypes[i];
            ia = field->at1[i];
            ja = field->at2[i];
            r[k-1][cur_index[k-1]] = distance(ia, ja, atm, bx);
            cur_index[k - 1]++;
        }

   FILE *f = fopen(fname, "w");
   fprintf(f, "n");
   for (i = 1; i < field->nBdata; i++)
     fprintf(f, "\t%d%s-%s", i, field->snames[field->bdata[i].spec1], field->snames[field->bdata[i].spec2]);
   fprintf(f, "\n");

   for (i = 0; i < mx; i++)
   {
      fprintf(f, "%d", i);
      for (j = 1; j < field->nBdata; j++)
        if (i < field->bdata[j].number)
          fprintf(f, "\t%f", r[j-1][i]);
        else
          fprintf(f, "\t");

      fprintf(f, "\n");
   }
   fclose(f);


   for (i = 1; i < field->nBdata; i++)
       delete[] r[i - 1];
   delete[] r;
}
