// UNTI for SYSTEM initialization

#include <stdlib.h>     // malloc, alloc, rand, NULL
#include <stdio.h>      // FILE
#include <math.h>       // log, sqrt, fabs
#include <string.h>     //strcmp

#include "utils.h"
#include "const.h"
#include "dataStruct.h"  // Sim, Box, Atoms ....
#include "vdw.h"
#include "elec.h"
#include "box.h"
#include "cell_list.h"
#include "bonds.h"
#include "angles.h"
#include "sys_init.h"
#include "temperature.h"
#include "integrators.h"
#include "ejump.h"
#include "rdf.h"

int spec_by_name(Field *field, char *name, int &id)
// seek a specie's id by its name and save it in id-variable. return success or not
{
    int i;

    for (i = 0; i < field->nSpec; i++)
      if (strcmp(field->snames[i], name) == 0)
      {
         id = i;
         return 1;
      }

    return 0;
}

int nucl_by_name(Field* field, char* name, int& id)
// seek a nucleus id by its name and save it in id-variable. return success or not
{
    int i;

    for (i = 0; i < field->nNucl; i++)
        if (strcmp(field->nnames[i], name) == 0)
        {
            id = i;
            return 1;
        }

    return 0;
}

int twospec_by_name(Field *field, char *name1, char *name2, int &id1, int &id2)
// seek two specie's id by their names and save them in id-variable. return success or not
{
    int i, suc = 0;

    for (i = 0; i < field->nSpec; i++)
    {
      if (strcmp(field->snames[i], name1) == 0)
      {
         id1 = i;
         suc++;
      }
      if (strcmp(field->snames[i], name2) == 0)
      {
         id2 = i;
         suc++;
      }
      if (suc > 1)
        return 1;
    }

    return 0;
}
// end 'twospec_by_name' function

int read_spec(int line, FILE* f, Spec* spec, char** name, Field *fld)
{
    int i, j;
    char nucl[8];

    fscanf(f, "%s %s %lf %lf %lf", spec->name, nucl, &(spec->mass), &(spec->charge), &(spec->energy));
    *name = spec->name;
    spec->number = 0;
    spec->varNumber = 0;

    // define nucleus (or add new)
    j = 1;
    for (i = 0; i < fld->nNucl; i++)
        if (strcmp(fld->nnames[i], nucl) == 0)
        {
            spec->nuclei = i;
            j = 0;
            break;
        }
    if (j)
    {
        strcpy(fld->nnames[fld->nNucl], nucl);
        fld->nnumbers[fld->nNucl] = 0;
        spec->nuclei = fld->nNucl;
        fld->nNucl++;
    }

    //! recalculate parameters in OUR units
    spec->mass *= m_scale;
    spec->charge *= q_scale;

    // flag of charged
    if (fabs(spec->charge) < 1.0E-10)
        spec->charged = 0;
    else
        spec->charged = 1;

    //counters and some flags
    spec->donacc = 0;
    spec->redForm = 0;
    spec->oxForm = 0;
    spec->pOyz = 0;
    spec->nOyz = 0;
    //spec->canBond = 0;
    spec->bondKeys = NULL;
    spec->angleType = 0;    // ability to form angles
    spec->frozen = 0;

    //flags for CN calculations
    spec->idCentral = 0;
    spec->idCounter = 0;

    //printf("spec[%d]:'%s': m=%f, q=%f, E=%f\n", line + 1, spec->name, spec->mass, spec->charge, spec->energy);
    fscanf(f, " \n ");
    return 1;
}
// end 'read_spec' function

int read_redox_seq(int line, FILE* f, Field* fld)
// read redox sequence (for example: "4 V5+ V4+ V3+ V2+")
{
    int at1, at2, j, k;
    char aname[8], bname[8];
    int res = 1;            // function result

    // read the number of species in a red-ox sequence and the first specie:
    fscanf(f, " %d %8s", &k, aname);
    if (!spec_by_name(fld, aname, at1))
    {
        printf("ERROR[402] unknown the first specie (%s) in red-ox line %d\n", aname, line + 1);
        res = 0;
    }
    fld->species[at1].varNumber = 1;   // flag of variable number of species

    for (j = 1; j < k; j++)
    {
        fscanf(f, " %8s", bname);
        if (!spec_by_name(fld, bname, at2))
        {
            printf("ERROR[403] unknown specie (%s) in red-ox line %d\n", bname, line + 1);
            res = 0;
        }

        // the next species are automatically donors:
        fld->species[at1].donacc |= (1 << bfAcceptor);   // binary flag means that the specie is an acceptor
        fld->species[at2].donacc |= (1 << bfDonor);   // binary flag means that the specie is a donor
        fld->species[at1].redForm = at2 + 1;
        fld->species[at2].oxForm = at1 + 1;
        fld->species[at2].varNumber = 1; // flag of variable number of species
        printf("donacc1: %d; donacc2: %d\n", fld->species[at1].donacc, fld->species[at2].donacc);
        at1 = at2;    // the donor in this pair becomes an acceptor in the next one
    }
    fscanf(f, " \n");
    return res;
}

int read_field(Field *field, Sim *sim)
// read field parameters from field.txt (and reset counters)
{
  int i, j, n, at1, k;
  FILE *f;
  char aname[8];
  int res = 1;      // function result

  f = fopen("field.txt", "r");
  if (f == NULL)
  {
      printf("ERROR[001]! Fatal Error. Can't open Field.txt file\n");
      return 0;
  }

  //RESET COUNTERS and ETC..
  field->nBonds = 0;
  field->nAngles = 0;
  sim->use_bnd = 0;
  sim->use_angl = 0;

  //READ SPECIES:
  field->charged_spec = 0;
  if (n = find_number(f, " spec %d"))
  {
      field->nSpec = n;
      field->nPair = (field->nSpec) * (field->nSpec - 1) / 2 + field->nSpec;   // the number of pairs (specie-specie)
      field->species = (Spec*)malloc(field->nSpec * (int)spec_size);      // array of speices
      field->snames = (char**)malloc(field->nSpec * pointer_size);   // array of specie names
      field->nnames = (name8*)malloc(field->nSpec * sizeof(name8));
      field->nnumbers = (int*)malloc(field->nSpec * int_size);
      field->nNucl = 0;
      for (i = 0; i < field->nSpec; i++)
      {
          read_spec(i, f, &(field->species[i]), &(field->snames[i]), field);
          if (field->species[i].charge != 0.0)
              field->charged_spec = 1;
      }
      //printf("charged spec = %d\n", field->charged_spec);
  }
  else
  {
      printf("ERROR[004]! There is no 'spec' section in the Field.txt file\n");
      res = 0;
  }

  //READ DONOR-ACCEPTORS
  if (n = find_number(f, " red-ox %d"))
  {
      for (i = 0; i < n; i++)
          read_redox_seq(i, f, field);

      //define nFreeElectrons for all species
      // for an example in a sequence (V5+ -> V4+ -> V3+ -> V2+) V5+ has 3 "free" electrons
      for (i = 0; i < field->nSpec; i++)
      {
          field->species[i].nFreeEl = 0;
          j = i;
          while (field->species[j].oxForm && ((field->species[j].donacc >> 0) & 1))  // while oxForm exists (species can be donor)
          {
              field->species[i].nFreeEl++;
              j = field->species[j].oxForm - 1;
          }
      }
  } // end red-ox Section

  // READ "FROZEN" ATOM TYPES
  if (n = find_number(f, " frozensp %d"))
  {
      for (i = 0; i < n; i++)
      {
          fscanf(f, "%s", aname);
          if (spec_by_name(field, aname, j))
          {
              field->species[j].frozen = 1;
          }
          else
          {
              printf("WARNING[b001] Unknown atom type (%s) in 'frozensp' section!\n", aname);
          }
      }
  }

  //READ VAN DER WAALS INTERACTIONS
  field->minRvdw = 999999.9;
  field->maxRvdw = 0.0;
  if (n = find_number(f, " vdw %d"))
  {
     //vdws array allocation    (vdws[iSpec][iSpec] = pair pot between them)
     field->vdws = (VdW***)malloc(field->nSpec * pointer_size);
     for (i = 0; i < field->nSpec; i++)
     {
       field->vdws[i] = (VdW**)malloc(field->nSpec * pointer_size);
       for (j = 0; j < field->nSpec; j++)
         field->vdws[i][j] = NULL;
     }

     field->nVdW = n;
     field->pairpots = (VdW*)malloc(n * sizeof(VdW));
     for (i = 0; i < n; i++)
     {
         read_vdw(i, f, field, sim);
     }
     //save maximal cuttoff radii for cell_list method:
     //sim->double_pars[dpMaxCutVdW] = field->maxRvdw;
     field->maxR2vdw = field->maxRvdw * field->maxRvdw;
  }
  else
  {
      field->nVdW = 0;
      printf("WARNING[001] no Van-der-Waals iteractions!\n");
  }
  //sim->mxRvdw2 = sim->double_pars[dpMaxCutVdW] * sim->double_pars[dpMaxCutVdW];

  
  //READ BOND TYPES
  field->nBdata = 0;
  if (n = find_number(f, " bonds %d"))
  {
     sim->use_bnd = 1;
     field->nBdata = n + 1;     // [0] element is reserved as 'empty bond'
     field->bdata = (Bond*)malloc(field->nBdata * sizeof(Bond));

     // matrix for default bonds (for new bonds formation)
     field->bond_matrix = (int**)malloc(field->nSpec * pointer_size);
     field->bonding_matr = (int**)malloc(field->nSpec * pointer_size);
     field->bindR2matrix = (double**)malloc(field->nSpec * pointer_size);
     for (i = 0; i < field->nSpec; i++)
     {
        field->bond_matrix[i] = (int*)malloc(field->nSpec * int_size);
        field->bonding_matr[i] = (int*)malloc(field->nSpec * int_size);
        field->bindR2matrix[i] = (double*)malloc(field->nSpec * double_size);
        for (j = 0; j < field->nSpec; j++)
        {
            field->bond_matrix[i][j] = 0;
            field->bonding_matr[i][j] = 0;
        }
     }
     // read bonds from file:
     for (i = 1; i < field->nBdata; i++)    // from 1 as [0] is reserved for 'empty bond'
       read_bond(i, f, field, sim);

     // update new_spec for mutable bonds
     for (i = 1; i < field->nBdata; i++)    // from 1 as [0] is reserved for 'empty bond'
     {
         if (field->bdata[i].mnEx)
             if (field->bdata[i].new_type[0])
             {
                 n = field->bdata[i].new_type[0];
                 if (n < 0)
                 {
                     field->bdata[i].new_spec1[0] = field->bdata[n].spec2;
                     field->bdata[i].new_spec2[0] = field->bdata[n].spec1;
                 }
                 else
                 {
                     field->bdata[i].new_spec1[0] = field->bdata[n].spec1;
                     field->bdata[i].new_spec2[0] = field->bdata[n].spec2;
                 }

             }
         if (field->bdata[i].mxEx)
             if (field->bdata[i].new_type[1])
             {
                 n = field->bdata[i].new_type[1];
                 if (n < 0)
                 {
                     field->bdata[i].new_spec1[1] = field->bdata[n].spec2;
                     field->bdata[i].new_spec2[1] = field->bdata[n].spec1;
                 }
                 else
                 {
                     field->bdata[i].new_spec1[1] = field->bdata[n].spec1;
                     field->bdata[i].new_spec2[1] = field->bdata[n].spec2;
                 }

             }
         //printf("bnd[%d]%s-%s. min:%d(%d-%d) max:%d(%d-%d)\n", i, field->snames[field->bdata[i].spec1], field->snames[field->bdata[i].spec2], field->bdata[i].new_type[0], field->bdata[i].new_spec1[0], field->bdata[i].new_spec2[0], field->bdata[i].new_type[1], field->bdata[i].new_spec1[1], field->bdata[i].new_spec2[1]);
     }
  }

  //READ EVOL FLAGS FOR BONDS
  if (n = find_number(f, " evol_bonds %d"))
  {
      for (i = 0; i < n; i++)
      {
          fscanf(f, "%d-%d", &j, &k);
          if ((j < 1) || (j >= field->nBdata))
          {
              printf("ERROR[b015] Wrong bond type in 'evol-bonds' section, pair[%d] %d-%d!\n", i, j, k);
              res = 0;
          }
          else if ((k < 1) || (k >= field->nBdata))
          {
              printf("ERROR[b016] Wrong evol bond type in 'evol-bonds' section, pair[%d] %d-%d!\n", i, j, k);
              res = 0;
          }
          else
              field->bdata[j].evol = k;
      }

  }

  //READ H-BONDS SECTION
  if (n = find_number(f, " h-bonds %d"))
  {
      for (i = 0; i < n; i++)
      {
          fscanf(f, "%d %s", &k, aname);
          if ((k < 1) || (k >= field->nBdata))
          {
              printf("ERROR[b012] Wrong bond number in 'h-bonds' section, pair %d-%s!\n", k, aname);
              res = 0;
          }
          else
          {
              if (!spec_by_name(field, aname, at1))
              {
                  printf("ERROR[b013] Wrong atom type(%s) in 'h-bonds' section, pair %d-%s!\n", aname, k, aname);
                  res = 0;
              }
              else
              {
                  if ((at1 != field->bdata[k].spec1)&&(at1 != field->bdata[k].spec2))
                  {
                      printf("ERROR[b014] The bond type[%d] does not contain specie %s, pair %d-%s!\n", k, aname, k, aname);
                      res = 0;
                  }
                  else  // everything is ok
                  {
                      field->bdata[k].hatom = at1;
                  }
              }
          }
      }
  }

  //READ VALENT ANGLE TYPES
  if (n = find_number(f, " angles %d "))
  {
      field->nAdata = n + 1; //[0] reserved for empty angle
      field->adata = (Angle*)malloc(field->nAdata * sizeof(Angle));

      // read angles from file (angle[0] reserved for no angle):
      field->adata[0].type = 0;
      for (i = 1; i < field->nAdata; i++)   // from 1 as [0] is reserved for 'empty angle'
          read_angle(i, f, field);
      sim->use_angl = 1;
  }
  else
  {
      sim->use_angl = 0;
      field->nAdata = 0;
  }

  //READ AUTO FORMING ANGLES
  if (n = find_number(f, " angle_forming %d "))
  {
      if (sim->use_angl)
      {
          for (i = 0; i < n; i++)
          {
              fscanf(f, "%s %d", aname, &k);
              if (spec_by_name(field, aname, at1))
                  field->species[at1].angleType = k;
              else
              {
                  printf("ERROR[017] wrong species(%s) in angle_formin section(%d)\n", aname, i + 1);
                  res = 0;
              }
          }
          sim->use_angl = 2;    // flag of variable angles
      }
      else
          printf("WARNING[b002] Declaration of angle_forming is ignored because there is no angles defintion\n");
  }

  // define ability of bond creation
  sim->use_linkage = 0;
  if (n = find_number(f, " linkage %d"))
  {
      if (!field->nBdata)
          printf("WARINING[b005] Linkage declaration is ignored, because there are no bond types\n");
      else
          if (!read_linkage(f, n, field, field->nBdata))
              res = 0;
          else
          {
              sim->use_bnd = 2;     // если все хорошо, переводим флаг use_bnd в значение "измен€ющиес€ св€зи"
              sim->use_linkage = 1;
          }
  }

  // read variable radii (number of lines must be equal to number of species)
  if (find_int(f, " radii %d", j))
      for (i = 0; i < field->nSpec; i++)
      {
          fscanf(f, "%s", aname);
          if (spec_by_name(field, aname, at1))
              fscanf(f, "%lf %lf %lf", &field->species[at1].radA, &field->species[at1].radB, &field->species[at1].mxEng);
          else
          {
              printf("ERROR[b018] wrong species(%s) in radii section(%d)\n", aname, i + 1);
              res = 0;
          }

      }

  fclose(f);
  return res;
}
// end 'read_field' function

int read_atoms_box(Atoms* atm, Field *field, Sim *sim, Box *box)
//read box parameters and atoms from atoms.xyz (and reset counters)
// and calculate nFreeElectron
{
   int i, j, n;
   int res = 1;
   char aname[8];
   double x, y, z;

   FILE  *f = fopen("atoms.xyz", "r");
   if (f == NULL)
   {
         printf("ERROR[007] Can't open file 'atoms.xyz'\n");
         return 0;
   }

   fscanf(f, "%d", &n);
   atm->nAt = n;

   // box reading (the second line)
   if (!read_box(f, box))
     res = 0;

   // array allocates
   atm->types = (int*)malloc(int_size * n);
   atm->xs = (double*)malloc(double_size * n);
   atm->ys = (double*)malloc(double_size * n);
   atm->zs = (double*)malloc(double_size * n);
   atm->vxs = (double*)malloc(double_size * n);
   atm->vys = (double*)malloc(double_size * n);
   atm->vzs = (double*)malloc(double_size * n);
   atm->fxs = (double*)malloc(double_size * n);
   atm->fys = (double*)malloc(double_size * n);
   atm->fzs = (double*)malloc(double_size * n);
   atm->x0s = (double*)malloc(double_size * n);
   atm->y0s = (double*)malloc(double_size * n);
   atm->z0s = (double*)malloc(double_size * n);
   atm->vx0 = (double*)malloc(double_size * n);
   atm->vy0 = (double*)malloc(double_size * n);
   atm->vz0 = (double*)malloc(double_size * n);
   atm->nBonds = (int*)malloc(int_size * n);
   atm->parents = (int*)malloc(int_size * n);

   //Reading atoms:
   sim->nFreeEl = 0;
   for (i = 0; i < n; i++)
   {
        fscanf(f, "%s %lf %lf %lf", aname, &x, &y, &z);
        if (spec_by_name(field, aname, j))
        {
           atm->types[i] = j;
           field->species[j].number++;
           field->nnumbers[field->species[j].nuclei]++;
           sim->nFreeEl += field->species[j].nFreeEl;

           atm->xs[i] = x;
           atm->ys[i] = y;
           atm->zs[i] = z;
           atm->x0s[i] = x;
           atm->y0s[i] = y;
           atm->z0s[i] = z;
           atm->vxs[i] = 0;
           atm->vys[i] = 0;
           atm->vzs[i] = 0;
           atm->fxs[i] = 0;
           atm->fys[i] = 0;
           atm->fzs[i] = 0;
           atm->nBonds[i] = 0;
           atm->parents[i] = -1;
        }
        else
        {
              printf("ERROR[009]! unknown atom[%d] type=%s in atoms.xyz file\n", i+1, aname);
              res = 0;
        }
   }  // cycle by read atoms
   fclose(f);
   return res;
}

void free_atoms(Atoms *atm)
// deallocate atom arrays
{
   free (atm->types);
   free (atm->xs);
   free (atm->ys);
   free (atm->zs);
   free(atm->vxs);
   free(atm->vys);
   free(atm->vzs);
   free(atm->fxs);
   free(atm->fys);
   free(atm->fzs);
   free(atm->x0s);
   free(atm->y0s);
   free(atm->z0s);
   free(atm->vx0);
   free(atm->vy0);
   free(atm->vz0);
   free(atm->nBonds);
   free(atm->parents);
}

int read_sim(Atoms *atm, Field *field, Sim *sim, Elec *elec, TStat *tstat)
// set up starting MD parameters
{
  int i, j, n, k;
  double x;
  char s[8];
  FILE *f;
  int res = 1; // result of function 0 - wrong 1 - true

  // RESET all counters and some calculated parameters
  sim->nBonds = 0;
  sim->pEjump = 0;
  sim->nEjump = 0;
  sim->pTotJump = 0;
  sim->nTotJump = 0;
  sim->engKin = 0.0;
  sim->Temp = 0.0;
  sim->pTotJump = 0;
  sim->prevTime = 0.0;
  sim->flags = 0;
  sim->nHead = 0;
  sim->nBndBr = 0;
  sim->nBndForm = 0;
  sim->flags = 0;
  //sim->double_pars[dpMaxCut] = 0.0;

  // READ BONDs AND ANGLEs LISTS
  //! it can't be placed into read_field, because it needs atoms, which are read after read_field
  //! maybe to replace these flags to control.txt ?
  f = fopen("field.txt", "r");
  if (f == NULL)
  {
      printf("ERROR[409]! Can't open 'field.txt' file (in read_sim)\n");
      return 0;
  }

  // pre-defined bond list
  if (find_number(f, " bond_list %d"))
  {
      if (!read_bondlist(atm, field, sim))
        res = 0;
      else
      {
          printf("the list of bonds is used(N=%d)!\n", field->nBonds);
          
          // situation, then bond_list exists but empty
          if (!field->nBonds)
          {
              if (sim->use_linkage)
                  alloc_bonds(5000, field); //! define this default value somewhere
              else
                  sim->use_bnd = 0;   // если св€зей сразу нет и нет возможности их образовыватьт, скидвываем флаг на 0
          }
      }
  }
  else
  {
     field->nBonds = 0;

     // if bonds can be formed, allocate memory
     if (sim->use_linkage)
         alloc_bonds(5000, field); //! define this default value somewhere
     else
         sim->use_bnd = 0;   // если св€зей сразу нет и нет возможности их образовыватьт, скидвываем флаг на 0
  }

  // pre-defined angle list
  if (find_number(f, " angle_list %d"))
  {
      if (field->nAdata)
      {
          read_anglelist(atm, field, sim);
          printf("the list of angles is used(N=%d)!\n", field->nAngles);
      }
      else
          printf("WARNING[b006] 'anlge_list' directive is ignored, because there are no angle type defintions\n");
  }
  else
  {
      field->nAngles = 0;
      // if angle types exist, allocate some memory for possible angles
      if (field->nAdata)
        alloc_angles(5000, field);  //! define this default value somewhere
  }
  fclose(f);

  f = fopen("control.txt", "r");
  if (f == NULL)
  {
        printf("ERROR[410]: Can't open file 'control.txt'\n");
        return 0;
  }

  if (!find_double(f, " timestep %lf ", sim->tSt))
  {
       printf("ERROR[411]: timestep must be declared in control.txt file!\n");
       res = 0;
  }

  //! 'timesim' directive take precedence over 'nstep' directive
  if (!find_double(f, " timesim %lf ", sim->tSim))
  {
      if (!find_int(f, " nstep %d", sim->nSt))
      {
         printf("ERROR[412]: no 'nstep' or 'timesim' directives in control.txt file!\n");
         res = 0;
      }
      else
         sim->tSim = double(sim->nSt * sim->tSt);
  }
  else
    sim->nSt = sim->tSim / sim->tSt;

  // seeking 'timeequil' or 'nequil'
  //! 'timesim' directive take precedence over 'nstep' directive
  if (!find_double(f, " timeequil %lf ", sim->tEq))
  {
       if (!(sim->nEq = find_number(f, " nequil %d ")))
       {
           printf("WARNING[002]: no 'nequil' or 'timeequil' directives in control.txt file - there is no equilibration period!\n");
       }
       else
          sim->tEq = double(sim->nEq * sim->tSt);
  }
  else
    sim->nEq = sim->tEq / sim->tSt;

  // t-scaling frequency during equlibration period
  if (sim->nEq)
  {
     if (!(sim->freqEq = find_number(f, " eqfreq %d ")))
     {
         printf("WARNING[003]: no t-Scale during equlibration period!\n");
     }
  }

  //Temperature and thermostat parameters
  if (read_tstat(f, tstat, atm->nAt))
  {
     sim->tTemp = tstat->Temp;  //! вообщем-то это ненужный параметр, в sim - фактическа€ температура, в термостате - целева€
  }
  else
     res = 0;

  //Electrostatic calculation parameters
  if (read_elec(f, elec, field))
    {
        // for cell list initialization:
        //sim->double_pars[dpMaxCutElec] = elec->rReal;
    }
  else
    res = 0;

  // seeking 'permittivity'
  if (!find_double_def(f, " permittivity %lf ", elec->eps, 1.0))
  {
      printf("WARNING[131]: permittivity was not defined in control.txt file. used value of 1.0!\n");
  }

  // a way of initial velocities determination
  if (find_str(f, " init_vel %s", s))
  {
     if (strcmp(s, "zero") == 0)        // v[i] = 0.0;
       sim->int_pars[ipInitVel] = tpInitVelZero;
     else if (strcmp(s, "gaus") == 0)   // according to gaussian routine
       sim->int_pars[ipInitVel] = tpInitVelGauss;
     else if (strcmp(s, "const") == 0)   // constant value for specific purposes
     {
         sim->int_pars[ipInitVel] = tpInitVelConst;
         fscanf(f, "%lf %lf %lf", &(atm->vxs[0]), &(atm->vys[0]), &(atm->vzs[0]));
         for (i = 1; i < atm->nAt; i++)
         {
             atm->vxs[i] = atm->vxs[0];
             atm->vys[i] = atm->vys[0];
             atm->vzs[i] = atm->vzs[0];
         }
     }
     else if (strcmp(s, "keng") == 0)   // according to specified kinetic energy
     {
         sim->int_pars[ipInitVel] = tpInitVelEng;

         double ekin, phi, theta, cost, vel;
         fscanf(f, "%lf", &ekin);
         //printf("ekin = %f\n", ekin);
         // verification: velocity summ:
         double vx = 0.0, vy = 0.0, vz = 0.0;
         for (i = 0; i < atm->nAt; i++)
         {
             vel = sqrt(2.0 * ekin / field->species[atm->types[i]].mass);
             //phi = rand01() * twopi;
             //theta = rand01() * pi - 0.5 * pi;
             phi = double(rand() % 32) / 32.0 * twopi;
             theta = double(rand() % 32) / 32.0 * twopi;// -0.5 * pi;// -0.5 * pi;
             sincos(theta, atm->vzs[i], cost);
             sincos(phi, atm->vys[i], atm->vxs[i]);
             atm->vxs[i] *= cost * vel;
             atm->vys[i] *= cost * vel;
             atm->vzs[i] *= vel;
             vx += atm->vxs[i];
             vy += atm->vys[i];
             vz += atm->vzs[i];
             //printf("phi=%f theta=%f x=%f y=%f z=%f\n", phi, theta, atm->vxs[i], atm->vys[i], atm->vzs[i]);
         }
         printf("sum velocity: (%f %f %f)\n", vx, vy, vz);
     }
     else
     {
       printf("ERROR[407] Unknown value of init_vel directive in control.txt file!\n");
       res = 0;
     }
  }
  else
  {
     printf("ERROR[406] No init_vel directive in control.txt file!\n");
     res = 0;
  }

  // Parameters for electron hopping
  //! move to ejump.cpp ?
  sim->ejtype = tpJumpNone;
  sim->eJump = find_number(f, " eJump %d ");
  if (sim->eJump != 0)
  {
        fscanf(f, "%lf %s ", &sim->rElec, s);
        if (strcmp(s, "eq") == 0)
        {
          sim->ejtype = tpJumpEq;
          // read admissible energy deviation from zero
          fscanf(f, "%lf", &sim->dEjump);
        }
        else if (strcmp(s, "min") == 0)
          sim->ejtype = tpJumpMin;
        else if (strcmp(s, "metr") == 0)
          sim->ejtype = tpJumpMetr;
        else
        {
              printf("ERROR[121]: unknown electron jump type in control file!\n");
              res = 0;
        }

        sim->rElec *= r_scale;     
        //sim->r2Elec *= r_scale;     //! убрать из sim это поле //! опечатка? rElec instead of r2Elec? corrected above
        //sim->double_pars[dpMaxCutJump] = sim->r2Elec;
        sim->r2Elec = sim->rElec * sim->rElec;

        //! если есть св€зи, то они могут разрушатьс€ или измен€тьс€ в ходе эл переноса
        if (sim->use_bnd == 1)
            sim->use_bnd = 2;
  }

  //EXTERNAL FIELDS
  // external electric field gradient along x direction (Ux = dU/dx)
  if (find_double_def(f, " elecfield %lf ", sim->Ux, 0.0))
  {
      fscanf(f, " %lf %lf ", &sim->Uy, &sim->Uz);
  }
  else
  {
      sim->Uy = 0; sim->Uz = 0;
  }

  // variables for shifting
  if (find_double_def(f, " shiftX %lf ", sim->shiftX, 0.0))
    fscanf(f, " %lf ", &sim->shiftVal);
  else
    sim->shiftVal = 0.0;

  sim->reset_vels = find_number(f, " reset_vels %d ");
  if (sim->reset_vels)
      printf("use reset vels every %d timestep\n", sim->reset_vels);

  // flag to use cell list acceleration technique
  if (find_double(f, " cell_list %lf ", sim->desired_cell_size))
  {
     sim->flags |= 1 << bfSimUseCList;
  }

  //RDF output
  res *= read_rdf(f, sim);

  // history output freqency (period)
  //! probably we dismiss this file and associtated variables
  if (!find_int_def(f, " hist %d ", sim->hist, 0))
  {

  }

  // statistic output freqency (period)
  if (!find_int_def(f, " stat %d ", sim->stat, 1000))
  {
      printf("WARNING[133]: stat directive is not specified, default value of 1000 is used\n");
  }

  // flag to output velocity autocorrelation function (VAF)
  find_int_def(f, " vaf %d ", sim->vaf, 0);

  // period of intermidate configurations output (no output if zero)
  find_int_def(f, " revcon %d ", sim->revcon, 0);

  //Coordination number (CN) output
  if (find_double(f, " outCN %lf ", x))   // outCN <radius> <Ncentral> <name1> <name2> ... <Nligand> <name1> <name2>..
  {
     sim->r2CN = x * x; // radius for CN calculation
     k = 1;             // index for central atom array
     fscanf(f, "%d", &n); // number of species(central) for CN calculating
     sim->nCentrCN = n;

     for (i = 0; i < n; i++)
     {
        fscanf(f, "%s", s);
        if (spec_by_name(field, s, j))
        {
          field->species[j].idCentral = k;
          k++;
        }
        else
        {
           printf("ERROR[201] Unknown species in outCN directive!\n");
           res = 0;
        }
     }
     k = 1; // index for ligand array
     fscanf(f, "%d", &n); // number of species(ligands) for CN calculating
     sim->nCountCN = n;
     for (i = 0; i < n; i++)
     {
        fscanf(f, "%s", s);
        if (spec_by_name(field, s, j))
        {
          field->species[j].idCounter = k;
          k++;
        }
        else
        {
           printf("ERROR[202] Unknown species in outCN directive!\n");
           res = 0;
        }

     }
     sim->outCN = 1;
  }
  else
    sim->outCN = 0;

  //Trajectories output
  if (find_int(f, " traj %d ", n))
  {
     sim->stTraj = n;   // start collect trajectories from this step
     fscanf(f, "%d %d %d", &sim->frTraj, &n, &k);   // start atom, end atom
     sim->at1Traj = n;
     sim->at2Traj = k + 1;  // for strong unequality
     //sim->nTraj = k - n + 1;    // number of atoms to output
  }
  else
   sim->frTraj = 0;

  //Bind trajectories output
  if (find_str(f, " bindtraj %s ", s))
  {
      // seek corresponding nuclei
      if (nucl_by_name(field, s, k))
      {
          n = 0;
          fscanf(f, "%d %d", &sim->bindTrajStart, &sim->bindTrajFreq);
          for (i = 0; i < field->nSpec; i++)
              if (field->species[i].nuclei == k)
                  n += field->species[i].number;
          if (n)
          {
              sim->nBindTrajAtoms = n;
              sim->bindTrajAtoms = (int*)malloc(n * int_size);
              j = 0;
              for (i = 0; i < atm->nAt; i++)
                  if (field->species[atm->types[i]].nuclei == k)
                  {
                      sim->bindTrajAtoms[j] = i;
                      j++;
                  }
          }
          else
              printf("WARNING[b008] Nuclei %s is specified for bindtraj output, but there are no such atoms\n", s);
      }
      else
      {
          printf("ERROR[b017] Unknown nuclei %s in bindtraj directive!\n", s);
          res = 0;
      }
  }
  else
      sim->nBindTrajAtoms = 0;

  // maximal neighbors of atoms (for ejump routine, for example)
  if (!find_int_def(f, " max_neigh %d ", sim->maxNbors, 50))
  {
      printf("WARNING[113]: max_neigh is not specified. Set to default value, 50!\n");
  }

  fclose(f);
  return res;
}
// end 'init_md' function

void alloc_neighbors(Atoms *atm, Sim *sim)
// create neighbors arrays for different purposes
{
   int i, j;

   sim->nNbors = (int*)malloc(atm->nAt * int_size);                 // the number of neighbors
   sim->nbors = (int**)malloc(atm->nAt * pointer_size);             // indexes of neighbors
   sim->distances = (double**)malloc(atm->nAt * pointer_size);      // r to neighbors
   sim->tnbors = (int**)malloc(atm->nAt * pointer_size);            // type of neighbors
   for (i = 0; i < atm->nAt; i++)
   {
      sim->nNbors[i] = 0;
      sim->nbors[i] = (int*)malloc(sim->maxNbors * int_size);
      sim->distances[i] = (double*)malloc(sim->maxNbors * double_size);
      sim->tnbors[i] = (int*)malloc(sim->maxNbors * int_size);
      for (j = 0; j < sim->maxNbors; j++)
        sim->tnbors[i][j] = 0;
   }
}
// end 'alloc_neighbors' function

void free_neighbors(Atoms *atm, Sim *sim)
// free neighbors arrays
{
   //int i;
   //printf("nAt=%d\n", atm->nAt);

   /*
   for (i = 0; i < atm->nAt; i++)
   {

      delete[] sim->nbors[i];
      delete[] sim->distances[i];
      delete[] sim->tnbors[i];
   }
   */

   free(sim->nNbors);
   free(sim->nbors);
   free(sim->distances);
   free(sim->tnbors);
}
// end 'free_neighbors' function

int init_md(Atoms *atm, Field *field, Sim *sim, Elec *elec, TStat *tstat, Box *bx)
{
    int i, k;
    int res = 1;

    // READ FROM FILES AND RESET COUNTERS
    res *= read_field(field, sim);
    res *= read_atoms_box(atm, field, sim, bx);
    res *= read_sim(atm, field, sim, elec, tstat);

    init_rdf(sim, bx);

    // PREPARE STRUCTURES
    //! functions 'prepare_...' calculate derived parameters based on read by functions 'read_...'
    prepare_box(bx);
    prepare_elec(atm, field, elec, sim, bx); // constant part of Ewald summation is calculated here too


    // calculated derived parameters of a force field (0.5*dt/mass)
    //! может перенести в read_field? 
    for (i = 0; i < field->nSpec; i++)
        field->species[i].rMass_hdt = 0.5 * sim->tSt / field->species[i].mass;

    //find maximal cuttoff
    sim->rMax = 0.0;
    if (elec->type)
        sim->rMax = elec->rReal;
    else
    {
        if (field->nVdW)
            sim->rMax = field->maxRvdw;
        if (sim->use_bnd == 2)
            sim->rMax = max(sim->rMax, field->maxRbind);
        if (sim->eJump != 0)
            sim->rMax = max(sim->rMax, sim->rElec);
    }
    sim->r2Max = sim->rMax * sim->rMax;
    /*
    sim->double_pars[dpMaxCut] = sim->double_pars[dpMaxCutVdW];
    for (i = 1; i < 4; i++)   //! DANGER, a dependence on constant definition
        if (sim->double_pars[dpMaxCut] < sim->double_pars[i])
            sim->double_pars[dpMaxCut] = sim->double_pars[i];

    sim->r2Max = sim->double_pars[dpMaxCut] * sim->double_pars[dpMaxCut];
    //! what's about cutoff distances for bond forming?
    */

  // variable species identifiers
    sim->nVarSpec = 0;
    for (i = 0; i < field->nSpec; i++)
        if (field->species[i].varNumber)
            sim->nVarSpec++;

    sim->varSpecs = (int*)malloc(sim->nVarSpec * int_size);
    k = 0;
    for (i = 0; i < field->nSpec; i++)
        if (field->species[i].varNumber)
        {
            sim->varSpecs[k] = i; k++;
        }

    //FINAL CALCULATION FROM INPUT PARAMETERS
    // degree of freedom:
    sim->degFree = 3 * atm->nAt - sim->nBonds;    // 3N - nBonds
    if (tstat->type)  // thermostating is used
        sim->degFree--; // temperature is constant, so minus one degree of freedom

    sim->revDegFree = (double)(1.0 / sim->degFree);

    // thermostat setting
    tstat->tKin = 0.5 * sim->tTemp * kB * sim->degFree;
    if (tstat->type == tpTermNose)
    {
        tstat->qMass = 2 * tstat->tKin * tstat->tau * tstat->tau;
        tstat->rQmass = 0.5 / tstat->tKin / tstat->tau / tstat->tau; // 1/Qmass = 1/(2*sigma*tau^2)
        tstat->qMassTau2 = 2 * tstat->tKin; // qMass / tauT^2
    }

    // initial velocities setting (in case of gauss, because needed in thermostat setting and kinetic energy
    if (sim->int_pars[ipInitVel] == tpInitVelGauss)
        gauss_temp(atm, field->species, tstat, sim);

    return res;
}
// end 'prepare_md' function

int init_serial(Atoms* atm, Field* field, Sim* sim, Elec* elec, TStat* tstat, Box* bx)
// allocate auxilary arrays for serial code, define function variables
//! вынести эту функцию в модуль отдельный дл€ серийной версии, чтобы подключать по необходимости только еЄ
{
    int res = 1;    // function result
    //! здесь же инициализации rdf, массивов дл€ Ёвальда и т.д.

    // try to initialize cell list:
    if ((sim->flags >> bfSimUseCList) & 1)
    {
        init_clist(atm, sim, bx, sim->rMax /*sim->double_pars[dpMaxCut]*/);
    }

    // ALLOCATE ARRAYS
    //! functions 'init_...' initialize some arrays
    alloc_rdf(sim, field, bx);
    init_elec(elec, bx, sim, atm);
    if (sim->ejtype)
    {
        init_ejump(atm, field, sim);
        alloc_neighbors(atm, sim);  //! now neighbor list is used only ejumps functions...
    }

    center_box(atm, bx);


    //SET OPERATING FUNCTIONS
    if (sim->nHead)  // use cell list method (nHead is the number of cells)
    {
        sim->integrator1 = integrate1_clst;
        sim->forcefield = cell_list;
    }
    else // don't use cell list method
    {
        sim->integrator1 = integrate1;
        sim->forcefield = all_pairs;
    }

    if (sim->ejtype)
    {
        sim->ejumper = jumps[sim->ejtype];   // type of jumping
        sim->pair = pair_inter_lst;       // neighbors keeping

        //superstructure of electron jumping
        if (sim->eJump < 0)
            sim->do_jump = jmp_rare;
        else
            sim->do_jump = jmp_oft;
    }
    else
    {
        sim->pair = pair_inter;
        sim->do_jump = jmp_none;
    }

    sim->pair_elec = pair_elecs[elec->type];
    sim->add_elec = add_elecs[elec->type];

    //INITIAL STEP CALCULATIONS
    reset_chars(sim);
    clear_force(atm, field->species, sim, bx);
    sim->add_elec(atm, field, elec, bx, sim);
    all_pairs(atm, field, elec, bx, sim);

    return res;
}
// end 'init_serial' function

void info_md(Atoms *atm, Sim *sim)
// some information to output for user
{
   //printf("box[%d](%f x %f x %f)\n", box->type, box->ax, box->by, box->cz);
   printf("MD long %d timesteps of %f ps\n", sim->nSt, sim->tSt);

   //! add thermostating info

   if (sim->int_pars[ipInitVel] == tpInitVelGauss)
     printf("initital velocities are scaled with gaussian\n");

   if (sim->nHead)
     printf("used cell list method with %d cells\n", sim->nHead);

   //! add electrostatic info

   if (sim->eJump != 0)
   {
       printf("electron jumps (max r: %f A, type: %s) ", sim->rElec, nmsJumpType[sim->ejtype]);
       if (sim->eJump < 0)
         printf("every %d-th timestep", sim->eJump);
       else
         printf("%d times per timestep", sim->eJump);

       if (sim->ejtype == tpJumpEq)
         printf(" dE=%f\n", sim->dEjump);
       else
         printf("\n");

       printf("nFreeEl: %d\n", sim->nFreeEl);
   }

   //! add vdw info

   //! add bonded info
   /*
   printf("Bonded interactions:\n");
   for (i = 0; i < sim->nBtypes; i++)
    printf("Bonds[%d]: %d %d %d %d\n", i+1, bTypes[i].spec1, bTypes[i].spec2, bTypes[i].spec1br, bTypes[i].spec2br);
   */

   /*
   printf("Species that can bond:\n");
   for (i = 0; i < sim->nSpec; i++)
    if (species[i].canBond)
      printf("%s[%d,%d,%d]\n", species[i].name, species[i].bondKeys[0], species[i].bondKeys[1], species[i].bondKeys[2]);
   */

   //printf("initial : degFree=%d revDegFree=%f\n", ssim->degFree, sim->revDegFree);
   printf("initial values:\n");
   printf("x0=%4.3f vx=%4.3f fx=%4.3f VdW=%E Real_coul=%E Ewald=%E\n", atm->xs[0], atm->vxs[0], atm->fxs[0], sim->engVdW, sim->engElec3, /*potE1*/ sim->engElec2);
   //printf("rReal=%f r2Max=%f alpha=%f\n", sim->r2Real, sim->r2Max, sim->alpha);
}
// end 'info_md' function

void free_field(Field *field)
// deallocate arrays assigned with a force field
{
    int i;

    if (field->nBdata)
    {
        for (i = 0; i < field->nSpec; i++)
        {
            free(field->bond_matrix[i]);
            free(field->bonding_matr[i]);
            free(field->bindR2matrix[i]);
        }
        free(field->bond_matrix);
        free(field->bonding_matr);
        free(field->bindR2matrix);

        free(field->bdata);
        if (field->nBonds)
        {
            free(field->at1);
            free(field->at2);
            free(field->bTypes);
        }
    }

    free_angles(field);     // all verifiyng are inside

    if (field->nVdW)
    {
        for (i = 0; i < field->nSpec; i++)
            free(field->vdws[i]);
        free(field->vdws);
        free(field->pairpots);
    }

    free(field->snames);
    free(field->nnames);
    free(field->nnumbers);
    free(field->species);
}
// end 'free_field' function

void free_serial(Atoms *atm, Field *field, Elec *elec, Sim *sim)
// deallocate arrays
{
   free_rdf(sim, field);
   free_elec(elec, atm);

   if (sim->ejtype)
   {
       free_ejump(sim, field);
       free_neighbors(atm, sim);    //! neighbors are used only for ejump functions
   }

   if (sim->nHead)
     free_clist(sim);

   if (sim->nVarSpec)
       free(sim->varSpecs);
}
// end 'free_serial' function

void free_md(Atoms* atm, Field* field, TStat *tstat)
// deallocate arrays
{
    free_tstat(tstat);
    free_field(field);
    free_atoms(atm);   // free atoms-associated arrays
}
// end 'free_md' function


