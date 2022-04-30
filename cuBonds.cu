#include <stdio.h>

#include "dataStruct.h"
#include "defines.h"
#include "cuStruct.h"
#include "cuMDfunc.h"
#include "cuBonds.h"
#include "utils.h"
#include "cuUtils.h"

void init_cuda_bonds(Atoms* atm, Field* fld, Sim* sim, cudaMD* hmd, hostManagMD* man)
// copy bonds data to device, hmd - host exemplar of cudaMD struct
{
    int i, j;
    int nsize = atm->mxAt * int_size;

    hmd->mxBond = fld->mxBonds;
    hmd->nBond = fld->nBonds;

    man->bndPerBlock = ceil((double)fld->mxBonds / (double)man->nMultProc);
    man->bndPerThread = ceil((double)man->bndPerBlock / (double)man->nSingProc);
    if (man->bndPerBlock < (man->bndPerThread * man->nSingProc))
        man->bndPerBlock = man->bndPerThread * man->nSingProc;    // but not less 


    cudaBond* bndTypes = (cudaBond*)malloc(fld->nBdata * sizeof(cudaBond));
    hmd->nBndTypes = fld->nBdata;
    for (i = 1; i < fld->nBdata; i++)   // i = 0 reserved for empty(deleted) bond
    {
        bndTypes[i].type = fld->bdata[i].type;
        bndTypes[i].spec1 = fld->bdata[i].spec1;
        bndTypes[i].spec2 = fld->bdata[i].spec2;
        bndTypes[i].mxEx = fld->bdata[i].mxEx;
        bndTypes[i].mnEx = fld->bdata[i].mnEx;
        bndTypes[i].new_type[0] = fld->bdata[i].new_type[0];
        bndTypes[i].new_type[1] = fld->bdata[i].new_type[1];

        if (bndTypes[i].mnEx)
        {
            j = bndTypes[i].new_type[0];
            if (j < 0)  // invert species
            {
                bndTypes[i].new_spec1[0] = fld->bdata[-j].spec2;
                bndTypes[i].new_spec2[0] = fld->bdata[-j].spec1;
            }
            else
            {
                bndTypes[i].new_spec1[0] = fld->bdata[j].spec1;
                bndTypes[i].new_spec2[0] = fld->bdata[j].spec2;
            }
        }

        if ((bndTypes[i].mxEx) && (bndTypes[i].new_type[1] != 0))   // не удаляем
        {
            j = bndTypes[i].new_type[1];
            if (j < 0)  // invert species
            {
                bndTypes[i].new_spec1[1] = fld->bdata[-j].spec2;
                bndTypes[i].new_spec2[1] = fld->bdata[-j].spec1;
            }
            else
            {
                bndTypes[i].new_spec1[1] = fld->bdata[j].spec1;
                bndTypes[i].new_spec2[1] = fld->bdata[j].spec2;
            }
        }


        bndTypes[i].new_spec1[0] = fld->bdata[i].new_spec1[0];
        bndTypes[i].new_spec1[1] = fld->bdata[i].new_spec1[1];
        bndTypes[i].new_spec2[0] = fld->bdata[i].new_spec2[0];
        bndTypes[i].new_spec2[1] = fld->bdata[i].new_spec2[1];
        bndTypes[i].hatom = fld->bdata[i].hatom;
        bndTypes[i].evol = fld->bdata[i].evol;

        //! переопределить spec2 для мин и макс, если макс не удалить

        bndTypes[i].p0 = (float)fld->bdata[i].p0;
        bndTypes[i].p1 = (float)fld->bdata[i].p1;
        bndTypes[i].p2 = (float)fld->bdata[i].p2;
        bndTypes[i].p3 = (float)fld->bdata[i].p3;
        bndTypes[i].p4 = (float)fld->bdata[i].p4;
        bndTypes[i].r2min = (float)fld->bdata[i].r2min;
        bndTypes[i].r2max = (float)fld->bdata[i].r2max;
        bndTypes[i].count = fld->bdata[i].number;
        bndTypes[i].rSumm = 0.f;
        bndTypes[i].rCount = 0;
        bndTypes[i].ltSumm = 0;
        bndTypes[i].ltCount = 0;
    }
    data_to_device((void**)&(hmd->bondTypes), bndTypes, fld->nBdata * sizeof(cudaBond));
    free(bndTypes);

    int4* bnds = (int4*)malloc(fld->nBonds * int4_size);
    for (i = 0; i < fld->nBonds; i++)
    {
        bnds[i] = make_int4(fld->at1[i], fld->at2[i], fld->bTypes[i], 0);
    }
    cudaMalloc((void**)&(hmd->bonds), fld->mxBonds * int4_size);
    cudaMemcpy(hmd->bonds, bnds, fld->nBonds * int4_size, cudaMemcpyHostToDevice);
    free(bnds);

    data_to_device((void**)&(hmd->nbonds), atm->nBonds, nsize);

    //int* int_array;
    int** int_int_array;
    float** fl_fl_array;
    int_int_array = (int**)malloc(fld->nSpec * pointer_size);
    for (i = 0; i < fld->nSpec; i++)
    {
        data_to_device((void**)&(int_int_array[i]), fld->bond_matrix[i], fld->nSpec * int_size);
    }
    data_to_device((void**)&(hmd->def_bonds), int_int_array, fld->nSpec * pointer_size);
    free(int_int_array);

    data_to_device((void**)&(hmd->parents), atm->parents, nsize);

    if (sim->use_bnd == 2)  // binding
    {
        cudaMalloc((void**)&(hmd->neighToBind), nsize);
        cudaMalloc((void**)&(hmd->canBind), nsize);
        cudaMalloc((void**)&(hmd->r2Min), nsize);


        int_int_array = (int**)malloc(fld->nSpec * pointer_size);
        fl_fl_array = (float**)malloc(fld->nSpec * pointer_size);
        float* fl_array = (float*)malloc(fld->nSpec * sizeof(float));
        for (i = 0; i < fld->nSpec; i++)
        {
            data_to_device((void**)&(int_int_array[i]), fld->bonding_matr[i], fld->nSpec * int_size);
            for (j = 0; j < fld->nSpec; j++)
                fl_array[j] = (float)fld->bindR2matrix[i][j];
            data_to_device((void**)&(fl_fl_array[i]), fl_array, fld->nSpec * float_size);
        }
        data_to_device((void**)&(hmd->bindBonds), int_int_array, fld->nSpec * pointer_size);
        data_to_device((void**)&(hmd->bindR2), fl_fl_array, fld->nSpec * pointer_size);

        free(int_int_array);
        free(fl_fl_array);
        free(fl_array);
    }
}

__device__ void try_to_bind(float r2, int id1, int id2, int spec1, int spec2, cudaMD *md)
{
    int r2Int;      //  (int)r2 * const

#ifdef DEBUG_MODE
    if ((spec1 < 0) || (spec1 >= MX_SPEC) || (spec2 < 0) || (spec2 >= MX_SPEC))
        printf("ERROR[002] In func 'try_to_bind'. Atom types have wrong values: %d and %d\n", spec1, spec2);
    if (id1 == id2)
        printf("try bind: the same atom indexes: %d\n", id1);
#endif
    //printf("try to bind\n");

    // надо бы исключить самосвязывание, хотя бы попытаться
    if (md->parents[id1] == id2)
        return;
    if (md->parents[id2] == id1)
        return;

    if (md->bindBonds[spec1][spec2] != 0)
    {
        if (r2 < md->bindR2[spec1][spec2])
        {
            //printf("try to bind\n");
            r2Int = (int)(r2 * 100);

            if (atomicMin(&(md->r2Min[id1]), r2Int) > r2Int)    // replace was sucessfull
            {
                // тут есть опасность, что пока мы приступили к этому, кто-то уже перезаписал минимум, ну принебрежем этим, кто успел тот и успел
                //atomicExch(&(md->neighToBind[id1]), id2);   
                
                // !можно наверно и без атомик:
                md->neighToBind[id1] = id2 + 1;     // as 0 is reserved for no bind
                md->canBind[id1] = 1;
                //atomicExch(&(md->canBind[id1]), 1);
            }

            // similar for the second atom
            if (atomicMin(&(md->r2Min[id2]), r2Int) > r2Int)    // replace was sucessfull
            {
                // тут есть опасность, что пока мы приступили к этому, кто-то уже перезаписал минимум, ну принебрежем этим, кто успел тот и успел
                //atomicExch(&(md->neighToBind[id2]), id1);   

                // !можно наверно и без атомик:
                md->neighToBind[id2] = id1 + 1;     // as 0 is reserved for no bind
                md->canBind[id2] = 1;
                //atomicExch(&(md->canBind[id2]), 1);
            }

            //printf("at[%d](sp=%d) and [%d](sp=%d) can be bonded\n", id1, spec1, id2, spec2);
        }
    }
}

__device__ void invert_bond(int &id1, int &id2, int &spec1, int &spec2, int4 *bnd)
// change places of atoms in the bond
{
    int tmp;

    bnd->x = id2;
    bnd->y = id1;
    tmp = spec1;
    spec1 = spec2;
    spec2 = tmp;
    tmp = id1;
    id1 = id2;
    id2 = tmp;
}

__device__ void keep_bndlifetime(int iStep, int4 *bond, cudaBond *type)
// add lifetime info to type record
{
    if (iStep != bond->w)
    {
        atomicAdd(&(type->ltSumm), (iStep - bond->w));
        atomicAdd(&(type->ltCount), 1);
    }

    // start new lifetime
    bond->w = iStep;

}

__device__ void exclude_parents(int id1, int id2, cudaMD* md)
// exclude id1 and id2 from parents of each other (if they are)
//  and seek other parents if able
{
    // flags to clear parent
    int clear_1 = 0;    
    int clear_2 = 0;
    int i, flag;

    //printf("_----- EXCLUDE PARENTS -------\n");


    if (md->parents[id1] == id2) 
    {
        if (md->nbonds[id1] > 0)
            clear_1 = 1;
        else // if there are no bonds with the atom, just reset its parent
            md->parents[id1] = -1;
    }
    if (md->parents[id2] == id1)
    {
        if (md->nbonds[id2] > 0)
            clear_2 = 1;
        else // if there are no bonds with the atom, just reset its parent
            md->parents[id2] = -1;
    }

/*
    if (clear_1 || clear_2)
    {
        for (i = 0; i < md->nBond; i++)
            if (md->bonds[i].z != 0)    // сюда же входит и связь id1-id2, поскольку она предварительно была обнулена в apply_bonds
            {
                if (md->bonds[i].x == id1)
                {
                    if (clear_1)
                    {
                        md->parents[id1] = md->bonds[i].y;
                        clear_1 = 0;
                        if (!clear_2)
                            break;
                        continue;   // because it can't be bonded with id2 (id1-id2 has already excluded), and can't be bonded with id1 (bond id1-id1 in this case)
                    }
                }
                if (md->bonds[i].y == id1)
                {
                    if (clear_1)
                    {
                        md->parents[id1] = md->bonds[i].x;
                        clear_1 = 0;
                        if (!clear_2)
                            break;
                        continue;   // see reasons above
                    }
                }
                if (md->bonds[i].x == id2)
                {
                    if (clear_2)
                    {
                        md->parents[id2] = md->bonds[i].y;
                        clear_2 = 0;
                        if (!clear_1)
                            break;
                        continue;   // see reasons above
                    }
                }
                if (md->bonds[i].y == id2)
                {
                    if (clear_2)
                    {
                        md->parents[id2] = md->bonds[i].x;
                        clear_2 = 0;
                        if (!clear_1)
                            break;
                        continue;   // see reasons above
                    }
                }

            }

        // end loop by replacing parents
        if (clear_1)    // если даже после этого не очистились - зануляем
            md->parents[id1] = -1;
        if (clear_2)    
            md->parents[id2] = -1;
    }

*/

    i = 0;
    while ((i < md->nBond) && (clear_1 || clear_2))
    {
        if (md->bonds[i].z != 0)
////////////  H-bonds addition
            if (md->bondTypes[md->bonds[i].z].hatom == -1)      //! H-bond extension
////////////  end H-bonds addition
            {
                flag = 0;
                if (clear_1)
                {
                    if (md->bonds[i].x == id1)
                    {
                        md->parents[id1] = md->bonds[i].y;
                        flag = 1;
                    }
                    else if (md->bonds[i].y == id1)
                    {
                        md->parents[id1] = md->bonds[i].x;
                        flag = 1;
                    }

                    if (flag)
                    {
                        clear_1 = 0;
                        i++;
                        continue;
                    }
                }
                if (clear_2)
                {
                    if (md->bonds[i].x == id2)
                    {
                        md->parents[id2] = md->bonds[i].y;
                        flag = 1;
                    }
                    else if (md->bonds[i].y == id2)
                    {
                        md->parents[id2] = md->bonds[i].x;
                        flag = 1;
                    }

                    if (flag)
                    {
                        clear_2 = 0;
                        i++;
                        continue;
                    }
                }
            }
        i++;
    }
    
    //! на всякий случай (на самом деле, если такое случилось, что-то пошло не так, завернуть в DEBUG)
    if (clear_1)    // если даже после этого не очистились - зануляем
        md->parents[id1] = -1;
    if (clear_2)
        md->parents[id2] = -1;
}

__device__ void exclude_H_parent(int hid, int pid, cudaMD* md)
// analog of exclude_parents function for hydrogen bonds
//  exclude pid from parent of H-atom (atoms[hid]), seek another parent for hid if acceptable
{
    int i;

    if (md->parents[hid] == pid)
    {
        if (md->nbonds[hid] > 0)    // there are bonds with this H-atom, seek a new parent
        {
            for (i = 0; i < md->nBond; i++)
                if (md->bonds[i].z != 0)
                {
                    if (md->bonds[i].x == hid)
                    {
                        md->parents[hid] = md->bonds[i].y;
                        break;
                    }
                    else if (md->bonds[i].y == hid)
                    {
                        md->parents[hid] = md->bonds[i].x;
                        break;
                    }
                }

            //! на всякий случай (вдруг что-то пошло не так, завернуть в дебаг)
            if (md->parents[hid] == pid)
            {
                md->parents[hid] = -1;      // если даже после этого не очистились - зануляем
            }
        }
        else // if there are no bonds with the atom, just reset its parent
            md->parents[hid] = -1;
    } // otherwise we need to do nothing
}

__device__ int evol_bondtype(cudaBond *btype, int spec1, int spec2, cudaMD* md)
// return new bond type to which btype must evolute when consisting atoms change their types
{
    if (btype->evol != 0) // try to use 'evol' default bond type
    {
        cudaBond* new_bnd = &(md->bondTypes[btype->evol]);
        if ((spec1 == new_bnd->spec1) && (spec2 == new_bnd->spec2))
            return btype->evol;
        else
            if ((spec1 == new_bnd->spec2) && (spec2 == new_bnd->spec1))
                return -btype->evol;
            else
                return md->def_bonds[spec1][spec2];
    }
    else // use matrix of default bonds
      return  md->def_bonds[spec1][spec2];
}

__device__ cudaBond* evol_bondtype_addr(cudaBond* old_bnd, int spec1, int spec2, cudaMD* md)
// return pointer to evolved bond type
{
    int def = evol_bondtype(old_bnd, spec1, spec2, md);
    if (def == 0)     // these atom types do not form a bond
        return NULL;
    else
    {
        if (def < 0)    def = -def;
        return &(md->bondTypes[def]);
    }
}

__global__ void apply_bonds(int iStep, int bndPerBlock, int bndPerThread, cudaMD* md)
{
    int def;
    int id1, id2;       // atom indexes
    int old, old_spec2, spec1, spec2, new_spec1, new_spec2;     // atom types
    int new_bond_type;
    
    int save_lt, need_r, loop;    // flags to save lifetime, to need to calculate r^2 and to be in while loop
    int mnmx;   // flag minimum or maximum
    int action; // flag: 0 - do nothing, 1 - delete bond, 2 - transform bond
    cudaBond *old_bnd, *cur_bnd;
    float dx, dy, dz, r2, r;
    float f, eng = 0.0f;
    __shared__ float shEng;
#ifdef DEBUG_MODE
    int cnt, loop_cnt;    // count of change spec2 loops and while(loop)
#endif

    if (threadIdx.x == 0)
    {
        shEng = 0.f;
    }
    __syncthreads();

    int id0 = blockIdx.x * bndPerBlock + threadIdx.x * bndPerThread;
    int N = min(id0 + bndPerThread, md->nBond);
    int iBnd;

    for (iBnd = id0; iBnd < N; iBnd++)

      if (md->bonds[iBnd].z)  // the bond is not broken
      {
          //printf("bnd[%d] type=%d exists (nBond=%d), thread=%d\n", iBnd, md->bonds[iBnd].z, md->nBond, threadIdx.x);

          // atom indexes
          id1 = md->bonds[iBnd].x;
          id2 = md->bonds[iBnd].y;

          // atom types
          spec1 = md->types[id1];
          spec2 = md->types[id2];

          old_bnd = &(md->bondTypes[md->bonds[iBnd].z]);
          cur_bnd = old_bnd;
/////     H-bonds addition
          int was_hatom = -1;    // save atom index, which was an hydrogen atom:
          if (cur_bnd->hatom == spec1)
              was_hatom = id1;
          else if (cur_bnd->hatom == spec2)
              was_hatom = id2;
/////     end H-bonds addition

          save_lt = 0;
          need_r = 1;
          loop = 1;
#ifdef DEBUG_MODE
          cnt = 0;
          loop_cnt = 0;
#endif

          //bnd = &(md->bondTypes[old_bond_type]);
          //here:
          if ((cur_bnd->spec1 == spec1)&&(cur_bnd->spec2 == spec2))
          {
              //ok
          }
          else
              if ((cur_bnd->spec1 == spec2) && (cur_bnd->spec2 == spec1) && (spec1 != spec2))
              {
                  invert_bond(id1, id2, spec1, spec2, &(md->bonds[iBnd]));
                  //... then ok
#ifdef DEBUG_MODE
                  //printf("inverted bond detected\n");
#endif
              }
              else // atom types do not correspond to bond types
              {
                  save_lt = 1;
#ifdef DEBUG_MODE
                  //printf("modified bond type detected\n");
#endif
              }

          // it's actual for half-periodic box, some atoms can escape box and we need to delete such bonds:
          //! maybe it's necessary to add verify if (box_type == tpBoxRect) .... to avoid meaningless check
          if ((md->xyz[id1].z < 0) || (md->xyz[id2].z < 0) || (md->xyz[id1].z > md->leng.z) || (md->xyz[id2].z > md->leng.z))
          {
              save_lt = 1;  // save lifetime
              action = 1;   // delete this bond
              loop = 0;     // do not perform the cycle (see further code)
          }


          // end initial stage
          while (loop)
          {
             if (save_lt)       // этот флаг кроме всего прочего, символизирует, что вначале нам уже потребовалось переопределить связь, при повторном прохождении цикла, он всегда будет true
             {
                 //def = md->def_bonds[spec1][spec2];
                 def = evol_bondtype(cur_bnd, spec1, spec2, md);
                 if (def == 0)     // these atom types do not form a bond
                 {
#ifdef DEBUG_MODE
                     printf("probably, something goes wrong\n");
#endif
                     action = 1;   // delete
                     break;
                 }
                 else
                 {
                     //! меняем связь и поехали дальше
                     if (def < 0)  // если обратная, меняем опять же порядок атомов в связи
                     {
                         invert_bond(id1, id2, spec1, spec2, &(md->bonds[iBnd]));
                         def = -def;
                     }
#ifdef DEBUG_MODE
                     if (md->def_bonds[spec1][spec2] != def)
                         printf("still wrong bond type between %d and %d = %d! Must be %d!\n", spec1, spec2, def, md->def_bonds[spec1][spec2]);
#endif

                     md->bonds[iBnd].z = def;
                     cur_bnd = &(md->bondTypes[def]);
                 }
             }  // end if (save_lt)

             // calculate distance (only once)
             if (need_r)
             {
                dx = md->xyz[id1].x - md->xyz[id2].x;
                dy = md->xyz[id1].y - md->xyz[id2].y;
                dz = md->xyz[id1].z - md->xyz[id2].z;
                //delta_periodic_orth(dx, dy, dz, md);
                md->funcDeltaPer(dx, dy, dz, md);
                r2 = dx * dx + dy * dy + dz * dz;
                need_r = 0;
             }

             action = 0;   // 0 - just cultivate bond 1 - delete bond 2 - transformate bond
             
             if ((cur_bnd->mxEx) && (r2 > cur_bnd->r2max))
             {
                 mnmx = 1;
                 if (cur_bnd->new_type[mnmx] == 0)  // delete bond
                   action = 1;
                else
                   action = 2;   // modify bond
             }
             else if ((cur_bnd->mnEx) && (r2 < cur_bnd->r2min))
             {
                 mnmx = 0;
                 action = 2;   // at minimum only modify bond
             }
             // end select action
             

             // try to change atom types (if needed)
             if (action)
             {
                 save_lt = 1;
                 new_spec1 = cur_bnd->new_spec1[mnmx];
                 new_spec2 = cur_bnd->new_spec2[mnmx];
#ifdef DEBUG_MODE
                 if ((new_spec1 < 0)||(new_spec1 >= 15)||(new_spec2 < 0)||(new_spec2 >= 15))
                     printf("ERROR[003]: in apply_bonds, change atom types. There are no one of such types: %d and %d. Minmax=%d cur_bnd=%p\n", new_spec1, new_spec2, mnmx, cur_bnd);
#endif
                 //the first atom
                 old = atomicCAS(&(md->types[id1]), spec1, new_spec1);
                 if (old != spec1)
                 {
                     spec1 = old;
                     spec2 = md->types[id2];   // обновляю значение 2го типа атома, вдруг оно тоже изменилось
                     // return to begin of the cycle
                 }
                 else      // types[id1] have been changed
                 {
                     //if (md->use_angl == 2) //      variable angles
                         atomicCAS(&(md->oldTypes[id1]), -1, spec1);
                     old_spec2 = spec2;
                     while ((old = atomicCAS(&(md->types[id2]), old_spec2, new_spec2)) != old_spec2)
                     {
                         //! наихудший вариант: А успели поменять, а Б - нет
                         // представляем, что у нас была связь A-old и она реагирует также на такую длину связи

                         //def = md->def_bonds[spec1][old];
                         def = evol_bondtype(cur_bnd, spec1, old, md);
#ifdef DEBUG_MODE
                         if (def == 0)
                         {
                             printf("UBEH[001]: in apply_bonds, change atom types. There are no bond types between Species[%d] and [%d]\n", spec1, old);
                             break;
                         }
#endif
                         if (def < 0)  // spec1 -> new_spec2 spec2 -> newSpec1
                         {
                             cur_bnd = &(md->bondTypes[-def]);
                             new_spec2 = cur_bnd->new_spec1[mnmx];
                         }
                         else // direct order
                         {
                             cur_bnd = &(md->bondTypes[def]);
                             new_spec2 = cur_bnd->new_spec2[mnmx];
                         }
                         old_spec2 = old;
#ifdef DEBUG_MODE
                         if ((new_spec2 < 0) || (new_spec2 >= MX_SPEC))
                             printf("ERROR[004]: step %d, apply_bonds function: try to set wrong type value:%d (bond_type:%d, mnmx:%d)\n", iStep, new_spec2, def, mnmx);
                         cnt++;
                         if (cnt > 10)
                         {
                             printf("UBEH[002]: too many atempst to change spec2 = %d\n", spec2);
                             break;
                         }
#endif
                     }
                     // сохраняем, что мы изменяли этот тип атома, если это не было сохранено ранее
                     //if (md->use_angl == 2) // variable angles
                         atomicCAS(&(md->oldTypes[id2]), -1, spec2);
                     loop = 0;
                 }

                 //end change types

             } // end if (action)
             else
               loop = 0;    // action == 0, out of cycle

#ifdef DEBUG_MODE
             loop_cnt++;
#endif
          }  // end while(loop)


          // 0 - just cultivate bond 1 - delete bond 2 - transformate bond
          // единственный случай, когда нужно обновить bnd
          if (action == 2)
          {
#ifdef DEBUG_MODE
              if (cur_bnd->new_type[mnmx] > md->nBndTypes)
                  printf("new_type = %d\n", cur_bnd->new_type[mnmx]);
              if (cur_bnd->new_type[mnmx] < -md->nBndTypes)
                  printf("new_type = %d\n", cur_bnd->new_type[mnmx]);
#endif
////////////  H-bonds addition
              int old_hatom = cur_bnd->hatom;
////////////  end H-bonds addition
              new_bond_type = cur_bnd->new_type[mnmx];
              if (new_bond_type < 0)
              {
                  md->bonds[iBnd].x = id2;
                  md->bonds[iBnd].y = id1;
                  new_bond_type = -new_bond_type;
              }
              md->bonds[iBnd].z = new_bond_type;
              cur_bnd = &(md->bondTypes[new_bond_type]);
////////////  H-bonds addition
              int cur_hatom = cur_bnd->hatom;
              if ((old_hatom == -1) && (cur_hatom != -1))   // the covalent bond transforms into an H-bond
              {
                  // atom types have already been changed
                  if (md->types[id1] == cur_hatom)
                  {
                      atomicSub(&(md->nbonds[id2]), 1);
                      exclude_H_parent(id1, id2, md);
                  }
                  else
                  {
                      atomicSub(&(md->nbonds[id1]), 1);
                      exclude_H_parent(id2, id1, md);
                  }
              }
              else if ((old_hatom != -1) && (cur_hatom == -1))   // the H-bond bond transforms into a covalent bond
              {
                  // atom types have already been changed and now this is a problem, because we need to identify which atom was an hydrogen atom
                  if (id1 == was_hatom)
                  {
                      atomicAdd(&(md->nbonds[id2]), 1);
                      if (md->parents[id2] == -1)
                          md->parents[id2] = id1;
                  }
                  else
                  {
                      atomicAdd(&(md->nbonds[id1]), 1);
                      if (md->parents[id1] == -1)
                          md->parents[id1] = id2;
                  }
              }
////////////  end H-bonds addition

              //atomicAdd(&(cur_bnd->count), 1);    // теперь в if (save_lt)
          }

          // perform calculations and save mean distance
          if (action != 1)  // not delete
          {
              r = sqrt(r2);
              //printf("bnd[%d] type=%d (spec1=%d) func=%p harmfunc=%p act=%d, thread=%d r2=%f\n", iBnd, cur_bnd->type, cur_bnd->spec1, (cur_bnd->force_eng), &bond_harm, action, threadIdx.x, r2);
              
              
              f = cur_bnd->force_eng(r2, r, eng, cur_bnd);

              //! ВРЕМЕННО! для уравновешивания связей
              /*
              if ((f > -0.1f) && (f < 0.1f))
              {
                  md->vls[id1] = make_float3(0.f, 0.f, 0.f);
                  md->vls[id2] = make_float3(0.f, 0.f, 0.f);
              }
              */

              //printf("f*dx=%f\n", f* dx);
              atomicAdd(&(md->frs[id1].x), f * dx);
              atomicAdd(&(md->frs[id2].x), -f * dx);
              atomicAdd(&(md->frs[id1].y), f * dy);
              atomicAdd(&(md->frs[id2].y), -f * dy);
              atomicAdd(&(md->frs[id1].z), f * dz);
              atomicAdd(&(md->frs[id2].z), -f * dz);

              atomicAdd(&(cur_bnd->rSumm), r);
              atomicAdd(&(cur_bnd->rCount), 1);
              
          }
          else      //delete bond
          {
              //printf("delete bond[%d] atm: %d-%d\n", iBnd, id1, id2);
////////////  H-bonds addition
              if ((old_bnd->hatom == -1)||((old_bnd->hatom != -1)&&(id1 == was_hatom)))
////////////  end H-bonds addition
                  atomicSub(&(md->nbonds[id1]), 1);
////////////  H-bonds addition
              if ((old_bnd->hatom == -1) || ((old_bnd->hatom != -1) && (id2 == was_hatom)))
////////////  end H-bonds addition
                  atomicSub(&(md->nbonds[id2]), 1);
              md->bonds[iBnd].z = 0;
              //atomicSub(&(old_bnd->count), 1);    // теперь в if (save_lt)

              //! здесь и везде надо сделать пересчет количеств специев

              // change parents
////////////  H-bonds addition
              if (old_bnd->hatom == -1)
////////////  end H-bonds addition
                  exclude_parents(id1, id2, md);
////////////  H-bonds addition
              else
              {
                  if (id1 == was_hatom)
                      exclude_H_parent(id1, id2, md);
                  else
                      exclude_H_parent(id2, id1, md);
              }
////////////  end H-bonds addition
          }

          
          if (save_lt)
          {
              keep_bndlifetime(iStep, &(md->bonds[iBnd]), old_bnd);
              if (action != 1) // not delete
                atomicAdd(&(cur_bnd->count), 1);
              atomicSub(&(old_bnd->count), 1);
          }
          
      } // end main loop

      // split energy to shared and then to global memory
      atomicAdd(&shEng, eng);
      __syncthreads();
      if (threadIdx.x == 0)
          atomicAdd(&(md->engBond), shEng);
}
// end 'apply_bonds' function


__global__ void apply_const_bonds(int iStep, int bndPerBlock, int bndPerThread, cudaMD* md)
// similar to apply_bonds, but if number and type of bonds are constant
{
    int id1, id2;       // atom indexes
//    int spec1, spec2;     // atom types

    cudaBond* cur_bnd;
    float dx, dy, dz, r2, r;
    float f, eng = 0.0f;
    __shared__ float shEng;

    if (threadIdx.x == 0)
    {
        shEng = 0.f;
    }
    __syncthreads();

    int id0 = blockIdx.x * bndPerBlock + threadIdx.x * bndPerThread;
    int N = min(id0 + bndPerThread, md->nBond);
    int iBnd;

    for (iBnd = id0; iBnd < N; iBnd++)
            if (md->bonds[iBnd].z)  //! вообще здесь эту проверку можно и опустить
            {
                // atom indexes
                id1 = md->bonds[iBnd].x;
                id2 = md->bonds[iBnd].y;
                // atom types
                //spec1 = md->types[id1];
                //spec2 = md->types[id2];
                cur_bnd = &(md->bondTypes[md->bonds[iBnd].z]);


                dx = md->xyz[id1].x - md->xyz[id2].x;
                dy = md->xyz[id1].y - md->xyz[id2].y;
                dz = md->xyz[id1].z - md->xyz[id2].z;
                //delta_periodic_orth(dx, dy, dz, md);
                md->funcDeltaPer(dx, dy, dz, md);
                r2 = dx * dx + dy * dy + dz * dz;
                r = sqrt(r2);
                f = cur_bnd->force_eng(r2, r, eng, cur_bnd);

                //! debugging
               // if (isnan(f))
                 //   printf("apply_const_bonds, f is nan\n");

                //! ВРЕМЕННО! для уравновешивания связей
                //if ((f > -0.1f) && (f < 0.1f))
                //{
                  //  md->vls[id1] = make_float3(0.f, 0.f, 0.f);
                  //  md->vls[id2] = make_float3(0.f, 0.f, 0.f);
                //}

                //! temp
                //if ((r < 1.f) || (r > 2.2f))
                  //  printf("length(%d=%d-%d)[%d]=%f, frc=%f dx=%f at1(%f,) at2(%f,) fx1=%f fx2=%f\n", iBnd, id1, id2, cur_bnd->type, r, f, dx, md->xyz[id1].x, md->xyz[id2].x, md->frs[id1].x, md->frs[id2].x);

                //printf("f*dx=%f\n", f* dx);
                atomicAdd(&(md->frs[id1].x), f * dx);
                atomicAdd(&(md->frs[id2].x), -f * dx);
                atomicAdd(&(md->frs[id1].y), f * dy);
                atomicAdd(&(md->frs[id2].y), -f * dy);
                atomicAdd(&(md->frs[id1].z), f * dz);
                atomicAdd(&(md->frs[id2].z), -f * dz);

                //if ((md->vls[id1].x < -10.f) || (md->vls[id1].x > 10.f))
                //{
                 //   printf("high speed bond(%d)[%d]length=%.2f f=%.2f dx=%.3f x1(%.3f) x2(%.3f) fx1=%.3f fx2=%.3f vx1=%.2f vx2=%.2f\n", iBnd, cur_bnd->type, r, f, dx, md->xyz[id1].x, md->xyz[id2].x, md->frs[id1].x, md->frs[id2].x, md->vls[id1].x, md->vls[id2].x);
                //}

                //if ((r < 1.f) || (r > 2.2f))
                //{
                //    printf("aft force upd: bond(%d)[%d]length=%.2f, force=%.2f dx=%.3f x1(%.3f) x2(%.3f) fx1=%.3f fx2=%.3f vx1=%.2f vx2=%.2f\n", iBnd, cur_bnd->type, r, f, dx, md->xyz[id1].x, md->xyz[id2].x, md->frs[id1].x, md->frs[id2].x, md->vls[id1].x, md->vls[id2].x);
                    //md->vls[i] = make_float3(0.f, 0.f, 0.f);
                    //md->vls[i] = make_float3(0.f, 0.f, 0.f);
                //}

                atomicAdd(&(cur_bnd->rSumm), r);
                atomicAdd(&(cur_bnd->rCount), 1);

            } // end main loop

            // split energy to shared and then to global memory
    atomicAdd(&shEng, eng);
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(&(md->engBond), shEng);
}
// end 'apply_const_bonds' function


__global__ void fix_bonds(int bndPerBlock, int bndPerThread, cudaMD* md)
// correct bond types for output
{
    int def;
    int id1, id2;       // atom indexes
    int spec1, spec2;     // atom types

    cudaBond* bnd;

    int id0 = blockIdx.x * bndPerBlock + threadIdx.x * bndPerThread;
    int N = min(id0 + bndPerThread, md->nBond);
    int iBnd;

    for (iBnd = id0; iBnd < N; iBnd++)
    {
        
        // atom indexes
        id1 = md->bonds[iBnd].x;
        id2 = md->bonds[iBnd].y;

        // atom types
        spec1 = md->types[id1];
        spec2 = md->types[id2];

        bnd = &(md->bondTypes[md->bonds[iBnd].z]);

        if ((bnd->spec1 == spec1) && (bnd->spec2 == spec2))
        {
            continue;   // everething is ok!
        }

        if ((bnd->spec1 == spec2) && (bnd->spec2 == spec1))
        {
            //invert bond
            md->bonds[iBnd].x = id2;
            md->bonds[iBnd].y = id1;
        }
        else // atom types do not correspond to bond types
        {
            atomicSub(&(bnd->count), 1);
            //def = md->def_bonds[spec1][spec2];
            def = evol_bondtype(bnd, spec1, spec2, md);
            if (def == 0)     // these atom types do not form a bond
            {
                printf("ERROR[007] in fix_bonds: species %d and %d can't be bonded\n", spec1, spec2);
            }
            else
            {
                //! меняем связь и поехали дальше
                //printf("fix bond\n");
                if (def < 0)  // если обратная, меняем опять же порядок атомов в связи
                {
                    md->bonds[iBnd].x = id2;
                    md->bonds[iBnd].y = id1;
                    def = -def;
                }

                md->bonds[iBnd].z = def;
                atomicAdd(&(md->bondTypes[def].count), 1);
            }
        }

    }   // end loop by bonds
}

__global__ void clear_bonds(cudaMD* md)
// clear bonds with .z == 0
{
    // не знаю, как сделать параллельный, вот серийный вариант:
    int i = 0;
    int j = md->nBond - 1;

    while (i < j)
    {
        while ((md->bonds[j].z == 0) && (j > i))
            j--;
        while ((md->bonds[i].z != 0) && (i < j))
            i++;
#ifdef DEBUG_MODE
        if ((i < 0) || (j < 0) || (i >= md->nBond) || (j >= md->nBond))
            printf("clear bonds: i=%d j=%d\n", i, j);
#endif
        if (i < j)
        {
            md->bonds[i] = md->bonds[j];
            md->bonds[j].z = 0;
            i++;
            j--;
        }
    }

    if ((i == j) && (md->bonds[i].z == 0))
        md->nBond = j;
    else
        md->nBond = j + 1;

    //printf("end clear bond\n");
}

__global__ void create_bonds(int iStep, int atPerBlock, int atPerThread, cudaMD* md)
// connect atoms which are selected to form bonds
{
    int id1, id2, nei;    // neighbour index
    int btype, bind;    // bond type index and index
    cudaBond* bnd;
    int spec1, spec2;   // species indexes

    //if (threadIdx.x == 0)
      //  if (blockIdx.x == 0)
        //    printf("creat_bonds(%d)\n", iStep);
    
    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);
    int iat;

    //! temp
    //if (md->nBond >= md->mxBond)
      //  return;

    for (iat = id0; iat < N; iat++)
    {
        nei = md->neighToBind[iat];
        if (nei)    // neighbour exists
        {
            nei--;  // because 0 means no neighbour (nei = atom_index + 1)
            //printf("step:%d at: %d(par: %d) nei: %d(par: %d)\n", iStep, iat, md->parents[iat], nei, md->parents[nei]);
            //if (iat == nei)
              //  printf("iat == nei!!!\n");

            //! тут сразу проверим и на parents (исключаем двойное связывание):
            //! вообще эта проверка уже выполняется в try to bind, так что тут может и сканает без нее
            //if (md->parents[iat] == nei)
              //  continue;
            //if (md->parents[nei] == iat)
              //  continue;

            //printf("create bond at atom[%d](%d)-[%d](%d)\n", iat, md->canBind[iat], nei, md->canBind[nei]);

            //! тут такая штука, если оставить как было, первым обратывается атом iat, а вторым nei
            //!  то у нас разные потоки беруется за это дело с разных сторон и получается один лочит первый атом в паре, другой - второй
            //!  и когда они добираются до 2ой проверки получают фолс и выходят, поэтому сразу упорядочим индексы
            if (iat < nei)
            {
                id1 = iat;
                id2 = nei;
            }
            else
            {
                //continue;   // попробуем просто выйти
                id1 = nei;
                id2 = iat;
            }

            spec1 = md->types[id1];
            spec2 = md->types[id2];
#ifdef DEBUG_MODE
            if ((spec1 < 0) || (spec1 >= MX_SPEC) || (spec2 < 0) || (spec2 >= MX_SPEC))
                printf("ERROR[005] %d step. Atom types have wrong values: %d[%d] and %d[%d]. \n", iStep, spec1, id1, spec2, id2);
#endif
            btype = md->bindBonds[spec1][spec2];
#ifdef DEBUG_MODE
            if (btype > md->nBndTypes)
                printf("ERROR[006] %step wrong bond type between species %d[%d] and %d[%d]\n", iStep, spec1, id1, spec2, id2);
            else
            {
                //printf("new type is ok\n");
            }
#endif

            // сразу проверим, могут ли эти атомы образовать связь, если нет - выходим
            if (btype == 0)
            {
                printf("0 btype! tot %d bonds\n", md->nBond);
                continue;
            }

            
            // try to lock the first atom
            if (atomicCAS(&(md->canBind[id1]), 1, 0) == 0)  // the atom is already used
                continue;

            // try to lock the second atom
            if (atomicCAS(&(md->canBind[id2]), 1, 0) == 0)  // the atom is already used
            {
                // unlock the first one back
                atomicExch(&(md->canBind[id1]), 1);
                continue;
            }

            // create bond iat-nei
            bind = atomicAdd(&(md->nBond), 1);
            //printf("(%d) th[%d] add bond %d between %d and %d\n", iStep, threadIdx.x, bind, id1, id2);
#ifdef DEBUG_MODE
            if (bind >= md->mxBond)
            {
                printf("UBEH[003]: Exceed maximal number of bonds, %d\n", md->mxBond);
            }
#endif
            // сохраняем, что мы изменяли эти тип атома, если это не было сохранено ранее
            //if (md->use_angl == 2)      // variable angles
            //{
                atomicCAS(&(md->oldTypes[id1]), -1, spec1);
                atomicCAS(&(md->oldTypes[id2]), -1, spec2);
            //}

            if (btype < 0)
            {
                // invert atoms order
                md->bonds[bind].x = id2;
                md->bonds[bind].y = id1;
                md->bonds[bind].z = -btype;
                bnd = &(md->bondTypes[-btype]);
                // change atom types according the formed bond
#ifdef DEBUG_MODE
                if (bnd->spec2 > MX_SPEC)
                    printf("wrong spec2 field (%d) has bnd(%d)\n", bnd->spec2, -btype);
#endif
                md->types[id1] = bnd->spec2;
                md->types[id2] = bnd->spec1;
            }
            else 
            {
                md->bonds[bind].x = id1;
                md->bonds[bind].y = id2;
                md->bonds[bind].z = btype;
                bnd = &(md->bondTypes[btype]);
#ifdef DEBUG_MODE
                if (bnd->spec2 > MX_SPEC)
                    printf("wrong spec2 field (%d) has bnd(%d)\n", bnd->spec2, btype);
#endif
                // change atom types according the formed bond
                md->types[id1] = bnd->spec1;
                md->types[id2] = bnd->spec2;
            }
            //printf("(%d) bond[%d] between %d and %d created, species (%d -> %d and %d -> %d)\n", iStep, bind, id1, id2, spec1, bnd->spec1, spec2, bnd->spec2);
#ifdef DEBUG_MODE
            if ((md->types[id1] < 0) || (md->types[id1] >= MX_SPEC) || (md->types[id2] < 0) || (md->types[id2] >= MX_SPEC))
                printf("ERROR[006] %d step. A bond was created with wrong atom types: %d[%d] and %d[%d]. bndType[%d].spec1,2=(%d, %d) \n", iStep, md->types[id1], id1, md->types[id2], id2, btype, bnd->spec1, bnd->spec2);
#endif
            /*
            int k;
            for (k = 0; k < bind; k++)
                if (md->bonds[k].x == md->bonds[bind].x)
                    if (md->bonds[k].y == md->bonds[bind].y)
                    {
                        printf("New bond %d step %d. The bond between %d(p=%d) and %d(p=%d) has been already creted (%d)!\n", bind, iStep, id1, md->parents[id1], id2, md->parents[id2], k);
                        break;
                    }
            */
            
            atomicAdd((&bnd->count), 1);
            md->bonds[bind].w = iStep;  // keep time of creating for lifetime calculation

            /*
            float x, y, z, r2;
            x = md->xyz[id1].x - md->xyz[id2].x;
            y = md->xyz[id1].y - md->xyz[id2].y;
            z = md->xyz[id1].z - md->xyz[id2].z;
            delta_periodic(x, y, z, md);
            r2 = x * x + y * y + z * z;
            */
            float r2 = md->funcDist2Per(id1, id2, md);

            // some verification
            //if ((r2 > 4.0) || (r2 < 1.0))
              //  printf("abnormal bonding length=%f r2Max=%f\n", sqrt(r2), md->r2Max);

            // попробуем занулить скорости частиц, образующих связь
            //md->vls[id1] = make_float3(0.f, 0.f, 0.f);
            //md->vls[id2] = make_float3(0.f, 0.f, 0.f);

            ////////////  H-bonds addition
            if (bnd->hatom != md->types[id2])  // another atom is not hydrogen (this is hydrogen)
            ////////////  end H-bonds addition
                atomicAdd(&(md->nbonds[id1]), 1);
            ////////////  H-bonds addition
            //else
              //  printf("except\n");
            if (bnd->hatom != md->types[id1])  // another atom is not hydrogen (this is hydrogen)
            ////////////  end H-bonds addition
                atomicAdd(&(md->nbonds[id2]), 1);
            
            // replace parents if none:
            //atomicCAS(&(md->parents[id1]), -1, id2);
            //atomicCAS(&(md->parents[id2]), -1, id1);
            // побольшому счету неважно, оставляем ли мы уже замененные или нет
            // так что можно заменить CAS на EXCH
            ////////////  H-bonds addition
            if (bnd->hatom != md->types[id2])  // another atom is not hydrogen (this is hydrogen)
            ////////////  end H-bonds addition
                atomicExch(&(md->parents[id1]), id2);
            ////////////  H-bonds addition
            if (bnd->hatom != md->types[id1])  // another atom is not hydrogen (this is hydrogen)
            ////////////  end H-bonds addition
                atomicExch(&(md->parents[id2]), id1);
            //! (?) а может быть атомик и не нужен
            //! объединить с предыдущим блоком, тогда можно избавится от двух if
        }

    }    // end loop by atoms
}
// end 'create_bonds' function

__global__ void refresh_atomTypes(int iStep, int atPerBlock, int atPerThread, cudaMD* md)
// recalculate number of atom types (функция сделана на тот случай, если не нужно обновлять углы, а типы частиц все равно меняются)
{
    int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
    int N = min(id0 + atPerThread, md->nAt);

    int iat;
    for (iat = id0; iat < N; iat++)
        if (md->oldTypes[iat] != -1)
        {
            if (md->oldTypes[iat] != md->types[iat])
            {
                atomicAdd(&(md->specs[md->types[iat]].number), 1);
                atomicSub(&(md->specs[md->oldTypes[iat]].number), 1);
            }
            md->oldTypes[iat] = -1;
        }
}
// end function 'refresh_atomTypes'

// bond potential functions:
__device__ float bond_harm(float r2, float r, float& eng, cudaBond* bnd)
// Harmonic bond potential:
// U = 1/2 k (r-r0)^2   (k = p0, r0 = p1)
{
    float x = r - bnd->p1; // r - r0
    eng += 0.5f * bnd->p0 * x * x;
    return -bnd->p0 / r * x;
}

__device__ float bond_harm_eknr(float r2, float r, cudaBond* bnd)
// harmonic bond energy by r2 and exactly known r
{
    float x = r - bnd->p1; // r - r0
    return 0.5f * bnd->p0 * x * x;
}

__device__ float bond_morse(float r2, float r, float& eng, cudaBond* bnd)
// Morse bond potential:
// U = D[1 - exp(-a(r-r0))]^2 - C   (D = p0, a = p1, r0 = p2, C = p3)
{
    float x = r - bnd->p2; // r - r0
    x = exp(-bnd->p1 * x); // exp(-a(r-r0))
    float y = 1 - x;

    eng += bnd->p0 * y * y - bnd->p3;
#ifdef DEBUG_MODE
    if (eng > 1e5)
        printf("too high morse eng, %f\n", eng);
    if (eng < -1e5)
        printf("too high morse eng, %f\n", eng);
#endif
    return -2.0 * bnd->p0 * bnd->p1 * x * y / r;
}

__device__ float bond_morse_eknr(float r2, float r, cudaBond* bnd)
// Morse bond energy by r2 and exactly known r
{
    float x = r - bnd->p2; // r - r0
    x = exp(-bnd->p1 * x); // exp(-a(r-r0))
    float y = 1 - x;

    return bnd->p0 * y * y - bnd->p3;
}

__device__ float bond_pedone(float r2, float r, float& eng, cudaBond* bnd)
// potential from paper of Alphonso Pedone (morse + E/r^12):
// U = D[1 - exp(-a(r-r0))]^2 - C - E/r^12  (D = p0, a = p1, r0 = p2, C = p3, E=p4)
{
    float x = r - bnd->p2; // r - r0
    x = exp(-bnd->p1 * x); // exp(-a(r-r0))
    float y = 1 - x;
    float ir2 = 1.f / r2;   // 1/r^2
    float irn = ir2 * ir2;  // 1/r^4
    irn = irn * irn * irn;  // 1/r^12

    eng += bnd->p0 * y * y - bnd->p3 - bnd->p4 * irn;
    return -2.0 * bnd->p0 * bnd->p1 * x * y / r - 12.0 * bnd->p4 * irn * ir2;   // -dU/dr * (1/r)
}

__device__ float bond_pedone_eknr(float r2, float r, cudaBond* bnd)
// Pedone bond energy by r2 and exactly known r
{
    float x = r - bnd->p2; // r - r0
    x = exp(-bnd->p1 * x); // exp(-a(r-r0))
    float y = 1 - x;
    float ir2 = 1.f / r2;   // 1/r^2
    float irn = ir2 * ir2;  // 1/r^4
    irn = irn * irn * irn;  // /r^12

    return bnd->p0 * y * y - bnd->p3 - bnd->p4 * irn;
}
        
__device__ float bond_buck(float r2, float r, float& eng, cudaBond* bnd)
// Buckingham:
// U = A exp(-r/ro) - C/r^6  (A = p0, ro = p1, C = p2)
{
    float ir2 = 1.0 / r2;
    float irn = ir2 * ir2;   // 1/r^4

    eng += bnd->p0 * exp(-r / bnd->p1) - bnd->p2 * irn * ir2;
    return bnd->p0 * exp(-r / bnd->p1) / r / bnd->p1 - 6.f * bnd->p2 * irn * irn;
}

__device__ float bond_buck_eknr(float r2, float r, cudaBond* bnd)
// Buckingham bond energy by r2 and exactly known r
{
    float ir2 = 1.0 / r2;
    float irn = ir2 * ir2;   // 1/r^4

    return bnd->p0 * exp(-r / bnd->p1) - bnd->p2 * irn * ir2;
}

__device__ float bond_e6812(float r2, float r, float& eng, cudaBond* bnd)
// exp - 6 - 8 -12:
// U = A exp(-r/ro) - C/r^6 - D/r^8 - F/r^12  (A = p0, ro = p1, C = p2, D = p3, F = p4)
{
    float ir2 = 1.0 / r2;
    float irn = ir2 * ir2;   // 1/r^4

    eng += bnd->p0 * exp(-r / bnd->p1) - bnd->p2 * irn * ir2 - bnd->p3 * irn * irn - bnd->p4 * irn * irn * irn;
    return bnd->p0 * exp(-r / bnd->p1) / r / bnd->p1 - 6.f * bnd->p2 * irn * irn - 8.f * bnd->p3 * irn * irn * ir2 - 12.0 * bnd->p4 * irn * irn * irn * ir2;
}

__device__ float bond_e6812_eknr(float r2, float r, cudaBond* bnd)
// "exp-6-8-12" bond energy by r2 and exactly known r
{
    float ir2 = 1.0 / r2;
    float irn = ir2 * ir2;   // 1/r^4

    return bnd->p0 * exp(-r / bnd->p1) - bnd->p2 * irn * ir2 - bnd->p3 * irn * irn - bnd->p4 * irn * irn * irn;
}


__global__ void define_bond_potential(cudaMD* md)
// set address of function for each bond
{
    cudaBond* bnd;
    //printf("def bonds, thradId=%d\n", threadIdx.x);
    bnd = &(md->bondTypes[threadIdx.x + 1]);    // + 1 because 0th bond is reserved as 'no_bond'

    switch (bnd->type)
    {
    case 1:
        bnd->force_eng = &bond_harm;
        //printf("harm(%p): bnd[%d]->func=%p\n", bond_harm, threadIdx.x, bnd->force_eng);
        bnd->eng_knr = &bond_harm_eknr;
        break;
    case 2:
        bnd->force_eng = &bond_morse;
        bnd->eng_knr = &bond_morse_eknr;
        break;
    case 3:
        bnd->force_eng = &bond_pedone;
        bnd->eng_knr = &bond_pedone_eknr;
        break;
    case 4:
        bnd->force_eng = &bond_buck;
        bnd->eng_knr = &bond_buck_eknr;
        break;
    case 5:
        bnd->force_eng = &bond_e6812;
        bnd->eng_knr = &bond_e6812_eknr;
        break;
    }
/*
    printf("bind_matr[0][0]=%d\n", md->bindBonds[0][0]);
    printf("bind_matr[0][1]=%d\n", md->bindBonds[0][1]);
    printf("bind_matr[0][2]=%d\n", md->bindBonds[0][2]);
    printf("bind_matr[0][3]=%d\n", md->bindBonds[0][3]);
    printf("bind_matr[1][0]=%d\n", md->bindBonds[1][0]);
    printf("bind_matr[2][0]=%d\n", md->bindBonds[2][0]);
    printf("bind_matr[3][0]=%d\n", md->bindBonds[3][0]);
*/

}
