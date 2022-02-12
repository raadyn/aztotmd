#include "dataStruct.h"
#include "cuStruct.h"
#include "cuEjump.h"
#include "utils.h"
#include "cuUtils.h"
#include "ejump.h"
#include "cuMDfunc.h"
#include "cuBonds.h"

extern __constant__ const float d_Fcoul_scale;  //! не помню точно

void init_cuda_ejump(Sim *sim, Atoms *atm, cudaMD *hmd)
{
	hmd->use_ejump = sim->ejtype;
	if (sim->ejtype)
	{
        hmd->r2Jump = (float)sim->r2Elec;
        hmd->dEjump = (float)sim->dEjump;
        hmd->mxFreeEls = sim->nFreeEl;
		cudaMalloc((void**)&(hmd->electrons), sim->nFreeEl * int_size);
		cudaMalloc((void**)&(hmd->accIds), atm->mxAt * int_size);
		cudaMalloc((void**)&(hmd->r2Jumps), atm->mxAt * int_size);
        hmd->nJump = 0;
        hmd->posBxJump = make_int3(0, 0, 0);
        hmd->negBxJump = make_int3(0, 0, 0);
    }
}

void free_cuda_ejump(cudaMD* hmd)
{
	if (hmd->use_ejump)
	{
		cudaFree(hmd->electrons);
		cudaFree(hmd->accIds);
		cudaFree(hmd->r2Jumps);
	}
}

__device__ void try_to_jump(float r2, int id1, int id2, int spec1, int spec2, cudaMD* md)
// verify that the given pair of atoms can be exchanged by electrons and keep this info
{
    if (r2 > md->r2Jump)
        return;

    int i;
    int r2Int = (int)(r2 * 1000);      // unfortunatelly, atomicMin works only for integers

    if ((md->specs[spec1].donacc >> bfDonor) & 1) // atom 1 is an e-donor
        if ((md->specs[spec2].donacc >> bfAcceptor) & 1) // atom 2 is an e-acceptor
        {
            if (atomicMin(&(md->r2Jumps[id1]), r2) > r2)    // replace was sucessfull, we've found new nearest acceptor for the donor
            {
                if (atomicExch(&(md->accIds[id1]), id2) == -1)      // this is the first time donor cultivation
                {
                    // save the donor in electron list
                    i = atomicAdd(&(md->nFreeEls), 1);
                    md->electrons[i] = id1;
                }
            }
        }

    // similar to the second atom
    if ((md->specs[spec2].donacc >> bfDonor) & 1) // atom 1 is an e-donor
        if ((md->specs[spec1].donacc >> bfAcceptor) & 1) // atom 2 is an e-acceptor
        {
            if (atomicMin(&(md->r2Jumps[id2]), r2) > r2)    // replace was sucessfull, we've found new nearest acceptor for the donor
            {
                if (atomicExch(&(md->accIds[id2]), id1) == -1)      // this is the first time donor cultivation
                {
                    // save the donor in electron list
                    i = atomicAdd(&(md->nFreeEls), 1);
                    md->electrons[i] = id2;
                }
            }
        }
}

__global__ void cuda_ejump(int bndPerThread, cudaMD* md)
{
    // one electron - one block, escape if out of range
    if (blockIdx.x >= md->nFreeEls)
        return;

    __shared__ int donor;       // donor atom id
    __shared__ int acceptor;    
    __shared__ int nbonds;      // the number of bonds of the acceptor and donor
    __shared__ float sh_dU;     // shared energy difference
    __shared__ int cultBnd;     // number of cultivated bonds

    if (threadIdx.x == 0)
    {
        donor = md->electrons[blockIdx.x];
        acceptor = md->accIds[donor];
        nbonds = md->nbonds[donor] + md->nbonds[acceptor];
        //! maybe exclude from shared variables the previous?
        sh_dU = 0.f;
        cultBnd = 0;
    }
    __syncthreads();


    int don_type = md->types[donor];
    int acc_type = md->types[acceptor];
    // verify that the donor is still a donor, and the acceptor is still an acceptor:
    if ((~md->specs[don_type].donacc >> bfDonor) & 1)
        return;
    if ((~md->specs[acc_type].donacc >> bfAcceptor) & 1)
        return;

    // don_type -> oxForm; acc_type -> redForm:
    //! ox and red Forms keep with +1 because 0 was reserved, maybe we need to change it
    int oxForm = md->specs[don_type].oxForm - 1;
    int redForm = md->specs[acc_type].redForm - 1;

    //! simplification: calculate dU only for bonded surrounding
    // every thread recieves a range of bonds for verification
    int id0 = threadIdx.x * bndPerThread;
    int N = min(id0 + bndPerThread, md->nBond);
    int i = id0;

    int nei, nei_type;       // neighbour atom index and its type
    cudaVdW* vdw;
    cudaBond* bnd;
    float r2, r, dU = 0.f;

    
    while ((i < N) && (cultBnd < nbonds))   // end cycle then all bonds became cultivated (or end of interval)
    {
        if (md->bonds[i].z == 0)    // deleted(broken) bond
        {
            i++;
            continue;
        }

        //! тут бы надо учесть возможности образования водородных связей, тогда nbonds может не соответствовать фактическому кол-ву
        //! но для простоты мы положим, что атомы доноров/акцепторов не могут учавствовать в водородных связях
        //! хотя возможно, если мы будем записывать водородные связи в nbonds обоих атомов, как обычные связи это не нарушит общности? ( я видимо сделал, чтобы валентные углы не портились)
        //! наверное, это можно пофиксить, если сделать угол зависимый от лиганда

        nei = -1;
        if (md->bonds[i].x == donor)
            nei = md->bonds[i].y;
        else 
            if (md->bonds[i].y == donor)
                nei = md->bonds[i].x;

        if (nei > -1)
        {
            nei_type = md->types[nei];
            r2 = md->funcDist2Per(donor, nei, md);
            r = sqrt(r2);    //  determine it in VDW func

            //van der Waals energy difference:
            vdw = md->vdws[don_type][nei_type];
            if (vdw != NULL)
                dU -= vdw->eng_r(r2, r, vdw);   //! нужно ввести функцию, когда r точно известно, без одной проверки r
            vdw = md->vdws[oxForm][nei_type];
            if (vdw != NULL)
                dU += vdw->eng_r(r2, r, vdw);

            //Coulombic energy difference (in simplest assumption)
            dU += d_Fcoul_scale * md->specs[nei].charge / r * (md->specs[oxForm].charge - md->specs[don_type].charge);

            //Bonded energy difference
            bnd = &(md->bondTypes[md->bonds[i].z]);
            dU -= bnd->eng_knr(r2, r, bnd);
            bnd = evol_bondtype_addr(bnd, oxForm, nei_type, md);
            if (bnd != NULL)
                dU += bnd->eng_knr(r2, r, bnd);

            atomicAdd(&cultBnd, 1);
        }

        //similar for the acceptor atom
        nei = -1;
        if (md->bonds[i].x == acceptor)
            nei = md->bonds[i].y;
        else
            if (md->bonds[i].y == acceptor)
                nei = md->bonds[i].x;

        if (nei > -1)
        {
            nei_type = md->types[nei];
            r2 = md->funcDist2Per(acceptor, nei, md);
            r = sqrt(r2);    //  determine it in VDW func

            //van der Waals energy difference:
            vdw = md->vdws[acc_type][nei_type];
            if (vdw != NULL)
                dU -= vdw->eng_r(r2, r, vdw);   //! нужно ввести функцию, когда r точно известно, без одной проверки r
            vdw = md->vdws[redForm][nei_type];
            if (vdw != NULL)
                dU += vdw->eng_r(r2, r, vdw);

            //Coulombic energy difference (in simplest assumption)
            dU += d_Fcoul_scale * md->specs[nei].charge / r * (md->specs[redForm].charge - md->specs[acc_type].charge);

            //Bonded energy difference
            bnd = &(md->bondTypes[md->bonds[i].z]);
            dU -= bnd->eng_knr(r2, r, bnd);
            bnd = evol_bondtype_addr(bnd, redForm, nei_type, md);
            if (bnd != NULL)
                dU += bnd->eng_knr(r2, r, bnd);

            atomicAdd(&cultBnd, 1);
        }


        i++;
        //if ((i-id0) > 30)
          //  printf("too long loop: i=%d id0=%d N=%d cultBnd=%d nBnd=%d\n", i, id0, N, cultBnd, nbonds);
    }
    atomicAdd(&sh_dU, dU);
    __syncthreads();
    

    if (threadIdx.x == 0)
    {
        //! add own energy!

        // external electric field energy difference (temp: only x-dimension):
        int px, py, pz;
        pass_periodic(donor, acceptor, md, px, py, pz);
        //sh_dU = 0.f;
        sh_dU += md->elecField.x * (md->xyz[donor].x * (md->specs[oxForm].charge - md->specs[don_type].charge) + (md->xyz[acceptor].x + px * md->leng.x) * (md->specs[redForm].charge - md->specs[acc_type].charge));

        // determine jump or not according to selected ejump type
        int do_jmp = 0;
        if (md->use_ejump == tpJumpEq)  // equality (Frank-Condon principle)
        {
            do_jmp = (((sh_dU > -md->dEjump))&&(sh_dU < md->dEjump));
        }
        else if (md->use_ejump == tpJumpMin)    // minimization
        {
            do_jmp = (sh_dU < 0.f);
        }
        else if (md->use_ejump == tpJumpMetr)   // Metropolis scheme (comparision with kB*T)
        {
            if (sh_dU < 0.f)
                do_jmp = 1;
            else
            {
                //! add Metropolis sheme here
            }
        }

        // perform an electron jump, try to change atom types
        if (do_jmp)
        {
            // verify, that acceptor atom is still of original type
            if (atomicCAS(&(md->types[acceptor]), acc_type, redForm) == acc_type)
            {
                if (atomicCAS(&(md->types[donor]), don_type, oxForm) == don_type)
                { } // sucesfully changed both types
                else
                {
                    // the worst variant, we've changed acceptor, but can't change donor
                    //?! how to solve this? but fortunatelly, this can be in very rare cases (several free electrons on one donor, or one type can be both a donor and acceptor
                    printf("Critical error in the electron jump routine: we've changed acceptor, but can't change donor\n");
                    do_jmp = 0;
                }
            }
            else
                do_jmp = 0;
        }

        // types have changed sucessfully, perform the rest actions:
        if (do_jmp)
        {
            // now electron sits on atom which was the acceptor 
            //! не знаю пока зачем, в отличие от серийной версии тут все равно не сделаешь несколько электронных шагов за один обычный.... но на будущее
            md->electrons[blockIdx.x] = acceptor;

            // keep old types, if they have not changed still
            atomicCAS(&(md->oldTypes[donor]), -1, don_type);
            atomicCAS(&(md->oldTypes[acceptor]), -1, acc_type);

            //! number of particles are refreshed in refresh_atomTypes or refresh_angles routines

            //! refresh bonds or wait the next apply bonds? (for the first case its better to save ids of corresponding bonds from the cycle above
            //change_bonds(iat, jat, ti2, tj2, atm, field);

            // STATISTICS
            atomicAdd(&(md->nJump), 1);              //  total number of jumps       

            // between defined donor-acceptor pair
            //sim->jumps[ti1][tj1]++;

            if (px > 0) // px is flag of box edge crossing (-1, 0, 1)
            {
                atomicAdd(&(md->posBxJump.x), 1);
                //sim->pTotJump++;
            }
            else
                if (px < 0)
                {
                    atomicAdd(&(md->negBxJump.x), 1);
                    //sim->nTotJump++;
                }
                else // px == 0
                {
                    //if (atm->xs[jat] > atm->xs[iat])
                      //  sim->pTotJump++;
                    //else if (atm->xs[jat] < atm->xs[iat])
                      //  sim->nTotJump++;
                    //! temp!
                    if (md->xyz[acceptor].x > md->xyz[donor].x)
                        atomicAdd(&(md->posBxJump.x), 1);
                    else if (md->xyz[acceptor].x < md->xyz[donor].x)
                        atomicAdd(&(md->negBxJump.x), 1);
                }
            //! add other dimensions?

            /*
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
            */
        }
    }
}


