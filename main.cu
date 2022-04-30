
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "device_functions.h"
//#include "device_atomic_functions.h"
#include <time.h>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "defines.h"
#include "cuStruct.h"
#include "dataStruct.h" 
#include "sys_init.h"
#include "vdw.h"
#include "cuBonds.h"
#include "cuAngles.h"
#include "cuElec.h"
#include "out_md.h"
#include "const.h"  // kB
#include "cuMDfunc.h"
#include "cuInit.h"
#include "cuVdW.h"
#include "angles.h"
#include "bonds.h"
#include "box.h"        // tpBoxRect, tpBoxHalf
#include "cuStat.h"
#include "cuPairs.h"
#include "cuSort.h"     // count_cell
#include "cuTemp.h"
#include "cuUtils.h"
#include "temperature.h"
#include "utils.h"
#include "elec.h"
#include "cuEjump.h"

//! PHYSICAL CONSTANTS
//__constant__ const float PI = 3.1415926;
extern __constant__ const float d_Fcoul_scale = 14.3996f;  
extern __constant__ const float d_sqrtpi = 1.772453f;  //   sqrt(PI)
extern __constant__ const float d_2pi = 6.283185307f;  //   2 * (PI)
__constant__ const float d_rkB = 11604.524844f;     //      invert Bolzman constant in program units

// Textues:
//texture<float> rMassHdt;
texture<float, 2, cudaReadModeElementType> qProd;

int out_thermalchar(Atoms* atm, Field* field, char* fname, cudaMD* md)
// write thermal characteristics (thermal energy and radii) in file [only for radiative thermostat]
{
    int i, j, t, mx = 0;
    int* curInds;    // currentIndex for saving
    FILE* f;

    float* cu_engs = (float*)malloc(atm->nAt * float_size);
    float* cu_rads = (float*)malloc(atm->nAt * float_size);

    cudaMemcpy(cu_engs, md->engs, atm->nAt * float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(cu_rads, md->radii, atm->nAt * float_size, cudaMemcpyDeviceToHost);

    // array of radii and thermal energies
    double** rads = (double**)malloc(field->nSpec * pointer_size);
    double** engs = (double**)malloc(field->nSpec * pointer_size);
    curInds = (int*)malloc(field->nSpec * int_size);
    for (i = 0; i < field->nSpec; i++)
    {
        rads[i] = (double*)malloc(field->species[i].number * double_size);
        engs[i] = (double*)malloc(field->species[i].number * double_size);
        curInds[i] = 0;
        mx = max(mx, field->species[i].number);
    }

    for (i = 0; i < atm->nAt; i++)
    {
        t = atm->types[i];
        rads[t][curInds[t]] = cu_rads[i];
        engs[t][curInds[t]] = cu_engs[i];
        curInds[t]++;
    }

    free(cu_engs);
    free(cu_rads);

    f = fopen(fname, "w");
    if (f == NULL)
        return 0; // error

    fprintf(f, "No");
    for (i = 0; i < field->nSpec; i++)
        fprintf(f, "\t%s_eng\t%s_rad", field->species[i].name, field->species[i].name);
    fprintf(f, "\n");

    for (i = 0; i < mx; i++)
    {
        fprintf(f, "%d", i + 1);
        for (j = 0; j < field->nSpec; j++)
            if (i < field->species[j].number)
                fprintf(f, "\t%f\t%f", engs[j][i], rads[j][i]);
            else
                fprintf(f, "\t\t");
        fprintf(f, "\n");
    }
    fclose(f);

    for (i = 0; i < field->nSpec; i++)
        if (field->species[i].number)
        {
            free(engs[i]);
            free(rads[i]);
        }
    free(engs);
    free(rads);
    free(curInds);
    return 1;
}

__global__ void calc_quantities(int iStep, cudaMD* md, int box_type)
// calculate derived parameters as total energy, pressure, mean bonds lifetime and etc
{
    int i;
    float k, f;

    md->engCoulTot = md->engCoul1 + md->engCoul2 + md->engCoul3;
    md->engPot = md->engCoulTot + md->engVdW + md->engBond + md->engAngl + md->engElecField;
    md->engPotKin = md->engPot + md->engKin;
    md->engTot = md->engPotKin + md->engTemp;

    // calculate 'kinetic' temperature: T = 2/kB * K/3N, but 3N is replaced to f - number of freedom degree, and f will be calculated as 3N - Nbond:
    f = 3.f * md->nAt - md->nBond;
    md->kinTemp = 2.f * d_rkB * md->engKin / f;

    // pressure calculation
    if (iStep >= md->nMom - 1)
    {
        i = md->iMom;
        k = 2.f * 1.58e6f / (md->tSt * (md->nMom - 1));             // I'm not sure in factor 2. Its derived for the case the particle change your velocity to opposite after collision with wall
        md->posPres.x = k * (md->posMom.x - md->posMomBuf[i].x) * md->revEdgeArea.x;
        md->posPres.y = k * (md->posMom.y - md->posMomBuf[i].y) * md->revEdgeArea.y;
        md->posPres.z = k * (md->posMom.z - md->posMomBuf[i].z) * md->revEdgeArea.z;
        md->negPres.x = k * (md->negMom.x - md->negMomBuf[i].x) * md->revEdgeArea.x;
        md->negPres.y = k * (md->negMom.y - md->negMomBuf[i].y) * md->revEdgeArea.y;
        md->negPres.z = k * (md->negMom.z - md->negMomBuf[i].z) * md->revEdgeArea.z;
        i--;
        if (i < 0)
            i = md->nMom - 1;
        md->posMomBuf[i] = md->posMom;
        md->negMomBuf[i] = md->negMom;
        md->iMom++;
        if (md->iMom >= md->nMom)
            md->iMom = 0;
        
        //md->jMom++;
        //if (md->jMom >= md->nMom)
        //    md->jMom = 0;

        // calculate pressure in all dimensions
        md->pressDim.x = 0.5f * (md->posPres.x + md->negPres.x);
        md->pressDim.y = 0.5f * (md->posPres.y + md->negPres.y);
        md->pressDim.z = 0.5f * (md->posPres.z + md->negPres.z);

        if (box_type == tpBoxHalf)
            md->pressure = 0.5f * (md->pressDim.x + md->pressDim.y);
        else
            md->pressure = (md->pressDim.x + md->pressDim.y + md->pressDim.z) / 3.f;
    }
    else
    {
        md->posMomBuf[iStep] = md->posMom;
        md->negMomBuf[iStep] = md->negMom;
        //md->jMom++;
    }

    
    if (md->use_bnd)
    {
        for (i = 1; i < md->nBndTypes; i++)
        {
            if (md->bondTypes[i].ltCount)
                md->bondTypes[i].ltMean = (float)md->bondTypes[i].ltSumm * md->tSt / md->bondTypes[i].ltCount;
            else // эту ветку надо убрать, задав дефолтное значение
                md->bondTypes[i].ltMean = 0.f;
            
            if (md->bondTypes[i].rCount)
                md->bondTypes[i].rMean = (float)md->bondTypes[i].rSumm / md->bondTypes[i].rCount;
            else // эту ветку надо убрать, задав дефолтное значение
                md->bondTypes[i].rMean = 0.f;

            //printf("ltMena[%d]=%f rMean = %f\n", i, md->bondTypes[i].ltMean, md->bondTypes[i].rMean);
        }
    }
}

__global__ void define_global_func(int boxtype, cudaMD *md)
{
    switch (md->use_coul)
    {
      case 0:
          md->funcCoul = &no_coul;
          break;
      case 1:   // direct Coulomb
          md->funcCoul = &direct_coul;
          break;
      case 2:   // Ewald
          md->funcCoul = &real_ewald;
          //md->funcCoul = &real_ewald_tex;
          break;
      case 3:   // Fennel
          md->funcCoul = &fennel;
          break;
      default:
          printf("ERROR[b008] Something wrong: wrong value of use_coul variable (%d)\n", md->use_coul);
    }
    //printf("arr[%d]=%f fetch=%f\n", 250, md->coulEng[250], fetch1D);

    switch (boxtype)
    {
      case 1:   // orthorombic boundary conditions
        md->funcDeltaPer = &delta_periodic_orth;
        md->funcDist2Per = &dist2_periodic_orth;

        md->funcPutPer = &put_periodic;
        md->funcAtToCell = &count_cell;
        // old variant of cell list, without sorting:
        //md->funcAtToCell = &keep_in_cell;

        break;
      case 2:   //
          md->funcDeltaPer = &delta_periodic_half;
          md->funcDist2Per = &dist2_periodic_half;

          md->funcPutPer = &put_halfperiodic;
          md->funcAtToCell = &count_cell_halfper;
          break;
      default:
        printf("ERROR[b019] Unknown box type (%d)\n", boxtype);
    }
}

int main()
{
    char c;         // for anykey pressing
    int start_time = time(NULL);
    srand(time(NULL));

    // the same part as for serial version: allocate memory and load data
    Elec* elec = (Elec*)malloc(sizeof(Elec));
    TStat* tstat = (TStat*)malloc(sizeof(TStat));
    Box* box = (Box*)malloc(sizeof(Box));
    Field* field = (Field*)malloc(sizeof(Field));
    Atoms* atoms = (Atoms*)malloc(sizeof(Atoms));
    Sim* sim = (Sim*)malloc(sizeof(Sim));
    int res = init_md(atoms, field, sim, elec, tstat, box);
    if (!res)
    {
        scanf("%c", &c);
        exit(1);
    }

    // allocate memory for CUDA version
    hostManagMD* man = (hostManagMD*)malloc(sizeof(hostManagMD));
    cudaMD* hostMD = (cudaMD*)malloc(sizeof(cudaMD));                               // a host exemplar of cudaMD
    cudaMD* devMD = init_cudaMD(atoms, field, sim, tstat, box, elec, man, hostMD);     // a device exemplar

    cuda_info();    // print some information about videocard

    define_global_func << <1, 1 >> > (box->type, devMD);
    define_vdw_func << <1, field->nVdW >> > (devMD);
    define_bond_potential << <1, field->nBdata >> > (devMD);
    define_ang_potential << <1, field->nAdata >> > (devMD);
    prepare_stat_addr << < 1, 1>> > (devMD);

    int iStep = 0;
    start_stat(man, field, sim, tstat, elec);
    if (sim->frTraj)
        start_traj(atoms, man, field, sim);
    if (sim->nBindTrajAtoms)
        start_bindtraj(man, field, sim);

    while (iStep < sim->nSt)
    {
        reset_quantities << <1, 1 >> > (devMD);
#ifndef USE_ALLPAIR
        clear_clist <<< man->nMultProc, man->nSingProc >>> (devMD);
#endif
#ifdef DEBUG_MODE
        //clear_ncult << < 1, 1 >> > (cumd);
#endif
        cudaThreadSynchronize();
        
        apply_pre_tstat(iStep, tstat, sim, devMD, man);
        verlet_1stage <<<man->nAtBlock, man->nAtThread/*man->nMultProc, man->nSingProc*/ >>> (iStep, man->atPerBlock, man->atPerThread, devMD);
        cudaThreadSynchronize();

        //verify_clist << < man->nMultProc, man->nSingProc >> > (devMD);
        //cudaThreadSynchronize();
        if (sim->use_bnd == 1)  // constant bonds
        {
            apply_const_bonds << <man->nMultProc, man->nSingProc >> > (iStep, man->bndPerBlock, man->bndPerThread, devMD);
            cudaThreadSynchronize();
        }
        else if (sim->use_bnd == 2) // variable bonds
        {
            apply_bonds << <man->nMultProc, man->nSingProc >> > (iStep, man->bndPerBlock, man->bndPerThread, devMD);
            cudaThreadSynchronize();
            clear_bonds << <1, 1 >> > (devMD);
            cudaThreadSynchronize();
        }

        iter_fastCellList(iStep, field, box, devMD, man);
        //verify_forces << <man->nAtBlock, man->nAtThread >> > (man->atPerBlock, man->atPerThread, iStep, devMD, 2);
        if (elec->type == tpElecEwald)
        {
            recip_ewald << <man->nMultProc, man->nSingProc, man->memRecEwald >> > (man->atPerBlock, man->atPerThread, devMD);
            cudaThreadSynchronize();
            ewald_force << <man->nMultProc, man->nSingProc >> > (man->atPerBlock, man->atPerThread, devMD);
            cudaThreadSynchronize();
        }

        if (sim->use_bnd == 2)   // variable bonds
        {
            create_bonds << <man->nAtBlock, man->nAtThread/*man->nMultProc, man->nSingProc*/ >> > (iStep, man->atPerBlock, man->atPerThread, devMD);
            cudaThreadSynchronize();
        }

        if (sim->ejtype)    // electron jumps
        {
            //! temporary: each timestep
            //! now for ejump is necessary bonds
            //! сложности будут для переменного числа свободных электронов
            cuda_ejump<<<sim->nFreeEl, 32>>>(man->bndPerThreadEjump, devMD);
            cudaThreadSynchronize();
        }

        if (sim->use_angl)  // there are valent angles
        {
            if (sim->use_angl == 2)     // variable angles
            {
                refresh_angles << <man->nAtBlock, man->nAtThread/*man->nMultProc, man->nSingProc*/ >> > (iStep, man->atPerBlock, man->atPerThread, devMD);
                cudaThreadSynchronize();
                clear_angles << <1, 1 >> > (devMD);
                cudaThreadSynchronize();
            }
            apply_angles << <man->nAtBlock, man->nAtThread/*man->nMultProc, man->nSingProc*/ >> > (iStep, man->angPerBlock, man->angPerThread, devMD);
            cudaThreadSynchronize();
        }
        // поскольку типы атомов обновляются в процедуре refresh_angles, если у нас нет углов, то нужно обновление типов атомов вызвать отдельно
        if ((sim->use_angl < 2) && ((sim->use_bnd == 2) || (sim->ejtype)))
            refresh_atomTypes << <man->nAtBlock, man->nAtThread >> > (iStep, man->atPerBlock, man->atPerThread, devMD);

        // 2 stage of velocity verlet integrator or reset velocities
        zero_engKin << <1, 1 >> > (devMD);
        cudaThreadSynchronize();
        if (sim->reset_vels)
            if (iStep % sim->reset_vels == 0)
                zero_vel<< < man->nAtBlock, man->nAtThread/*man->nMultProc, man->nSingProc*/ >> > (man->atPerBlock, man->atPerThread, iStep, devMD);
            else
                verlet_2stage << < man->nAtBlock, man->nAtThread/*man->nMultProc, man->nSingProc*/ >> > (man->atPerBlock, man->atPerThread, iStep, devMD);
        else
            verlet_2stage <<< man->nAtBlock, man->nAtThread/*man->nMultProc, man->nSingProc*/ >>> (man->atPerBlock, man->atPerThread, iStep, devMD);

        cudaThreadSynchronize();
        apply_tstat(iStep, tstat, sim, devMD, man);
        calc_quantities<<<1, 1>>>(iStep, devMD, box->type);
        cudaThreadSynchronize();

        // statistics block:
        if (iStep % sim->stat == 0)
            print_stat << <1, 1 >> > (iStep, devMD);
        stat_iter(iStep, man, &(man->stat), devMD, hostMD, sim->tSt);
        if (sim->use_bnd)
            stat_iter(iStep, man, &(man->sbnd), devMD, hostMD, sim->tSt);
        stat_iter(iStep, man, &(man->smsd), devMD, hostMD, sim->tSt);
        if (sim->ejtype)
            stat_iter(iStep, man, &(man->sjmp), devMD, hostMD, sim->tSt);
        if (sim->nuclei_rdf)
            nrdf_iter(iStep, field, sim, man, hostMD, devMD);
        else
            rdf_iter(iStep, field, sim, man, hostMD, devMD);
        if (sim->frTraj)
            traj_iter(iStep, man, devMD, hostMD, sim, atoms);
        if (sim->nBindTrajAtoms)
            bindtraj_iter(iStep, man, devMD, hostMD, sim, box);
#ifdef DEBUG_MODE
        //verify_ncult << <1, 1 >> > (3, 1, 334, cumd);
        //cudaThreadSynchronize();
#endif
        iStep++;
        if (keyPress(0x1B))
        {
            printf("halt program by Esc press!\n");
            break;
        }
    }
    print_stat << <1, 1 >> > (iStep, devMD);    // final output to screen
    // final statistics output:
    end_stat(man, field, sim, hostMD, sim->tSt);
    if (sim->nuclei_rdf)
        copy_nrdf(field, sim, man, hostMD, "rdf.dat", "rdf_n.dat");  
    else
        copy_rdf(field, sim, man, hostMD, "rdf.dat");
    if (sim->frTraj)
        end_traj(man, hostMD, sim, atoms);
    if (sim->nBindTrajAtoms)
        end_bindtraj(man, hostMD, sim, box);

    if (sim->use_bnd == 2) // for non-constant bonds
    {
        fix_bonds << < man->nMultProc, man->nSingProc >> > (man->bndPerBlock, man->bndPerThread, devMD);
        cudaThreadSynchronize();
    }

    md_to_host(atoms, field, hostMD, devMD, man);
    if (tstat->type == tpTermRadi)
    {
        out_thermalchar(atoms, field, "tchars.dat", hostMD);
        // output radiated photon energies:
        int nbin = hostMD->numbPhEngBin;
        double bin = (double)hostMD->phEngBin;
        int* phEngs = (int*)malloc(nbin * int_size);
        cudaMemcpy(phEngs, hostMD->phEngs, nbin * int_size, cudaMemcpyDeviceToHost);
        FILE* f = fopen("radiated_engs.dat", "w");
        fprintf(f, "eng\tnumber\neng, eV\tnumber\n");
        int i;
        for (i = 0; i < nbin; i++)
            fprintf(f, "%f\t%d\n", i * bin + 0.5 * bin, phEngs[i]);
        fclose(f);
        free(phEngs);
    }

    free_device_md(devMD, man, sim, field, tstat, hostMD);
    free(man);

    out_atoms(atoms, atoms->nAt, field->species, box, "revcon.xyz");
    if (field->nBdata)
    {
        save_bondlist("revbonds.txt", field);
        bond_out(atoms, field, box, "lengths.dat");
    }
    if (field->nAdata)
        save_anglelist("revangles.txt", field);
    //out_bonds()
    out_velocities(atoms, field, "velocities.dat");
    if (sim->outCN)
        out_cn(atoms, field, box, sim, "CN.dat");
    out_ncn(atoms, field, box, sim, "nCN.dat");
    if (res)
        free_md(atoms, field, tstat);

    free(box);
    free(atoms);
    free(elec);
    free(tstat);
    free(field);
    free(sim);

    int final_time = time(NULL);
    int res_time = final_time - start_time;
    printf("Finish. Elapsed time: %d s\n", res_time);
    scanf("%c", &c);
}
