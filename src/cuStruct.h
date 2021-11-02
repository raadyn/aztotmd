// define structures for CUDA version of azTotMD
#ifndef CUSTRUCT_H
#define CUSTRUCT_H

#include <stdio.h>
#include "defines.h"

struct cudaMD;

struct cudaSpec
{
    //VARIABLES
    int number; // êîëè÷åñòâî ÷àñòèö äàííîãî ñîðòà

    double displ; // displacement (for MSD calculation)
    double vaf;   // velocity autocorrelation function

    //CONSTANTS
    int nuclei;     // id of nuclei
    double mass;
    double charge;
    double energy; // own energy for dE calculation during jump

    int charged; // (f) 0 - neitral
    int donacc; // donor/acceptor  binary flags:  01 - donor 10 - acceptor 00 - no donor, no acceptor, 11 - both
    //Spec *oxForm;    // link to Spec after e-donoring
    //Spec *redForm;   //  ....              e-accepting
    int oxForm;
    int redForm;  // index of spec - redForm of this Spec
    int varNumber;    // (f)  number of particles is variable
    int frozen;     // (f)  "frozen" atom type. Such atoms do not move during simulation

    int nFreeEl;  // number of electron available for donoring

    //int canBond;  // flag: can create bond with some spec or not
    //int* bondKeys; // array of keys+1 for bonds that can be assigned to species with corresponding index
    //int angleType;    // (f) possibility to form angle (=0 - can not form angle, >0 - angle with id=angleType


    //variables for CN (coord number) output
    int idCentral;   // flag for counting of CN (0 - do not calculate, n - index )
    int idCounter;   // flag for counterion (0 - not counting, n - index of array place)

    // radii
    double radA, radB, mxEng;

};

// struct for pair potential
struct cudaVdW
{
    // all fields are constant during the simulation
    int type;
    int use_radii;              // flag to use radii in radiative thermostat
    float p0, p1, p2, p3, p4;
    float r2cut;    //square of cutoff
    float (*eng)(float r2, cudaVdW* vdw); // function to calculate energy
    float (*feng)(float r2, cudaVdW* vdw, float& eng); // function to calculate force (return) & energy (save in eng)
    float (*feng_r)(float r2, float &r, cudaVdW* vdw, float& eng); // function to calculate force (return) & energy (save in eng) if r may be known
    float (*eng_r)(float r2, float &r, cudaVdW* vdw); // return energy by r and r^2

    float (*radi_func)(float r2, float rad1, float rad2, cudaVdW* vdw, float& eng);     // radiative potential, which use radii
};

struct statStruct
{
    int ndata;      // the number of variables in one sampling
    int dstep;      // period of statistic sampling (number of timesteps between sampling)
    double dtime;   // time between sampling (= dstep * timestep)
    int nstep;      // the maximal number of sampling in device buffer
    int step0;      // number of the first timestep in the current statistic buffer
    int count;      // iterator (the current number of sampling)
    int size;       // size of one sampling
    int position;   // position (in bytes) of the first sampling in device buffer
    int typeind;    // the first type index in host array of types
    FILE* out_file;
};

// ñòðóêòóðà äëÿ ñîõðàíåíèÿ òåõ ïàðàìåòðîâ CUDA, êîòîðûå íåîáõîäèìî õðàíèòü íà êîìïå: 
//  ÷èñëî àòîìîâ íà áëîê/ïîòîê è ò.ä.
struct hostManagMD
{
    int atPerThread, atPerBlock;        // number of atoms per thread and per block
    int nAtBlock, nAtThread;            // number of atoms blocks and thread
    int bndPerThread, bndPerBlock;      /// number of bonds per block and thread
    int angPerThread, angPerBlock;
    int cellPerThread, cellPerBlock, nCellThread, nCellBlock;
    int atStep;     // step for all functions applied to all atoms:     int step = ceil((double)md->nAt / (double)blockDim.x / (double)gridDim.x);


    int nPair1Block, nPair2Block;
    int nMultProc, nSingProc;
    int totCores;
    int memPerPairBlock;    // shared memory required for cell_list2a and similar functions
    int memPerPairsBlock;   // for cell_list3a
    int pairPerBlock;       // for cell_list3a: number of pairs cultivalted by one block

    // for FastList (cell_list4a and 4b functions)
    //int nBlock4a, nThread4a, mem4a, nBlock4b, nThread4b, mem4b, cellPerBlock4a;

    // for pairs bypass implementation
    int pairMemA, pairMemB, pairBlockA, pairBlockB, pairThreadA, pairThreadB;
    int cellPerBlockA, cellPerBlockB;

    //for pairs:
    int div_type, list_type, bypass_type;

    // for EWALD
    int memRecEwald;    // memory for shared (float2 sh_sums[]) in 'recip_ewald' function

    // for radiative thermostat
    int tstat_sign;

    // statistics output
    char* stat_buffer;
    int* stat_types;            // types of statistic data
    int tot_ndata;      // summ of ndata for all statistics
    statStruct stat, sbnd, smsd, sjmp;    // general statistics, bond statistic, msd statistic, e-jump statistics

    // rdf output
    float* rdf_buffer, *nrdf_buffer;
    int rdf_size, nrdf_size;   // total size of rdf buffers (simple and by nuclei) (on both device and host sides)
    int rdf_count;  // counter of rdf sampling

    // trajectroies output
    int nstep_traj;          // number of steps keeped in device (read in cuda.txt)
    float* traj_buffer;
    int traj_size, traj_count;  // size of one step of trajectory buffer, number of steps keeped on device
    int traj_step0;
    int traj_dstep;         // delta step for trajectories
    double traj_dtime;         // dstep * timestep
    FILE* traj_file;

    // bind trajectories output
    int nstep_bindtraj;          // number of steps keeped in device (read in cuda.txt)
    char* bindtraj_buffer;
    int bindtraj_size, bindtraj_count;  // size of one step of trajectory buffer, number of steps keeped on device
    int bindtraj_step0;
    int bindtraj_dstep;         // delta step for trajectories
    double bindtraj_dtime;         // dstep * timestep
    FILE* bindtraj_file;
    // number of threads and block for bind_trajectory statistics:
    int nBindTrajBlock, nBindTrajThread, bindTrajPerBlock, bindTrajPerThread;


    // pointers to some fields of cudaMD structure
    int* nAtPtr;        // & cudaMD->nAt
    int* nBndPtr;       // & cudaMD->nBond
    int* nAngPtr;       // & cudaMD->nAng

};

// structure for bond type
struct cudaBond
{
    // constants
    int type;  // type of potential
    int spec1, spec2; // type of atoms that connected by this bond type
    int new_type[2];      // (id) bond type after mutation
    int new_spec1[2], new_spec2[2];
    int mxEx, mnEx;     // flags: maximum or minimum of bond length exists
    int hatom;          // flag/index (= -1 for covalent bonds, = spec1/spec2 for hydrogen bonds, = index of H-atom type) 
    int evol;           // flag/index (= 0 or new type of bond, to which transforms this)

    float p0, p1, p2, p3, p4;    // potential parameters
    float r2min, r2max;          // square of minimal and maximal bond length
    float (*force_eng)(float r2, float r, float &eng, cudaBond *bond); // return energy by r and r^2
    float (*eng_knr)(float r2, float r, cudaBond* bond);        // energy by r2 and exactly known r

    // variables
    int count;         // quantity of such bonds
    float rSumm;        // summ of lentghs (for mean length calculation)
    int rCount;         // number of measured lengths (for mean length calculation)
    int ltSumm, ltCount;    // for calculation of lifetime. Integer because time in timestep count
    float rMean, ltMean;    // mean length and lifetime
};

struct cudaAngle
{
    int type; // potential type (now = 1 harmonic cosinus)
    //int central;  // of central atom species type
    float p0, p1, p2;    // parameters

    // execute functions:
    void (*force_eng)(int4* angle, cudaAngle* type, cudaMD* md, float& eng);
};

struct cudaMD
{
    //VARIABLES
    int nAt;    // the number of atoms
    int *nnumbers;       // nuclei numbers

    //atoms:
    float3* xyz; // coordinates
    float3* vls; // velocities
    float3* frs; // forces
    float* engs;     // atom heat energies (for radiative thermostat only)
    float* radii;    // atom's radii (for radiative thermostat only)
    int* radstep;       // number of timestep when atom radiates photon (for radiative thermostat only)
    int* types;  // types (index of species array)
//#ifdef USE_FASTLIST
    // sorted arrays:
    float3* sort_xyz; // coordinates
    float3* sort_vls; // velocities
    float3* sort_frs;   // forces
    float* sort_engs;    // atom heat energies (for radiative thermostat only)
    float* sort_radii;    // atom's radii (for radiative thermostat only)
    int* sort_radstep;      // number of timestep when atom radiates photon (for radiative thermostat only)
    int* sort_types;  // types (index of species array)
    int* sort_parents;
    float* sort_rMasshdT;    // 1/m * 0.5 timestep for each atom (atom mass does practically not change during simulation, even if atom's type do)
    float* sort_masses;
    int* sort_oldTypes;
    int* sort_ind;        // atom indexes in sorted order
    int* sort_nbonds;
    int* sort_nangles;
    int* sort_trajs;        // indexes of atoms for trajectory output
    int* cellIndexes;         // index of cell in which atom is stored
    int* insideCellIndex;       // index of atom inside the cell
    int* firstAtomInCell;   // index of the first atom in the cell
    int* nAtInCell;         // number of atom in the cell

    // for singleAtom cell list:
    int* nNeighCell;        // number of neighboring cells for a given cell
    int3** neighCells;      // list a neighboring cells for a given cell (.x keeps cell index, .y - type of shift, .z - type of interaction
//#endif

    //energies (fact):
    float engKin, engPot, engTot, engElecField;
    float engCoul1, engCoul2, engCoul3, engCoulTot; // Coulombic energies (for Ewald: 1 - real, 2 - recipropal, 3 - const and total)
    float engBond, engAngl;
    float engVdW;
    float teKin;    // target kinetic energy
    float engTemp;          // thermal energy (for radiative thermostat only)

    //other characteristics
    float3 posMom;  // momentum per box edge in positive directions
    float3 negMom;  // momentum per box edge in negative directions
    // buffers for momentum
    float3* posMomBuf;
    float3* negMomBuf;
    int nMom;         // number of data of momentum kept in buffer
    int iMom/*, jMom*/;           // index of current element of buffer
    float3 posPres; // pressure on box edge in positive directions
    float3 negPres; // pressure on box edge in negative directions
    float pressure; // mean pressure = (posPres + negPres)/6

    //thermostat and temperature data
    float chit, consInt;
    float temp;     // temperature
    float tscale;   // temperature scaling for velocities (auxiliary variable)
    int curEng, curVect;      // variables for radiation thermostat: current index of photon energies array and random vectors array
    int rnd;        // variable for spending threads
    uint4 ui4rnd;   // variable for random number generator

    //box characteristics
    float3 leng, halfLeng, revLeng; // length, half of length and reversible length
    float3 edgeArea;        // areas of box edges perpendicular to corresponding direction (for example edgeArea.x = S(Oyz edge))
    float3 revEdgeArea;     // = 1 / edgeArea
    float volume;

    //cell list
    int** cells;    //the first index - number of the cell, cell[i][0] - a count of atoms in the i-th cell, cell[i][j] (j > 0) index of atom in the cell

    //particle counters
    int3* specAcBoxPos;    // number of particle of given type (by []) passed throw the box in positive directions
    int3* specAcBoxNeg;    // number of particle of given type (by []) passed throw the box in negative directions

    //Ewald data
    float2* qDens;   // summ (q iexp(kr)) for each k-vector, complex value
    float2** qiexp;     // q * iexp (kr) [nAt][nKvec]

    //bonds:
    int nBond;      // number of bonds
    int4* bonds;    // array of bonds .x, .y are atom indexes, .z - type (z = 0 is reserved for deleted bonds), .w - timestep of creating
    int* nbonds;    // count of bond for given atom
    int* neighToBind;   // neightbour of given atom for binding
    int* canBind;       // flags that atom[iat] can be bind
    int* r2Min;         // distances for nearest neighbor (used for binding)
    int* parents;       // indexes of one of the atom bonded with a given

    //angles:
    int nAngle;    // number of angles
    int4* angles;   // array of angles  .x - central atom, .y, .z - ligands, .w - type (=0 for deleted)
    int* nangles;        // number of angles for given atom
    int* oldTypes;      // [iAt] = types[iAt] if types[iAt] was changed during this timestep, else = -1 

    //e-jumps (some of them are constant)
    int use_ejump;              //(f) to use ejump and its type
    float r2Jump;               // cutoff^2 for electron jumping
    float dEjump;              // criteria then dE is assumed to zero
    int* electrons;             //[mxFreeEls] ids of atoms with free electron
    int nFreeEls, mxFreeEls;    // current number of three electrons and their maximal number
    int* accIds;                //[nAt] ids of accepter for a given donor
    int* r2Jumps;             //[nAt] distance for the nearest acceptor
    //stat for e-jump:
    int nJump;              //  total number of jumps       
    int3 posBxJump;         // number of positive box crossing
    int3 negBxJump;         // number of negative box crossing

    //exteranl fields
    float3 elecField;   // electrical field gradient (dU/dr) in Volts per angstrom

    //statistics output (c) - const
    char* stat_buf;         // statistics buffer (where we keep the data)
    void** stat_addr;       // (c) array of pointers to statistic data
    int* stat_shifts;       // (c) array of shifts (in bytes) for each element in buffer array
    int* stat_types;        // (c) 0 - int, else - float data
    //rdf
    float* rdf;             // rdf data
    float* nrdf;            // nuclei rdf data
    //trajectories
    float* traj_buf;        // buffer for trajectories (x, y, z)
    //bind trajectories
    int nBindTrajAtm;       // number of atoms to output
    int* bindtraj_atoms;    // indexes of atoms to output
    char* bindtraj_buf;     // the data buffer

    // functions!
    //float (*funcCoul)(float r2, float& r, float chprd, float alpha, float& eng);     // function for coulombic interaction in the system (in pair calculations)
    float (*funcCoul)(float r2, float& r, float chprd, cudaMD *md, float& eng);     // function for coulombic interaction in the system (in pair calculations)



    //CONSTANTS
    float tSt;      // timestep

    // general
    float r2Max;    // maximal distance of interaction (=r2Elec if where is electrostatic)
    // flags to use Coulombic interaction, for using bonds and angles =0(no bonds/angles) =1(constant bonds/angles, the quantity is const, the type can change) =2 (variable bonds/angles)
    int use_coul, use_bnd, use_angl;

    //thermostat and temperature data
    int tstat;                  // termostat type
    float rQmass, qMassTau2;
    float* engPhotons;          // photon energies for radiative thermostat
    float3* randVects;          // random unit vectors for radiatvive thermostat
    float3* uvects;             // unit vectors

    //cell list
    int maxAtPerCell;
    int maxAtPerBlock;
    //int float3PerCell = sizeof(float3) * maxAtPerCell;
    //int intPerCell = sizeof(int) * maxAtPerCell;
    float3 cSize;   // cell size in x, y and z-directions
    float3 cRevSize;    // 1/cell size
    int3 cNumber;   // cell numbers in x, y and z-directions
    int cnYZ;       // cNumber.y * cNumber.z

    
    // For bypass of cell list
    //! ïåðâûå ïàðû (0-1, 2-3, 4-5 è ò.ä.)
    int nCell;          // the number of cells
    int nPair1, nPair;  // the number of the first cell pairs (cultivated by cell_list2a) and total cell pairs
    int4* cellPairs;   // x - index of the first cell in the pair, y - the second and z - type of pair, w - type of shift
    float3* cellShifts; // delta coordinates to take into account periodic boundaries between cells
    //!!! NEW VARIANT OF CELL LIST
    int* nFirstPairs;       // npairs for each block cultivated cell_list4a function
    int4** firstPairs;      // .z - reserved for type
    float3** firstShifts;
    int4* cellBlocks;       // .x - index of the first cell, .y - index of the second cell, .z - number of cell from second to last, .w - number of pairs
    //int3** secPairs;        // .z - reserved for type
    float3** secShifts;

    //species
    int nSpec;          // the number of species
    cudaSpec* specs;    // species properties
    float** chProd; // type[i]*charge * type[j].charge
    cudaArray* texChProd;   // òîæå ñàìîå â âèäå ïîïðîáóåì îðãàíèçîâàòü â âèäå òåêñòóðíîé ïàìÿòè
    //! ìîæåò ñðàçó äîìíîæèòü íà Fcoul_scale ?

    // electrostaitc
    //float* coulEng; // Coulomb energy as a function of distance
    //float* coulFrc; // Coulomb force ...
    cudaArray* coulEng; // Coulomb energy as a function of distance
    cudaArray* coulFrc; // Coulomb force ...

    //van der Waals
    cudaVdW* pairpots;  // all pair potentials
    cudaVdW*** vdws;    // pointers to pair potential between [i][j] particles types

    // atoms
    float* rMasshdT;    // 1/m * 0.5 timestep for each atom (atom mass does practically not change during simulation, even if atom's type do)
    float* masses;

    //Elec (including) Ewald parameters
    float rElec, r2Elec;       // cutoff for Electrostatic: if Ewald = cutoff for real part; if direct = cutoff for direct, etc
    float alpha;        // alpha in Ewald summation in some other techniques
    float daipi2;          // 2 * alpha / sqrt(pi)    - used in Ewald and Fennel
    float elC1, elC2;       // some constansts (used for Fennel)
    int3 nk;            // number of k-vectors in each direction
    int nKvec;          // total number of k-vectors
    float rKcut2;       // cutoff by k-vectors
    float3* rk;         // components of rk-vector
    //float* rk2;         // rk^2 = rkx^2 + rky^2 + rkz^2
    float* exprk2;      // exp(-rk2/4a2)/rk2    k-depent factor for Ewald summ
    float ewEscale, ewFscale;       // constant factors for energy and force calculations

    //bonds
    int mxBond;          // maximal number of bonds
    int nBndTypes;      // number of bond types (0 is reserved!)
    cudaBond* bondTypes; // [0] reserved as deleted (!some fields in this struct is not constant)
    int** def_bonds;    // array[nSpec][nSpec] of default bond types
    int** bindBonds;    // array[nSpec][nSpec] bond types created by binding
    float** bindR2;        // square of binding distance [nSpec][nSpec]

    //angles
    int mxAngle;
    cudaAngle* angleTypes;  // [0] reserved as deleted
    int* specAngles;    // a[nSp] angle type formed by given species

#ifdef DEBUG_MODE
    //DEBUGGING
    int* nCult;     // number of atoms cultivation
    int atInList, dublInList;   // number of atoms (and dublicates) in cell list
    int* nPairCult;  // number of cultivated pairs
    int* nCelCult;       // number of cultivated cells
    int nFCall, nVdWcall; // n call of force function, of vdw part
    int nAllPair;   // n of all pair iteraction
    float sqrCoul;  // sqr of coulomb energy
#endif
};

#endif // CUSTRUCT_H
