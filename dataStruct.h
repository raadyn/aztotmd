#ifndef DATASTR_H
#define DATASTR_H

typedef char name8[8];

// initial velocities setup types:
const int tpInitVelZero     = 0;    // zero
const int tpInitVelGauss    = 1;    // gaussian
const int tpInitVelConst = 2;       // constant value of initial velocity
const int tpInitVelEng = 3;         // in such way that user set 1/2mv2 with random direction

// const for binary flags for sim->flag
const int bfSimUseCList = 0;    // use cList

// const for int_params in Sim
const int nIntParam = 1;
const int ipInitVel     = 0; // way of initial velocities setup

/*
// const for double_params in Sim
const int nDoubleParam = 4;

const int dpMaxCutVdW  = 0; // maximal VdW cuttoff
const int dpMaxCutElec = 1; // maximal electrostatic cuttoff
const int dpMaxCutJump = 2; // maximal ejump cuttoff
const int dpMaxCut     = 3; // maximal cuttoff
*/

// pre definitions...
struct TStat;
struct Atoms;
struct VdW;
struct Spec;
struct Bond;
struct Box;
struct Field;
struct Elec;

// structure for simulation parameters
struct Sim
{
   double tSt;    // (d-p) value of timestep (in MD units)      (d-p) directive - parameter
   //! не обязательно его хранить
   double tSim;   // (d-p) length of simulation (in MD time units)
   int nSt;       // (d-p) number of timestep

   //! не обязательно его хранить
   double tEq;    // (d-p) length of equilibration period (in MD time units)
   int nEq;       // (d-p) number of timestep in equilibration period
   int freqEq;    // (d-p) freqence of equilibration (equilibration will be called every freqEq step)

   // general flags
   int use_bnd, use_angl; // flags for using bonds and angles =0(no bonds/angles) =1(constant bonds/angles, the quantity is const, the type can change) =2 (variable bonds/angles)
   int use_linkage;         // (f) use or not new bonds forming (temporary flag)

   //SYSTEM PROPERTIES:
   //double engPot;           // (pr) potential energy
   double engTot;           // (pr) total energy
   double engKin;           // (pr) kinetic energy
   double engVdW;           // (pr) Van der Waals energy
   double engElec1;    // (pr) 'Constant' part of Ewald Sum (depends on volume and charge of system)
   double engElec2;      // (pr) Recipropal part of Ewald Sum
   double engElec3;     // (pr) Real part of Ewald Sum
   double engElecTot;   // (pr) total electrostatic energy
   double engElecField;     // (pr) Energy of electric field
   double engBond;          // (pr) Energy of bonded interaction
   double engAngle;         //  (pr) Energy of valent angles
   double engOwn;            // own energy of particles
   double Temp;             // (pr) Instanteous temperature
   double presXn, presXp, presYn, presYp, presZn, presZp;   // pressure in different directions
   double prevTime;         // previous time (for dt calculation)


   double tTemp;   // (d-p) target temperature (in K)
   //double tKin;    // (dp) target kinetic energy, according to temperature (sigma in DL_POLY source)  (dp) - derived parameter
   int degFree;     // (p) number of degree freedom (3*N) in the simplest case
   double revDegFree;       // (p)  1/degFree   p - parameter

   //vars for cell-list method
   double desired_cell_size;    // a desired length of the cell
   int *clist;        // (a)    atom list [NAtom] for cell-list method  (a) - array
   int *chead;     //   (a) head list [Ncell] for cell-list method
   int nHead;         // (n)    the number of cells (cnX*cnY*cnZ)
   int *nHeadNeig;      // (a)  the numbers of cells neighbors
   int **lstHNeig;      // (a)  lists of the cells neighbors
   int cnX, cnY, cnZ; // (n)    numbers of cells in X, Y and Z-directions
   int cnYZ;            // (n)  cnY * cnZ
   double clX, clY, clZ;  // (dp)    cell length in X,Y and Z-directions

   //vars for neighbors list
   int *nNbors;      // numbers[nAt] of neighbors of atoms
   int **nbors;        // indexes of neighbors of atom
   int **tnbors;        // types of neighbors (VdW, Coulomb, bonding, e-jump, bond formation)
   double **distances;     // distances to neighbors
   int maxNbors;        // maximal number of neighbors in list

   //max VdW range (square):
   //double mxRvdw2;      // maximal cutoff distance for VdW interactions
   double rMax;         // maximal cutoff distance (among VdW, real Ewald, etc)
   double r2Max;        // rMax^2

   //vars for Electrostatic
   //double alpha;    // alpha in Ewald technique
   //double r2Real;   //r^2 of cutoff radii for real part of Ewald summ

   // SOME LESS USED PARAMETERS (parameters which are used only in initialization or very rare)
   int flags;                           // binary flag for some purpose (see constant of bits)
   int int_pars[nIntParam];             // array of integer parameters
   //double double_pars[nDoubleParam];    // array of double parameters

   //vars for E-Jump
   int eJump;      //    the number of eJumps routine at every timestep (if negative - every Nth step)
   double rElec;    // (d-p)    maximal length of electron jump
   double r2Elec; //    (dp)    square of rElec
   double dEjump; // criteria then dE is assumed to zero
   int ejtype;     // (d-f) type of electron jump (see constants above)
   //double *EngDifs;     // for saving energy difference between jumps
   //int nJumpVar;        // the number for jumps variants

   //int varSpecNumb; // (f)  the variable number of atoms of one specie
   int *varSpecs;   // indexes of species with variable quantity
   int nVarSpec;    // the number of species with variable quantity

   int nFreeEl;     // number of free electrons in the system (number of electrons avaliable for jumps)
   int* electrons;      // array of electron number sites
   int nJump;       // counter of electron jumps
   int **jumps; // array for keeping nJump between different donor-acceptor pairs
   int pBxEjump, nBxEjump;  // (c)  the numbers of electron jumped in positive and negative directions through the Oyz box edge
   int pEjump, nEjump;  // (c)  the number of electron jumped in positive and negative direction through the center of box
   int pTotJump, nTotJump;  // (c) total jumps in positive and negative directions

   //vars for bonding
   //int Bonding;  // (d-f) flag for bonding during MD run
   //double rBond, r2Bond; // r and r^2 for bonding distance
   //int *bondAtoms;      // array for keeping indexes of bonded atom. -1 value if no bonding (for only_twoatomic)
   int *bondTypes;      // array for keeping bond types (for only_twoatomic)
   int nBonds;          // number of bonds in the system
   int nBndForm, nBndBr;    // (c) the number of formed and breaking bonds (counters)
   int maxCanBondAtm;  // (d-m) the maximal number of the atoms which can form bond
   int nCanBondAtm;    // the number of atoms which can form bond
   int *canBondAtms;    // indexes of atoms which can form bond

   // OTHER GENERAL DIRECTIVES
   //var for External Field
   double Ux, Uy, Uz;       // (d-p)    field gradient in x,y and z-directions: dU/dx and etc
   //! for shifts
   double shiftX, shiftVal;     // if (x > shiftX) fz = fz + shiftVal

   int reset_vels;      // (d) set velocities as zero every Nth step




   // OUTPUT VARS

   //vars for RDF
   double dRDF;     // width of RDF step
   double idRDF;    // 1/dRDF
   double rRDF;     // maximal length of RDF, but if box size is lesser, rdf will ouputs only to box size
   double r2RDF;    // rRDF^2
   int nRDF;        // number of RDF points (maxR / dRDF)
   int frRDF;       // frequency of RDF statistics
   int frRDFout;    // frequence of RDF output (if 0 - no intermediate output)
   int nRDFout;     // the number of RDF sampling
   int nuclei_rdf;  // (f) to use RDF by nuclei
   double **rdf; 


   //coordination number (CN)
   int outCN;   // (f) to output CN (coord number) info
   double r2CN;  // CN radius^2
   int nCountCN;        // number of counter species for CN output
   int nCentrCN;        // number of central atom species for CN output

   int hist;      // as freq will be outputed history file (every 'hist' step)
   int revcon;    // frequency of REVCON
   int stat;     // frequency of stat file output
   int vaf;     // frequency of VAF

   //trajectories
   int stTraj;      //  the timestep number from which trajectories are collected
   int frTraj;      // frequency of Traj output (every frTraj timestep)
   int at1Traj, at2Traj;    // initial and last atoms for trajectory output

   //bind trajectories
   int nBindTrajAtoms;      // number of atoms to output
   int bindTrajFreq, bindTrajStart;     // frequency and start from
   int* bindTrajAtoms;      // array of indexes for binded trajectories

   //functions for operating!
     //enery of i-j pair calculations (используется для обхода пар в функциях cell_list или all_pair)
   void (*pair)(int i, int j, Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim);

   void (*integrator1)(Atoms *atm, Spec *spec, Sim *sim, Box *box, TStat *tstat);

   void (*forcefield)(Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim);
   void (*add_elec)(Atoms *atm, Field *field, Elec *elec, Box *bx, Sim *sim);
   void (*pair_elec)(Spec *sp, int it, int jt, double r2, double r, Elec *elec, Sim *sim, double &force);

   int (*ejumper)(Atoms *atm, Field *field, Sim *sim, Box *bx);
   void (*do_jump)(Atoms *atm, Field *field, Sim *sim, Box *bx, int step);
};

// box structure
struct Box
{
   int type;


   double la, lb, lc;   // length of box vectors (a, b, c)
   double maxLength;    // maximal length
   double ra, rb, rc;   //  invert length of box vectors (a, b, c)
   double ha, hb, hc;   // half of length a, b, c
   double nha, nhb, nhc;// negative ha, hb, hc = -0.5*la, ...

   double vol; // volume
   double rvol; // 1/volume

   double revSOxy, revSOyz, revSOxz;    // 1/S of box edges...

   // a,b,c vectors
   double ax, ay, az;
   double bx, by, bz;
   double cx, cy, cz;

   // invert vectors:
   double iax, iay, iaz;
   double ibx, iby, ibz;
   double icx, icy, icz;

   //cell perpendicular widths for the invert matrix:
   double ip1, ip2, ip3;

   //double cosA, cosB, cosC; // cos of angles between a,b,c.  A - between b and c; B - a and c; C - a and b


   // Impulses from particles for pressure determination
   // momentums in X,Y,Z coord in positive(p) or negative (n) direction
   double momXp, momXn, momYp, momYn, momZp, momZn;
   double momXp0, momXn0, momYp0, momYn0, momZp0, momZn0;       // previous values of momentum
};

// structure for Atom type (Species)
struct Spec
{
  int number; // количество частиц данного сорта

  char name[8];
  double mass;
  //double rmass; // 1/m
  double rMass_hdt; // 1/m * 0.5 dt (dt - timestep) in prepare_field()
  double charge;
  double energy; // own energy for dE calculation during jump
  int frozen;   // (f) "frozen" atom type. These atoms do not move during simulation
  int nuclei;   // index of nuclei to which belong the specie

  int charged; // (f) 0 - neitral
  int donacc; // donor/acceptor  binary flags:  01 - donor 10 - acceptor 00 - no donor, no acceptor, 11 - both
  //Spec *oxForm;    // link to Spec after e-donoring
  //Spec *redForm;   //  ....              e-accepting
  int oxForm;
  int redForm;  // index of spec - redForm of this Spec
  int varNumber;    // (f)  number of particles is variable

  int nFreeEl;  // number of electron available for donoring

  //int canBond;  // flag: can create bond with some spec or not
  int *bondKeys; // array of keys+1 for bonds that can be assigned to species with corresponding index
  int angleType;    // (f) possibility to form angle (=0 - can not form angle, >0 - angle with id=angleType


  //variables for CN (coord number) output
  int idCentral;   // flag for counting of CN (0 - do not calculate, n - index )
  int idCounter;   // flag for counterion (0 - not counting, n - index of array place)

  //counters for moving through edges of the box:
  int pOyz; // icrease, then particle away from box throw the Oyz plane in positive direction
  int nOyz; //                                                              negative

  int pOxz;
  int nOxz;
  int pOxy;
  int nOxy;

  double displ; // displacement (for MSD calculation)
  double vaf;   // velocity autocorrelation function

  // radii
  double radA, radB, mxEng;
};

//structure for short-range pair iteraction (Van-der-Waals)
struct VdW  // pair potential
{
   int type;
   int use_radii;       // (f) to use radii in potential calculation (only for radiative thermostat, which is only in CUDA version)
   double p0, p1, p2, p3, p4;
   double r2cut; // cuttoff^2
   double (*eng)(double r2, VdW *vdw); // function to calculate energy
   double (*feng)(double r2, VdW *vdw, double &eng); // function to calculate force (return) & energy (save in eng)
   double (*feng_r)(double r2, double &r, VdW *vdw, double &eng); // function to calculate force (return) & energy (save in eng) if r may be knonw
   double (*eng_r)(double r2, double r, VdW *vdw); // return energy by r and r^2
};

struct Atoms
{
   int nAt;
   int* types;  //! заменить на Spec*
   double *xs, *ys, *zs;
   double *vxs, *vys, *vzs;
   double *fxs, *fys, *fzs;
   int *nBonds;  // the number of bonds (provided by bond_list)
   int *parents;    //! index of atom connected with this, однозначно определен только для атомов с одной связью

   //initial coordinates (for MSD calculation)
   double *x0s, *y0s, *z0s;
   double *vx0, *vy0, *vz0;
};

// structure for bond type
struct Bond
{
  int type;  // type of potential
  //int breakable;  // flag breaking bond or not
  //int mut;          // (f) mutable bond: 0 - not, 1 - by exceed 2 - by lowering
  int new_type[2];      // (id) bond type after mutation: [0] - r2<r2min  [1] r2>r2max
  int spec1, spec2; // type of atoms that connected by this bond type
  //int spec1br, spec2br; // type of atoms after bond breaking
  int new_spec1[2], new_spec2[2];
  int mnEx, mxEx;
  int hatom;          // flag and index (= -1 for covalent bonds, = spec1/spec2 for hydrogen bonds, = index of H-atom) 
  int evol;           // flag/index (= 0 or new type of bond, to which transforms this)
  double p0, p1, p2, p3, p4;    // potential parameters
  double r2min, r2max;          // square of distance of bond breaking (if applicapble)
  //double energy;        // energy difference between and after bond breaking
  int number;           // quantity of such bonds
  //int ind;              // current index (вспомогательная переменная для вывода связей в файл)

};

struct Angle
{
  int type; // potential type (now = 1 harmonic cosinus)
  int central;  // of central atom species type
  double p0, p1, p2;    // parameters
};

// for electrostatic calculations
struct Elec
{
    int type;   // type of electrostatic calculations
                // see constants tpElec...
    double eps;   // permittivity,  e in 1/(4pi * e * e0)

    double alpha;
    int kx, ky, kz; // number of k-vectors (for Ewald technique)
    double scale, scale2;   // used in Ewald and Fennel (scale = erfc(aRc)/Rc, scale2 = erfc(aRc)/Rc^2 + ...)
    double daipi2;          // 2 * alpha / sqrt(pi)    - used in Ewald and Fennel
    double mr4a2;
    double rkcut, rkcut2;
    double rReal, r2Real;   // real space cuttoff and its square

    //arrays (for Ewald technique):
    double **elc, **els, **emc, **ems, **enc, **ens;
    double *lmc, *lms, *ckc, *cks;
};

// for force field keeping
struct Field
{
    int nSpec, nNucl;   // number of species and nuclei
    int nPair;       // (n)  number of pairs between Spec, including self pair: 0.5*Nsp*(Nsp-1) + Nsp;
    int nVdW;       // (n) total number of pair potentials
    Spec *species;
    char **snames;      // names of species (pointer to structure->name)
    name8* nnames;      // nuclei names
    int* nnumbers;      // number of neclei
    int charged_spec;   // (f) are there charged species or not

    VdW   *pairpots;    // array for keeping all possble pair potentials
    VdW ***vdws;    // matrix of pointer to pairpots: vdws[iSpec][iSpec] = &pairpot
    double minRvdw;     // minimal distance of VdW interaction //! для чего?
    double maxRvdw;     // maximal distance of VdW interaction
    double maxR2vdw;    // maxRvdw^2
    double maxRbind;    // maximal distance of atoms binding

    int nBdata;     // number of bond types
    Bond *bdata;    // array of bond types

    // arrays for bonds
    int nBonds;
    int mxBonds;  // maximal number of bonds (memory allocation)
    int *at1;
    int *at2;
    int *bTypes;

    int** bond_matrix;  // 0 = no bond, positive - indexes in bdata, negative - abs - indexes in bdata with inverted atom order
    int** bonding_matr; // similar to bond_matrix but for species which can form bond
    double** bindR2matrix;  // matrix of binding distance

    int nAdata;     // the number of angle types
    Angle *adata;   // array of angle types

    // valent angles part
    int nAngles;    // the number of valency angles
    int mxAngles;   // maximal number for angles allocation
    int *centrs;    // indexes of central atom
    int *lig1;      // indexes of the first ligand
    int *lig2;      // indexes of the second ligand
    int *angTypes;  // angles types

};

const int spec_size = sizeof(Spec);

#endif  /* DATASTR_H */
