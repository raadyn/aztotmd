#ifndef DEFINES_H
#define DEFINES_H

// Debugging
//#define DEBUG_MODE
#define DEBUG1_MODE		// mainly, velocities
#define MX_VEL	1e5		// maximal velocity component
// forces auto_cap
//#define AUTO_CAP
#define MX_FRC	1e4



// maximal number of species (for debugging)
#define MX_SPEC 15



//#define USE_ALLPAIR
//#define USE_CELL3
//#define USE_CELL2
#define USE_FASTLIST


//#define USE_CONST

// FOR EWALD:
// maximal number of k-vectors in one direction
#define NKVEC_MX	100
// maximal total number of k-vectors
#define NTOTKVEC	1200


//#define TX_CHARGE



#endif // DEFINES_H

