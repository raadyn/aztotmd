#ifndef BOX_H
#define BOX_H

const int tpBoxRect = 1;    // orthorombic periodical box
const int tpBoxHalf = 2;	// orthorombical half-periodic box (x,y are periodic, z is not)

//initialization:
int read_box(FILE *f, Box *bx);
void prepare_box(Box *box);

// periodic functions:
void delta_periodic(double &dx, double &dy, double &dz, Box *box);
void pass_periodic(int iat, int jat, Atoms *atm, Box *box, int &px, int &py, int &pz);
void put_periodic(Atoms *atm, int index, Spec *sp, Box *box);

// disnance calculations (between i-th and j-th atoms):
double distance(int i, int j, Atoms *atm, Box *bx);
double distance_by_coord(double x1, double y1, double z1, double x2, double y2, double z2, Box* bx);
double sqr_distance(int i, int j, Atoms *atm, Box *bx);
double sqr_distance_proj(int i, int j, Atoms *atm, Box *bx, double &dx, double &dy, double &dz);

// center functions:
void center_box(Atoms *atm, Box *bx);
void scale_and_centr_box(Atoms *atm, Box *bx, double scale);

//output
void save_box(FILE *f, Box *bx);

#endif  /* BOX_H */
