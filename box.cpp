#include <math.h>   // log, sqrt
#include <stdio.h>  // FILE

#include "dataStruct.h"  // Sim, Box, Atoms ....
#include "box.h"

void prepare_box(Box* bx);

int read_box(FILE *f, Box *bx)
// read box parameters from file
{
   int res = 1;

   fscanf(f, "%d", &bx->type);
   if ((bx->type == tpBoxRect) || (bx->type == tpBoxHalf))
   {
        fscanf(f, "%lf %lf %lf", &bx->la, &bx->lb, &bx->lc);
        prepare_box(bx);
   }
   else
   {
        printf("ERROR[008] Unknown box type!\n");
        res = 0;
   }

   return res;
}
// end 'read_box' function

void prepare_box(Box *bx)
// calculate derived properties of the box
{

   double axb1, axb2, axb3;
   double bxc1, bxc2, bxc3;
   double cxa1, cxa2, cxa3;

   // the same for invert matrix
   double iaxb1, iaxb2, iaxb3;
   double ibxc1, ibxc2, ibxc3;
   double icxa1, icxa2, icxa3;

   double det, rdet;

   // if rectangular box we read box lengths, otherwise we read vectors and convert them into lengths
   if (bx->type == tpBoxRect)
   {
       bx->ay = 0.0; bx->az = 0.0; bx->bx = 0.0; bx->bz = 0.0; bx->cx = 0.0; bx->cy = 0.0;
       bx->ax = bx->la;
       bx->by = bx->lb;
       bx->cz = bx->lc;
   }
   else
   {
       bx->la = sqrt(bx->ax*bx->ax + bx->ay*bx->ay + bx->az*bx->az);
       bx->lb = sqrt(bx->bx*bx->bx + bx->by*bx->by + bx->bz*bx->bz);
       bx->lc = sqrt(bx->cx*bx->cx + bx->cy*bx->cy + bx->cz*bx->cz);
   }

   // invert area for different box edges
   bx->revSOxy = 1.0 / (bx->la * bx->lb);
   bx->revSOxz = 1.0 / (bx->la * bx->lc);
   bx->revSOyz = 1.0 / (bx->lb * bx->lc);

   bx->maxLength = bx->la;
   if (bx->maxLength < bx->lb)
     bx->maxLength = bx->lb;
   if (bx->maxLength < bx->lc)
     bx->maxLength = bx->lc;

   // half of box vector length:
   bx->ha = bx->la * 0.5;
   bx->hb = bx->lb * 0.5;
   bx->hc = bx->lc * 0.5;

   // negative half of box vector length:
   bx->nha = -bx->ha;  // -half a
   bx->nhb = -bx->hb;
   bx->nhc = -bx->hc;

   //  invert box vector length
   bx->ra = 1.0 / bx->la;  // reversible a
   bx->rb = 1.0 / bx->lb;
   bx->rc = 1.0 / bx->lc;

   /*
   // cosines of cell angles
   bx->cosC = (bx->ax * bx->bx + bx->ay * bx->by + bx->az * bx->bz)/(bx->la * bx->lb);
   bx->cosB = (bx->ax * bx->cx + bx->ay * bx->cy + bx->az * bx->cz)/(bx->la * bx->lc);
   bx->cosA = (bx->bx * bx->cx + bx->by * bx->cy + bx->bz * bx->cz)/(bx->lb * bx->lc);
   */

   // calculate vector products of cell vectors (for rect they are just (0;0;la*lb) (lb*lc;0;0) (0;la*lc;0))
   axb1 = bx->ay * bx->bz - bx->az * bx->by;    // 0 for rectangular
   axb2 = bx->az * bx->bx - bx->ax * bx->bz;    // 0 for rectangular
   axb3 = bx->ax * bx->by - bx->ay * bx->bx;    // la*lb for rectangular (SOxy)

   bxc1 = bx->by * bx->cz - bx->bz * bx->cy;    // lb*lc for rectangular (SOyz)
   bxc2 = bx->bz * bx->cx - bx->bx * bx->cz;    // 0 for rectangular
   bxc3 = bx->bx * bx->cy - bx->by * bx->cx;    // 0 for rectangular

   cxa1 = bx->az * bx->cy - bx->ay * bx->cz;    // 0 for rectangular
   cxa2 = bx->ax * bx->cz - bx->az * bx->cx;    // la*lc for rectangular (SOxz)
   cxa3 = bx->ay * bx->cx - bx->ax * bx->cy;    // 0 for rectangular


   // volume and invert volume
   //! only for rectangular case
   if ((bx->type == tpBoxRect) || (bx->type == tpBoxHalf))
   {
      bx->vol = bx->la * bx->lb * bx->lc;
   }

   det = bx->ax * bxc1 + bx->bx * cxa1 + bx->cx * axb1;
   rdet = 0.0;
   if (fabs(det) > 0.0)
     rdet = 1.0 / det;

   bx->rvol = 1.0 / bx->vol;


   // inverse matrix
   bx->iax = rdet * bxc1;   // SOyz / V (rect geom) = 1/la (ra)
   bx->iay = rdet * cxa1;   // 0 (rect geom)
   bx->iaz = rdet * axb1;   // 0 (rect geom)

   bx->ibx = rdet * bxc2;   // 0 (rect geom)
   bx->iby = rdet * cxa2;   // 1/lb (rect geom) (rb)
   bx->ibz = rdet * axb2;   // 0 (rect geom)

   bx->icx = rdet * bxc3;
   bx->icy = rdet * cxa3;
   bx->icz = rdet * axb3;   // rc (rect geom)

   // calculate vector products of cell vectors for the invert matrix
   iaxb1 = bx->iay * bx->ibz - bx->iaz * bx->iby;   // 0  (for rect geom)
   iaxb2 = bx->iaz * bx->ibx - bx->iax * bx->ibz;   // 0  (for rect geom)
   iaxb3 = bx->iax * bx->iby - bx->iay * bx->ibx;   // ra*rb  (for rect geom)

   ibxc1 = bx->iby * bx->icz - bx->ibz * bx->icy;   // rb*rc  (for rect geom)
   ibxc2 = bx->ibz * bx->icx - bx->ibx * bx->icz;   // 0  (for rect geom)
   ibxc3 = bx->ibx * bx->icy - bx->iby * bx->icx;   // 0  (for rect geom)

   icxa1 = bx->iaz * bx->icy - bx->iay * bx->icz;   // 0  (for rect geom)
   icxa2 = bx->iax * bx->icz - bx->iaz * bx->icx;   // ra*rc  (for rect geom)
   icxa3 = bx->iay * bx->icx - bx->iax * bx->icy;   // 0  (for rect geom)

   // for Ewald summation
   bx->ip1 = bx->rvol / sqrt(ibxc1 * ibxc1 + ibxc2 * ibxc2 + ibxc3 * ibxc3);    // ra (for rect geom)
   bx->ip2 = bx->rvol / sqrt(icxa1 * icxa1 + icxa2 * icxa2 + icxa3 * icxa3);    // rb (for rect geom)
   bx->ip3 = bx->rvol / sqrt(iaxb1 * iaxb1 + iaxb2 * iaxb2 + iaxb3 * iaxb3);    // rc (for rect geom)

   // counters
   bx->momXp = 0.0;
   bx->momXn = 0.0;
   bx->momYp = 0.0;
   bx->momYn = 0.0;
   bx->momZp = 0.0;
   bx->momZn = 0.0;
   bx->momXp0 = 0.0;
   bx->momXn0 = 0.0;
   bx->momYp0 = 0.0;
   bx->momYn0 = 0.0;
   bx->momZp0 = 0.0;
   bx->momZn0 = 0.0;
}
// end 'box_prop' function

void save_box(FILE *f, Box *bx)
// save box parameter in file
{
   if (bx->type == tpBoxRect)
   {
        fprintf(f, "%d %f %f %f\n", bx->type, bx->la, bx->lb, bx->lc);
   }
}
// end 'save_box' function


void delta_periodic(double &dx, double &dy, double &dz, Box *box)
// apply periodic boundary to coordinate differences: dx, dy, dz
{
   //!Only for rectangular geometry!

   // x
   if (dx > box->ha)
     dx -= box->la;
   else
     if (dx < box->nha)
       dx += box->la;

   // y
   if (dy > box->hb)
     dy -= box->lb;
   else
     if (dy < box->nhb)
       dy += box->lb;

   // z
   if (dz > box->hc)
     dz -= box->lc;
   else
     if (dz < box->nhc)
       dz += box->lc;
}
// end 'rect_periodic' function

void pass_periodic(int iat, int jat, Atoms *atm, Box *box, int &px, int &py, int &pz)
//! function for electron jumps, determine there was jumps through box edge or not
{
   double dx = atm->xs[iat] - atm->xs[jat];
   //double dy = atm->ys[iat] - atm->ys[jat];
   //double dz = atm->zs[iat] - atm->zs[jat];

   if (dx > box->ha) // второй атом в отрицательном отображении
     {
       px = -1;
     }
   else
     if (dx < box->nha) //  второй атом в положительном отображении
       {
          px = 1;
       }
     else
       px = 0;

   //! add y- and z- directions
}

void put_periodic(Atoms *atm, int index, Spec *sp, Box *box)
// put atom[index] in periodic box.
{
   //! Only for rectangular geometry!
   //! momentum is kept without factor 2!

   int t = atm->types[index];
   if (atm->xs[index] < 0)
   {
      // (variant 1): not farhter than box length:
      //atm->xs[index] += box->la;

      // (variant 2): any length
      atm->xs[index] += ((int)(-atm->xs[index]*box->ra) + 1) * box->la;

      sp[t].nOyz++; // counter of crossing in a negative direction
      box->momXn += sp[t].mass * (-atm->vxs[index]); // we suppose that vx in this case is negative
   }
   else
     if (atm->xs[index] > box->la)
     {
        //atm->xs[index] -= box->la;
        atm->xs[index] -= ((int)(atm->xs[index]*box->ra)) * box->la;

        sp[t].pOyz++;
        box->momXp += sp[t].mass * atm->vxs[index];
     }


   if (atm->ys[index] < 0)
   {
      //atm->ys[index] += box->lb;
      atm->ys[index] += ((int)(-atm->ys[index]*box->rb) + 1) * box->lb;

      sp[t].nOxz++;
      box->momYn += sp[t].mass * (-atm->vys[index]); // we suppose that vy in this case is negative
   }
   else
     if (atm->ys[index] > box->lb)
     {
          //atm->ys[index] -= box -> lb;
          atm->ys[index] -= ((int)(atm->ys[index]*box->rb)) * box->lb;

          sp[t].pOxz++;
          box->momYp += sp[t].mass * atm->vys[index];
     }

   if (atm->zs[index] < 0)
   {
      //atm->zs[index] += box -> lc;
      atm->zs[index] += ((int)(-atm->zs[index]*box->rc) + 1) * box->lc;

      sp[t].nOxy++;
      box->momZn += sp[t].mass * (-atm->vzs[index]); // i suppose that vy in this case is negative
   }
   else
     if (atm->zs[index] > box -> lc)
     {
          //atm->zs[index] -= box -> lc;
          atm->zs[index] -= ((int)(atm->zs[index]*box->rc)) * box->lc;

          sp[t].pOxy++;
          box->momZp += sp[t].mass * atm->vzs[index];
     }
}
// end 'put_periodic' function

double sqr_distance(int i, int j, Atoms *atm, Box *bx)
// return square of distance between atoms j and i
{
    double  dx = atm->xs[i] - atm->xs[j];
    double  dy = atm->ys[i] - atm->ys[j];
    double  dz = atm->zs[i] - atm->zs[j];
    delta_periodic(dx, dy, dz, bx);
    return dx*dx + dy*dy + dz*dz;
}

double distance(int i, int j, Atoms *atm, Box *bx)
// return distance between atoms j and i
{
    double  dx = atm->xs[i] - atm->xs[j];
    double  dy = atm->ys[i] - atm->ys[j];
    double  dz = atm->zs[i] - atm->zs[j];
    delta_periodic(dx, dy, dz, bx);
    return sqrt(dx*dx + dy*dy + dz*dz);
}

double distance_by_coord(double x1, double y1, double z1, double x2, double y2, double z2, Box* bx)
// return distance between atoms by their coordinates
{
    double  dx = x1 - x2;
    double  dy = y1 - y2;
    double  dz = z1 - z2;
    delta_periodic(dx, dy, dz, bx);
    return sqrt(dx * dx + dy * dy + dz * dz);
}

double sqr_distance_proj(int i, int j, Atoms *atm, Box *bx, double &dx, double &dy, double &dz)
// return square of distance between atoms j and i and projections
{
    dx = atm->xs[i] - atm->xs[j];
    dy = atm->ys[i] - atm->ys[j];
    dz = atm->zs[i] - atm->zs[j];
    delta_periodic(dx, dy, dz, bx);
    return dx*dx + dy*dy + dz*dz;
}

void center_box(Atoms *atm, Box *bx)
// atom centering
{
    //! only for rectangular geometry!

    int i;
    //double cx, cy, cz;  // center coordinates
    double mxx, mxy, mxz;   // maximum values
    double mnx, mny, mnz;   // minimum values
    double dx, dy, dz;      // values for coordinates shift

    //cx = cy = cz = 0.0;
    mxx = mxy = mxz = 0.0;
    mnx = bx -> la;
    mny = bx -> lb;
    mnz = bx -> lc;
    for (i = 0; i < atm->nAt; i++)
    {
        /*
        cx += atm->xs[i];
        cy += atm->ys[i];
        cz += atm->zs[i];
        */

        if (atm->xs[i] > mxx)
          mxx = atm->xs[i];
        if (atm->xs[i] < mnx)
          mnx = atm->xs[i];

        if (atm->ys[i] > mxy)
          mxy = atm->ys[i];
        if (atm->ys[i] < mny)
          mny = atm->ys[i];

        if (atm->zs[i] > mxz)
          mxz = atm->zs[i];
        if (atm->zs[i] < mnz)
          mnz = atm->zs[i];
    }

    dx = 0.5 * (mxx - mnx) - bx->ha;
    dy = 0.5 * (mxy - mny) - bx->hb;
    dz = 0.5 * (mxz - mnz) - bx->hc;
    for (i = 0; i < atm->nAt; i++)
    {
       atm->xs[i] -= dx;
       atm->ys[i] -= dy;
       atm->zs[i] -= dz;
    }
}

void scale_and_centr_box(Atoms *atm, Box *bx, double scale)
// scale (by multipliyng on scale) and centering
{
    //! only for rectangular geometry!

    int i;
    //double cx, cy, cz;  // center coordinates
    double mxx, mxy, mxz;   // maximum values
    double mnx, mny, mnz;   // minimum values
    double dx, dy, dz;      // values for coordinates shift

    //cx = cy = cz = 0.0;
    mxx = mxy = mxz = 0.0;
    bx->la *= scale;
    bx->lb *= scale;
    bx->lc *= scale;
    mnx = bx->la;
    mny = bx->lb;
    mnz = bx->lc;

    // refresh remaining box propertes (rectangular geometry case)
    bx->revSOxy = 1.0 / (bx->la * bx->lb);
    bx->revSOxz = 1.0 / (bx->la * bx->lc);
    bx->revSOyz = 1.0 / (bx->lb * bx->lc);
    bx->maxLength *= scale;
    bx->ha = bx->la * 0.5;
    bx->hb = bx->lb * 0.5;
    bx->hc = bx->lc * 0.5;
    bx->nha = -bx->ha;  // -half a
    bx->nhb = -bx->hb;
    bx->nhc = -bx->hc;
    bx->ra = 1.0 / bx->la;  // reversible a
    bx->rb = 1.0 / bx->lb;
    bx->rc = 1.0 / bx->lc;
    bx->vol = bx->la * bx->lb * bx->lc;
    bx->rvol = 1.0 / bx->vol;
    bx->ip1 = bx->ra;
    bx->ip2 = bx->rb;
    bx->ip3 = bx->rc;

    for (i = 0; i < atm->nAt; i++)
    {
        /*
        cx += atm->xs[i];
        cy += atm->ys[i];
        cz += atm->zs[i];
        */
        atm->xs[i] *= scale;
        atm->ys[i] *= scale;
        atm->zs[i] *= scale;

        if (atm->xs[i] > mxx)
          mxx = atm->xs[i];
        if (atm->xs[i] < mnx)
          mnx = atm->xs[i];

        if (atm->ys[i] > mxy)
          mxy = atm->ys[i];
        if (atm->ys[i] < mny)
          mny = atm->ys[i];

        if (atm->zs[i] > mxz)
          mxz = atm->zs[i];
        if (atm->zs[i] < mnz)
          mnz = atm->zs[i];
    }

    dx = 0.5 * (mxx - mnx) - bx->ha;
    dy = 0.5 * (mxy - mny) - bx->hb;
    dz = 0.5 * (mxz - mnz) - bx->hc;
    for (i = 0; i < atm->nAt; i++)
    {
       atm->xs[i] -= dx;
       atm->ys[i] -= dy;
       atm->zs[i] -= dz;
    }
}
