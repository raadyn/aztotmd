#include "cuFlow.h"
#include "cuStruct.h"

__global__ void add_flow(float cs, float sn, float dx, float dy, cudaMD *md)
// "create" new atoms of flow and set their coordinates,
//		cs, sn, dx, dy are cosinus and sinus of rotation angle and x- and y- shifts
{
	int iat = threadIdx.x + md->nAt;

	// set coordinates (rotation matrix and shift applied to coordinates of the preset flow atoms):
	md->xyz[iat].x = cs * md->flow_xyz[threadIdx.x].x - sn * md->flow_xyz[threadIdx.x].y + dx;
	md->xyz[iat].y = sn * md->flow_xyz[threadIdx.x].x + cs * md->flow_xyz[threadIdx.x].y + dy;
	md->xyz[iat].z = md->flow_xyz[threadIdx.x].z;


}