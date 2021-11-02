#include <stdio.h>		// printf

#include "defines.h"
#include "cuStruct.h"
#include "cuMDfunc.h"	// delta_periodic
#include "cuAngles.h"

__global__ void refresh_angles(int iStep, int atPerBlock, int atPerThread, cudaMD *md)
// delete old angles and create new ones for atoms which change their type
{
	int i, j, n, t, ang;
	int nei[8];		// bonded neighbors of given atom
	int cnt;		

	//if (threadIdx.x == 0)
		//if (blockIdx.x == 0)
			//printf("refresh_ang(%d)\n", iStep);

	int id0 = blockIdx.x * atPerBlock + threadIdx.x * atPerThread;
	int N = min(id0 + atPerThread, md->nAt);
	
	int iat;
	for (iat = id0; iat < N; iat++)
		if (md->oldTypes[iat] != -1)
		{
			//printf("modified atom[%d]\n", iat);
			// delete old angles:
			i = 0;
			n = md->nangles[iat];
			//if (md->types[iat] > 1)
			  //printf("nangl=%d\n", n);
			while (n && (i < md->nAngle))
			{
				if (md->angles[i].w)
					if (md->angles[i].x == iat)
					{
						n--;
						md->angles[i].w = 0;
						//printf("try to del\n");
					}
				i++;
			}
			//md->nangles[iat] = n;	// dublicated action

			// create new angles
			t = md->specAngles[md->types[iat]];		// get type of angle, which formed by current atom type
			//if ((t != 1)&&(t != 0))
				//printf("impossible angle type=%d, spec=%d iat=%d\n", t, md->types[iat], iat);
			if (t && (md->nbonds[iat] > 1))		// atom type supports angle creating and number of bonds is enough
			{
				//printf("atom[%d](%d) tries to create angle(%d)\n", iat, md->types[iat], t);
				// search of neighbors by bonds
				i = 0; cnt = 0;
				n = md->nbonds[iat];
				while (n && (i < md->nBond))
				{
					if (md->bonds[i].z)		// if bond isn't deleted
					{
						if (md->bonds[i].x == iat)
						{
							//printf("bnd %d = (%d %d)\n", i, md->bonds[i].x, md->bonds[i].y);
							nei[cnt] = md->bonds[i].y;
							cnt++;
							n--;	// ÷òîáû âûéòè êîãäà îáðàáîòàåì äîñòàòî÷íîå ÷èñëî ñâÿçåé
						}
						else if (md->bonds[i].y == iat)
						{
							//printf("bnd %d = (%d %d)\n", i, md->bonds[i].x, md->bonds[i].y);
							nei[cnt] = md->bonds[i].x;
							cnt++;
							n--;
						}
					}
					i++;
				}

				// add new angles based on found neighbors:
				//printf("cnt=%d\n", cnt);
				for (i = 0; i < cnt-1; i++)
					for (j = i + 1; j < cnt; j++)
					{
						ang = atomicAdd(&(md->nAngle), 1);
#ifdef DEBUG_MODE
						if (nei[i] == nei[j])
						{
							printf("the valent angle has the same ligands\n");
							int k = 0;
							n = md->nbonds[iat];
							while (n && (k < md->nBond))
							{
								if (md->bonds[k].z)
									if ((md->bonds[k].x == iat) || (md->bonds[k].y == iat))
									{
										n--;
										printf("step(%d) bnd %d = (%d[%d] %d[%d]) created at %d step\n", iStep, k, md->bonds[k].x, md->types[md->bonds[k].x], md->bonds[k].y, md->types[md->bonds[k].y], md->bonds[k].w);
									}
								k++;
							}
							
						}
#endif
						//! â äåáàãå - ïðîâåðêó íà ìàêñèìóì
						md->angles[ang] = make_int4(iat, nei[i], nei[j], t);
						//printf("(%d) thred[%d] i=%d j=%d\n", iStep, threadIdx.x, i, j);
					}

				n = (cnt * (cnt - 1)) / 2;
				//printf("n=%d\n", n);
			}
			md->nangles[iat] = n;

			// recalculate number of particels and reset flag
			//! âîîáùå ïëîõî, ÷òî ïåðåñ÷åò êîëè÷åñòâà ÷àñòèö èäåò çäåñü, â óãëàõ. Åñëè óãëîâ íåò, êîë-âà ÷àñòèö íå áóäóò îáíîâëÿòüñÿ
			if (md->oldTypes[iat] != md->types[iat])
			{
				atomicAdd(&(md->specs[md->types[iat]].number), 1);
				atomicSub(&(md->specs[md->oldTypes[iat]].number), 1);
			}
			md->oldTypes[iat] = -1;
		}	
}
// end function 'refresh_angles'

__global__ void clear_angles(cudaMD* md)
// remove angles with .w = 0
{
	//printf("clear angles\n");
	
	// íå çíàþ, êàê ñäåëàòü ïàðàëëåëüíûé, âîò ñåðèéíûé âàðèàíò:
	int i = 0;
	int j = md->nAngle - 1;

	//for debug:
	//int cnt = 0;
	//int j0 = j;

	while (i < j)
	{
		while ((md->angles[j].w == 0) && (j > i))
			j--;
		while ((md->angles[i].w != 0) && (i < j))
			i++;
		if (i < j)
		{
			
			//if ((md->angles[i].w > 1) || (md->angles[j].w > 1))
				//printf("wrong anles: [%d]=%d [%d]=%d\n", i, md->angles[i].w, j, md->angles[j].w);

			md->angles[i] = md->angles[j];
			md->angles[j].w = 0;
			i++;
			j--;
			//printf("delete angle[%d]\n", i);
			//cnt++;
		}
	}

	if ((i == j) &&  (md->angles[i].w == 0))
   		md->nAngle = j;
	else
		md->nAngle = j + 1;

	//printf("end clear angles\n");
}
// end function 'clear_angles'

__device__ void angle_hcos(int4* angle, cudaAngle* type, cudaMD* md, float& eng);

__global__ void apply_angles(int iStep, int angPerBlock, int angPerThread, cudaMD* md)
// apply valence angle potentials
{
	cudaAngle* ang;

	// energies of angle potential	
	float eng = 0.f;
	__shared__ float shEng;

	//if (threadIdx.x == 0)
		//if (blockIdx.x == 0)
		//	printf("apply_angles(%d)[%d]: perBlock=%d perThread=%d\n", iStep, blockIdx.x, angPerBlock, angPerThread);

	if (threadIdx.x == 0)
		shEng = 0.f;
	__syncthreads();

	int id0 = blockIdx.x * angPerBlock + threadIdx.x * angPerThread;
	int N = min(id0 + angPerThread, md->nAngle);

	int i;
	for (i = id0; i < N; i++)
		if (md->angles[i].w)
		{
/*  ê ñîæàëåíèþ ïðèéäåòñÿ ïåðåíåñòè ýòó ÷àñòü ñðàçó ïîñëå ñîðòèðîâêè, ïîñêîëüêó îíà âàæíà äëÿ ïðîöåäóðû refresh_angles, ÷òî èäåò âíà÷àëå
			// take sorting into account
#ifdef USE_FASTLIST
			md->angles[i].x = md->sort_ind[md->angles[i].x];
			md->angles[i].y = md->sort_ind[md->angles[i].y];
			md->angles[i].z = md->sort_ind[md->angles[i].z];
#endif
*/
			
			if (md->angles[i].y == md->angles[i].z)
				printf("Angle[%d] at atom[%d](type: %d)(nbnd=%d)(par=%d) has the same ligands: %d(%d)(nbnd=%d par=%d)!\n", i, md->angles[i].x, md->types[md->angles[i].x], md->nbonds[md->angles[i].x], md->parents[md->angles[i].x], md->angles[i].y, md->types[md->angles[i].y], md->nbonds[md->angles[i].y], md->parents[md->angles[i].y]);
			
			// old verification
			//if (md->angles[i].w != 1)
			//{
			//	printf("wrong angle\n");
				//printf("ang=%p ang->type=%d func=%p hcos=%p ang1=%p w=%d\n", ang, ang->type, ang->force_eng, &angle_hcos, &(md->angleTypes[1]), md->angles[i].w);
			//}
			
			ang = &(md->angleTypes[md->angles[i].w]);
			//printf("ang=%p ang->type=%d func=%p hcos=%p ang1=%p w=%d\n", ang, ang->type, ang->force_eng, &angle_hcos, &(md->angleTypes[1]), md->angles[i].w);
			ang->force_eng(&(md->angles[i]), ang, md, eng);
		}

	// split energy to shared and then to global memory
	//printf("eng=%f\n", eng);
	atomicAdd(&shEng, eng);
	__syncthreads();
	if (threadIdx.x == 0)
		atomicAdd(&(md->engAngl), shEng);

	//if (threadIdx.x == 0)
		//if (blockIdx.x == 0)
		//printf("end apply_angles(%d)[%d]: perBlock=%d perThread=%d\n", iStep, blockIdx.x, angPerBlock, angPerThread);

}

__device__ void angle_hcos(int4* angle, cudaAngle* type, cudaMD* md, float& eng)
// harmonic cosine valent angle potential:
// U = k / 2 * (cos(th)-cos(th0))^2
{
	float k = type->p0;
	float cos0 = type->p1;
	//printf("hcos: k=%f cos0=%f\n", k, cos0);

	// indexes of central atom and ligands:
	int c = angle->x;
	int l1 = angle->y;
	int l2 = angle->z;

	//! è òóò åù¸ ìîæíî ñõèòðèòü, ñðàçó âçÿòü ðàññòîÿíèÿ èç bonds, âåäü angle ìîæåò áûòü òîëüêî ìåæäó bonds

	// vector ij
	float xij = md->xyz[l1].x - md->xyz[c].x;
	float yij = md->xyz[l1].y - md->xyz[c].y;
	float zij = md->xyz[l1].z - md->xyz[c].z;
	//printf("angle: (%d-%d-%d), dij=(%f, %f, %f)\n", c, l1, l2, xij, yij, zij);
	delta_periodic(xij, yij, zij, md);
	float r2ij = xij * xij + yij * yij + zij * zij;
	float rij = sqrt(r2ij);

	// vector ik
	float xik = md->xyz[l2].x - md->xyz[c].x;
	float yik = md->xyz[l2].y - md->xyz[c].y;
	float zik = md->xyz[l2].z - md->xyz[c].z;
	delta_periodic(xik, yik, zik, md);
	float r2ik = xik * xik + yik * yik + zik * zik;
	float rik = sqrt(r2ik);

	float cos_th = (xij * xik + yij * yik + zij * zik) / rij / rik;
	float dCos = cos_th - cos0; // delta cosinus

	float c1 = -k * dCos;
	float c2 = 1.0 / rij / rik;
	//printf("rij=%f rik=%f cos=%f dcos=%f c1=%f c2=%f\n", rij, rik, cos_th, dCos, c1, c2);

	//printf("f[c]x = %f\n", -c1 * (xik * c2 + xij * c2 - cos_th * (xij / r2ij + xik / r2ik)));
	atomicAdd(&(md->frs[c].x), -c1 * (xik * c2 + xij * c2 - cos_th * (xij / r2ij + xik / r2ik)));
	atomicAdd(&(md->frs[c].y), -c1 * (yik * c2 + yij * c2 - cos_th * (yij / r2ij + yik / r2ik)));
	atomicAdd(&(md->frs[c].z), -c1 * (zik * c2 + zij * c2 - cos_th * (zij / r2ij + zik / r2ik)));

	atomicAdd(&(md->frs[l1].x), c1 * (xik * c2 - cos_th * xij / r2ij));
	atomicAdd(&(md->frs[l1].y), c1 * (yik * c2 - cos_th * yij / r2ij));
	atomicAdd(&(md->frs[l1].z), c1 * (zik * c2 - cos_th * zij / r2ij));

	atomicAdd(&(md->frs[l2].x), c1 * (xij * c2 - cos_th * xik / r2ik));
	atomicAdd(&(md->frs[l2].y), c1 * (yij * c2 - cos_th * yik / r2ik));
	atomicAdd(&(md->frs[l2].z), c1 * (zij * c2 - cos_th * zik / r2ik));

	//printf("hcos eng: %f\n", 0.5f * k * dCos * dCos);
	eng += 0.5f * k * dCos * dCos;
}

__global__ void define_ang_potential(cudaMD* md)
{
	cudaAngle* ang;
	//printf("def bonds, thradId=%d\n", threadIdx.x);
	ang = &(md->angleTypes[threadIdx.x + 1]);    // + 1 because 0th bond is reserved as 'no_angle'

	switch (ang->type)
	{
	case 1:
		ang->force_eng = &angle_hcos;
		//printf("set as ang cos\n");
		break;
	}
}
