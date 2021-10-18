// rdf.h
//  заголовочный файл модуля, работающего с радиальной функцией распределения
#ifndef RDF_H
#define RDF_H

int read_rdf(FILE* f, Sim* sim);
void init_rdf(Sim* sim, Box* box);
int alloc_rdf(Sim *sim, Field *field, Box *box);
//void clear_rdf(double **rdf, Sim *sim, Field *field, int &nRDF);
void get_rdf(Atoms *atm, Sim *sim, Field *field, Box *box);
int out_rdf(Field *field, Box *box, Sim *sim, char *fname);
void free_rdf(Sim *sim, Field *field);

#endif  /* RDF_H */
