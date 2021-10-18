// наиболее часто нужные полезные вещи
//  которые нужны не только дл€ ћƒ, но и вообще
#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>   // *FILE

//sizes for memory allocation
const int int_size = sizeof(int);
const int double_size = sizeof(double);
const int pointer_size = sizeof(void*);

// input/output
int keyPress(int key);	// verify that key is pressed

// simple but useful functions
double min(double x1, double x2);
double max(double x1, double x2);
int max(int x1, int x2);
int del_and_rest(int a, int b, int& rest);
double sqr_sum(double x, double y, double z);
int npairs(int n);

void sincos(double arg, double& s, double& c);


// random number from 0 to 1
double rand01();

// functions for searching in a text file
int find_int(FILE *f, const char *templ, int &value);
int find_number(FILE *f, const char *templ);
int find_int_def(FILE *f, const char *templ, int &value, int def_val);
int find_double(FILE *f, const char *templ, double &value);
int find_double_def(FILE *f, const char *templ, double &value, double def_value);
int find_str(FILE *f, const char *templ, char *str);
//int find_flag(FILE *f, const char *templ);

#endif  /* UTILS_H */
