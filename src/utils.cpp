#include <stdlib.h>     // malloc, alloc, rand, NULL
#include <stdio.h>      // *FILE
#include <math.h>       // sin, cos
#include <conio.h>      // _kbhit, _getch

#include "utils.h"

int keyPress(int key)
// verify that key is pressed
{
    if (_kbhit())
    {
        int c = _getch();
        if (c == key)
            return 1;
        else
            return 0;
    }
    else
        return 0;
}

double min(double x1, double x2)    
{
    if (x2 < x1)
        return x2;
    else
        return x1;
}

double max(double x1, double x2)
{
    if (x2 > x1)
        return x2;
    else
        return x1;
}

int max(int x1, int x2)
{
    if (x2 > x1)
        return x2;
    else
        return x1;
}

int del_and_rest(int a, int b, int& rest)
// return (a/b) and save rest
{
    rest = a % b;
    return a / b;
}

double sqr_sum(double x, double y, double z)	
{
    return x * x + y * y + z * z;
}

int npairs(int n)
// number of pairs for n elements
{
    return n * (n - 1) / 2;
}

void sincos(double arg, double& s, double& c)
// function of simultaneous calculation sin and cos (faster then sin and cos separately)
{
    //asm("fsincos" : [c]"=t"(c), [s]"=u"(s) : "0"(arg));
    //sincos();

/*
    __asm
    {
        fsincos : [c]=t(c), [s]=u(s) : 0(arg));
    }
*/
//! assembler is not supported, so:
    s = sin(arg);
    c = cos(arg);
}

//FUNCTIONS FOR SEEKING IN FILES

//default all - ok  (result = 1), but if something wrong we must set res as 0
//! òóò õèòðàÿ êîíñòðóêöèÿ, ÿ å¸ ñàì ïðèäóìàë, íî íå óâåðåí, êàê îíà ðàáîòàåò,ïîõîæå â i êëàä¸òñÿ ñðàâíåíèå fscanf <= 0
//! ïîýòîìó åñëè ïî îêîí÷àíèþ öèêàë i = 1, ò.å. òðó, òî öèêë íå íàø¸ë òîãî, ÷òî èñêàë
int find_int(FILE *f, const char *templ, int &value)
{
  int i, x;
  char bufer[100];

  rewind(f);
  while (!feof(f) && (i = fscanf(f, templ, &x) <= 0))
      fscanf_s(f, "%s", bufer, 100);

  if (i)
    return 0;
  else
    {
        value = x;
        return 1;
    }
}

int find_number(FILE *f, const char *templ)
// similar to find_int but return searching result or zero if no result
{
  int i, x;
  char bufer[100];

  rewind(f);
  while (!feof(f) && (i = fscanf(f, templ, &x) <= 0))
    fscanf_s(f, "%s", bufer, 100);

  if (i)
    return 0;
  else
    return x;
}

int find_int_def(FILE *f, const char *templ, int &value, int def_val)
// with default value
{
  int i, x;
  char bufer[100];

  rewind(f);
  while (!feof(f) && (i = fscanf(f, templ, &x) <= 0))
      fscanf_s(f, "%s", bufer, 100);

  if (i)
    {
        value = def_val;
        return 0;
    }
  else
    {
        value = x;
        return 1;
    }
}

int find_double(FILE *f, const char *templ, double &value)
{
  int i;
  char bufer[100];

  rewind(f);
  while (!feof(f) && (i = fscanf(f, templ, &value) <= 0))
      fscanf_s(f, "%s", bufer, 100);

  if (i)
    return 0;
  else
    return 1;
}

int find_double_def(FILE *f, const char *templ, double &value, double def_value)
// with default value
{
  int i;
  double x;
  char bufer[100];

  rewind(f);
  while (!feof(f) && (i = fscanf(f, templ, &x) <= 0))
      fscanf_s(f, "%s", bufer, 100);

  if (i)
    {
        value = def_value;
        return 0;
    }
  else
    {
        value = x;
        return 1;
    }
}

int find_str(FILE *f, const char *templ, char *str)
// return 1 if there are such string or 0 elsewhere
{
  int i;
  char bufer[100];

  rewind(f);
  while (!feof(f) && (i = fscanf(f, templ, str) <= 0))
      fscanf_s(f, "%s", bufer, 100);

  if (i)
    return 0;
  else
    return 1;
}

double rand01()
// generate random number from 0 to 1
{
   return double(rand() % 10000)/10000;
}
