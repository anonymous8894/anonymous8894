


#ifndef __STDLIB__HEADER
#define __STDLIB__HEADER 


#include "_fake_defines.h"
#include "_fake_typedefs.h"

void *malloc(unsigned long size);
void* calloc(unsigned long num, unsigned long size);
void free(void *ptr);
int abs(int x);
void exit(int status);

#endif