
#include <iostream>
#include <math.h>
#include  "stest.hpp"
void geometry::prnt_name(char *a){
   std::cout<< "I am in: " << a <<std::endl;
}

double geometry::circle( double r){

   prnt_name( "circle area");

   return  M_1_PI * r *r;
}