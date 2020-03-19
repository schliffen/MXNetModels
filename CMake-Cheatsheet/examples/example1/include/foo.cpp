#include "foo.hpp"
#include <iostream>

double foo::hello( double s){
    double d = s *s + 1;
    std::cout << "the value of d: "<< d << std::endl;
    return d;
}

double foo2::divisor(double a, double b){
   //pname( "divisor" );

   if (b == 0 )
     return a/b ;
   else
     return 1/(b+.00001); 
}