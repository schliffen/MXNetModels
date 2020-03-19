//
// Created by Morten Nobel-Jørgensen on 01/09/2017.
//
#include<iostream>
#include "foo.hpp"

int main(){
    double inp=2.1;
 double res = foo::hello(inp);
 int a = 4;   
 std::cout<< a *res << " testing. \n";
 
 std::cout<< "operator plus" << operator +(3,3) << std::endl;
 
    return 0;
} 