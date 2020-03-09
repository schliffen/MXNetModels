#include <iostream>
#include <string>
#include "Hello.h"


void Hello::print()
{
    std::cout << "Hello Headers!" << std::endl;
}
void C_MATRIX::pstate(char  a){
    std::cout<<"name: " << a << std::endl;
}
float C_MATRIX::multiply(float a, float b){
    char ca[9] = "multiply";
    C_MATRIX::pstate(*ca);
    return a*b;

}
