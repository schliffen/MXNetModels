#include <iostream>
#include <string>
#include "../src/Hello.h"

using namespace std;


int main(int argc, char *argv[])
{
    Hello hi;
    C_MATRIX cb;
    hi.print();
    float ab = cb.multiply(1.2, 3.1);

    cout<<" multiplication: "<< ab<< endl;

    return 0;
}