#include "cnn_top.h"
#include <iostream>
#include <cstdlib>

int main() {
    data_t input[3][IMG_SIZE][IMG_SIZE];
    data_t output[NUM_CLASSES];

    // Random input for testing
    for(int c=0;c<3;c++)
        for(int i=0;i<IMG_SIZE;i++)
            for(int j=0;j<IMG_SIZE;j++)
                input[c][i][j] = (rand()%256)/255.0;  // simulate normalized image

    cnn_top(input, output);

    std::cout << "CNN output:\n";
    for(int i=0;i<NUM_CLASSES;i++)
        std::cout << "Class " << i << ": " << (float)output[i] << "\n";

    return 0;
}