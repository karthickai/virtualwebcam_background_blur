//
// Created by karthick on 10/09/20.
//

#include <iostream>
#include <tensorflow/c/c_api.h>

int main() {
    std::cout << "TensorFlow library version " << TF_Version() << std::endl;
    return 0;
}
