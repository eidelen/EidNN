#include "neuron.h"
#include <cmath>

float Neuron::sigmoid( const float& z )
{
    return 1.0 / ( 1.0 + std::exp(-z) );
}

float Neuron::d_sigmoid( const float& z )
{
    return sigmoid(z) * ( 1.0 - sigmoid(z) );
}
