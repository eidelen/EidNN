/****************************************************************************
** Copyright (c) 2017 Adrian Schneider
**
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and associated documentation files (the "Software"),
** to deal in the Software without restriction, including without limitation
** the rights to use, copy, modify, merge, publish, distribute, sublicense,
** and/or sell copies of the Software, and to permit persons to whom the
** Software is furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
** FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
** DEALINGS IN THE SOFTWARE.
**
*****************************************************************************/

#include "neuron.h"
#include <cmath>
#include <iostream>
#include <random>

Neuron::Neuron( const unsigned int& nbr_of_input ) :
    m_nbr_of_inputs( nbr_of_input )
{
    m_weights = Eigen::VectorXf(nbr_of_input);
    m_bias = 0;

    setRandomWeights( 0.0, 1.0 );
    setRandomBias( 0.0, 1.0 );
}

Neuron::~Neuron()
{

}

bool Neuron::feedForward(const Eigen::VectorXf& x_in , float& z, float &activation)
{
    if( x_in.rows() != m_nbr_of_inputs )
    {
        std::cout << "Error: Input vector size mismatch" << std::endl;
        return false;
    }

    z = x_in.dot(m_weights) + m_bias;
    activation = sigmoid(z);

    return true;
}

bool Neuron::setWeights( const Eigen::VectorXf& weights )
{
    if( weights.rows() != m_nbr_of_inputs )
    {
        std::cout << "Error: Weight vector size mismatch" << std::endl;
        return false;
    }

    m_weights = weights;
    return true;
}

void Neuron::setRandomWeights( const float& mean, const float& deviation )
{
    std::default_random_engine randomGenerator;
    std::normal_distribution<float> gNoise(mean, deviation);

    for( unsigned int i = 0; i < m_weights.rows(); i++ )
    {
        m_weights(i) = gNoise(randomGenerator);
    }
}

void Neuron::setBias( const float& bias )
{
    m_bias = bias;
}

void Neuron::setRandomBias( const float& mean, const float& deviation )
{
    std::default_random_engine randomGenerator;
    std::normal_distribution<float> gNoise(mean, deviation);

    setBias( gNoise(randomGenerator) );
}

float Neuron::sigmoid( const float& z )
{
    return 1.0 / ( 1.0 + std::exp(-z) );
}

float Neuron::d_sigmoid( const float& z )
{
    return sigmoid(z) * ( 1.0 - sigmoid(z) );
}
