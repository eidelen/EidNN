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

#include <iostream>
#include "layer.h"
#include "neuron.h"

using namespace std;

Layer::Layer(const uint& nbr_of_neurons , const uint &nbr_of_inputs) :
    m_nbr_of_neurons( nbr_of_neurons ),
    m_nbr_of_inputs( nbr_of_inputs )
{
    initLayer();
}

Layer::Layer( const uint& nbr_of_inputs, const vector<Eigen::VectorXf>& weights, const vector<float>& biases ) :
    Layer::Layer( weights.size(), nbr_of_inputs )
{
    assert( weights.size() ==  biases.size() );

    for( unsigned int n = 0; n < m_neurons.size(); n++ )
    {
        shared_ptr<Neuron> nr = m_neurons.at(n);
        nr->setWeights( weights.at(n) );
        nr->setBias( biases.at(n) );
    }
}

Layer::~Layer()
{
    // used smart pointers
}

bool Layer::feedForward( const Eigen::VectorXf& x_in )
{
    if( x_in.rows() != m_nbr_of_inputs )
    {
        std::cout << "Error: Layer input vector size mismatch" << std::endl;
        return false;
    }

    // feed input to every neuron in layer
    for( unsigned int n = 0; n < m_neurons.size(); n++ )
    {
        shared_ptr<Neuron> nr = m_neurons.at(n);
        float z; float activation;
        nr->feedForward( x_in, z, activation );

        m_activation_out(n) = activation;
        m_z_weighted_input(n) = z;
    }
}

// init vectors and neurons
void Layer::initLayer()
{
    for( unsigned int n = 0; n < m_nbr_of_neurons; n++ )
        m_neurons.push_back( shared_ptr<Neuron>( new Neuron( m_nbr_of_inputs ) ) );

    m_activation_out = Eigen::VectorXf( m_nbr_of_neurons );
    m_z_weighted_input = Eigen::VectorXf( m_nbr_of_neurons );
}
