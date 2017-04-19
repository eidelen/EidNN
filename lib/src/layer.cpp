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

    m_activation_in = x_in;

    updateWeightMatrixAndBiasVector(); // updates m_weightMatrix and m_biasVector

    m_z_weighted_input = m_weightMatrix * x_in + m_biasVector;

    // compute sigmoid for weighted input vector
    for( unsigned int n = 0; n < m_z_weighted_input.rows(); n++ )
    {
        m_activation_out(n) = Neuron::sigmoid(m_z_weighted_input(n));
    }

    return true;
}

// Neuron wise computation
bool Layer::feedForwardSlow( const Eigen::VectorXf& x_in )
{
    if( x_in.rows() != m_nbr_of_inputs )
    {
        std::cout << "Error: Layer input vector size mismatch" << std::endl;
        return false;
    }

    m_activation_in = x_in;

    // feed input to every neuron in layer
    for( unsigned int n = 0; n < m_neurons.size(); n++ )
    {
        shared_ptr<Neuron> nr = m_neurons.at(n);
        float z; float activation;
        nr->feedForward( x_in, z, activation );

        m_activation_out(n) = activation;
        m_z_weighted_input(n) = z;
    }

    return true;
}


// init vectors and neurons
void Layer::initLayer()
{
    for( unsigned int n = 0; n < m_nbr_of_neurons; n++ )
        m_neurons.push_back( shared_ptr<Neuron>( new Neuron( m_nbr_of_inputs ) ) );

    m_activation_in = Eigen::VectorXf( m_nbr_of_inputs );
    m_activation_out = Eigen::VectorXf( m_nbr_of_neurons );
    m_z_weighted_input = Eigen::VectorXf( m_nbr_of_neurons );
    m_weightMatrix = Eigen::MatrixXf( m_nbr_of_neurons , m_nbr_of_inputs );
    m_biasVector = Eigen::VectorXf( m_nbr_of_neurons );
    m_bias_partialDerivatives = Eigen::VectorXf( m_nbr_of_neurons );
    m_weight_partialDerivatives = Eigen::MatrixXf( m_nbr_of_neurons , m_nbr_of_inputs );
}


bool Layer::setWeights( const vector<Eigen::VectorXf>& weights )
{
    if( weights.size() != getNbrOfNeurons() )
    {
        std::cout << "Error: Weights vector size mismatches number of neurons" << std::endl;
        return false;
    }

    for( unsigned int k = 0; k < getNbrOfNeurons(); k++ )
        if( ! m_neurons.at(k)->setWeights( weights.at(k) ) )
            return false;

    updateWeightMatrixAndBiasVector();

    return true;
}

bool Layer::setBiases( const vector<float>& biases )
{
    if( biases.size() != getNbrOfNeurons() )
    {
        std::cout << "Error: Bias vector size mismatches number of neurons" << std::endl;
        return false;
    }

    for( unsigned int k = 0; k < getNbrOfNeurons(); k++ )
        m_neurons.at(k)->setBias( biases.at(k) );

    return true;
}

shared_ptr<Neuron> Layer::getNeuron( const unsigned int& nIdx )
{
    if( nIdx >= m_neurons.size() )
    {
        std::cout << "Error: Neuron index exceeds vector" << std::endl;
        return std::shared_ptr<Neuron>(NULL);
    }

    return m_neurons.at(nIdx);
}

void Layer::setWeight( const float& weight )
{
    Eigen::VectorXf uniformWeight = Eigen::VectorXf::Constant(getNbrOfNeuronInputs(), weight);

    for( shared_ptr<Neuron>& neuron : m_neurons )
        neuron->setWeights( uniformWeight );

    updateWeightMatrixAndBiasVector();
}

void Layer::setBias( const float& bias )
{
    for( shared_ptr<Neuron>& neuron : m_neurons )
        neuron->setBias( bias );

    updateWeightMatrixAndBiasVector();
}

void Layer::resetRandomlyWeightsAndBiases()
{
    for( shared_ptr<Neuron>& neuron : m_neurons )
    {
        neuron->setRandomWeights( 0.0, 1.0 );
        neuron->setRandomBias( 0.0, 1.0 );
    }

    updateWeightMatrixAndBiasVector();
}


bool Layer::setActivationOutput( const Eigen::VectorXf& activation_out )
{
    if( activation_out.rows() != m_activation_out.rows() )
    {
        std::cout << "Error: Layer activation output mismatch" << std::endl;
        return false;
    }

    m_activation_out = activation_out;
    return true;
}

bool Layer::computeBackpropagationOutputLayerError( const Eigen::VectorXf& expectedNetworkOutput )
{
    if( m_activation_out.rows() != expectedNetworkOutput.rows() )
    {
        std::cout << "Error: Layer activation output to label mismatch" << std::endl;
        return false;
    }

    m_backpropagationError = ((m_activation_out - expectedNetworkOutput).array() * d_sigmoid( m_z_weighted_input ).array()).matrix();
    return true;
}

bool Layer::computeBackprogationError( const Eigen::VectorXf& errorNextLayer, const Eigen::MatrixXf& weightMatrixNextLayer )
{
    if( m_z_weighted_input.rows() != weightMatrixNextLayer.cols()  ||  errorNextLayer.rows() != weightMatrixNextLayer.rows() )
    {
        std::cout << "Error: computeBackprogationError Layer dimension mismatch" << std::endl;
        return false;
    }

    m_backpropagationError = ((weightMatrixNextLayer.transpose() * errorNextLayer).array() * d_sigmoid( m_z_weighted_input ).array()).matrix();
    return true;
}


const Eigen::VectorXf Layer::d_sigmoid( const Eigen::VectorXf& z )
{
    unsigned int nbrOfComponents = z.rows();
    Eigen::VectorXf res = Eigen::VectorXf( nbrOfComponents );

    for( unsigned int k = 0; k < nbrOfComponents; k++ )
        res(k) = Neuron::d_sigmoid( z(k) );

    return res;
}

void Layer::updateWeightMatrixAndBiasVector()
{
    // assemble weight matrix based on weights in all neurons
    for( unsigned int k = 0; k < m_nbr_of_neurons; k++ )
    {
        std::shared_ptr<Neuron> kNeuron = getNeuron(k);
        m_weightMatrix.row(k) = kNeuron->getWeights().transpose();
        m_biasVector(k) = kNeuron->getBias();
    }
}


void Layer::computePartialDerivatives()
{
    Eigen::VectorXf error = getBackpropagationError();

    // bias
    m_bias_partialDerivatives = error;

    // weights
    Eigen::VectorXf activation_in = getInputActivation(); 
    m_weight_partialDerivatives = error * activation_in.transpose();
}

void Layer::updateWeightsAndBiases()
{
    const Eigen::VectorXf partialDerivativesBiases = getBackpropagationError();
}


