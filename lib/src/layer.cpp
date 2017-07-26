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
#include "helpers.h"
#include "costFunction.h"
#include "quadraticCost.h"

using namespace std;

Layer::Layer(const uint& nbr_of_neurons , const uint &nbr_of_inputs, const LayerOutputType& type) :
    m_nbr_of_neurons( nbr_of_neurons ),
    m_nbr_of_inputs( nbr_of_inputs ),
    m_layer_type(type)
{
    initLayer();
}

Layer::Layer( const uint& nbr_of_inputs, const vector<Eigen::VectorXd>& weights, const vector<double>& biases, const LayerOutputType& type ) :
    Layer::Layer( uint(weights.size()), nbr_of_inputs, type )
{
    assert( weights.size() ==  biases.size() );

    // write weight matrix and bias vector
    for( unsigned int n = 0; n < weights.size(); n++ )
    {
        m_weightMatrix.row(n) = weights.at(n).transpose();
        m_biasVector(n,0) = biases.at(n);
    }
}

Layer::Layer( const Layer& l ) : Layer( l.getNbrOfNeurons(), l.getNbrOfNeuronInputs(), l.getLayerType() )
{
    // Note: Temporary results like activations and derivatives are not copied.
    m_weightMatrix = l.getWeigtMatrix();
    m_biasVector = l.getBiasVector();
}


// init vectors and neurons
void Layer::initLayer()
{
    m_weightMatrix = Eigen::MatrixXd( m_nbr_of_neurons , m_nbr_of_inputs );
    m_biasVector = Eigen::MatrixXd( m_nbr_of_neurons, 1 );
    resetRandomlyWeightsAndBiases();

    // init with size 1 -> dimensionso of these matrices will change corrsponding to input signal
    m_activation_in = Eigen::MatrixXd( 1, 1 );
    m_activation_out = Eigen::MatrixXd( 1, 1 );
    m_z_weighted_input = Eigen::MatrixXd( 1, 1 );
    m_backpropagationError =  Eigen::MatrixXd( 1 , 1 );

    m_costFunction.reset( new QuadraticCost() );
}


Layer::~Layer()
{
    // used smart pointers
}

bool Layer::feedForward(const Eigen::MatrixXd &x_in )
{
    if( x_in.rows() != m_nbr_of_inputs )
    {
        std::cout << "Error: Layer input vector size mismatch" << std::endl;
        return false;
    }

    m_activation_in = x_in;
    m_z_weighted_input = m_weightMatrix * x_in + m_biasVector.replicate(1, x_in.cols());
    m_activation_out = Eigen::MatrixXd(m_z_weighted_input.rows(), m_z_weighted_input.cols());

    if( m_layer_type == Sigmoid )
    {
        // compute sigmoid of weighted input matrix
        for( unsigned int m = 0; m < m_z_weighted_input.rows(); m++ )
            for( unsigned int n = 0; n < m_z_weighted_input.cols(); n++ )
                m_activation_out(m,n) = Neuron::sigmoid( m_z_weighted_input(m,n) );
    }
    else if( m_layer_type == Softmax )
    {
        // compute softmax of weighted input matrix
        Eigen::MatrixXd expZ = (m_z_weighted_input.array().exp()).matrix();
        Eigen::MatrixXd expSums = expZ.colwise().sum();

        for( unsigned int n = 0; n < m_z_weighted_input.cols(); n++ ) // each sample
            for( unsigned int m = 0; m < m_z_weighted_input.rows(); m++ ) // each neuron
                m_activation_out(m,n) = expZ(m,n) / expSums(0,n);
    }

    return true;
}


bool Layer::setWeights( const vector<Eigen::VectorXd>& weights )
{
    if( weights.size() != getNbrOfNeurons() )
    {
        std::cout << "Error: Weights vector size mismatches number of neurons" << std::endl;
        return false;
    }

    for( unsigned int n = 0; n < weights.size(); n++ )
        m_weightMatrix.row(n) = weights.at(n).transpose();

    return true;
}

bool Layer::setWeights( const Eigen::MatrixXd& weights )
{
    if( weights.rows() != m_weightMatrix.rows() || weights.cols() != m_weightMatrix.cols() )
    {
        std::cout << "Error: Weights matrix size mismatches" << std::endl;
        return false;
    }

    m_weightMatrix = weights;
    return true;
}

bool Layer::setBiases( const vector<double>& biases )
{
    if( biases.size() != getNbrOfNeurons() )
    {
        std::cout << "Error: Bias vector size mismatches number of neurons" << std::endl;
        return false;
    }

    for( unsigned int n = 0; n < biases.size(); n++ )
        m_biasVector(n,0) = biases.at(n);

    return true;
}

bool Layer::setBiases( const Eigen::MatrixXd& biases )
{
    if( biases.rows() != getNbrOfNeurons() || biases.cols() != 1 )
    {
        std::cout << "Error: Bias Eigen vector size mismatches" << std::endl;
        return false;
    }

    m_biasVector = biases;

    return true;
}

void Layer::setWeight( const double& weight )
{
    Eigen::VectorXd uniformWeight = Eigen::VectorXd::Constant(getNbrOfNeuronInputs(), weight);

    for( unsigned int n = 0; n < getNbrOfNeurons(); n++ )
        m_weightMatrix.row(n) = uniformWeight.transpose();
}


void Layer::setBias(const double &bias )
{
    m_biasVector = Eigen::MatrixXd::Constant(getNbrOfNeurons(), 1, bias);
}


void Layer::resetRandomlyWeightsAndBiases()
{
    std::default_random_engine randomGenerator;
    std::normal_distribution<double> gNoise(0.0, 1.0);

    for( unsigned int i = 0; i < getNbrOfNeurons(); i++ )
    {
        m_biasVector(i,0) = gNoise(randomGenerator);

        Eigen::VectorXd thisWeights = Eigen::VectorXd( getNbrOfNeuronInputs() );
        for( unsigned int k = 0; k < getNbrOfNeuronInputs(); k++ )
            thisWeights(k) = gNoise(randomGenerator);

        m_weightMatrix.row(i) = thisWeights.transpose();
    }
}


bool Layer::setActivationOutput( const Eigen::MatrixXd& activation_out )
{
    if( activation_out.rows() != getNbrOfNeurons() )
    {
        std::cout << "Error: Activation output signal mismatch" << std::endl;
        return false;
    }

    m_activation_out = activation_out;
    return true;
}

bool Layer::computeBackpropagationOutputLayerError(const Eigen::MatrixXd &expectedNetworkOutput )
{
    if( m_activation_out.rows() != expectedNetworkOutput.rows() ||
            m_activation_out.cols() != expectedNetworkOutput.cols())
    {
        std::cout << "Error: Layer activation output to label mismatch" << std::endl;
        return false;
    }

    if( m_layer_type == Sigmoid )
        m_backpropagationError = m_costFunction->delta(m_z_weighted_input, m_activation_out, expectedNetworkOutput );
    else if( m_layer_type == Softmax )
        m_backpropagationError = m_activation_out - expectedNetworkOutput;

    return true;
}

bool Layer::computeBackprogationError(const Eigen::MatrixXd &errorNextLayer, const Eigen::MatrixXd& weightMatrixNextLayer )
{
    if( m_z_weighted_input.rows() != weightMatrixNextLayer.cols()  ||  errorNextLayer.rows() != weightMatrixNextLayer.rows() )
    {
        std::cout << "Error: computeBackprogationError Layer dimension mismatch" << std::endl;
        return false;
    }

    m_backpropagationError = ((weightMatrixNextLayer.transpose() * errorNextLayer).array() * Neuron::d_sigmoid( m_z_weighted_input ).array()).matrix();
    return true;
}

void Layer::computePartialDerivatives()
{
    Eigen::MatrixXd delta = getBackpropagationError();

    m_bias_partialDerivatives.clear();
    m_weight_partialDerivatives.clear();

    // compute derivatives for each passed sample
    for( unsigned int k = 0; k < delta.cols(); k++ )
    {
        Eigen::MatrixXd thisDelta = delta.col(k);
        m_bias_partialDerivatives.push_back( thisDelta );

        Eigen::MatrixXd thisInputActivation = getInputActivation().col(k);
        m_weight_partialDerivatives.push_back( delta.col(k) * thisInputActivation.transpose() ); // This is different from the 4th-equation? Study!
    }
}

void Layer::updateWeightsAndBiases(const double &eta, const unsigned int& sampleIdx )
{
    updateWeightsAndBiases(eta * getPartialDerivativesBiases().at(sampleIdx), eta * getPartialDerivativesWeights().at(sampleIdx) );
}

void Layer::updateWeightsAndBiases(const Eigen::MatrixXd& deltaBias, const Eigen::MatrixXd& deltaWeight)
{

    const Eigen::MatrixXd newBiases = getBiasVector() - deltaBias;
    setBiases( newBiases );

    const Eigen::MatrixXd newWeights = getWeigtMatrix() - deltaWeight;
    setWeights( newWeights );
}

void Layer::print() const
{
    Eigen::VectorXd a;

    Helpers::printVector(getBiasVector(),"Biases");
    Helpers::printMatrix(getWeigtMatrix(),"Weights");
    Helpers::printVector(getBackpropagationError(),"Error");
}

std::string Layer::serialize( ) const
{
    unsigned int* topoBuf = new unsigned int[3];
    topoBuf[0] = m_nbr_of_neurons;
    topoBuf[1] = m_nbr_of_inputs;
    topoBuf[2] = static_cast<unsigned int>(m_layer_type);

    size_t nbrOfDoublesWeightMatrix = m_nbr_of_neurons * m_nbr_of_inputs; // + m_nbr_of_neurons;
    double* weightBuf = new double[ nbrOfDoublesWeightMatrix ];
    for( size_t m = 0; m < m_nbr_of_neurons; m++ )
        for( size_t n = 0; n < m_nbr_of_inputs; n++ )
            weightBuf[ m*m_nbr_of_inputs + n ] = m_weightMatrix( long(m), long(n) );

    size_t nbrOfDoublesBias = m_nbr_of_neurons;
    double* biasBuf = new double[ nbrOfDoublesBias ];
    for( size_t m = 0; m < nbrOfDoublesBias; m++ )
        biasBuf[ m ] = m_biasVector( long(m), 0 );

    string retBuffer;
    retBuffer.append( string( (char*)topoBuf, 3*sizeof(unsigned int) ) );
    retBuffer.append( string( (char*)weightBuf, nbrOfDoublesWeightMatrix*sizeof(double) ) );
    retBuffer.append( string( (char*)biasBuf, nbrOfDoublesBias*sizeof(double) ) );

    delete[] topoBuf;
    delete[] weightBuf;
    delete[] biasBuf;

    return retBuffer;
}

Layer* Layer::deserialize( const string& buffer )
{
    const char* buf = buffer.c_str();

    unsigned int nbrOfNeurons = ((unsigned int*)(buf))[0];
    unsigned int nbrOfInputs = ((unsigned int*)(buf))[1];
    Layer::LayerOutputType lType = static_cast<Layer::LayerOutputType>(((unsigned int*)(buf))[2]);

    size_t offset = 3 * sizeof(unsigned int);

    Eigen::MatrixXd weightMatrix = Eigen::MatrixXd( nbrOfNeurons , nbrOfInputs );
    const double* weightBuf = (const double*)((buf + offset));
    for( size_t m = 0; m < nbrOfNeurons; m++ )
        for( size_t n = 0; n < nbrOfInputs; n++ )
            weightMatrix( long(m), long(n) ) = weightBuf[ m*nbrOfInputs + n ] ;

    offset = offset + nbrOfNeurons*nbrOfInputs*sizeof(double);

    Eigen::MatrixXd biasVector = Eigen::MatrixXd( nbrOfNeurons, 1 );
    const double* biasBuf = (const double*)((buf + offset));
    for( size_t m = 0; m < nbrOfNeurons; m++ )
        biasVector( long(m), 0 ) = biasBuf[ m ];

    Layer* l = new Layer( nbrOfNeurons, nbrOfInputs, lType );
    l->setBiases( biasVector );
    l->setWeights( weightMatrix );

    return l;
}

Layer::LayerOutputType Layer::getLayerType() const
{
    return m_layer_type;
}

void Layer::setLayerType( const LayerOutputType& type)
{
    m_layer_type = type;
}



