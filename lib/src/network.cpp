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

#include "network.h"
#include "layer.h"

#include <random>
#include <iostream>

using namespace std;

Network::Network( const vector<unsigned int> networkStructure ) :
    m_NetworkStructure( networkStructure ), m_oberserver( NULL ), m_asyncOperation{}, m_operationInProgress( false )
{
    initNetwork();
}

Network::~Network()
{
    if( m_asyncOperation.joinable() )
        m_asyncOperation.detach();
}

void Network::initNetwork()
{
    unsigned int nbrOfInputs = 0; // for input layer, there is no input needed.

    for( unsigned int nbrOfNeuronsInLayer : m_NetworkStructure )
    {
        m_Layers.push_back( shared_ptr<Layer>( new Layer(nbrOfNeuronsInLayer, nbrOfInputs) ) );
        nbrOfInputs = nbrOfNeuronsInLayer; // the next layer has same number of inputs as neurons in this layer.
    }

    m_activation_out = Eigen::MatrixXd( 1, 1 ); // dimension will be updated based on nbr of input samples
}

bool Network::feedForward( const Eigen::MatrixXd& x_in )
{
    // first layer does not perform any operation. It's activation output is just x_in.
    if( ! getLayer(0)->setActivationOutput(x_in) )
        return false;

    for( unsigned int k = 1; k < m_Layers.size(); k++ )
    {
        // Pass output signal from former layer to next layer.
        if( ! getLayer( k )->feedForward(  getLayer( k-1 )->getOutputActivation()  ) )
        {
            cout << "Error: Outpt-Input signal size mismatch" << endl;
            return false;
        }
    }

    // network output signal is in the last layer
    m_activation_out = m_Layers.back()->getOutputActivation();

    return true;
}

unsigned int Network::getNumberOfLayer() const
{
    return unsigned(m_NetworkStructure.size());
}

shared_ptr<Layer> Network::getLayer( const unsigned int& layerIdx )
{
    if( layerIdx >= getNumberOfLayer() )
    {
        cout << "Error: getLayer out of index" << endl;
        return shared_ptr<Layer>(NULL);
    }

    return m_Layers.at(layerIdx);
}

bool Network::gradientDescent( const Eigen::MatrixXd& x_in, const Eigen::MatrixXd& y_out, const double& eta )
{
    if( ! doFeedforwardAndBackpropagation(x_in, y_out ) )
        return false;

    // Update weights and biases with the computed derivatives and learning rate.
    // First layer does not need to be updated -> it is just input layer
    for( unsigned int k = 1; k < getNumberOfLayer(); k++ )
        getLayer(k)->updateWeightsAndBiases( eta );

    return true;
}

bool Network::stochasticGradientDescentAsync(const std::vector<Eigen::MatrixXd> &samples, const std::vector<Eigen::MatrixXd> &lables,
                                             const unsigned int& batchsize, const double& eta)
{
    if( !prepareForNextAsynchronousOperation() )
        return false;

    m_asyncOperation = std::thread(&Network::stochasticGradientDescent, this,  samples, lables, batchsize, eta);
    return true;
}

bool Network::stochasticGradientDescent(const std::vector<Eigen::MatrixXd>& samples, const std::vector<Eigen::MatrixXd>& lables,
                                        const unsigned int& batchsize, const double& eta)
{    
    bool retValue = false;
    size_t nbrOfSamples = samples.size();

    if( samples.size() != lables.size() )
    {
        cout << "Error: number of samples and lables mismatch" << endl;
        sendProg2Obs( NetworkOperationCallback::OpStochasticGradientDescent, NetworkOperationCallback::OpResultErr, 1.0 );
    }
    else if( nbrOfSamples < batchsize )
    {
        cout << "Error: batchsize exceeds number of available smaples" << endl;
        sendProg2Obs( NetworkOperationCallback::OpStochasticGradientDescent, NetworkOperationCallback::OpResultErr, 1.0 );
    }
    else
    {
        // one epoch
        unsigned long nbrOfBatches = nbrOfSamples / batchsize;

        // random generator
        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_int_distribution<> iDist(0, int(nbrOfSamples)-1);

        Eigen::MatrixXd batch_in( samples.at(0).rows(), batchsize );
        Eigen::MatrixXd batch_out( lables.at(0).rows(), batchsize );

        for( unsigned int batch = 0; batch < nbrOfBatches; batch++ )
        {
            // generate a random sample set
            for( unsigned int b = 0; b < batchsize; b++ )
            {
                size_t rIdx = size_t( iDist(e2) );
                batch_in.col(b) = samples.at(rIdx);
                batch_out.col(b) = lables.at(rIdx);
            }

            doStochasticGradientDescentBatch(batch_in, batch_out, eta);
            sendProg2Obs( NetworkOperationCallback::OpStochasticGradientDescent, NetworkOperationCallback::OpInProgress, double(batch)/double(nbrOfBatches) );
        }

        retValue = true;
        sendProg2Obs( NetworkOperationCallback::OpStochasticGradientDescent, NetworkOperationCallback::OpResultOk, 1.0 );
    }

    m_operationInProgress = false;
    return retValue;
}

bool Network::doStochasticGradientDescentBatch(const Eigen::MatrixXd& batch_in, const Eigen::MatrixXd& batch_out, const double& eta )
{
    // this feedforwards the whole batch at once
    if( !doFeedforwardAndBackpropagation( batch_in, batch_out ) )
        return false;

    long batchsize = batch_in.cols();

    // compute average partial derivatives over all samples in all layers
    for( unsigned int j = 1; j < getNumberOfLayer(); j++ )
    {
        const std::shared_ptr<Layer>& l = getLayer(j);

        Eigen::MatrixXd biasSum = Eigen::MatrixXd::Constant( l->getNbrOfNeurons(), 1, 0.0 );
        Eigen::MatrixXd weightSum = Eigen::MatrixXd::Constant( l->getNbrOfNeurons(), l->getNbrOfNeuronInputs(), 0.0 );

        vector<Eigen::MatrixXd> pd_biases = l->getPartialDerivativesBiases();
        vector<Eigen::MatrixXd> pd_weigths = l->getPartialDerivativesWeights();

        for( unsigned int k = 0; k < pd_biases.size(); k++ )
        {
            biasSum = biasSum + pd_biases.at(k);
            weightSum = weightSum + pd_weigths.at(k);
        }

        // update weights and biases in layer
        Eigen::MatrixXd avgPDBias = biasSum * ( eta / double(batchsize) );
        Eigen::MatrixXd avgPDWeights = weightSum * ( eta / double(batchsize) );
        l->updateWeightsAndBiases(avgPDBias, avgPDWeights);
    }

    return true;
}

shared_ptr<Layer> Network::getOutputLayer()
{
    return getLayer( getNumberOfLayer() - 1 );
}

double Network::getNetworkErrorMagnitude()
{
    Eigen::VectorXd oErr =  getOutputLayer()->getBackpropagationError();
    return oErr.norm();
}

void Network::print()
{
    // skip first layer -> input
    for( unsigned int i = 1; i < getNumberOfLayer(); i++ )
    {
        shared_ptr<Layer> l = getLayer( i );

        std::cout << "Layer " << i << ":" << std::endl;
        getLayer( i )->print();
        std::cout << std::endl << std::endl;
    }
}

bool Network::doFeedforwardAndBackpropagation( const Eigen::MatrixXd& x_in, const Eigen::MatrixXd& y_out )
{
    // updates output in all layers
    if( ! feedForward(x_in) )
        return false;

    if( getOutputActivation().rows() != y_out.rows() )
    {
        cout << "Error: desired output signal mismatching dimension" << endl;
        return false;
    }

    // Compute output error in the last layer
    std::shared_ptr<Layer> layerAfter = getOutputLayer();
    layerAfter->computeBackpropagationOutputLayerError( y_out );
    layerAfter->computePartialDerivatives();

    // Compute error and partial derivatives in all remaining layers, but not input layer
    for( int k = int(getNumberOfLayer()) - 2; k > 0; k-- )
    {
        std::shared_ptr<Layer> thisLayer = getLayer( unsigned(k) );
        thisLayer->computeBackprogationError( layerAfter->getBackpropagationError(), layerAfter->getWeigtMatrix() );
        thisLayer->computePartialDerivatives();

        layerAfter = thisLayer;
    }

    return true;
}

void Network::sendProg2Obs( const NetworkOperationCallback::NetworkOperationId& opId,
                                      const NetworkOperationCallback::NetworkOperationStatus& opStatus,
                                      const double& progress  )
{
    if( m_oberserver != NULL )
        m_oberserver->networkOperationProgress( opId, opStatus, progress );
}

bool Network::prepareForNextAsynchronousOperation()
{
    if( isOperationInProgress() )
    {
        cout << "Error: Asynchronous operation already in progress" << endl;
        return false;
    }

    m_operationInProgress = true;
    if( m_asyncOperation.joinable() ) // Join former thread if done yet
        m_asyncOperation.join();

    return true;
}
