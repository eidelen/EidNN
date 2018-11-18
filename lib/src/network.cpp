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
#include "helpers.h"
#include "crossEntropyCost.h"
#include "quadraticCost.h"

#include <random>
#include <iostream>
#include <fstream>

using namespace std;

Network::Network( const vector<unsigned int> networkStructure ) :
    m_NetworkStructure( networkStructure ), m_oberserver( NULL ), m_asyncOperation{}, m_operationInProgress( false )
{
    initNetwork();
}

Network::Network( const Network& n ) :
    m_NetworkStructure( n.getNetworkStructure() ), m_oberserver( n.m_oberserver ), m_asyncOperation{}, m_operationInProgress( false )
{
    // copy layers
    m_Layers.clear();
    for( unsigned int l = 0; l < n.getNumberOfLayer(); l++ )
    {
        const shared_ptr<const Layer> layer = n.getLayer(l);
        shared_ptr<Layer> cp_layer( new Layer( *(layer.get()) ) );
        m_Layers.push_back( cp_layer );
    }

    m_activation_out = Eigen::MatrixXd( 1, 1 ); // dimension will be updated based on nbr of input samples
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

shared_ptr<const Layer> Network::getLayer(const unsigned int &layerIdx ) const
{
    if( layerIdx >= getNumberOfLayer() )
    {
        cout << "Error: getLayer out of index" << endl;
        return shared_ptr<const Layer>(NULL);
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
    double cost = 0.0;
    size_t nbrOfSamples = samples.size();

    if( samples.size() != lables.size() )
    {
        cout << "Error: number of samples and lables mismatch" << endl;
    }
    else if( nbrOfSamples < batchsize )
    {
        cout << "Error: batchsize exceeds number of available smaples" << endl;
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

            double batchCost = 0.0;
            doStochasticGradientDescentBatch(batch_in, batch_out, eta, batchCost);
            cost += batchCost;

            sendProg2Obs( NetworkOperationCallback::OpStochasticGradientDescent, NetworkOperationCallback::OpInProgress, double(batch)/double(nbrOfBatches) );
        }

        cost = cost / static_cast<double>(nbrOfBatches);

        retValue = true;
    }

    m_operationInProgress = false;

    if( retValue )
    {
        if( m_oberserver != NULL )
        {
            // test network with training data
            double successRateEuclidean; double successRateMaxIdx; double euclideanDistanceThreshold; double avgCost; std::vector<size_t> failedSamples;
            testNetwork( samples, lables, euclideanDistanceThreshold, successRateEuclidean, successRateMaxIdx, avgCost, failedSamples );
            m_oberserver->networkTrainingResults( successRateEuclidean, successRateMaxIdx, avgCost);
        }

        sendProg2Obs(NetworkOperationCallback::OpStochasticGradientDescent, NetworkOperationCallback::OpResultOk, 1.0);
    }
    else
    {
        sendProg2Obs(NetworkOperationCallback::OpStochasticGradientDescent, NetworkOperationCallback::OpResultErr, 1.0);
    }

    return retValue;
}

bool Network::doStochasticGradientDescentBatch(const Eigen::MatrixXd& batch_in, const Eigen::MatrixXd& batch_out, const double& eta, double& cost )
{
    // this feedforwards the whole batch at once
    if( !doFeedforwardAndBackpropagation( batch_in, batch_out ) )
        return false;

    cost = getOutputLayer()->getCost();

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

std::shared_ptr<const Layer> Network::getOutputLayer() const
{
    return getLayer( getNumberOfLayer() - 1 );
}

double Network::getNetworkErrorMagnitude() const
{
    Eigen::MatrixXd oErr =  getOutputLayer()->getBackpropagationError();

    size_t n = oErr.cols();

    if( n == 0 )
        return 0;

    double accumError = 0.0;
    for( size_t i = 0; i < n; i++ )
    {
        accumError += oErr.col(i).norm();
    }

    return accumError / n;
}

double Network::getNetworkCost() const
{
    return getOutputLayer()->getCost();
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

bool Network::testNetworkAsync( const std::vector<Eigen::MatrixXd>& samples, const std::vector<Eigen::MatrixXd>& lables,
                       const double& euclideanDistanceThreshold )
{
    if( !prepareForNextAsynchronousOperation() )
        return false;

    m_asyncOperation = std::thread(&Network::doTestAsync, this,  samples, lables, euclideanDistanceThreshold);
    return true;
}

// this intermediate function is necessary because testNetwork results are passed by reference
void Network::doTestAsync( const std::vector<Eigen::MatrixXd>& samples, const std::vector<Eigen::MatrixXd>& lables,
                           const double& euclideanDistanceThreshold )
{
    double successRateEuclidean; double successRateMaxIdx; double avgCost; std::vector<size_t> failedSamples;
    bool res = testNetwork( samples, lables, euclideanDistanceThreshold, successRateEuclidean, successRateMaxIdx, avgCost, failedSamples );

    m_operationInProgress = false;

    if( m_oberserver != NULL )
    {
        if( res )
        {
            m_oberserver->networkOperationProgress( NetworkOperationCallback::OpTestNetwork, NetworkOperationCallback::OpResultOk, 1.0 );
            m_oberserver->networkTestResults( successRateEuclidean, successRateMaxIdx, avgCost, failedSamples );
        }
        else
        {
            m_oberserver->networkOperationProgress( NetworkOperationCallback::OpTestNetwork, NetworkOperationCallback::OpResultErr, 1.0 );
        }
    }
}

bool Network::testNetwork(  const std::vector<Eigen::MatrixXd>& samples, const std::vector<Eigen::MatrixXd>& lables,
                            const double& euclideanDistanceThreshold, double& successRateEuclideanDistance,
                            double& successRateIdenticalMax, double& avgCost, std::vector<size_t>& failedSamplesIdx )
{
    if( samples.size() != lables.size() )
    {
        cout << "Error: samples and lables size mismatch" << endl;
        return false;
    }

    failedSamplesIdx.clear();

    size_t nbrOfTestSamples = samples.size();
    successRateEuclideanDistance = 0.0; successRateIdenticalMax = 0.0; avgCost = 0.0;

    for( size_t t = 0; t < nbrOfTestSamples; t++ )
    {
        if( !feedForward(samples.at(t)) )
            return false;

        Eigen::MatrixXd outputSignal = getOutputActivation();
        Eigen::MatrixXd expectedSignal = lables.at(t);

        getOutputLayer()->computeBackpropagationOutputLayerError( expectedSignal );
        avgCost += getOutputLayer()->getCost();

        // Test Euclidean distance
        double euclideanDistance = (outputSignal-expectedSignal).norm();
        bool euclideanRequirement = euclideanDistance < euclideanDistanceThreshold;
        if( euclideanRequirement )
            successRateEuclideanDistance = successRateEuclideanDistance + 1.0;

        // Test max elements identical
        unsigned long expected_m, expected_n, out_m, out_n; double maxElem;
        Helpers::maxElement(expectedSignal, expected_m, expected_n, maxElem);
        Helpers::maxElement(outputSignal, out_m, out_n, maxElem);
        bool maxIdenticalRequirement = out_m == expected_m;
        if( maxIdenticalRequirement ) // since vector, both n are anyway 0
            successRateIdenticalMax = successRateIdenticalMax + 1.0;

        // overall failed -> if classification failed
        if( !maxIdenticalRequirement )
            failedSamplesIdx.push_back( t );

        if( t % 10 == 0 ) // send progress only for every 10th sample
            sendProg2Obs( NetworkOperationCallback::OpTestNetwork, NetworkOperationCallback::OpInProgress, double(t)/double(nbrOfTestSamples) );
    }

    avgCost = avgCost / double(nbrOfTestSamples);

    // normalise success rates
    successRateEuclideanDistance = successRateEuclideanDistance / double(nbrOfTestSamples);
    successRateIdenticalMax = successRateIdenticalMax / double(nbrOfTestSamples);

    sendProg2Obs( NetworkOperationCallback::OpTestNetwork, NetworkOperationCallback::OpResultOk, 1.0 );

    return true;
}

string Network::serialize() const
{
    string retBuf;

    size_t nbrOfTopoElements = m_NetworkStructure.size() + 1;
    unsigned int* topoBuf = new unsigned int[ nbrOfTopoElements ];
    topoBuf[0] = getNumberOfLayer();
    for( size_t i = 0; i < m_NetworkStructure.size(); i++ )
        topoBuf[i+1] = m_NetworkStructure.at(i);

    retBuf.append( string( (char*)topoBuf, nbrOfTopoElements*sizeof(unsigned int) ) );

    for( shared_ptr<const Layer> l : m_Layers )
    {
        string lBuf = l->serialize();
        unsigned int lBufSize = lBuf.size();
        retBuf.append( string( (char*)(&lBufSize), 1*sizeof(unsigned int) ) );
        retBuf.append(lBuf);
    }

    delete [] topoBuf;

    return retBuf;
}

Network* Network::deserialize( const string& buffer )
{
    const char* buf = buffer.c_str();

    unsigned int nbrOfLayers = ((unsigned int*)buf)[0];
    std::vector<unsigned int> networkStructure;
    for( unsigned int i = 0; i < nbrOfLayers; i++ )
        networkStructure.push_back( ((unsigned int*)buf)[i+1] );

    size_t offset = (nbrOfLayers+1) * sizeof(unsigned int);

    Network* n = new Network( networkStructure );

    for( unsigned int i = 0; i < nbrOfLayers; i++ )
    {
        const char* layerBuf = buf + offset;

        unsigned int sizeOfThisLayer = ((unsigned int*)layerBuf)[0];
        string layerData( layerBuf + sizeof(unsigned int), sizeOfThisLayer );
        Layer* l = Layer::deserialize( layerData );

        n->getLayer(i)->setBiases( l->getBiasVector() );
        n->getLayer(i)->setWeights( l->getWeigtMatrix() );
        n->getLayer(i)->setLayerType( l->getLayerType() );

        delete l;

        offset = offset + sizeOfThisLayer + sizeof(unsigned int);
    }

    return n;
}


bool Network::save( const string& filePath )
{
    ofstream netFile;
    netFile.open( filePath );

    if( ! netFile.is_open() )
        return false;

    netFile << serialize();
    netFile.close();
    return true;
}


Network* Network::load( const string& filePath )
{
    ifstream netFile( filePath );
    if( ! netFile.is_open() )
        return NULL;

    // read the whole file
    std::string netAsBuffer((std::istreambuf_iterator<char>(netFile)),
                             std::istreambuf_iterator<char>());

    netFile.close();

    return Network::deserialize( netAsBuffer );
}

void Network::setCostFunction( const ECostFunction& function )
{
    std::shared_ptr<CostFunction> cf;
    if( function == CrossEntropy )
        cf.reset( new CrossEntropyCost() );
    else
        cf.reset( new QuadraticCost() );

    getOutputLayer()->setCostFunction( cf );
}

void Network::setSoftmaxOutput( const bool& enable )
{
    if( enable )
        getOutputLayer()->setLayerType(Layer::Softmax);
    else
        getOutputLayer()->setLayerType(Layer::Sigmoid);
}

bool Network::isSoftmaxOutputEnabled() const
{
    return getOutputLayer()->getLayerType() == Layer::Softmax;
}
