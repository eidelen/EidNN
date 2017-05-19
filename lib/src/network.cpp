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

#include <iostream>

using namespace std;

Network::Network( const vector<unsigned int> networkStructure ) :
    m_NetworkStructure( networkStructure )
{
    initNetwork();
}

Network::~Network()
{

}

void Network::initNetwork()
{
    unsigned int nbrOfInputs = 0; // for input layer, there is no input needed.

    for( unsigned int nbrOfNeuronsInLayer : m_NetworkStructure )
    {
        m_Layers.push_back( shared_ptr<Layer>( new Layer(nbrOfNeuronsInLayer, nbrOfInputs) ) );
        nbrOfInputs = nbrOfNeuronsInLayer; // the next layer has same number of inputs as neurons in this layer.
    }

    m_activation_out = Eigen::VectorXf( m_NetworkStructure.back() ); // output activation vector has the size of the last layer (output layer)
}

bool Network::feedForward( const Eigen::VectorXf& x_in )
{
    // first layer does not perform any operation. It's activation output is just x_in.
    if( ! getLayer(0)->setActivationOutput(x_in) )
    {
        cout << "Error: Input signal size mismatch" << endl;
        return false;
    }

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

size_t Network::getNumberOfLayer()
{
    return m_NetworkStructure.size();
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

bool Network::backpropagation( const Eigen::VectorXf x_in, const Eigen::VectorXf& y_out, const float& eta )
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

    // Compute error and partial derivatives in all remaining layers
    for( int k = getNumberOfLayer() - 2; k >= 0; k-- )
    {
        std::shared_ptr<Layer> thisLayer = getLayer(k);
        thisLayer->computeBackprogationError( layerAfter->getBackpropagationError(), layerAfter->getWeigtMatrix() );
        thisLayer->computePartialDerivatives();

        layerAfter = thisLayer;
    }

    // Update weights and biases with the computed derivatives and learning rate.
    // First layer doew not need to be updated -> no effect in input layer
    for( size_t k = 1; k < getNumberOfLayer(); k++ )
        getLayer(k)->updateWeightsAndBiases( eta );

    return true;
}

shared_ptr<Layer> Network::getOutputLayer()
{
    return getLayer( getNumberOfLayer() - 1 );
}
