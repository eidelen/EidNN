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

#include <gtest/gtest.h>
#include "layer.h"
#include "neuron.h"


TEST(LayerTest, ConstructAndSize)
{
    Layer* l = new Layer(10,100);

    ASSERT_EQ( 10, l->getNbrOfNeurons() );
    ASSERT_EQ( 100, l->getNbrOfNeuronInputs() );

    delete l;
}

TEST(LayerTest, ActivationVector)
{
    // make a layer with two neurons and two inputs
    // neuron 0: w0 = 1, w1 = 2, b = 0
    // neuron 1: w0 = 3, w1 = 4, b = 2

    std::vector<Eigen::VectorXf> weights;
    std::vector<float> biases;

    Eigen::VectorXf w_n0(2);  w_n0 << 1, 2;
    weights.push_back( w_n0 );
    biases.push_back( 0 );

    Eigen::VectorXf w_n1(2);  w_n1 << 3, 4;
    weights.push_back( w_n1 );
    biases.push_back( 2 );

    Layer* l = new Layer( 2, weights, biases );

    ASSERT_EQ( 2, l->getNbrOfNeurons() );
    ASSERT_EQ( 2, l->getNbrOfNeuronInputs() );


    Eigen::VectorXf x1(2);  x1 << 0, 0;
    // for this input, the output should be:
    // n0: 0  and n1: 2   ->  [ 0, 2 ]
    l->feedForward(x1);
    const Eigen::VectorXf z1 = l->getWeightedInputZ();
    const Eigen::VectorXf a1 = l->getOutputActivation();
    ASSERT_EQ( z1.rows(), 2 );
    ASSERT_NEAR( z1(0), 0, 0.0001 );
    ASSERT_NEAR( a1(0), Neuron::sigmoid(0), 0.0001 );
    ASSERT_NEAR( z1(1), 2, 0.0001 );
    ASSERT_NEAR( a1(1), Neuron::sigmoid(2), 0.0001 );

    Eigen::VectorXf x2(2);  x2 << 1, 2;
    // n0: 1*1 + 2*2 + 0 = 5    n1: 1*3 + 2*4 + 2 = 13
    l->feedForward(x2);
    const Eigen::VectorXf z2 = l->getWeightedInputZ();
    const Eigen::VectorXf a2 = l->getOutputActivation();
    ASSERT_NEAR( z2(0), 5, 0.0001 );
    ASSERT_NEAR( a2(0), Neuron::sigmoid(5), 0.0001 );
    ASSERT_NEAR( z2(1), 13, 0.0001 );
    ASSERT_NEAR( a2(1), Neuron::sigmoid(13), 0.0001 );

    delete l;
}

TEST(LayerTest, SetWeightsAndBiases)
{
    Layer* l = new Layer( 2, 2 );

    std::vector<Eigen::VectorXf> wV;
    Eigen::VectorXf w_n0(2);  w_n0 << 1, 2;
    Eigen::VectorXf w_n1(2);  w_n1 << 3, 4;
    wV.push_back( w_n0 ); wV.push_back( w_n1 );

    std::vector<float> bV; bV.push_back(5); bV.push_back(6);

    ASSERT_TRUE( l->setBiases( bV ) );
    ASSERT_TRUE( l->setWeights( wV ) );

    bV.push_back(7);
    ASSERT_FALSE( l->setBiases( bV ) );

    // check some weights and biases
    ASSERT_NEAR( (l->getNeuron(0)->getWeights())(0), 1, 0.0001 );
    ASSERT_NEAR( (l->getNeuron(1)->getWeights())(1), 4 , 0.0001 );
    ASSERT_NEAR( l->getNeuron(1)->getBias(), 6 , 0.0001 );

    // set uniform neuron settings
    l->setBias( 10.00 );
    l->setWeight( 13.00 );
    ASSERT_NEAR( (l->getNeuron(0)->getWeights())(0), 13.00, 0.0001 );
    ASSERT_NEAR( (l->getNeuron(1)->getWeights())(1), 13.00, 0.0001 );
    ASSERT_NEAR( l->getNeuron(1)->getBias(), 10.00 , 0.0001 );
    ASSERT_NEAR( l->getNeuron(0)->getBias(), 10.00 , 0.0001 );

    l->resetRandomlyWeightsAndBiases();
    ASSERT_TRUE( fabs( l->getNeuron(0)->getBias() ) < 2.0 ); // Theoretically, this could fail. But this is very unlikely.

    delete l;
}

TEST(LayerTest, D_Sigmoid)
{
    Eigen::VectorXf x(3);  x << 0, -100, +100;
    Eigen::VectorXf f = Layer::d_sigmoid(x);

    ASSERT_NEAR( f(0), 0.25, 0.0001 );
    ASSERT_NEAR( f(1), 0.0, 0.0001 );
    ASSERT_NEAR( f(2), 0.0, 0.0001 );
}

TEST(LayerTest, ComputeOutputError)
{
    Layer* l = new Layer( 2, 2 );
    l->setBias(0.0);
    l->setWeight(0.0);

    Eigen::VectorXf x(2);  x << 0, 0;
    l->feedForward( x ); // output should be computed equal to 0.5;

    // if expected outcome is 0.5, the error of the last layer is 0.0.
    Eigen::VectorXf y(2);  y << 0.5, 0.5;
    ASSERT_TRUE( l->computeOutputLayerError(y) );
    ASSERT_NEAR( l->getOutputLayerError()(0), 0.0, 0.0001 );
    ASSERT_NEAR( l->getOutputLayerError()(1), 0.0, 0.0001 );

    // wrong dimension
    Eigen::VectorXf y_wrong(3);  y_wrong << 0.5, 0.5, 0.5;
    ASSERT_FALSE( l->computeOutputLayerError(y_wrong) );

    // actual error
    Eigen::VectorXf y_next(2);  y_next << 0.5, 1.5;
    ASSERT_TRUE( l->computeOutputLayerError(y_next) );
    ASSERT_NEAR( l->getOutputLayerError()(0), 0.0, 0.0001 );
    ASSERT_NEAR( l->getOutputLayerError()(1), -0.25, 0.0001 );

    delete l;
}

