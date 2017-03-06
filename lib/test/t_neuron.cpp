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
#include "neuron.h"

TEST(NeuronTest, SigmoidFunction)
{
    ASSERT_FLOAT_EQ( Neuron::sigmoid(0.0), 0.5);
    ASSERT_NEAR( Neuron::sigmoid(4) + Neuron::sigmoid(-4), 1.0, 0.0001);  // symmetric
    ASSERT_NEAR( Neuron::sigmoid(10), 1.0, 0.001);
    ASSERT_NEAR( Neuron::sigmoid(-10), 0.0, 0.001);
}

TEST(NeuronTest, DerivationSigmoidFunction)
{
    ASSERT_NEAR( Neuron::d_sigmoid(10), 0.0, 0.0001);
    ASSERT_NEAR( Neuron::d_sigmoid(10), 0.0, 0.0001);
    ASSERT_NEAR( Neuron::d_sigmoid(0), 0.25, 0.0001);
}

TEST(NeuronTest, SetWeightsAndInput)
{
    int nbr_input = 2;
    float dummy;

    Neuron* n = new Neuron( nbr_input );

    ASSERT_EQ( n->getNbrOfInputs(), nbr_input );

    Eigen::VectorXf wrongSize = Eigen::VectorXf(3);
    ASSERT_FALSE( n->setWeights(wrongSize) );
    ASSERT_FALSE( n->feedForward(wrongSize,dummy,dummy) );

    Eigen::VectorXf rightSize = Eigen::VectorXf(nbr_input);
    ASSERT_TRUE( n->setWeights(rightSize) );
    ASSERT_TRUE( n->feedForward(rightSize,dummy,dummy) );

    delete n;
}


TEST(NeuronTest, Activation)
{
    Neuron* n = new Neuron( 2 );

    Eigen::VectorXf weights(2);
    weights << 1.0, 2.0;

    n->setBias(1.0);    n->setWeights(weights);


    float activation, z;
    Eigen::VectorXf x_in(2);
    x_in << 0.0, 0.0;
    n->feedForward(x_in, z, activation); // z = 0*1 + 0*2 + 1 = 1

    ASSERT_NEAR(z, 1.0, 0.0001 );
    ASSERT_NEAR(activation, Neuron::sigmoid(1.0), 0.0001 );


    x_in << 1.0, 3.0;
    n->feedForward(x_in, z, activation); // z = 1*1 + 3*2 + 1 = 8.0
    ASSERT_NEAR(z, 8.0, 0.0001 );
    ASSERT_NEAR(activation, Neuron::sigmoid(8.0), 0.0001 );


    x_in << -1.0, 3.0;
    n->feedForward(x_in, z, activation); // z = -1*1 + 3*2 + 1 = 6.0
    ASSERT_NEAR(z, 6.0, 0.0001 );
    ASSERT_NEAR(activation, Neuron::sigmoid(6.0), 0.0001 );


    delete n;
}
