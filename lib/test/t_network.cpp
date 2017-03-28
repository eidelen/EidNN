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
#include "network.h"
#include "layer.h"
#include "neuron.h"


TEST(NetworkTest, ConstructNetwork)
{
    std::vector<unsigned int> map = {1,2,2};
    Network* net = new Network(map);

    ASSERT_EQ( net->getNumberOfLayer(), 3 );
    ASSERT_EQ( net->getLayer(0)->getNbrOfNeurons(), 1 );

    ASSERT_EQ( net->getLayer(1)->getNbrOfNeurons(), 2 );
    ASSERT_EQ( net->getLayer(1)->getNbrOfNeuronInputs(), 1 );

    ASSERT_EQ( net->getLayer(2)->getNbrOfNeurons(), 2 );
    ASSERT_EQ( net->getLayer(2)->getNbrOfNeuronInputs(), 2 );

    ASSERT_TRUE( net->getLayer(3).get() == NULL );

    delete net;
}

TEST(NetworkTest, FeedForward_1)
{
    std::vector<unsigned int> map1 = {1,20,26,20,3};
    Network* net1 = new Network(map1);

    // set all neural network to zero -> sigmoid(0) = 0.5
    for( unsigned int k = 0; k < net1->getNumberOfLayer(); k++ )
    {
        std::shared_ptr<Layer> l = net1->getLayer( k );
        l->setBias( 0.0 );
        l->setWeight( 0.0 );
    }

    Eigen::VectorXf x1(1); x1 << 0.0;
    ASSERT_TRUE(net1->feedForward(x1));
    const Eigen::VectorXf out1 = net1->getOutputActivation();
    ASSERT_NEAR( out1(0), 0.5, 0.0001 );
    ASSERT_NEAR( out1(1), 0.5, 0.0001 );


    Eigen::VectorXf x2(1); x2 << 10000.0;
    ASSERT_TRUE(net1->feedForward(x2));
    const Eigen::VectorXf out2 = net1->getOutputActivation();
    ASSERT_NEAR( out2(0), 0.5, 0.0001 );
    ASSERT_NEAR( out2(1), 0.5, 0.0001 );


    delete net1;
}

TEST(NetworkTest, FeedForward_Reference)
{
    std::vector<unsigned int> map = {1,2,1};
    Network* net = new Network(map);

    for( unsigned int k = 0; k < net->getNumberOfLayer(); k++ )
    {
        std::shared_ptr<Layer> l = net->getLayer( k );
        l->setWeight( 1.0 );
    }

    std::vector<float> layer1_b; layer1_b.push_back(0.2); layer1_b.push_back(-0.3);
    net->getLayer( 1 )->setBiases(layer1_b);

    std::vector<float> layer2_b; layer2_b.push_back(0.4);
    net->getLayer( 2 )->setBiases(layer2_b);

    /*      IN              L1                                  L2
     *
     *               0.0*1.0+0.2 = 0.2 -> o(0.2)
     *      0.0                                            o(  o(0.2)*1.0+o(-0.3)*1.0  +  0.4)
     *               0.0*1.0-0.3 = -0.3 -> o(-0.3)
     *
     */

    Eigen::VectorXf x(1); x << 0.0;
    ASSERT_TRUE(net->feedForward(x));
    const Eigen::VectorXf out = net->getOutputActivation();
    float outputShouldBe = Neuron::sigmoid( Neuron::sigmoid(0.2) + Neuron::sigmoid(-0.3) + 0.4);
    ASSERT_NEAR( out(0),  outputShouldBe, 0.0001 );

    delete net;
}

TEST(NetworkTest, Backpropagation_input)
{
    std::vector<unsigned int> map = {1,4,2};
    Network* net = new Network(map);

    Eigen::VectorXf x(1); x << 0.0;
    Eigen::VectorXf x_wrong_dimension(2); x_wrong_dimension << 0.0, 0.0;
    Eigen::VectorXf y(2); y << 0.0, 0.0;
    Eigen::VectorXf y_wrong_dimension(1); y_wrong_dimension << 0.0;

    ASSERT_TRUE( net->backpropagation(x, y));
    ASSERT_FALSE( net->backpropagation(x_wrong_dimension, y ));
    ASSERT_FALSE( net->backpropagation( x, y_wrong_dimension ));

    delete net;

    Eigen::VectorXf a(2); a << 0.0, 1.0;
    Eigen::VectorXf b(2); b << 2.0, 3.0;

    std::cout << a + b << std::endl;

}




