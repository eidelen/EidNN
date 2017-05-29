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

#include <random>
#include <gtest/gtest.h>
#include "network.h"
#include "layer.h"
#include "neuron.h"
#include "helpers.h"


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

    Eigen::VectorXd x1(1); x1 << 0.0;
    ASSERT_TRUE(net1->feedForward(x1));
    const Eigen::VectorXd out1 = net1->getOutputActivation();
    ASSERT_NEAR( out1(0), 0.5, 0.0001 );
    ASSERT_NEAR( out1(1), 0.5, 0.0001 );


    Eigen::VectorXd x2(1); x2 << 10000.0;
    ASSERT_TRUE(net1->feedForward(x2));
    const Eigen::VectorXd out2 = net1->getOutputActivation();
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

    std::vector<double> layer1_b; layer1_b.push_back(0.2); layer1_b.push_back(-0.3);
    net->getLayer( 1 )->setBiases(layer1_b);

    std::vector<double> layer2_b; layer2_b.push_back(0.4);
    net->getLayer( 2 )->setBiases(layer2_b);

    /*      IN              L1                                  L2
     *
     *               0.0*1.0+0.2 = 0.2 -> o(0.2)
     *      0.0                                            o(  o(0.2)*1.0+o(-0.3)*1.0  +  0.4)
     *               0.0*1.0-0.3 = -0.3 -> o(-0.3)
     *
     */


    Eigen::MatrixXd x(1,1); x << 0.0;
    ASSERT_TRUE(net->feedForward(x));
    const Eigen::MatrixXd out = net->getOutputActivation();
    double outputShouldBe = Neuron::sigmoid( Neuron::sigmoid(0.2) + Neuron::sigmoid(-0.3) + 0.4);
    ASSERT_NEAR( out(0,0),  outputShouldBe, 0.0001 );

    // multiple input
    Eigen::MatrixXd xM(1,3); xM << 0.0, 0.0, 0.0;
    ASSERT_TRUE(net->feedForward(xM));
    const Eigen::MatrixXd out2 = net->getOutputActivation();
    for( unsigned int k = 0; k < 3; k++ )
        ASSERT_NEAR( out2(0,k),  outputShouldBe, 0.0001 );


    delete net;
}

TEST(NetworkTest, Backpropagation_input)
{
    std::vector<unsigned int> map = {1,4,2};
    Network* net = new Network(map);

    Eigen::VectorXd x(1); x << 0.0;
    Eigen::VectorXd x_wrong_dimension(2); x_wrong_dimension << 0.0, 0.0;
    Eigen::VectorXd y(2); y << 0.0, 0.0;
    Eigen::VectorXd y_wrong_dimension(1); y_wrong_dimension << 0.0;

    // check invalid dimensions
    ASSERT_TRUE( net->gradientDescent(x, y, 1.0 ));
    ASSERT_FALSE( net->gradientDescent(x_wrong_dimension, y, 1.0 ));
    ASSERT_FALSE( net->gradientDescent( x, y_wrong_dimension, 1.0 ));


    // check results

    // set all neural network to zero -> sigmoid(0) = 0.5
    for( unsigned int k = 0; k < net->getNumberOfLayer(); k++ )
    {
        std::shared_ptr<Layer> l = net->getLayer( k );
        l->setBias( 0.0 );
        l->setWeight( 0.0 );
    }

    // set expected outcome to 0.5. Therefore all errors and all partial derivatives in the network are 0.0
    Eigen::VectorXd y_zero_error(2); y_zero_error << 0.5, 0.5;
    net->gradientDescent( x, y_zero_error, 1.0 );

    for( unsigned int u = 1; u < net->getNumberOfLayer(); u++ )
    {
        std::shared_ptr<Layer> ll = net->getLayer( u );

        Eigen::VectorXd err;
        err = ll->getBackpropagationError();

        // check that all are zero
        for( unsigned int q = 0; q < err.rows(); q++ )
            ASSERT_NEAR( err(q),  0.0, 0.0001 );

        Eigen::VectorXd pd_biases = ll->getPartialDerivativesBiases();
        for( unsigned int q = 0; q < pd_biases.rows(); q++ )
            ASSERT_NEAR( pd_biases(q),  0.0, 0.0001 );

        Eigen::MatrixXd pd_weights = ll->getPartialDerivativesWeights();
        for( unsigned int q = 0; q < pd_weights.rows(); q++ )
            for( unsigned int p = 0; p < pd_weights.cols(); p++ )
                ASSERT_NEAR( pd_weights(q,p),  0.0, 0.0001 );
    }

    delete net;
}

TEST(NetworkTest, Backpropagation_Errors)
{
    std::vector<unsigned int> map = {1,2};
    Network* net = new Network(map);

    // set whole neural network weights to 0.1 and biases to 0.2
    for( unsigned int k = 0; k < net->getNumberOfLayer(); k++ )
    {
        std::shared_ptr<Layer> l = net->getLayer( k );
        l->setBias( 0.1 );
        l->setWeight( 0.2 );
    }

    // input & output
    Eigen::VectorXd x(1); x << 0.0;  // this leads to network output of [0.52498, 0.52498]



    //Test 1) create an expected value equal to the output -> no error
    net->feedForward( x ); Eigen::VectorXd y = net->getOutputActivation();
    net->gradientDescent(x, y, 1.0);

    // expected errors in outputlayer = [0.0, 0.0]
    ASSERT_NEAR( net->getOutputLayer()->getBackpropagationError()(0), 0.0, 0.0001 );
    ASSERT_NEAR( net->getOutputLayer()->getBackpropagationError()(1), 0.0, 0.0001 );
    ASSERT_NEAR( net->getNetworkErrorMagnitude(), 0.0, 0.00001 );

    // partial derivatives are [0.0, 0.0]
    ASSERT_NEAR( net->getOutputLayer()->getPartialDerivativesWeights()(0,0), 0.0, 0.0001 );
    ASSERT_NEAR( net->getOutputLayer()->getPartialDerivativesWeights()(1,0), 0.0, 0.0001 );
    ASSERT_NEAR( net->getOutputLayer()->getPartialDerivativesBiases()(0), 0.0, 0.0001 );
    ASSERT_NEAR( net->getOutputLayer()->getPartialDerivativesBiases()(1), 0.0, 0.0001 );



    //Test 2) create an expected value where the first element is equal to the expected output -> no error
    //        but the second element differs from the expectation
    x << 2.0;
    net->feedForward( x ); y = net->getOutputActivation();
    y(1) = y(1) - 0.1; // alter output a bit
    net->gradientDescent(x,y, 1.0);

    // check error
    double errMag = net->getNetworkErrorMagnitude();
    ASSERT_NEAR( errMag, 0.02350037, 0.00001 );


    // expected errors and derivatives for neuron 0 is 0.0
    Eigen::VectorXd kx = net->getOutputLayer()->getBackpropagationError();
    ASSERT_NEAR( net->getOutputLayer()->getBackpropagationError()(0), 0.0, 0.0001 );
    ASSERT_NEAR( net->getOutputLayer()->getPartialDerivativesWeights()(0,0), 0.0, 0.0001 );
    ASSERT_NEAR( net->getOutputLayer()->getPartialDerivativesBiases()(0), 0.0, 0.0001 );

    delete net;
}

TEST(NetworkTest, Backpropagate_Simple_Example)
{
    // recognize positive numbers and negative numbers in the range of -100 to 100
    std::vector<unsigned int> map = {1,5,2};
    Network* net = new Network(map);

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-10000, +10000);

    double bestResult = -1.0;
    for( int evolutions = 0; evolutions < 60; evolutions++ )
    {
        for( int ts = 0; ts < 1000; ts++ )
        {
            double valIn = dist(e2);
            Eigen::VectorXd x(1);
            x(0) = valIn;

            Eigen::VectorXd y(2);
            if( valIn > 0 )
                y << 1.0, 0.0;
            else
                y << 0.0, 1.0;

            net->gradientDescent( x, y, 0.25 );
        }

        double nbrOfTests = 100;
        double nbrSuccessful = 0;

        for( int test_s = 0; test_s < nbrOfTests; test_s++ )
        {
            double testVal = dist(e2);
            Eigen::VectorXd x(1);
            x(0) = testVal;

            net->feedForward( x );
            Eigen::VectorXd o_activation = net->getOutputActivation();

            if( testVal > 0 )
            {
                if( o_activation(0) > o_activation(1) && o_activation(0) > 0.99 ) // at least 99% sure that it is positive
                    nbrSuccessful = nbrSuccessful + 1.0f;
            }
            else
            {
                if( o_activation(1) > o_activation(0) && o_activation(1) > 0.99 ) // at least 99% sure that it is negative
                    nbrSuccessful = nbrSuccessful + 1.0f;
            }
        }

        double thisSuccessRate = 100 * nbrSuccessful / nbrOfTests;
        if( bestResult < thisSuccessRate )
            bestResult = thisSuccessRate;

        std::cout << "Evolution " << evolutions <<  ": success rate = " << 100 * nbrSuccessful / nbrOfTests << "%" << std::endl;
    }

    std::cout << "Best Result =  " << bestResult <<  "%" << std::endl;
    delete  net;

    ASSERT_GT( bestResult, 90 );
}

TEST(NetworkTest, Backpropagate_Multilayer_Input_Example)
{
    // find the index of the biggest element in a vector
    std::vector<unsigned int> map = {3,10,3};
    Network* net = new Network(map);

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 9);

    std::vector<int> choice;
    for( int inp = 0; inp < 10; inp++ )
        choice.push_back( inp );


    double bestResult = -1.0;
    for( int evolutions = 0; evolutions < 60; evolutions++ )
    {
        for( int ts = 0; ts < 1000; ts++ )
        {

            // generate input
            Eigen::VectorXd x(3);
            std:vector<int> set = choice;
            for( int i = 0; i < 3; i++ )
            {
                while( true )
                {
                    uint elemIdx = uint( floor(dist(e2)) );
                    if( elemIdx < set.size() )
                    {
                        x(i) = set.at( elemIdx );
                        set.erase( set.begin() + elemIdx );
                        break;
                    }
                }
            }

            // ... and corresponding output (index of maximum value)
            Eigen::VectorXd y(3); y << 0, 0, 0;
            int maxIdx = 0;
            double maxValue = -1;
            for( int i = 0; i < 3; i++ )
            {
                if( x(i) > maxValue )
                {
                    maxValue = x(i);
                    maxIdx = i;
                }
            }
            y( maxIdx ) = 1.0;

            net->gradientDescent( x, y, 0.1f );
        }


        double nbrOfTests = 100;
        double nbrSuccessful = 0;

        for( int test_s = 0; test_s < nbrOfTests; test_s++ )
        {
            // generate input
            Eigen::VectorXd x(3);
            vector<int> set = choice;
            for( int i = 0; i < 3; i++ )
            {
                while( true )
                {
                    uint elemIdx = uint( floor(dist(e2)) );
                    if( elemIdx < set.size() )
                    {
                        x(i) = set.at( elemIdx );
                        set.erase( set.begin() + elemIdx );
                        break;
                    }
                }
            }

            // ... and corresponding output (index of maximum value)
            Eigen::VectorXd y(3); y << 0, 0, 0;
            int maxIdx = 0;
            double maxValue = -1;
            for( int i = 0; i < 3; i++ )
            {
                if( x(i) > maxValue )
                {
                    maxValue = x(i);
                    maxIdx = i;
                }
            }
            y( maxIdx ) = 1.0;

            net->feedForward( x );

            if( ( y - net->getOutputActivation() ).norm() < 0.10 )
                nbrSuccessful = nbrSuccessful + 1.0f;
        }

        double thisSuccessRate = 100 * nbrSuccessful / nbrOfTests;
        std::cout << "Evolution " << evolutions <<  ": success rate = " << thisSuccessRate << "%" << std::endl;

        if( bestResult < thisSuccessRate )
            bestResult = thisSuccessRate;
    }

    std::cout << "Best Result =  " << bestResult <<  "%" << std::endl;
    delete  net;

    ASSERT_GT( bestResult, 90 );
}

TEST(NetworkTest, Backpropagate_StochasticGD)
{
    // recognize positive numbers and negative numbers with stochastic gradient descent
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-100, +100);

    // create training set
    std::vector<Eigen::VectorXd> xin;
    std::vector<Eigen::VectorXd> yout;
    for( uint k = 0; k < 1000; k++ )
    {
        double value = dist(e2);
        Eigen::VectorXd thisSample(1);
        Eigen::VectorXd thisLable(2);

        thisSample(0) = value;

        if( value > 0 )
            thisLable << 1.0, 0.0;
        else
            thisLable << 0.0, 1.0;

        xin.push_back( thisSample );
        yout.push_back( thisLable );
    }

    // create test set
    std::vector<Eigen::VectorXd> t_xin;
    std::vector<Eigen::VectorXd> t_yout;
    for( uint k = 0; k < 100; k++ )
    {
        double value = dist(e2);
        Eigen::VectorXd thisSample(1);
        Eigen::VectorXd thisLable(2);

        thisSample(0) = value;

        if( value > 0 )
            thisLable << 1.0, 0.0;
        else
            thisLable << 0.0, 1.0;

        t_xin.push_back( thisSample );
        t_yout.push_back( thisLable );
    }

    std::vector<unsigned int> map = {1,5,2};
    Network* net = new Network(map);
    unsigned int nbrEpochs = 60;
    unsigned int miniBatch = 20;
    double bestResult = -1.0f;

    for( unsigned int epoch = 0; epoch < nbrEpochs; epoch++ )
    {
        //training
        for( size_t k = 0; k < xin.size() / miniBatch ; k++ )    // one epoch runs the number of overall samples.
            net->stochasticGradientDescent( xin, yout, miniBatch, 0.1 );

        // testing
        double nbrSuccessful = 0;
        for( size_t i = 0; i < t_xin.size(); i++ )
        {
            Eigen::VectorXd tx = t_xin.at(i);
            Eigen::VectorXd ty = t_yout.at(i);

            net->feedForward( tx );
            double diff = (net->getOutputActivation() - ty).norm();

            if( diff < 0.1 )
                nbrSuccessful = nbrSuccessful + 1.0;
        }

        double thisSuccessRate = 100 * nbrSuccessful / double(t_xin.size());
        if( bestResult < thisSuccessRate )
            bestResult = thisSuccessRate;
        std::cout << "Epoch " << epoch <<  ": success rate = " << 100 * nbrSuccessful / double(t_xin.size()) << "%" << std::endl;
    }

    std::cout << "Best Result =  " << bestResult <<  "%" << std::endl;

    delete  net;

    ASSERT_GT( bestResult, 90 );
}



