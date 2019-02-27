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
    Layer* l = new Layer(10,100,Layer::Softmax);

    ASSERT_EQ( 10, l->getNbrOfNeurons() );
    ASSERT_EQ( 100, l->getNbrOfNeuronInputs() );
    ASSERT_EQ( Layer::Softmax, l->getLayerType() );

    delete l;
}

TEST(LayerTest, CopyConstructor)
{
    Layer* l = new Layer(2,2,Layer::Softmax);
    l->setBias(0.2); l->setWeight(0.8);

    Layer* lcopy = new Layer( *l );
    ASSERT_NEAR( lcopy->getBiasVector()(0,0), 0.2, 0.0001 );
    ASSERT_NEAR( lcopy->getBiasVector()(1,0), 0.2, 0.0001 );
    ASSERT_NEAR(lcopy->getWeightMatrix()(0,0), 0.8, 0.0001 );
    ASSERT_NEAR(lcopy->getWeightMatrix()(0,1), 0.8, 0.0001 );
    ASSERT_NEAR(lcopy->getWeightMatrix()(1,0), 0.8, 0.0001 );
    ASSERT_NEAR(lcopy->getWeightMatrix()(1,1), 0.8, 0.0001 );
    ASSERT_EQ( lcopy->getLayerType(), Layer::Softmax );

    delete l;
    delete lcopy;
}

TEST(LayerTest, Serialization)
{
    Layer* l = new Layer(2,2,Layer::Softmax);
    l->setBias(0.2); l->setWeight(0.8);

    std::string serializedBuf = l->serialize( );
    Layer* lcopy = Layer::deserialize( serializedBuf );

    ASSERT_NEAR( lcopy->getBiasVector()(0,0), 0.2, 0.0001 );
    ASSERT_NEAR( lcopy->getBiasVector()(1,0), 0.2, 0.0001 );
    ASSERT_NEAR(lcopy->getWeightMatrix()(0,0), 0.8, 0.0001 );
    ASSERT_NEAR(lcopy->getWeightMatrix()(0,1), 0.8, 0.0001 );
    ASSERT_NEAR(lcopy->getWeightMatrix()(1,0), 0.8, 0.0001 );
    ASSERT_NEAR(lcopy->getWeightMatrix()(1,1), 0.8, 0.0001 );
    ASSERT_EQ( lcopy->getLayerType(), Layer::Softmax );

    delete l;
    delete lcopy;
}



TEST(LayerTest, ActivationVectorSigmoid)
{
    // make a layer with two neurons and two inputs
    // neuron 0: w0 = 1, w1 = 2, b = 0
    // neuron 1: w0 = 3, w1 = 4, b = 2

    std::vector<Eigen::VectorXd> weights;
    std::vector<double> biases;

    Eigen::VectorXd w_n0(2);  w_n0 << 1, 2;
    weights.push_back( w_n0 );
    biases.push_back( 0 );

    Eigen::VectorXd w_n1(2);  w_n1 << 3, 4;
    weights.push_back( w_n1 );
    biases.push_back( 2 );

    Layer* l = new Layer( 2, weights, biases, Layer::Sigmoid );

    ASSERT_EQ( 2, l->getNbrOfNeurons() );
    ASSERT_EQ( 2, l->getNbrOfNeuronInputs() );


    Eigen::MatrixXd x1(2,1);  x1 << 0, 0;
    // for this input, the output should be:
    // n0: 0  and n1: 2   ->  [ 0, 2 ]
    l->feedForward(x1);
    const Eigen::MatrixXd z1 = l->getWeightedInputZ();
    const Eigen::MatrixXd a1 = l->getOutputActivation();
    ASSERT_EQ( z1.rows(), 2 );
    ASSERT_EQ( z1.cols(), 1 );
    ASSERT_NEAR( z1(0,0), 0, 0.0001 );
    ASSERT_NEAR( a1(0,0), Neuron::sigmoid(0), 0.0001 );
    ASSERT_NEAR( z1(1,0), 2, 0.0001 );
    ASSERT_NEAR( a1(1,0), Neuron::sigmoid(2), 0.0001 );

    Eigen::MatrixXd x2(2,1);  x2 << 1, 2;
    // n0: 1*1 + 2*2 + 0 = 5    n1: 1*3 + 2*4 + 2 = 13
    l->feedForward(x2);
    const Eigen::MatrixXd z2 = l->getWeightedInputZ();
    const Eigen::MatrixXd a2 = l->getOutputActivation();
    ASSERT_NEAR( z2(0,0), 5, 0.0001 );
    ASSERT_NEAR( a2(0,0), Neuron::sigmoid(5), 0.0001 );
    ASSERT_NEAR( z2(1,0), 13, 0.0001 );
    ASSERT_NEAR( a2(1,0), Neuron::sigmoid(13), 0.0001 );

    // check safing of input activation (should be x2)
    const Eigen::MatrixXd in_act = l->getInputActivation();
    ASSERT_NEAR( x2(0,0), in_act(0,0), 0.0001 );
    ASSERT_NEAR( x2(1,0), in_act(1,0), 0.0001 );


    // Multiple input / output -> same numbers as above.
    unsigned int nbrSamples = 3;
    Eigen::MatrixXd xM = x2.replicate(1,nbrSamples);
    l->feedForward(xM);

    const Eigen::MatrixXd zM = l->getWeightedInputZ();
    const Eigen::MatrixXd aM = l->getOutputActivation();
    const Eigen::MatrixXd in_actM = l->getInputActivation();

    ASSERT_EQ( zM.rows(), 2 );
    ASSERT_EQ( zM.cols(), nbrSamples );
    ASSERT_EQ( aM.rows(), 2 );
    ASSERT_EQ( aM.cols(), nbrSamples );
    ASSERT_EQ( in_actM.rows(), 2 );
    ASSERT_EQ( in_actM.cols(), nbrSamples );

    for( unsigned int k = 0; k < nbrSamples; k++ )
    {
        ASSERT_NEAR( zM(0,k), 5, 0.0001 );
        ASSERT_NEAR( aM(0,k), Neuron::sigmoid(5), 0.0001 );
        ASSERT_NEAR( zM(1,k), 13, 0.0001 );
        ASSERT_NEAR( aM(1,k), Neuron::sigmoid(13), 0.0001 );

        // check safing of input activation (should be x2)
        ASSERT_NEAR( xM(0,k), in_actM(0,k), 0.0001 );
        ASSERT_NEAR( xM(1,k), in_actM(1,k), 0.0001 );
    }

    delete l;
}

TEST(LayerTest, ActivationVectorSoftmax)
{
    // make a layer with two neurons and two inputs
    // neuron 0: w0 = 1, w1 = 2, b = 0
    // neuron 1: w0 = 3, w1 = 4, b = 2

    std::vector<Eigen::VectorXd> weights;
    std::vector<double> biases;

    Eigen::VectorXd w_n0(2);  w_n0 << 1, 2;
    weights.push_back( w_n0 );
    biases.push_back( 0 );

    Eigen::VectorXd w_n1(2);  w_n1 << 3, 4;
    weights.push_back( w_n1 );
    biases.push_back( 2 );

    Layer* l = new Layer( 2, weights, biases, Layer::Softmax );

    ASSERT_EQ( 2, l->getNbrOfNeurons() );
    ASSERT_EQ( 2, l->getNbrOfNeuronInputs() );


    Eigen::MatrixXd x1(2,3);  x1 << 0, 0, 0, 0, 0, 0;
    // for this input, the weighted output should be:  ->  [ 0, 2 ; 0, 2; 0, 2]
    // softmax activation should be
    // sum = e^(0) + e^(2) = 1 + 7.3891 = 8.3891
    // a0 = e^(0) / (e^(0) + e^(2)) = 0.11920
    // a1 = e^(2) / (e^(0) + e^(2)) = 0.88080


    l->feedForward(x1);
    const Eigen::MatrixXd z1 = l->getWeightedInputZ();
    const Eigen::MatrixXd a1 = l->getOutputActivation();
    ASSERT_EQ( z1.rows(), 2 );
    ASSERT_EQ( z1.cols(), 3 );
    ASSERT_NEAR( z1(0,0), 0, 0.0001 );
    ASSERT_NEAR( z1(0,1), 0, 0.0001 );
    ASSERT_NEAR( z1(0,2), 0, 0.0001 );

    ASSERT_NEAR( a1(0,0), 0.11920, 0.001 );
    ASSERT_NEAR( a1(0,1), 0.11920, 0.001 );
    ASSERT_NEAR( a1(0,2), 0.11920, 0.001 );

    ASSERT_NEAR( z1(1,0), 2, 0.0001 );
    ASSERT_NEAR( z1(1,1), 2, 0.0001 );
    ASSERT_NEAR( z1(1,2), 2, 0.0001 );

    ASSERT_NEAR( a1(1,0), 0.88080, 0.001 );
    ASSERT_NEAR( a1(1,1), 0.88080, 0.001 );
    ASSERT_NEAR( a1(1,2), 0.88080, 0.001 );

    ASSERT_NEAR( a1(0,0) + a1(1,0), 1.0000, 0.001 );

    delete l;

}

TEST(LayerTest, SetWeightsAndBiases)
{
    Layer* l = new Layer( 2, 2 );

    std::vector<Eigen::VectorXd> wV;
    Eigen::VectorXd w_n0(2);  w_n0 << 1, 2;
    Eigen::VectorXd w_n1(2);  w_n1 << 3, 4;
    wV.push_back( w_n0 ); wV.push_back( w_n1 );

    std::vector<double> bV; bV.push_back(5); bV.push_back(6);

    ASSERT_TRUE( l->setBiases( bV ) );
    ASSERT_TRUE( l->setWeights( wV ) );

    bV.push_back(7);
    ASSERT_FALSE( l->setBiases( bV ) );

    // check some weights and biases
    ASSERT_NEAR( (l->getWeightMatrix()(0,0)), 1, 0.0001 );
    ASSERT_NEAR( (l->getWeightMatrix()(1,1)), 4 , 0.0001 );
    ASSERT_NEAR( l->getBiasVector()(1), 6 , 0.0001 );

    // set uniform neuron settings
    l->setBias( 10.00 );
    l->setWeight( 13.00 );
    ASSERT_NEAR( (l->getWeightMatrix()(0,0)), 13.00, 0.0001 );
    ASSERT_NEAR( (l->getWeightMatrix()(1,1)), 13.00, 0.0001 );
    ASSERT_NEAR( l->getBiasVector()(1), 10.00 , 0.0001 );
    ASSERT_NEAR( l->getBiasVector()(0), 10.00 , 0.0001 );

    l->resetRandomlyWeightsAndBiases();
    ASSERT_TRUE( fabs( l->getBiasVector()(0) ) < 2.0 ); // Theoretically, this could fail. But this is very unlikely.

    // set directly vector
    Eigen::VectorXd bVector_wrongSize(3);  bVector_wrongSize << 1, 2, 3;
    ASSERT_FALSE( l->setBiases( bVector_wrongSize ) );
    Eigen::VectorXd bVector_rightSize(2);  bVector_rightSize << 6, 344;
    ASSERT_TRUE( l->setBiases( bVector_rightSize ) );
    ASSERT_NEAR( l->getBiasVector()(0), 6.0 , 0.0001 );
    ASSERT_NEAR( l->getBiasVector()(1), 344.0 , 0.0001 );

    // set directly matrix
    Eigen::MatrixXd wMatrixWrong = Eigen::MatrixXd( 2 , 3 ); wMatrixWrong << 1,2,3,4,5,6;
    ASSERT_FALSE( l->setWeights( wMatrixWrong ));
    Eigen::MatrixXd wMatrixRight = Eigen::MatrixXd( 2 , 2 ); wMatrixRight << 10, 11,   15, 16;
    ASSERT_TRUE( l->setWeights( wMatrixRight ));
    ASSERT_NEAR(l->getWeightMatrix()(0,0), 10.0, 0.0001 );
    ASSERT_NEAR(l->getWeightMatrix()(0,1), 11.0, 0.0001 );
    ASSERT_NEAR(l->getWeightMatrix()(1,0), 15.0, 0.0001 );
    ASSERT_NEAR(l->getWeightMatrix()(1,1), 16.0, 0.0001 );

    delete l;
}


TEST(LayerTest, ComputeOutputError)
{
    Layer* l = new Layer( 2, 2 );
    l->setBias(0.0);
    l->setWeight(0.0);

    Eigen::VectorXd x(2);  x << 0, 0;
    l->feedForward( x ); // output should be computed equal to 0.5;

    // if expected outcome is 0.5, the error of the last layer is 0.0.
    Eigen::VectorXd y(2);  y << 0.5, 0.5;
    ASSERT_TRUE( l->computeBackpropagationOutputLayerError(y) );
    ASSERT_NEAR( l->getBackpropagationError()(0), 0.0, 0.0001 );
    ASSERT_NEAR( l->getBackpropagationError()(1), 0.0, 0.0001 );

    // wrong dimension
    Eigen::VectorXd y_wrong(3);  y_wrong << 0.5, 0.5, 0.5;
    ASSERT_FALSE( l->computeBackpropagationOutputLayerError(y_wrong) );

    // actual error
    Eigen::VectorXd y_next(2);  y_next << 0.5, 1.5;
    ASSERT_TRUE( l->computeBackpropagationOutputLayerError(y_next) );
    ASSERT_NEAR( l->getBackpropagationError()(0), 0.0, 0.0001 );
    ASSERT_NEAR( l->getBackpropagationError()(1), -0.25, 0.0001 );

    delete l;
}

TEST(LayerTest, ComputeOutputErrorMultipleInput)
{
    Layer* l = new Layer( 2, 2 );
    l->setBias(0.0);
    l->setWeight(0.0);

    Eigen::MatrixXd x0(2,4);  x0 << 0, 0, 0,0, 0,0, 0,0;
    l->feedForward( x0 ); // output should be computed equal to 0.5;

    Eigen::MatrixXd x(2,3);  x << 0, 0, 0,0, 0,0;
    l->feedForward( x ); // output should be computed equal to 0.5;

    // if expected outcome is 0.5, the error of the last layer is 0.0.
    Eigen::MatrixXd y(2,3);  y << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
    ASSERT_TRUE( l->computeBackpropagationOutputLayerError(y) );
    for( unsigned int k = 0; k<3; k++ )
    {
        ASSERT_NEAR( l->getBackpropagationError()(0,k), 0.0, 0.0001 );
        ASSERT_NEAR( l->getBackpropagationError()(1,k), 0.0, 0.0001 );
    }

    // wrong dimension
    Eigen::MatrixXd y_wrong(3,3);  y_wrong << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
    ASSERT_FALSE( l->computeBackpropagationOutputLayerError(y_wrong) );

    Eigen::MatrixXd y_wrong2(2,2);  y_wrong2 << 0.5, 0.5, 0.5, 0.5;
    ASSERT_FALSE( l->computeBackpropagationOutputLayerError(y_wrong2) );

    // actual error
    Eigen::MatrixXd y_true(2,3);  y_true << 0.5, 0.5, 0.5,   1.5, 1.5, 1.5;
    ASSERT_TRUE( l->computeBackpropagationOutputLayerError(y_true) );

    for( unsigned int k = 0; k<3; k++ )
    {
        ASSERT_NEAR( l->getBackpropagationError()(0,k), 0.0, 0.0001 );
        ASSERT_NEAR( l->getBackpropagationError()(1,k), -0.25, 0.0001 );
    }

    delete l;
}


TEST(LayerTest, SetAndGetLayerType)
{
    Layer* l = new Layer(10,10);
    ASSERT_EQ( Layer::Sigmoid, l->getLayerType() ); // default
    l->setLayerType( Layer::Softmax );
    ASSERT_EQ( Layer::Softmax, l->getLayerType() );
    l->setLayerType( Layer::Sigmoid );
    ASSERT_EQ( Layer::Sigmoid, l->getLayerType() );

    delete l;
}

#include "quadraticCost.h"
#include "crossEntropyCost.h"

TEST(LayerTest, CostFunction)
{
    Layer* l = new Layer( 2, 2 );

    std::shared_ptr<CrossEntropyCost> ce( new CrossEntropyCost() );
    std::shared_ptr<QuadraticCost> qc( new QuadraticCost() );

    // default should be quadratic
    ASSERT_TRUE( l->getCostFunction()->name().compare( "quadraticcost" ) == 0 );

    l->setCostFunction( ce );
    ASSERT_TRUE( l->getCostFunction()->name().compare( "crossentropy" ) == 0 );

    delete l;
}

TEST(LayerTest, SumOfSquareWeights)
{
    Layer* l = new Layer(2,2);

    std::vector<Eigen::VectorXd> wV;
    Eigen::VectorXd w_n0(2);  w_n0 << 1, 2;
    Eigen::VectorXd w_n1(2);  w_n1 << 3, 4;
    wV.push_back( w_n0 ); wV.push_back( w_n1 );

    ASSERT_TRUE( l->setWeights( wV ) );

    double w_should = 1*1 + 2*2 + 3*3 + 4*4;

    ASSERT_FLOAT_EQ(l->getSumOfWeightSquares(), w_should);

    delete l;
}

TEST(LayerTest, Regularization)
{
    Layer* l = new Layer(2,2);

    std::shared_ptr<Regularization> reg( new Regularization(Regularization::RegularizationMethod::WeightDecay, 11 ));

    l->setRegularizationMethod( reg );

    ASSERT_FLOAT_EQ(l->getRegularizationMethod()->m_lamda, 11);
    ASSERT_EQ(l->getRegularizationMethod()->m_method, Regularization::RegularizationMethod::WeightDecay);


    // copy constructor

    Layer* l2 = new Layer(*l);

    ASSERT_FLOAT_EQ(l2->getRegularizationMethod()->m_lamda, 11);
    ASSERT_EQ(l2->getRegularizationMethod()->m_method, Regularization::RegularizationMethod::WeightDecay);

    delete l;
    delete l2;
}

