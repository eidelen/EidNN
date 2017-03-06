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
