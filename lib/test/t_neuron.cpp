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

