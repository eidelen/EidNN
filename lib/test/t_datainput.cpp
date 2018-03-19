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
#include "dataInput.h"

TEST(DataInput, addClear)
{
    DataInput* di = new DataInput();

    for( size_t i = 0; i < 100; i++ )
    {
        Eigen::MatrixXd input_test( 5, 1 );
        input_test.setConstant(0.0);
        input_test(0,0) = i;

        Eigen::MatrixXd output_test( 4, 1 );
        output_test.setConstant(0.0);
        output_test(1,0) = i;

        di->addTestSample(input_test,output_test);

        ASSERT_EQ(di->getNumberOfTestSamples(), i+1);


        Eigen::MatrixXd input_training = input_test;
        input_training(0,0) = i*2;

        Eigen::MatrixXd output_training = output_test;
        output_training(1,0) = i*2;

        di->addTrainingSample(input_training,output_training);

        ASSERT_EQ(di->getNumberOfTrainingSamples(), i+1);
    }

    for( size_t i = 0; i < 100; i++ )
    {
        DataElement dtest = di->m_test.at(i);
        DataElement dtraining = di->m_training.at(i);

        ASSERT_FLOAT_EQ(dtest.input(0,0), i);
        ASSERT_FLOAT_EQ(dtest.output(1,0), i);
        ASSERT_FLOAT_EQ(dtraining.input(0,0), i*2);
        ASSERT_FLOAT_EQ(dtraining.output(1,0), i*2);
    }

    di->clear();
    ASSERT_EQ(di->getNumberOfTestSamples(), 0);
    ASSERT_EQ(di->getNumberOfTrainingSamples(), 0);
}


