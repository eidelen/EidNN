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
        ASSERT_TRUE(dtest.outputSet);
        ASSERT_FALSE(dtest.lableSet);
        ASSERT_FLOAT_EQ(dtraining.input(0,0), i*2);
        ASSERT_FLOAT_EQ(dtraining.output(1,0), i*2);
        ASSERT_TRUE(dtraining.outputSet);
        ASSERT_FALSE(dtraining.lableSet);
    }

    di->clear();
    ASSERT_EQ(di->getNumberOfTestSamples(), 0);
    ASSERT_EQ(di->getNumberOfTrainingSamples(), 0);
}

TEST(DataInput, addLableGenerateOutput)
{
    DataInput *di = new DataInput();
    Eigen::MatrixXd in(5, 1); // does not matter for this test

    // 4 different lables 1,2,3,4
    di->addTestSample(in, 4);
    di->addTestSample(in, 1);
    di->addTestSample(in, 2);
    di->addTestSample(in, 2);
    di->addTestSample(in, 3);
    di->addTestSample(in, 3);

    di->addTrainingSample(in, 4);
    di->addTrainingSample(in, 1);
    di->addTrainingSample(in, 2);
    di->addTrainingSample(in, 2);
    di->addTrainingSample(in, 3);
    di->addTrainingSample(in, 2);
    di->addTrainingSample(in, 3);
    di->addTrainingSample(in, 2);
    di->addTrainingSample(in, 3);

    ASSERT_TRUE(di->generateFromLables());
    ASSERT_EQ(4, di->m_lables.size());

    // having only the raw input and output in a vector is often used.
    ASSERT_EQ(DataInput::getInputData(di->m_training).size(), di->m_training.size());
    ASSERT_EQ(DataInput::getOutputData(di->m_training).size(), di->m_training.size());
    ASSERT_EQ(DataInput::getInputData(di->m_test).size(), di->m_test.size());
    ASSERT_EQ(DataInput::getOutputData(di->m_test).size(), di->m_test.size());

    for (DataElement &de : di->m_test)
    {
        Eigen::MatrixXd outShould = Eigen::MatrixXd::Constant(4, 1, 0.0);
        outShould(de.lable - 1, 0) = 1.0;
        ASSERT_TRUE((outShould - de.output).isMuchSmallerThan(0.001));
    }

    for (DataElement &de : di->m_training)
    {
        Eigen::MatrixXd outShould = Eigen::MatrixXd::Constant(4, 1, 0.0);
        outShould(de.lable - 1, 0) = 1.0;
        ASSERT_TRUE((outShould - de.output).isMuchSmallerThan(0.001));
    }

    di->addTestSample(in, 10); // unknown lable among test samples
    ASSERT_FALSE(di->generateFromLables());
}

TEST(DataInput, normalize)
{
    DataInput* di = new DataInput();

    Eigen::MatrixXd in( 3, 1 );
    in << 0, 2, 4;
    Eigen::MatrixXd out( 2, 1 );
    out << 1,2;

    di->addTrainingSample(in,out);
    di->addTestSample(in,out);

    di->normalizeData();

    Eigen::MatrixXd shouldNorm( 3, 1 );
    shouldNorm << -1, 0, 1;

    ASSERT_TRUE((shouldNorm - di->m_training.at(0).input).isMuchSmallerThan(0.001));
    ASSERT_TRUE((shouldNorm - di->m_test.at(0).input).isMuchSmallerThan(0.001));

}


