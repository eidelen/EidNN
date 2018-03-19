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

#include "dataInput.h"

using namespace std;

DataInput::DataInput( )
{
}

DataInput::~DataInput()
{
}
void DataInput::addTrainingSample(const Eigen::MatrixXd &input, const Eigen::MatrixXd &expectedOutput)
{
    DataElement de;
    de.input = input;
    de.output = expectedOutput;
    m_training.push_back(de);
}

void DataInput::addTestSample(const Eigen::MatrixXd &input, const Eigen::MatrixXd &expectedOutput)
{
    DataElement de;
    de.input = input;
    de.output = expectedOutput;
    m_test.push_back(de);
}

void DataInput::clear()
{
    m_training.clear();
    m_test.clear();
}

size_t DataInput::getNumberOfTrainingSamples() const
{
    return m_training.size();
}

size_t DataInput::getNumberOfTestSamples() const
{
    return m_test.size();
}
