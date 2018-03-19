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

#ifndef DATAINPUT_H
#define DATAINPUT_H

#include <vector>
#include <eigen3/Eigen/Dense>

struct DataElement
{
    Eigen::MatrixXd input;
    Eigen::MatrixXd output;
};

class DataInput
{

public:
    DataInput();
    ~DataInput();

    /**
     * Add a training sample.
     * @param input Sample input.
     * @param expectedOutput Expected sample output.
     */
    void addTrainingSample( const Eigen::MatrixXd& input, const Eigen::MatrixXd& expectedOutput);

    /**
     * Add a test sample.
     * @param input Sample input.
     * @param expectedOutput Expected sample output.
     */
    void addTestSample( const Eigen::MatrixXd& input, const Eigen::MatrixXd& expectedOutput);

    /**
     * Clear all test and training data.
     */
    void clear();

    /**
     * Returns the number of training samples.
     * @return Number of training samples
     */
    size_t getNumberOfTrainingSamples() const;

    /**
     * Returns the number of test samples.
     * @return Number of test samples
     */
    size_t getNumberOfTestSamples() const;


public:

    std::vector<DataElement> m_training;
    std::vector<DataElement> m_test;

};

#endif //DATAINPUT_H
