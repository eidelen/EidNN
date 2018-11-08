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
#include <map>
#include <Eigen/Dense>

struct DataElement
{
    Eigen::MatrixXd input;

    Eigen::MatrixXd output;
    bool outputSet = false;

    int lable = 0;
    bool lableSet = false;
};

class DataLable
{
public:
    DataLable(int lab){lable = lab; }
    DataLable(int lab, const Eigen::MatrixXd& out ){lable = lab; output = out; }

    bool operator< (const DataLable &right) const
    {
        return lable < right.lable;
    }

    int lable = 0;
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
     * Add a training sample. When added last,
     * generateFromLables() needs to be called.
     * @param input Sample input.
     * @param expectedOutput Numeric sample lable.
     */
    void addTrainingSample( const Eigen::MatrixXd& input, int lable);

    /**
     * Add a test sample.
     * @param input Sample input.
     * @param expectedOutput Expected sample output.
     */
    void addTestSample( const Eigen::MatrixXd& input, const Eigen::MatrixXd& expectedOutput);

    /**
    * Add a test sample. When added last,
    * generateFromLables() needs to be called.
    * @param input Sample input.
    * @param expectedOutput Numeric sample lable.
    */
    void addTestSample( const Eigen::MatrixXd& input, int lable);

    /**
     * Normalize the input vectors. The applied normalization
     * method is "Mean 0 and standard deviation 1".
     */
    void normalizeData();

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

    /**
     * Generates sample output automatically based on
     * the number of different numeric lables which
     * were set. This function has to be called after
     * all smaples were added with
     * addTrainingSample( const Eigen::MatrixXd& input, int lable) or
     * addTestSample( const Eigen::MatrixXd& input, int lable);
     *
     * @return True if successfull. Otherwise false.
     */
    bool generateFromLables();

    /**
     * Returns a vector consisting only of the input vectors for the passed data set.
     * @param vector Data set.
     * @return
     */
    static std::vector<Eigen::MatrixXd> getInputData( const std::vector<DataElement>& vector );

    /**
     * Returns a vector consisting only of the output vectors for the passed data set.
     * @param vector Data set.
     * @return
     */
    static std::vector<Eigen::MatrixXd> getOutputData( const std::vector<DataElement>& vector );

    /**
     * Normalize an input vector.
     * @param in
     * @return
     */
    static Eigen::MatrixXd normalize0Mean1Std(const Eigen::MatrixXd& in);

    /**
     * Return the row index of the maximum element in out.
     * @param out
     * @return
     */
    static size_t getStrongestIdx(const Eigen::MatrixXd& out);


    struct DataInputValidation
    {
        bool valid = false;
        size_t inputDataLength = 0;
        size_t outputDataLength = 0;
    };

    /**
     * Checks that there is a common data input vector
     * size and a common data output vector size among
     * all training and testing samples.
     * @return Validation ok if consistent sizes. Otherwise false.
     */
    DataInputValidation validateData() const;


    /**
     * The input signal is a normalized vector, even though it might be an actual
     * image. This function returns a specific representation for this type of data.
     * This function is meant to be overwritten.
     * @param input The input vector.
     * @param representationAvailable Returns true if a dedicated representation is implemented.
     * @return Representation of data.
     */
    virtual Eigen::MatrixXd representation( const Eigen::MatrixXd& input, bool* representationAvailable  ) const;



private:
    bool assignOutput( std::vector<DataElement>& vector );



public:

    std::vector<DataElement> m_training;
    std::vector<DataElement> m_test;
    std::map<int,DataLable> m_lables;

};

#endif //DATAINPUT_H
