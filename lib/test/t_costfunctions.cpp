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
#include "crossEntropyCost.h"
#include "quadraticCost.h"
#include "neuron.h"

TEST(CostFunction, QuadraticCostDelta)
{
    // Test with 3 samples, each 2 outputs

    QuadraticCost* qc = new QuadraticCost();

    Eigen::MatrixXd z = Eigen::MatrixXd::Constant(2, 3, 0.0);
    Eigen::MatrixXd a = Eigen::MatrixXd::Constant(2, 3, 0.5);
    Eigen::MatrixXd y = Eigen::MatrixXd::Constant(2, 3, 0.6);

    // (0.5 - 0.6) * d_sig(0.0) = -0.1 * 0.25 = -0.025
    Eigen::MatrixXd delta = qc->delta(z,a,y);
    for( int m = 0; m < 2; m++ )
        for( int n = 0; n < 3; n++ )
            ASSERT_NEAR( delta(m,n), -0.025, 0.000001);

    delete qc;
}

TEST(CostFunction, QuadraticCost)
{
    // Test with 3 samples, each 2 outputs

    QuadraticCost* qc = new QuadraticCost();

    Eigen::MatrixXd a = Eigen::MatrixXd::Constant(2, 3, 1.0);
    Eigen::MatrixXd y = Eigen::MatrixXd::Constant(2, 3, 2.0);

    // 0.5 * (  ||[1,1] - [2,2]|| )^2  = 0.5 * 1.4142^2 = 0.5 * 2.0 = 1.0
    double cost = qc->cost(a,y);
    ASSERT_NEAR( cost, 1.0, 0.000001);

    delete qc;
}

TEST(CostFunction, CrossEntropyCostDelta)
{
    // Test with 3 samples, each 2 outputs

    CrossEntropyCost* cc = new CrossEntropyCost();

    Eigen::MatrixXd a = Eigen::MatrixXd::Constant(2, 3, 0.5);
    Eigen::MatrixXd y = Eigen::MatrixXd::Constant(2, 3, 0.6);
    Eigen::MatrixXd z_unused = Eigen::MatrixXd::Constant(2, 3, 0.0);

    // (0.5 - 0.6) = -0.1
    Eigen::MatrixXd delta = cc->delta(z_unused,a,y);
    for( int m = 0; m < 2; m++ )
        for( int n = 0; n < 3; n++ )
            ASSERT_NEAR( delta(m,n), -0.1, 0.000001);

    delete cc;
}

TEST(CostFunction, CrossEntropyCost)
{
    /// Test with 3 samples, each 2 outputs

    CrossEntropyCost* cc = new CrossEntropyCost();

    Eigen::MatrixXd a = Eigen::MatrixXd::Constant(2, 3, 0.5);
    Eigen::MatrixXd y = Eigen::MatrixXd::Constant(2, 3, 0.6);

    // - (  0.6*ln(0.5)+(1-0.6)*ln(1.0-0.5) )  = 0.69315
    double cost = cc->cost(a,y);
    ASSERT_NEAR( cost, 2.0 * 0.69315, 0.0001);

    delete cc;
}
