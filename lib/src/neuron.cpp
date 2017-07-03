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

#include "neuron.h"
#include <cmath>

double Neuron::sigmoid( const double& z )
{
    return 1.0 / ( 1.0 + std::exp(-z) );
}

double Neuron::d_sigmoid( const double& z )
{
    return sigmoid(z) * ( 1.0 - sigmoid(z) );
}

const Eigen::MatrixXd Neuron::d_sigmoid( const Eigen::MatrixXd& z )
{
    Eigen::MatrixXd res = Eigen::MatrixXd( z.rows(), z.cols() );
    for( unsigned int m = 0; m < z.rows(); m++ )
        for( unsigned int n = 0; n < z.cols(); n++ )
            res(m,n) = Neuron::d_sigmoid( z(m,n) );

    return res;
}
