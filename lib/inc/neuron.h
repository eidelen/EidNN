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

#ifndef NEURONHEADER
#define NEURONHEADER

#include <Eigen/Dense>

class Neuron
{

public:

    /**
    * Sigmoid function value of a given input z.
    */
    static double sigmoid(const double &z );

    /**
    * Sigmoid derivation function value of a given input z.
    */
    static double d_sigmoid(const double &z );

    /**
     * Computes component wise derrivative of the sigmoid function
     * for each component in the vector z.
     * @param z Vector in
     * @return Vector holding the result.
     */
    static const Eigen::MatrixXd d_sigmoid(const Eigen::MatrixXd &z );
};

#endif //NEURONHEADER
