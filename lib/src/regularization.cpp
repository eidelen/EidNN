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

#include <inc/regularization.h>

#include "regularization.h"

Regularization::Regularization(Regularization::RegularizationMethod method, double lamda) :
m_method(method), m_lamda(lamda), m_weightSum(1.0), m_nbrSamples(1)
{
}

Regularization::Regularization() : m_method(Regularization::RegularizationMethod::NoneRegularization), m_lamda(1.0), m_weightSum(1.0), m_nbrSamples(1)
{
}

Regularization::~Regularization()
{

}
std::string Regularization::toString() const
{
    std::string ret;

    switch( m_method )
    {
        case RegularizationMethod::NoneRegularization:
            ret = "None";
            break;

        case RegularizationMethod::WeightDecay:
            ret = "Weight Decay";
            break;
    }

    return ret;
}
double Regularization::regularizationCost() const
{
    double regCost = 0.0;

    switch( m_method )
    {
        case RegularizationMethod::NoneRegularization:
            break;

        case RegularizationMethod::WeightDecay:
            regCost = m_lamda / (2.0 * m_nbrSamples) * m_weightSum;
            break;
    }

    return regCost;
}
