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

#include "genetic.h"
#include "layer.h"

#include <iostream>
#include <random>

NetworkPtr Genetic::crossover(NetworkPtr a, NetworkPtr b, Genetic::CrossoverMethod method)
{
    auto kv = a->getNetworkStructure();
    auto qv = b->getNetworkStructure();

    if( ! std::equal( a->getNetworkStructure().begin(), a->getNetworkStructure().end(),  b->getNetworkStructure().begin() ) )
    {
        std::cout << "Genetic::crossover, Error mismatching network sizes" << std::endl;
        return std::shared_ptr<Network>(nullptr);
    }

    NetworkPtr cross = std::shared_ptr<Network>( new Network(*(a.get())) );

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for( unsigned int i = 0; i < cross->getNumberOfLayer(); i++ )
    {
        auto al = a->getLayer(i);
        auto bl = b->getLayer(i);
        auto crl = cross->getLayer(i);

        // crossover weight matrix
        auto aw = al->getWeightMatrix();
        auto bw = bl->getWeightMatrix();
        Eigen::MatrixXd crlw(aw.rows(), aw.cols());

        for( size_t m = 0; m < crlw.rows(); m++ )
        {
            for( size_t n = 0; n < crlw.cols(); n++ )
            {
                if( dis(gen) == 0 )
                    crlw(m,n) = aw(m,n);
                else
                    crlw(m,n) = bw(m,n);
            }
        }

        crl->setWeights(crlw);

        // crossover bias vector
        auto ab = al->getBiasVector();
        auto bb = bl->getBiasVector();
        Eigen::MatrixXd crlb(ab.rows(),1);

        for( size_t m = 0; m < crlb.rows(); m++ )
        {
            if( dis(gen) == 0 )
                crlb(m) = ab(m);
            else
                crlb(m) = bb(m);
        }

        crl->setBiases(crlb);
    }

    return cross;
}
