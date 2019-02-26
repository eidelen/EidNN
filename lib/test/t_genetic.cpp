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
#include <algorithm>

#include "genetic.h"
#include "layer.h"
#include "helpers.h"


TEST(Genetic, CrossoverEqualNetworks)
{
    auto a = std::shared_ptr<Network>(new Network({20,50}));
    a->resetWeights();
    auto b = std::shared_ptr<Network>(new Network(*(a.get())));

    auto c = Genetic::crossover(a,b,Genetic::Uniform);

    //b == a -> c == a

    ASSERT_TRUE( std::equal( c->getNetworkStructure().begin(), c->getNetworkStructure().end(),  a->getNetworkStructure().begin() ) );

    for(unsigned int k = 0; k < c->getNumberOfLayer(); k++ )
    {
        auto l_a = a->getLayer(k);
        auto l_c = c->getLayer(k);

        ASSERT_TRUE((l_a->getBiasVector() - l_c->getBiasVector()).isMuchSmallerThan(0.00001));
        ASSERT_TRUE((l_a->getWeightMatrix() - l_c->getWeightMatrix()).isMuchSmallerThan(0.00001));
    }
}

TEST(Genetic, Crossover)
{
    auto a = std::shared_ptr<Network>(new Network({20,50,30}));
    auto b = std::shared_ptr<Network>(new Network({20,50,30}));

    a->getLayer(1)->setWeight(0.0);
    a->getLayer(1)->setBias(1.0);
    a->getLayer(2)->setWeight(2.0);
    a->getLayer(2)->setBias(3.0);

    b->getLayer(1)->setWeight(4.0);
    b->getLayer(1)->setBias(5.0);
    b->getLayer(2)->setWeight(6.0);
    b->getLayer(2)->setBias(7.0);

    auto c = Genetic::crossover(a,b,Genetic::Uniform);

    //b == a -> c == a

    ASSERT_TRUE( std::equal( c->getNetworkStructure().begin(), c->getNetworkStructure().end(),  a->getNetworkStructure().begin() ) );

    auto l1 = c->getLayer(1);
    auto l1w = l1->getWeightMatrix();
    auto l1b = l1->getBiasVector();

    size_t cntA = 0;
    size_t cntB = 0;

    for( size_t m = 0; m < l1w.rows(); m++ )
    {
        for( size_t n = 0; n < l1w.cols(); n++ )
        {
            double w = l1w(m,n);

            if( w == 0.0 )
                cntA++;
            else if( w == 4.0 )
                cntB++;
            else
                ASSERT_TRUE( false );
        }
    }

    double ratio = ((double)(cntB)) / ((double)(cntA));
    ASSERT_NEAR( ratio, 1.0, 0.8 );



    cntA = 0;
    cntB = 0;
    for( size_t m = 0; m < l1b.rows(); m++ )
    {
        double b = l1b(m);
        if( b == 1.0 )
            cntA++;
        else if( b == 5.0 )
            cntB++;
        else
            ASSERT_TRUE( false );
    }

    ratio = ((double)(cntB)) / ((double)(cntA));
    ASSERT_NEAR( ratio, 1.0, 0.8 );



    auto l2 = c->getLayer(2);
    auto l2w = l2->getWeightMatrix();
    auto l2b = l2->getBiasVector();

    cntA = 0;
    cntB = 0;

    for( size_t m = 0; m < l2w.rows(); m++ )
    {
        for( size_t n = 0; n < l2w.cols(); n++ )
        {
            double w = l2w(m,n);

            if( w == 2.0 )
                cntA++;
            else if( w == 6.0 )
                cntB++;
            else
                ASSERT_TRUE( false );
        }
    }

    ratio = ((double)(cntB)) / ((double)(cntA));
    ASSERT_NEAR( ratio, 1.0, 0.8 );

    cntA = 0;
    cntB = 0;
    for( size_t m = 0; m < l2b.rows(); m++ )
    {
        double b = l2b(m);
        if( b == 3.0 )
            cntA++;
        else if( b == 7.0 )
            cntB++;
        else
            ASSERT_TRUE( false );
    }

    ratio = ((double)(cntB)) / ((double)(cntA));
    ASSERT_NEAR( ratio, 1.0, 0.8 );

}

TEST(Genetic, CrossoverMutation)
{
    auto a = std::shared_ptr<Network>(new Network({20,50}));
    a->resetWeights();
    auto b = std::shared_ptr<Network>(new Network(*(a.get())));

    auto c = Genetic::crossover(a,b,Genetic::Uniform,0.1);

    ASSERT_TRUE( std::equal( c->getNetworkStructure().begin(), c->getNetworkStructure().end(),  a->getNetworkStructure().begin() ) );

    for(unsigned int k = 1; k < c->getNumberOfLayer(); k++ )
    {
        auto l_a = a->getLayer(k);
        auto l_c = c->getLayer(k);

        ASSERT_FALSE((l_a->getBiasVector() - l_c->getBiasVector()).isMuchSmallerThan(0.00001));
        ASSERT_FALSE((l_a->getWeightMatrix() - l_c->getWeightMatrix()).isMuchSmallerThan(0.00001));
    }
}