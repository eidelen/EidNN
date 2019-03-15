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
#include "evolution.h"
#include "network.h"
#include "genetic.h"
#include "helpers.h"
#include <memory>


class OneStepSimulation: public Simulation
{
public:

    OneStepSimulation()
    {
        std::vector<unsigned int> map = {2,10,2};
        m_network = NetworkPtr( new Network(map) );
    }

    ~OneStepSimulation()
    {

    }

    double getFitness() override
    {
        Eigen::MatrixXd should(2,1);
        should(0,0) = 0.5;
        should(1,0) = 0.5;

        double diff = (should - m_network->getOutputActivation()).norm();
        double fitness = 1.0 / diff;
        return fitness;
    }

protected:

    void update() override
    {
        Eigen::MatrixXd mat(2,1);// = Eigen::MatrixXd::Random(2, 1);
        mat(0,0) = 0.2;
        mat(1,0) = 0.4;
        m_network->feedForward(mat);
        m_alive = false;
    }

};

class OneStepSimFactory: public SimulationFactory
{
public:
    OneStepSimFactory()
    {

    }

    ~OneStepSimFactory()
    {

    }

    std::shared_ptr<Simulation> createRandomSimulation() override
    {
        return std::shared_ptr<OneStepSimulation>(new OneStepSimulation());
    }

};



TEST(Evolution, InitialRun)
{
    std::shared_ptr<OneStepSimFactory> f(new OneStepSimFactory());

    Evolution* e = new Evolution(100,200,f);

    ASSERT_FALSE(e->isEpochOver());
    e->doEpoch();
    ASSERT_TRUE(e->isEpochOver());


    std::vector<SimulationPtr> simRes = e->getSimulationsOrderedByFitness();

    ASSERT_EQ(simRes.size(), 100);

    double bestF = std::numeric_limits<double>::max();
    for( auto s : simRes )
    {
        double tf = s->getFitness();
        ASSERT_LE( tf, bestF );
        bestF = tf;
    }

    std::cout << simRes[0]->getFitness() << std::endl;
    Helpers::printMatrix(simRes[0]->getNetwork()->getOutputActivation(), "best");

    delete e;
}

TEST(Evolution, MultiEpochs)
{
    std::shared_ptr<OneStepSimFactory> f(new OneStepSimFactory());

    Evolution* e = new Evolution(500,1000,f);
    e->setMutationRate(0.05);

    while(true)
    {
        e->doEpoch();

        SimulationPtr best = e->getSimulationsOrderedByFitness()[0];

        if( best->getFitness() > 200 )
        {
            Helpers::printMatrix( best->getNetwork()->getOutputActivation(), "best solution");
            std::cout << "Number of epochs: " << e->getNumberOfEpochs() << std::endl;
            break;
        }

        e->breed();
    }

    ASSERT_LE(e->getNumberOfEpochs(), 50);

    delete e;
}

TEST(Evolution, SaveAndLoad)
{
    std::shared_ptr<OneStepSimFactory> f(new OneStepSimFactory());

    Evolution* e = new Evolution(100,100,f);
    e->doStep();
    e->save("a.net", "b.net");

    Evolution* q = new Evolution(100,100,f);
    q->doStep();
    q->load("a.net", "b.net");


    NetworkPtr e_a = e->getSimulationsOrderedByFitness()[0]->getNetwork();
    NetworkPtr e_b = e->getSimulationsOrderedByFitness()[1]->getNetwork();

    NetworkPtr q_a = q->getSimulationsOrderedByFitness()[0]->getNetwork();
    NetworkPtr q_b = q->getSimulationsOrderedByFitness()[1]->getNetwork();

    // check that same activation leads to same output
    Eigen::MatrixXd input(2,1);// = Eigen::MatrixXd::Random(2, 1);
    input(0,0) = 0.5;
    input(1,0) = 0.5;

    e_a->feedForward(input);
    e_b->feedForward(input);
    q_a->feedForward(input);
    q_b->feedForward(input);

    ASSERT_TRUE( ( e_a->getOutputActivation() -  q_a->getOutputActivation() ).isZero(0.001) );
    ASSERT_TRUE( ( e_b->getOutputActivation() -  q_b->getOutputActivation() ).isZero(0.001) );



    delete e;
    delete q;
}