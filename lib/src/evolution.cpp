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


#include "evolution.h"
#include "layer.h"
#include "helpers.h"


#include <algorithm>
#include <iostream>
#include <inc/evolution.h>


Evolution::Evolution(size_t nInitial, size_t nNext, SimFactoryPtr simFactory)
: m_nInitials(nInitial), m_nOffsprings(nNext), m_simFactory(simFactory), m_epochOver(false), m_epochCount(0), m_mutationRate(0.0)
{
    std::generate_n(std::back_inserter(m_simulations), nInitial, [simFactory]()->SimulationPtr { return simFactory->createRandomSimulation(); });
}

Evolution::~Evolution()
{

}

void Evolution::doStep()
{
    bool anyLive = false;
    for( auto s: m_simulations )
    {
        if (s->isAlive())
        {
            s->doStep();
            anyLive = true;
        }
    }

    m_epochOver = !anyLive;
}

void Evolution::doEpoch()
{
    while( !isEpochOver() )
        doStep();

    m_epochCount++;

}

bool Evolution::isEpochOver()
{
    return m_epochOver;
}

std::vector<SimulationPtr > Evolution::getSimulationsOrderedByFitness()
{
    std::sort( m_simulations.begin(), m_simulations.end(), [](SimulationPtr a, SimulationPtr b) -> bool {
        return a->getFitness() > b->getFitness();
    } );
    return m_simulations;
}

void Evolution::breed()
{
    std::vector<SimulationPtr> ord = getSimulationsOrderedByFitness();
    SimulationPtr a = ord[0];
    SimulationPtr b = ord[1];

    m_simulations.clear();

    std::generate_n(std::back_inserter(m_simulations), m_nOffsprings, [=]()->SimulationPtr { return m_simFactory->createCrossover(a,b,m_mutationRate); });

    m_epochOver = false;
}

size_t Evolution::getNumberOfEpochs() const
{
    return m_epochCount;
}

double Evolution::getMutationRate() const
{
    return m_mutationRate;
}

void Evolution::setMutationRate(double mutationRate)
{
    m_mutationRate = mutationRate;
}

