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
#include <numeric>
#include <thread>


Evolution::Evolution(size_t nInitial, size_t nNext, SimFactoryPtr simFactory, unsigned int nThreads)
: m_nInitials(nInitial), m_nOffsprings(nNext), m_simFactory(simFactory), m_epochOver(false), m_epochCount(0), m_mutationRate(0.0),
  m_stepCounter(0), m_simSpeed(0.0), m_nbrThreads(nThreads)
{
    m_simSpeedTime = now();
    std::generate_n(std::back_inserter(m_simulations), nInitial, [simFactory]()->SimulationPtr { return simFactory->createRandomSimulation(); });
}

Evolution::~Evolution()
{

}

void Evolution::doStepOnFewSimulations( std::vector<SimulationPtr>& sims, std::atomic_bool& anyAlive, size_t start, size_t end )
{
    for( size_t k = start; k < end; k++ )
    {
        SimulationPtr s = sims[k];
        if (s->isAlive())
        {
            s->doStep();
            anyAlive = true;
        }
    }
}

void Evolution::doStep()
{
    m_stepCounter++;

    std::atomic_bool anyAlive = false;
    std::vector<std::thread> thv;
    size_t samplesPerThread = m_simulations.size() / m_nbrThreads;
    for(unsigned int th = 0; th < m_nbrThreads; th++)
    {
        size_t startPos = th * samplesPerThread;
        size_t endPos = startPos + samplesPerThread;
        if( th == m_nbrThreads - 1 ) // last thread till end
            endPos = m_simulations.size();

        thv.emplace_back(std::thread( doStepOnFewSimulations, std::ref(m_simulations), std::ref(anyAlive), startPos, endPos ));
    }

    for( std::thread& th : thv )
        th.join();

    if( !anyAlive )
    {
        m_epochOver = !anyAlive;
        m_epochCount++;
    }

    // update sim speed every 10 steps
    if( m_stepCounter % 20 == 0 )
    {
        std::chrono::milliseconds n = now();
        m_simSpeed = m_stepCounter / ( (now()-m_simSpeedTime).count() / 1000.0);
        m_stepCounter = 0;
        m_simSpeedTime = n;
    }
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

std::pair<size_t, size_t> Evolution::getNumberAliveAndDead() const
{
    size_t alive = 0;
    size_t dead = 0;


    for( auto m: m_simulations )
        if( m->isAlive() )
            alive++;
        else
            dead++;

    return std::pair<size_t,size_t>(alive,dead);
}

double Evolution::getSimulationsAverageAge() const
{
    return std::accumulate(m_simulations.begin(), m_simulations.end(), 0.0,   [] (double total, auto i) { return total + i->getAge(); }) / (double)m_simulations.size();
}

void Evolution::killAllSimulations()
{
    for( auto m: m_simulations )
        m->kill();
}

double Evolution::getSimulationStepsPerSecond() const
{
    return m_simSpeed;
}

std::chrono::milliseconds Evolution::now() const
{
    return std::chrono::duration_cast< std::chrono::milliseconds >(
            std::chrono::system_clock::now().time_since_epoch());
}


