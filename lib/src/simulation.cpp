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


#include <inc/simulation.h>

#include "simulation.h"
#include "genetic.h"


Simulation::Simulation()
{
    setLastUpdateTime(now());
    m_creation = now();
}

Simulation::~Simulation()
{

}

void Simulation::doStep()
{
    if( isAlive() )
    {
        update();
        setLastUpdateTime(now());
    }
}

void Simulation::update()
{
    // Do override
}

double Simulation::getFitness()
{
    return 0.0;
}

std::chrono::milliseconds Simulation::now() const
{
    return std::chrono::duration_cast< std::chrono::milliseconds >(
            std::chrono::system_clock::now().time_since_epoch());
}

const std::chrono::milliseconds &Simulation::getLastUpdateTime() const
{
    return m_lastUpdate;
}

void Simulation::setLastUpdateTime(const std::chrono::milliseconds &lastUpdate)
{
    m_lastUpdate = lastUpdate;
}

double Simulation::getTimeSinceLastUpdate() const
{
    return (now() - m_lastUpdate).count() / 1000.0;
}

const std::shared_ptr<Network>& Simulation::getNetwork() const
{
    return m_network;
}

void Simulation::setNetwork(const std::shared_ptr<Network> &network)
{
    m_network = network;
}

bool Simulation::isAlive() const
{
    return m_alive;
}

double Simulation::getAge() const
{
    return ((double)(m_lastUpdate - m_creation).count()) / 1000.0;
}

void Simulation::kill()
{
    m_alive = false;
}


// Factory

SimulationFactory::SimulationFactory()
{
}

SimulationFactory::~SimulationFactory()
{
}

SimulationPtr SimulationFactory::createRandomSimulation()
{
    return std::shared_ptr<Simulation>();
}

SimulationPtr SimulationFactory::createCrossover( SimulationPtr a, SimulationPtr b, double mutationRate)
{
    NetworkPtr cr = Genetic::crossover(a->getNetwork(), b->getNetwork(), Genetic::CrossoverMethod::Uniform, mutationRate);
    SimulationPtr crs = createRandomSimulation();
    crs->setNetwork(cr);
    return crs;
}

