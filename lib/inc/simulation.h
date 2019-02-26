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

#ifndef SIMULATION_H
#define SIMULATION_H

#include <chrono>
#include <memory>

#include "network.h"

#define SimulationPtr std::shared_ptr<Simulation>

class Simulation
{

public:

    Simulation();
    virtual ~Simulation();

    /**
     * Update the simulation.
     */
    void doStep();

    /**
     * Fitness is a measure performance.
     * @return Fitness.
     */
    virtual double getFitness();

    /**
     * Get time stamp of the last update.
     * @return Milliseconds since 1970.
     */
    virtual const std::chrono::milliseconds &getLastUpdateTime() const;

    /**
     * Set time stamp of last update.
     * @param lastUpdate
     */
    virtual void setLastUpdateTime(const std::chrono::milliseconds &lastUpdate);

    /**
     * Time since last update in seconds.
     * @return Elapsed time in seconds
     */
    virtual double getTimeSinceLastUpdate() const;

    /**
     * Get neuronal network.
     * @return NN
     */
    virtual const std::shared_ptr<Network> &getNetwork() const;

    /**
     * Set neuronal network.
     * @param network NN
     */
    virtual void setNetwork(const std::shared_ptr<Network> &network);

    /**
     * Is the simulation still alive.
     */
    virtual bool isAlive() const;

protected:

    std::chrono::milliseconds now() const;

    /**
     * Update the actual simulation.
     */
    virtual void update();


protected:

    std::chrono::milliseconds m_lastUpdate;
    bool m_alive = true;
    NetworkPtr m_network;
};

#define SimFactoryPtr std::shared_ptr<SimulationFactory>

class SimulationFactory
{
public:
    SimulationFactory();
    virtual ~SimulationFactory();

    virtual SimulationPtr createRandomSimulation();
    virtual SimulationPtr createCrossover( SimulationPtr a, SimulationPtr b, double mutationRate );
};


#endif // SIMULATION_H
