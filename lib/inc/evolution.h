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

#ifndef _EVOLUTION_H_
#define _EVOLUTION_H_

#include "simulation.h"

#include <memory>
#include <vector>

/**
 * This class runs simulations and evolutions, and keeps track of the
 * best networks.
 */
class Evolution
{

public:

    /**
     * Constructor
     * @param nInitial How many random initialized genoms (first epoch)
     * @param nNext How many offsprings generated among best genoms (further epochs)
     * @param simFactory Factory for simulations
     * @param nThreads Number of threads used for computation
     */
    Evolution( size_t nInitial, size_t nNext, SimFactoryPtr simFactory, unsigned int nThreads = 4 );

    virtual ~Evolution();

    /**
     * A single, discrete simulation step.
     */
    void doStep();

    /**
     * Run a whole epoch. An epoch ends when all simulations died.
     */
    void doEpoch();

    /**
     * Create the next generation.
     */
    void breed();

    /**
     * Checks if Epoch is over. Epoch ends when all simulations died.
     * @return
     */
    bool isEpochOver();

    /**
     * Get vector of simulations ordered by fitness. Best first.
     * @return Vector.
     */
    std::vector<SimulationPtr> getSimulationsOrderedByFitness();

    /**
     * Get number of run epochs.
     * @return Number of epochs.
     */
    size_t getNumberOfEpochs() const;

    /**
     * Get the nubmer of alive and dead simulations.
     */
    std::pair<size_t, size_t> getNumberAliveAndDead( ) const;

    /**
     * Average simulations age.
     * @return seconds.
     */
    double getSimulationsAverageAge() const;

    /**
     * Get mutation rate.
     * @return Mutation rate (0.0 - 1.0)
     */
    double getMutationRate() const;

    /**
     * Set mutation rate.
     * @param mutationRate 0.0 - 1.0
     */
    void setMutationRate(double mutationRate);

    /**
     * Are parents kept for the next epoch.
     * @return True if parents are kept for next epoch.
     */
    bool isKeepParents() const;

    /**
     * If true, parents are part of the next epoch.
     * @param keepParents True -> parents are part of next epoch. Otherwise false.
     */
    void setKeepParents(bool keepParents);

    /**
     * Kill all simulations.
     */
    void killAllSimulations();

    /**
     * Get the number of simulation steps achieved in 1 second.
     * @return Simulation rate.
     */
    double getSimulationStepsPerSecond() const;

    /**
     * Set a new factory object.
     * @param simFactory Factory.
     */
    void resetFactory(SimFactoryPtr simFactory);

    /**
     * Safe the evolution -> the two best
     * @param a_path Path to fittest
     * @param b_path Path to second fittest
     */
    void save(const std::string &a_path, const std::string &b_path);

    /**
     * Load the evolution -> the two best
     * @param a_path Path to fittest
     * @param b_path Path to second fittest
     * @return True if successful. Otherwise false.
     */
    bool load( const std::string& a_path, const std::string& b_path );

private:
    std::chrono::milliseconds now() const;
    static void doStepOnFewSimulations( std::vector<SimulationPtr>& sims, std::atomic_bool& anyAlive, size_t start, size_t end );


private:
    size_t m_nInitials;
    size_t m_nOffsprings;
    SimFactoryPtr m_simFactory;
    std::vector<SimulationPtr> m_simulations;
    bool m_epochOver;
    size_t m_epochCount;
    double m_mutationRate;
    size_t m_stepCounter;
    std::chrono::milliseconds m_simSpeedTime;
    double m_simSpeed;
    unsigned int m_nbrThreads;
    bool m_keepParents;
    SimulationPtr m_fittest;
};



#endif // _EVOLUTION_H_