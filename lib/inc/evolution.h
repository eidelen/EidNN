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
     */
    Evolution( size_t nInitial, size_t nNext, SimFactoryPtr simFactory );

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



private:
    size_t m_nInitials;
    size_t m_nOffsprings;
    SimFactoryPtr m_simFactory;
    std::vector<SimulationPtr> m_simulations;
    bool m_epochOver;
};



#endif // _EVOLUTION_H_