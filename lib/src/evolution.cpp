

#include "evolution.h"

#include <algorithm>
#include <inc/evolution.h>


Evolution::Evolution(size_t nInitial, size_t nNext, SimFactoryPtr simFactory)
: m_nInitials(nInitial), m_nOffsprings(nNext), m_simFactory(simFactory), m_epochOver(false)
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

    std::generate_n(std::back_inserter(m_simulations), m_nOffsprings, [=]()->SimulationPtr { return m_simFactory->createCrossover(a,b); });

}

