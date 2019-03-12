//
// Created by Adrian Schneider on 2019-03-13.
//

#ifndef EIDNN_CARFACTORY_H
#define EIDNN_CARFACTORY_H

#include "network.h"
#include "simulation.h"

class TrackMap;

class CarFactory: public SimulationFactory
{
public:
    CarFactory(std::shared_ptr<TrackMap> map);

    ~CarFactory() override;

    std::shared_ptr<Simulation> createRandomSimulation() override;

    SimulationPtr createCrossover( SimulationPtr a, SimulationPtr b, double mutationRate) override;

    SimulationPtr copy( SimulationPtr a ) override;

private:
    void setAllBiasToZero(NetworkPtr net);
    std::shared_ptr<TrackMap> m_map;

};


#endif //EIDNN_CARFACTORY_H
