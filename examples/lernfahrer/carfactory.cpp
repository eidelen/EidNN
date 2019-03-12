//
// Created by Adrian Schneider on 2019-03-13.
//

#include "carfactory.h"
#include "car.h"
#include "genetic.h"
#include "layer.h"
#include "trackmap.h"

CarFactory::CarFactory(std::shared_ptr<TrackMap> map)
{
    m_map = map;
}

CarFactory::~CarFactory()
{

}

std::shared_ptr<Simulation> CarFactory::createRandomSimulation()
{
    std::shared_ptr<Car> car( new Car( ) );

    car->setMap(m_map);
    car->setPosition(Eigen::Vector2d(400,345) );
    car->setDirection(Eigen::Vector2d(1,0));

    setAllBiasToZero(car->getNetwork());

    return car;
}

SimulationPtr CarFactory::createCrossover(SimulationPtr a, SimulationPtr b, double mutationRate)
{
    NetworkPtr cr = Genetic::crossover(a->getNetwork(), b->getNetwork(), Genetic::CrossoverMethod::Uniform, mutationRate);
    setAllBiasToZero(cr);

    SimulationPtr crs = createRandomSimulation();
    crs->setNetwork(cr);
    return crs;
}

void CarFactory::setAllBiasToZero(NetworkPtr net)
{
    for( unsigned int i = 0; i < net->getNumberOfLayer(); i++ )
        net->getLayer(i)->setBias(0.0);
}

SimulationPtr CarFactory::copy( SimulationPtr a )
{
    SimulationPtr crs = createRandomSimulation();
    crs->setNetwork(a->getNetwork());
    return crs;
}
