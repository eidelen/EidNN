
#include "car.h"
#include "genetic.h"
#include "layer.h"

#include <iostream>
#include <Eigen/Geometry>

Car::Car()
{
    setSpeed( 0.0 );
    setPosition( Eigen::Vector2d(0.0, 0.0));
    setAcceleration(0.0);
    setDirection(Eigen::Vector2d(1.0, 0.0));
    setRotationSpeed(0.0);

    m_rotationToOriginal = 0.0;
    m_mapSet = false;

    setMeasureAngles( {-80, -55.0, -25.0, 0.0, 25.0, 55.0, 80} );

    m_droveDistance = 0.0;

    std::vector<unsigned int> map = {7,2};
    m_network = NetworkPtr( new Network(map) );

    m_killer.start();
    m_formerDistance = 0.0;
}

Car::~Car()
{

}

double Car::getSpeed() const
{
    return m_speed;
}

void Car::setSpeed(double speed)
{
    m_speed = speed;
    m_lastSpeed = speed;
}

const Eigen::Vector2d &Car::getPosition() const
{
    return m_position;
}

void Car::setPosition(const Eigen::Vector2d &position)
{
    m_position = position;
}

double Car::getAcceleration() const
{
    return m_acceleration;
}

void Car::setAcceleration(double acceleration)
{
    m_acceleration = acceleration;
}

const Eigen::Vector2d &Car::getDirection() const
{
    return m_direction;
}

void Car::setDirection(const Eigen::Vector2d &direction)
{
    m_direction = direction;
    m_direction.normalize();
    m_rotationToOriginal = computeAngleBetweenVectors(Eigen::Vector2d(1.0,0.0), m_direction);
}

void Car::update()
{
    double animTime = getTimeSinceLastUpdate();

    // rotate first
    Eigen::Rotation2D<double> r( m_rotationSpeedRad *animTime );
    m_direction = r.toRotationMatrix() * m_direction;
    Eigen::Vector2d(1.0,0.0);
    m_rotationToOriginal = computeAngleBetweenVectors(Eigen::Vector2d(1.0,0.0), m_direction);


    double newSpeed = std::max(m_speed + animTime*getAcceleration(), 0.0);
    newSpeed = std::min(newSpeed,400.0);

    Eigen::Vector2d newSpeedVector = m_direction * newSpeed;
    Eigen::Vector2d oldSpeedVector = m_direction * m_lastSpeed;
    Eigen::Vector2d effectiveSpeed = (newSpeedVector + oldSpeedVector) * 0.5;

    Eigen::Vector2d newPosition = getPosition() + animTime * effectiveSpeed;

    newPosition = handleCollision(getPosition(), newPosition);

    m_droveDistance = m_droveDistance + (newPosition - getPosition()).norm();

    setPosition( newPosition );
    setSpeed( newSpeed );


    // decide what to do next
    m_measuredDistances = measureDistances();
    Eigen::MatrixXd nnInput = m_measuredDistances.col(0);

    // normalize input
    double maxValInput = nnInput.maxCoeff();
    nnInput = nnInput * 1.0/maxValInput;

    m_network->feedForward(nnInput);
    Eigen::MatrixXd nnOut = m_network->getOutputActivation();

    //std::cout << nnOut.transpose() << std::endl;


    double maxRotationSpeed = 360.0;
    double maxAcceleration = 100.0;

    // scale output from 0 - 1 to -1 to +1
    double speedActivation = (nnOut(0,0) - 0.5) * 2;
    double rotationActivation = (nnOut(1,0) - 0.5) * 2;
    setAcceleration(maxAcceleration*speedActivation);
    setRotationSpeed(maxRotationSpeed*rotationActivation);

    if( m_killer.elapsed() > 1000 )
    {
        considerSuicide();
        m_killer.restart();
    }
}

double Car::getFitness()
{
    return m_droveDistance;
}

double Car::getRotationSpeed() const
{
    return m_rotationSpeed;
}

void Car::setRotationSpeed(double rotationSpeed)
{
    m_rotationSpeed = rotationSpeed;
    m_rotationSpeedRad = -(M_PI / 180.0 * m_rotationSpeed);
}

double Car::getRotationRelativeToInitial() const
{
    return m_rotationToOriginal;
}

double Car::computeAngleBetweenVectors(const Eigen::Vector2d &a, const Eigen::Vector2d &b) const
{
    return std::acos( a.dot(b) / ( a.norm() * b.norm() ) ) / M_PI * 180.0;
}

const Eigen::MatrixXi& Car::getMap() const
{
    return m_map;
}

void Car::setMap(const Eigen::MatrixXi &map)
{
    m_map = map;
    m_mapSet = true;
}

Eigen::Vector2d Car::handleCollision(const Eigen::Vector2d& from, const Eigen::Vector2d& to)
{
    if( !m_mapSet )
        return to;

    if( isPositionValid(from) == 0 )
    {
        m_alive = false;
        return from;
    }

    Eigen::Vector2d dif = to - from;
    double l = dif.norm();

    if( l < 0.00000001 ) // no or very small move
        return to;

    Eigen::Vector2d du = dif * 1.0 / l;

    double tillEdge = distanceToEdge(from,du);

    if( tillEdge < l ) // collision
    {
        m_alive = false;
        return from;
    }

    return to;
}

double Car::distanceToEdge(const Eigen::Vector2d &pos, const Eigen::Vector2d &direction) const
{
    if( !m_mapSet )
        return 0.0;

    Eigen::Vector2d d = direction.normalized();
    Eigen::Vector2d end = pos;

    bool goOn = true;

    while(isPositionValid(end))
        end = end + d;

    return (end-pos).norm();
}

const std::vector<double> &Car::getMeasureAngles() const
{
    return m_measureAngles;
}

void Car::setMeasureAngles(const std::vector<double> &measureAngles)
{
    m_measureAngles = measureAngles;
}

Eigen::MatrixXd Car::measureDistances() const
{
    // distance, xP, yP
    Eigen::MatrixXd angs( m_measureAngles.size(), 3);
    for( size_t k = 0; k < m_measureAngles.size(); k++ )
    {
        double aRad = -(M_PI / 180.0 * m_measureAngles[k]);
        Eigen::Rotation2D<double> r( aRad );
        Eigen::Vector2d measureDir = r.toRotationMatrix() * m_direction;

        double dist = distanceToEdge(getPosition(),measureDir);

        Eigen::Vector2d edgePos = getPosition() + measureDir * dist;
        angs(k,0) = dist;
        angs(k,1) = edgePos(0);
        angs(k,2) = edgePos(1);
    }

    return angs;
}

Eigen::MatrixXd Car::getMeasuredDistances() const
{
    return m_measuredDistances;
}

void Car::considerSuicide()
{
    if(( getAge() > 1.0 && m_droveDistance < 3.0) || m_droveDistance < m_formerDistance * 1.02 )
    {
        m_alive = false;
    }
    else
    {
        m_formerDistance = m_droveDistance;
    }
}

bool Car::isPositionValid(const Eigen::Vector2d &pos) const
{
    if(pos(0) < 0.0 ||  pos(0) > m_map.cols()-1 || pos(1) < 0.0 ||  pos(1) > m_map.rows()-1)
        return false;

    return m_map( std::ceil(pos(1)) , std::ceil(pos(0))) == 1;
}


CarFactory::CarFactory(const Eigen::MatrixXi &map) : m_map(map)
{

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

    return car;
}

SimulationPtr CarFactory::createCrossover(SimulationPtr a, SimulationPtr b, double /*mutationRate*/)
{
    NetworkPtr cr = Genetic::crossover(a->getNetwork(), b->getNetwork(), Genetic::CrossoverMethod::Uniform, 0.02);

    // set all bias to zero
    for( size_t i = 0; i < cr->getNumberOfLayer(); i++ )
    {
        cr->getLayer(i)->setBias(0.0);
    }

    SimulationPtr crs = createRandomSimulation();
    crs->setNetwork(cr);
    return crs;
}


