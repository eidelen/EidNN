
#include "car.h"
#include "genetic.h"
#include "layer.h"

#include <iostream>
#include <Eigen/Geometry>

Car::Car(): m_rotationToOriginal(0.0), m_mapSet(false), m_droveDistance(0.0),
    m_formerDistance(0.0), m_accumulatedRotation(0.0)
{
    setSpeed( 0.0 );
    setPosition( Eigen::Vector2d(0.0, 0.0));
    setAcceleration(0.0);
    setDirection(Eigen::Vector2d(1.0, 0.0));
    setRotationSpeed(0.0);

    setMeasureAngles( {-80, -55.0, -25.0, 0.0, 25.0, 55.0, 80} );

    std::vector<unsigned int> map = {8,4,2};
    m_network = NetworkPtr( new Network(map) );

    m_killer.start();
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
    double thisRotation =  m_rotationSpeedRad * animTime;
    Eigen::Rotation2D<double> r( thisRotation );
    m_direction = r.toRotationMatrix() * m_direction;
    Eigen::Vector2d(1.0,0.0);
    m_rotationToOriginal = computeAngleBetweenVectors(Eigen::Vector2d(1.0,0.0), m_direction);

    // adjust speed
    double newSpeed = std::max(m_speed + animTime*getAcceleration(), 0.0);
    newSpeed = std::min(newSpeed,600.0);
    Eigen::Vector2d effectiveSpeed = m_direction * (newSpeed + m_speed) * 0.5;

    // set new position
    Eigen::Vector2d newPosition = m_position + animTime * effectiveSpeed;
    newPosition = handleCollision(m_position, newPosition);

    // adjust drove distance and accumulated rotation
    m_droveDistance += (newPosition - getPosition()).norm();
    m_accumulatedRotation += std::abs(thisRotation);

    navigate();

    // update
    m_speed = newSpeed;
    m_position = newPosition;
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

std::shared_ptr<TrackMap> Car::getMap() const
{
    return m_map;
}

void Car::setMap(std::shared_ptr<TrackMap> map)
{
    m_map = map;
    m_mapSet = true;
}

Eigen::Vector2d Car::handleCollision(const Eigen::Vector2d& from, const Eigen::Vector2d& to)
{
    if( !m_mapSet )
        return to;

    if( !( m_map->isPositionValid(from) ) )
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

    while(m_map->isPositionValid(end))
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

    if( m_accumulatedRotation / getAge() > M_PI / 2.0) //90° per seconds average
    {
        m_alive = false;
    }
}

void Car::navigate()
{
    // decide what to do next
    m_measuredDistances = measureDistances();
    Eigen::MatrixXd nnInput = m_measuredDistances.col(0);
    nnInput.conservativeResize(m_measuredDistances.rows()+1, 1); // additional input for speed
    nnInput(nnInput.rows()-1,0) = m_speed;

    // normalize input -> all values are positive -> scale them on a range -1 to +1
    double maxValInput = nnInput.maxCoeff();
    nnInput = (nnInput * 2.0/maxValInput).array() - 1.0;

    m_network->feedForward(nnInput);
    Eigen::MatrixXd nnOut = m_network->getOutputActivation();

    double maxRotationSpeed = 720.0;
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
