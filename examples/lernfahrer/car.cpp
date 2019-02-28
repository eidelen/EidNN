

#include "car.h"

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

    setMeasureAngles( {-45.0, 0.0, 45.0} );

    m_droveDistance = 0.0;
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


    double newSpeed = m_speed + animTime*getAcceleration();

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

    Eigen::Vector2d dif = to - from;
    double l = dif.norm();

    if( l < 0.00000001 ) // no or very small move
        return to;

    Eigen::Vector2d du = dif * 1.0 / l;

    double tillEdge = distanceToEdge(from,du);

    if( tillEdge < l ) // collision
    {
        m_alive = false;
        return from + du * tillEdge;
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

    while(goOn)
    {
        end = end + d;

        // check if within map
        if(end(0) < 0.0)
        {
            end(0) = 0.0;
            goOn = false;
        }

        if(end(1) < 0.0)
        {
            end(1) = 0.0;
            goOn = false;
        }

        if(end(0) > m_map.cols()-1)
        {
            end(0) = m_map.cols()-1;
            goOn = false;
        }

        if(end(1) > m_map.rows()-1)
        {
            end(1) = m_map.rows()-1;
            goOn = false;
        }

        if( m_map( std::ceil(end(1)) , std::ceil(end(0))) == 0 )
        {
            end(0) = std::ceil(end(0));
            end(1) = std::ceil(end(1));
            goOn = false;
        }
    }

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
    car->setPosition(Eigen::Vector2d(500,150) );
    car->setDirection(Eigen::Vector2d(1,0));

    return car;
}
