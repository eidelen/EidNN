

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
    m_lastDirection = m_direction;
}

void Car::update()
{
    double animTime = getTimeSinceLastUpdate();

    double newSpeed = m_speed + animTime*getAcceleration();

    Eigen::Vector2d newDirection = m_direction;

    Eigen::Vector2d newSpeedVector = m_direction * newSpeed;

    Eigen::Vector2d oldSpeedVector = m_lastDirection * m_lastSpeed;

    Eigen::Vector2d effectiveSpeed = (newSpeedVector + oldSpeedVector) * 0.5;

    setPosition( getPosition() + animTime * effectiveSpeed );
    setSpeed( newSpeed );
    setDirection( m_direction );

}

double Car::getFitness()
{
    return 0.0;
}

double Car::getRotationSpeed() const
{
    return m_rotationSpeed;
}

void Car::setRotationSpeed(double rotationSpeed)
{
    m_rotationSpeed = rotationSpeed;
}


