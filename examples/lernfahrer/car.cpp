

#include "car.h"

#include <iostream>

Car::Car()
{
    setLastUpdate(now());
    setSpeed( Eigen::Vector2d(0.0, 0.0) );
    setPosition( Eigen::Vector2d(0.0, 0.0));
    setAcceleration(0.0);

    m_speedLastUpdate = getSpeed();
}

Car::~Car()
{

}

const Eigen::Vector2d &Car::getSpeed() const
{
    return m_speed;
}

void Car::setSpeed(const Eigen::Vector2d &speed)
{
    m_speed = speed;
    m_speedLastUpdate = speed;
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

const std::chrono::milliseconds &Car::getLastUpdate() const
{
    return m_lastUpdate;
}

void Car::setLastUpdate(const std::chrono::milliseconds &lastUpdate)
{
    m_lastUpdate = lastUpdate;
}

std::chrono::milliseconds Car::now() const
{
    return std::chrono::duration_cast< std::chrono::milliseconds >(
            std::chrono::system_clock::now().time_since_epoch());
}

double Car::timeSinceLastUpdate() const
{
    return (now() - m_lastUpdate).count() / 1000.0;
}

void Car::update()
{
    double animTime = timeSinceLastUpdate();

    auto speed = getSpeed();
    double speedMagnitude = speed.norm();
    double newSpeedMagnitude = speedMagnitude + animTime*getAcceleration();
    auto currentSpeed = speed * newSpeedMagnitude/speedMagnitude;
    auto effectiveSpeed = (currentSpeed + m_speedLastUpdate) * 0.5;

    setPosition( getPosition() + animTime * effectiveSpeed );

    m_speedLastUpdate = currentSpeed;
    setLastUpdate(now());
}

