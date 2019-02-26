#ifndef EIDNN_CAR_H
#define EIDNN_CAR_H

#include "simulation.h"

#include <Eigen/Dense>
#include <chrono>

class Car: public Simulation
{
public:
    Car();
    virtual ~Car();

public:

    double getAcceleration() const;
    void setAcceleration(double acceleration);
    const Eigen::Vector2d &getPosition() const;
    void setPosition(const Eigen::Vector2d &position);
    double getSpeed() const;
    void setSpeed(double speed);
    const Eigen::Vector2d &getDirection() const;
    void setDirection(const Eigen::Vector2d &direction);

    /**
     * Rotation speed in degree per second.
     * @return Degree per second
     */
    double getRotationSpeed() const;

    /**
     * Rotation speed in degree per second.
     * @param rotationSpeed Degree per second
     */
    void setRotationSpeed(double rotationSpeed);

    /**
     * Rotation relative to initial.
     * @return Degree
     */
    double getRotationRelativeToInitial() const;


    double getFitness() override;

    double computeAngleBetweenVectors( const Eigen::Vector2d& a, const Eigen::Vector2d& b ) const;

private:
    void update() override;


private:
    Eigen::Vector2d m_direction;
    Eigen::Vector2d m_position;
    double m_acceleration;
    double m_speed;
    double m_rotationSpeed;
    double m_rotationSpeedRad;
    double m_rotationToOriginal;

    double m_lastSpeed;
};


#endif //EIDNN_CAR_H
