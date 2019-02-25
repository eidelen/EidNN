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

    double getFitness() override;

private:
    void update() override;


private:
    Eigen::Vector2d m_direction;
    Eigen::Vector2d m_position;
    double m_acceleration;
    double m_speed;


    double m_lastSpeed;
    Eigen::Vector2d m_lastDirection;

};


#endif //EIDNN_CAR_H
