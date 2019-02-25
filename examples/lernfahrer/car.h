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
    const Eigen::Vector2d &getSpeed() const;
    void setSpeed(const Eigen::Vector2d &speed);

    double getFitness() override;

private:
    void update() override;


private:
    Eigen::Vector2d m_speed;
    Eigen::Vector2d m_position;
    double m_acceleration;


    Eigen::Vector2d m_speedLastUpdate;

};


#endif //EIDNN_CAR_H
