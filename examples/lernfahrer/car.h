#ifndef EIDNN_CAR_H
#define EIDNN_CAR_H

#include <Eigen/Dense>
#include <chrono>

class Car
{
public:
    Car();
    virtual ~Car();

public:
    const std::chrono::milliseconds &getLastUpdate() const;
    void setLastUpdate(const std::chrono::milliseconds &lastUpdate);
    double getAcceleration() const;
    void setAcceleration(double acceleration);
    const Eigen::Vector2d &getPosition() const;
    void setPosition(const Eigen::Vector2d &position);
    const Eigen::Vector2d &getSpeed() const;
    void setSpeed(const Eigen::Vector2d &speed);

    /**
     * Time since last update in seconds.
     * @return Elapsed time in seconds
     */
    double timeSinceLastUpdate() const;

    void update();

private:
    std::chrono::milliseconds now() const;


private:
    Eigen::Vector2d m_speed;
    Eigen::Vector2d m_position;
    double m_acceleration;
    std::chrono::milliseconds m_lastUpdate;

    Eigen::Vector2d m_speedLastUpdate;

};


#endif //EIDNN_CAR_H
