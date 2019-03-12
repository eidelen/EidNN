#ifndef EIDNN_CAR_H
#define EIDNN_CAR_H

#include "simulation.h"
#include "trackmap.h"

#include <QTime>
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

    std::shared_ptr<TrackMap> getMap() const;

    void setMap(std::shared_ptr<TrackMap> map);

    double distanceToEdge(const Eigen::Vector2d& pos, const Eigen::Vector2d& direction) const;

    const std::vector<double> &getMeasureAngles() const;

    void setMeasureAngles(const std::vector<double>& measureAngles);

    Eigen::MatrixXd getMeasuredDistances() const;


private:
    void update() override;
    Eigen::Vector2d handleCollision(const Eigen::Vector2d& from, const Eigen::Vector2d& to);
    Eigen::MatrixXd measureDistances() const;
    void considerSuicide();
    void navigate();


private:
    Eigen::Vector2d m_direction;
    Eigen::Vector2d m_position;
    double m_acceleration;
    double m_speed;
    double m_rotationSpeed;
    double m_rotationSpeedRad;
    double m_rotationToOriginal;

    std::shared_ptr<TrackMap> m_map;
    bool m_mapSet;

    std::vector<double> m_measureAngles;
    Eigen::MatrixXd m_measuredDistances;

    double m_droveDistance;
    double m_accumulatedRotation;

    QTime m_killer;
    double m_formerDistance;
};

#endif //EIDNN_CAR_H
