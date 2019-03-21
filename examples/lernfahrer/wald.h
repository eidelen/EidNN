//
// Created by Adrian Schneider on 2019-03-17.
//

#ifndef EIDNN_WALD_H
#define EIDNN_WALD_H

#include "track.h"

class Wald: public Track
{
public:
    Wald(const QString &name, const QString &rscPath);
    virtual ~Wald();
    void draw(QPainter *painter, const std::vector<SimulationPtr > &simRes) override;

private:
    Eigen::MatrixXi m_originalMap;
    double m_anim;
};


#endif //EIDNN_WALD_H
