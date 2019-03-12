//
// Created by Adrian Schneider on 2019-03-12.
//

#ifndef EIDNN_STRANGE_H
#define EIDNN_STRANGE_H

#include "track.h"

class Strange: public Track
{
public:
    Strange(const QString &name, const QString &rscPath);
    virtual ~Strange();

    void draw(QPainter *painter, const std::vector<SimulationPtr > &simRes) override;

private:
    Eigen::MatrixXi m_originalMap;
};


#endif //EIDNN_STRANGE_H

