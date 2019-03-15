//
// Created by Adrian Schneider on 2019-03-13.
//

#ifndef EIDNN_TRACKMAP_H
#define EIDNN_TRACKMAP_H

#include <Eigen/Dense>

class TrackMap
{
public:
    TrackMap(const Eigen::MatrixXi& map);
    virtual ~TrackMap();

    void resetMap(const Eigen::MatrixXi& map);
    const Eigen::MatrixXi& getMap() const;

    void setDynamicMap(const Eigen::MatrixXi& map);
    void clearDynamicMap();

    int isPositionValid(const Eigen::Vector2d &pos) const;

    Eigen::MatrixXi createAllValidMap() const;

    Eigen::MatrixXi computeDistanceMap( const Eigen::MatrixXi& map ) const;


private:

    int isPositionValid(const Eigen::MatrixXi& map, const Eigen::Vector2d &pos) const;

    int mapPositionValue(size_t m, size_t n, const Eigen::MatrixXi& map) const;

    void getKnownSuroundingMapDistances( size_t m, size_t n, const Eigen::MatrixXi& map,  int& min, int& max ) const;


    Eigen::MatrixXi m_map;
    Eigen::MatrixXi m_dynamicMap;
    bool m_dynamicMapSet;
};



#endif //EIDNN_TRACKMAP_H
