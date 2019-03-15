//
// Created by Adrian Schneider on 2019-03-13.
//

#include "trackmap.h"

#include <iostream>


TrackMap::TrackMap(const Eigen::MatrixXi& map): m_dynamicMapSet(false)
{
    m_map = computeDistanceMap( map );
}

TrackMap::~TrackMap()
{

}

void TrackMap::resetMap(const Eigen::MatrixXi &map)
{
    m_map = computeDistanceMap( map );
}

const Eigen::MatrixXi &TrackMap::getMap() const
{
    return m_map;
}

int TrackMap::isPositionValid(const Eigen::Vector2d &pos) const
{
    int r = isPositionValid(m_map, pos);

    if( r!=0 && m_dynamicMapSet )
        r = std::min(r, isPositionValid(m_dynamicMap, pos));

    return r;
}

int TrackMap::isPositionValid(const Eigen::MatrixXi& map, const Eigen::Vector2d &pos) const
{
    if(pos(0) < 0.0 ||  pos(0) > map.cols()-1 || pos(1) < 0.0 ||  pos(1) > map.rows()-1)
        return 0;

    return map( std::ceil(pos(1)) , std::ceil(pos(0)));
}

void TrackMap::setDynamicMap(const Eigen::MatrixXi &map)
{
    m_dynamicMapSet = true;
    m_dynamicMap = map;
}

void TrackMap::clearDynamicMap()
{
    m_dynamicMapSet = false;
}

Eigen::MatrixXi TrackMap::createAllValidMap() const
{
    return Eigen::MatrixXi::Ones(m_map.rows(), m_map.cols());
}

Eigen::MatrixXi TrackMap::computeDistanceMap( const Eigen::MatrixXi& map ) const
{
    Eigen::MatrixXi dMap = map;

    for( size_t m = 0; m < dMap.rows(); m++ )
    {
        for( size_t n = 0; n < dMap.cols(); n++ )
        {

            if(m==2 && n==1 )
                int k = 0;

            if( mapPositionValue(m,n, dMap) > 0 )
            {
                // get initial surrounding info
                int iMax, iMin;
                getKnownSuroundingMapDistances(m,n,dMap,iMin,iMax);

                if( iMin == 0 )
                {
                    dMap(m,n) = 1;
                    continue;
                }

                // the max different among neighbours can be 2 -> if so, the current distance is in between.
                if( iMax - iMin == 2 )
                {
                    dMap(m, n) = iMin + 1;
                    continue;
                }


                // otherwise, we have to search in the right down region for a collision

                int r = iMax;
                while( true )
                {
                    bool collision = false;

                    // lower h line
                    for (int i = -r; i <= r && !collision; i++)
                        collision = collision || (mapPositionValue(m + r, n + i, dMap) == 0);

                    // left v line
                    for (int i = 1; i <= r && !collision; i++)
                        collision = collision || (mapPositionValue(m + i, n - r, dMap) == 0);

                    // right v line
                    for (int i = -r; i <= r && !collision; i++)
                        collision = collision || (mapPositionValue(m + i, n + r, dMap) == 0);

                    if (!collision)
                    {
                        r = r + 1;
                        break;
                    }
                    else
                    {
                        r = r - 1;
                    }
                }

                dMap(m, n) = r;

            }
            else
            {
                dMap(m, n) = 0;
            }
        }
    }

    return dMap;
}

int TrackMap::mapPositionValue(size_t m, size_t n, const Eigen::MatrixXi &map) const
{
    if(n < 0 ||  n > map.cols()-1 || m < 0 ||  m > map.rows()-1 )
        return 0;

    return map( m , n );
}

void TrackMap::getKnownSuroundingMapDistances(size_t m, size_t n, const Eigen::MatrixXi& map, int &min, int &max) const
{
    min = std::numeric_limits<int>::max();
    max = std::numeric_limits<int>::min();

    int mP = mapPositionValue(m-1,n-1, map);
    min = std::min( min, mP);
    max = std::max( max, mP);

    mP = mapPositionValue(m-1,n, map);
    min = std::min( min, mP);
    max = std::max( max, mP);

    mP = mapPositionValue(m-1,n+1, map);
    min = std::min( min, mP);
    max = std::max( max, mP);
}
