/****************************************************************************
** Copyright (c) 2017 Adrian Schneider
**
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and associated documentation files (the "Software"),
** to deal in the Software without restriction, including without limitation
** the rights to use, copy, modify, merge, publish, distribute, sublicense,
** and/or sell copies of the Software, and to permit persons to whom the
** Software is furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
** FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
** DEALINGS IN THE SOFTWARE.
**
*****************************************************************************/

#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include "car.h"
#include "trackmap.h"
#include "helpers.h"

TEST(Car, Init)
{
    auto c = new Car();

    ASSERT_FLOAT_EQ(c->getAcceleration(),0.0);
    ASSERT_FLOAT_EQ(c->getSpeed(),0.0);
    ASSERT_FLOAT_EQ(c->getPosition()(0),0.0);
    ASSERT_FLOAT_EQ(c->getPosition()(1),0.0);
    ASSERT_FLOAT_EQ(c->getDirection()(0),1.0);
    ASSERT_FLOAT_EQ(c->getDirection()(1),0.0);
    ASSERT_NEAR(c->getTimeSinceLastUpdate(),0.0,0.01);
    ASSERT_FLOAT_EQ(c->getRotationSpeed(),0.0);

    delete c;
}

TEST(Car, MoveConstVelocity)
{
    auto c = new Car();

    double speed = 10;
    c->setSpeed(10);
    c->setDirection(Eigen::Vector2d(1,0));
    c->setPosition(Eigen::Vector2d(0,0));
    Eigen::MatrixXi map(100,100);
    map.fill(1);

    std::shared_ptr<TrackMap> tmap( new TrackMap(map) );

    c->setMap(tmap);

    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    c->doStep();
    ASSERT_NEAR(c->getPosition()(0),speed*0.25,0.3);
    ASSERT_NEAR(c->getPosition()(1),0.0,0.1);

    ASSERT_NEAR(c->getFitness(),speed*0.25,0.3);


    delete c;
}

TEST(Car, Accelerate)
{
    auto c = new Car();

    c->setDirection(Eigen::Vector2d(1,0));
    c->setSpeed(0.0);
    c->setAcceleration(10.0);

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    c->doStep();

    ASSERT_NEAR(c->getSpeed(),10.0, 0.1);
    ASSERT_NEAR(c->getPosition()(0),5.0,0.1);
    ASSERT_NEAR(c->getPosition()(1),0.0,0.1);

    delete c;
}

TEST(Car, Rotation)
{
    auto c = new Car();

    c->setDirection(Eigen::Vector2d(1,0));
    c->setRotationSpeed(180);

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    c->doStep();

    // 0 -> 360 = 180
    ASSERT_NEAR(c->getDirection()(0),-1.0,0.2);
    ASSERT_NEAR(c->getDirection()(1),0.0,0.2);

    delete c;
}

TEST(Car, RelAngle)
{
    auto c = new Car();

    ASSERT_NEAR(0.0, c->computeAngleBetweenVectors(Eigen::Vector2d(1,0), Eigen::Vector2d(1,0)), 0.0);
    ASSERT_NEAR(0.0, c->computeAngleBetweenVectors(Eigen::Vector2d(1,0), Eigen::Vector2d(2,0)), 0.0);
    ASSERT_NEAR(0.0, c->computeAngleBetweenVectors(Eigen::Vector2d(1,0), Eigen::Vector2d(0,5)), 90.0);
    ASSERT_NEAR(0.0, c->computeAngleBetweenVectors(Eigen::Vector2d(1,0), Eigen::Vector2d(-2,0)), 180.0);
    ASSERT_NEAR(0.0, c->computeAngleBetweenVectors(Eigen::Vector2d(1,0), Eigen::Vector2d(0,-4)), 270.0);

    delete c;
}


TEST(Car, Collision)
{
    Eigen::MatrixXi map(20,20);
    map.fill(1);
    map.col(17).setZero();

    std::shared_ptr<TrackMap> tmap( new TrackMap(map) );

    auto c = new Car();
    c->setMap(tmap);
    c->setSpeed(100);
    c->setPosition(Eigen::Vector2d(0,0));
    c->setDirection(Eigen::Vector2d(1,0));

    ASSERT_TRUE(c->isAlive());

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    c->doStep();
    ASSERT_FALSE(c->isAlive());

    ASSERT_NEAR(c->getPosition()(0), 0.0, 0.1);
    ASSERT_NEAR(c->getPosition()(1), 0.0, 0.1);

    delete c;
}

TEST(Car, DistanceToEdge)
{
    Eigen::MatrixXi map(10,10);
    map.fill(1);
    map.col(5).setZero();

    std::shared_ptr<TrackMap> tmap( new TrackMap(map) );

    auto c = new Car();
    c->setMap(tmap);
    c->setPosition(Eigen::Vector2d(2,2));

    ASSERT_NEAR(c->distanceToEdge(Eigen::Vector2d(2,3) , Eigen::Vector2d(1,0)), 3.0, 0.1);
    ASSERT_NEAR(c->distanceToEdge(Eigen::Vector2d(2,3) , Eigen::Vector2d(-1,0)), 3.0, 0.1);
    ASSERT_NEAR(c->distanceToEdge(Eigen::Vector2d(2,3) , Eigen::Vector2d(0,1)), 7.0, 0.1);
    ASSERT_NEAR(c->distanceToEdge(Eigen::Vector2d(2,3) , Eigen::Vector2d(0,-1)), 4.0, 0.1);

    delete c;
}

TEST(Car, DistanceInput)
{
    Eigen::MatrixXi map(10,10);
    map.fill(1);
    map.col(5).setZero();

    std::shared_ptr<TrackMap> tmap( new TrackMap(map) );

    auto c = new Car();
    c->setMap(tmap);
    c->setPosition(Eigen::Vector2d(2,2));
    c->setMeasureAngles({0,90,-90, 0, 0, 0, 0});
    c->setDirection(Eigen::Vector2d(1,0));
    c->setSpeed(0.0);

    c->doStep();

    Eigen::MatrixXd mAng = c->getMeasuredDistances();

    ASSERT_EQ(mAng.rows(), 7);
    ASSERT_EQ(mAng.cols(), 3);

    ASSERT_NEAR(mAng(0,0) , 3.0, 0.1);
    ASSERT_NEAR(mAng(0,1) , 5.0, 0.1);
    ASSERT_NEAR(mAng(0,2) , 2.0, 0.1);

    ASSERT_NEAR(mAng(1,0) , 3.0, 0.1);
    ASSERT_NEAR(mAng(1,1) , 2.0, 0.1);
    ASSERT_NEAR(mAng(1,2) , -1.0, 0.1);

    ASSERT_NEAR(mAng(2,0) , 8.0, 0.1);
    ASSERT_NEAR(mAng(2,1) , 2.0, 0.1);
    ASSERT_NEAR(mAng(2,2) , 10.0, 0.1);

    delete c;
}

TEST(Car, DistanceMap)
{
    Eigen::MatrixXi map(5,5);
    map.fill(1);

    std::shared_ptr<TrackMap> tmap( new TrackMap(map) );

    Eigen::MatrixXi dmap = tmap->computeDistanceMap(map);

    ASSERT_EQ(dmap(2,2), 3);
    ASSERT_EQ(dmap(1,2), 2);
    ASSERT_EQ(dmap(3,2), 2);
    ASSERT_EQ(dmap(2,3), 2);
    ASSERT_EQ(dmap(2,1), 2);
    ASSERT_EQ(dmap(0,0), 1);
    ASSERT_EQ(dmap(1,1), 2);
}

TEST(Car, DistanceMapObst)
{
    Eigen::MatrixXi map(5,5);
    map.fill(1);
    map(1,1) = 0;

    std::shared_ptr<TrackMap> tmap( new TrackMap(map) );
    Eigen::MatrixXi dmap = tmap->computeDistanceMap(map);

    ASSERT_EQ(dmap(0,0), 1);
    ASSERT_EQ(dmap(1,1), 0);
    ASSERT_EQ(dmap(2,2), 1);
    ASSERT_EQ(dmap(3,3), 2);
    ASSERT_EQ(dmap(4,4), 1);

    ASSERT_EQ(dmap(2,0), 1);
    ASSERT_EQ(dmap(2,1), 1);
    ASSERT_EQ(dmap(2,2), 1);
    ASSERT_EQ(dmap(2,3), 2);
    ASSERT_EQ(dmap(2,4), 1);
}


TEST(Car, DistanceMapHeavyComp)
{
    // 3.3, 0.7

    Eigen::MatrixXi map(200,200);
    map.fill(1);

    std::shared_ptr<TrackMap> tmap( new TrackMap(map) );
    Eigen::MatrixXi dmap = tmap->computeDistanceMap(map);

    for( size_t i = 0; i < 200; i++ )
    {
        ASSERT_EQ(dmap(i, 0), 1);
        ASSERT_EQ(dmap(0, i), 1);
        ASSERT_EQ(dmap(i, 199), 1);
        ASSERT_EQ(dmap(199, i), 1);
    }

    for( size_t t = 0; t < 100; t++ )
    {
        ASSERT_EQ(dmap(t,t), t+1);
        ASSERT_EQ(dmap(199-t, 199-t), t+1);
        ASSERT_EQ(dmap(100,t), t+1);
        ASSERT_EQ(dmap(100,199-t), t+1);
    }
}


