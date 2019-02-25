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

    delete c;
}

TEST(Car, MoveConstVelocity)
{
    auto c = new Car();

    double speed = 10;
    c->setSpeed(10);
    c->setDirection(Eigen::Vector2d(1,0));

    for(int k = 0; k < 8; k++ )
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        c->doStep();
        ASSERT_NEAR(c->getPosition()(0),(k+1)*speed*0.25,0.3);
        ASSERT_NEAR(c->getPosition()(1),0.0,0.01);
    }

    delete c;
}

