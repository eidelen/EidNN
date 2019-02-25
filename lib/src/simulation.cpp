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

#include "simulation.h"

Simulation::Simulation()
{
    setLastUpdateTime(now());
}

Simulation::~Simulation()
{

}

void Simulation::doStep()
{
    update();
    setLastUpdateTime(now());
}

void Simulation::update()
{
    // Do override
}

double Simulation::getFitness()
{
    return 0.0;
}

std::chrono::milliseconds Simulation::now() const
{
    return std::chrono::duration_cast< std::chrono::milliseconds >(
            std::chrono::system_clock::now().time_since_epoch());
}

const std::chrono::milliseconds &Simulation::getLastUpdateTime() const
{
    return m_lastUpdate;
}

void Simulation::setLastUpdateTime(const std::chrono::milliseconds &lastUpdate)
{
    m_lastUpdate = lastUpdate;
}

double Simulation::getTimeSinceLastUpdate() const
{
    return (now() - m_lastUpdate).count() / 1000.0;
}