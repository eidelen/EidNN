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

#include "faceDataInput.h"
#include <QFile>
#include <QTextStream>
#include <iostream>

FaceDataInput::FaceDataInput()
{
}

FaceDataInput::~FaceDataInput()
{
}

bool FaceDataInput::addToTraining( const QString& path, int lable, size_t maxNbrOfSamples )
{
    return addToSet(ETraining, path, lable, maxNbrOfSamples);
}

bool FaceDataInput::addToTest( const QString& path, int lable, size_t maxNbrOfSamples )
{
    return addToSet(ETest, path, lable, maxNbrOfSamples);
}

bool FaceDataInput::parseImgSampleLine(const QString& line, std::vector<float>& values)
{
    QStringList valListStr = line.split(",");
    for (int i = 0; i < valListStr.size(); i++)
    {
        QString vStr = valListStr[i];

        bool okTrans = true;
        int val = vStr.toInt(&okTrans);
        if (!okTrans)
        {
            // check if "," was last
            if( i+1 == valListStr.size() )
            {
                // ignore
            }
            else
            {
                std::cerr << "Error to int: " << vStr.toStdString() << std::endl;
                return false;
            }
        }
        else
        {
            // normalize data
            values.push_back(static_cast<float>(val));
        }
    }
    return true;
}

bool FaceDataInput::addToSet(FaceDataInput::ESet set, const QString& path, int lable, size_t maxNbrOfSamples )
{
    QFile inputFile(path);
    if(! inputFile.open(QIODevice::ReadOnly) )
    {
        std::cerr << "Cannot open file: " << path.toStdString() << std::endl;
        return false;
    }

    QTextStream in(&inputFile);
    size_t  sample_count = 0;
    while(!in.atEnd())
    {
        QString line = in.readLine(); // each line is one sample
        std::vector<float> values;
        if( !parseImgSampleLine(line, values) )
        {
            std::cerr << "Parse error" << std::endl;
            return false;
        }

        Eigen::MatrixXd xIn = Eigen::MatrixXd(values.size(), 1);
        for(size_t u = 0; u < values.size(); u++)
            xIn(u, 0) = values.at(u);


        if( set == ETraining )
            addTrainingSample(xIn,lable);
        else if( set == ETest )
            addTestSample(xIn,lable);

        sample_count++;
        if( sample_count >= maxNbrOfSamples )
            break;
    }
    inputFile.close();

    return true;
}



