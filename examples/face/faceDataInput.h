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

#ifndef FACEDATAINPUT
#define FACEDATAINPUT

#include "dataInput.h"
#include <QString>
#include <limits>

class FaceDataInput: public DataInput
{

public:
    FaceDataInput( );
    ~FaceDataInput();

    bool addToTraining( const QString& path, int lable, size_t maxNbrOfSamples = std::numeric_limits<size_t>::max());
    bool addToTest( const QString& path, int lable, size_t maxNbrOfSamples = std::numeric_limits<size_t>::max() );

    DataElement getTestImageAsPixelValues( size_t idx ) const;


private:
    bool parseImgSampleLine(const QString& line, std::vector<float>& values);

    enum ESet
    {
        ETraining,
        ETest
    };
    bool addToSet(ESet set, const QString& path, int lable, size_t maxNbrOfSamples );

    std::vector<DataElement> m_testImages;

};

#endif //FACEDATAINPUT
