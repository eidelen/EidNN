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

#include "mnistDataInput.h"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

MnistDataInput::MnistDataInput()
{
    load();
}

MnistDataInput::~MnistDataInput()
{
}

bool MnistDataInput::loadMNISTSample( const std::vector<std::vector<double>>& imgSet, const std::vector<uint8_t>& lableSet,
                                      const size_t& idx, Eigen::MatrixXd& img, uint8_t& lable)
{
    if( std::max(imgSet.size(), lableSet.size()) <= idx )
        return false;

    lable = lableSet.at( idx );
    const std::vector<double>& imgV = imgSet.at( idx );

    img = Eigen::MatrixXd( imgV.size(), 1 );
    for( size_t i = 0; i < imgV.size(); i++ )
        img( int(i), 0 ) = imgV.at(i);

    return true;
}

void MnistDataInput::load()
{
    // MNIST_DATA_LOCATION passed by cmake
    auto mnistinputNormalized = mnist::read_dataset<std::vector, std::vector, double, uint8_t>(MNIST_DATA_LOCATION);
    mnist::normalize_dataset(mnistinputNormalized);

    // Load training data
    for( size_t k = 0; k < mnistinputNormalized.training_images.size(); k++ )
    {
        Eigen::MatrixXd xInNormalized; uint8_t lable;
        loadMNISTSample( mnistinputNormalized.training_images, mnistinputNormalized.training_labels, k, xInNormalized, lable );
        addTrainingSample(xInNormalized, static_cast<int>(lable));
    }

    // Load testing data
    for( size_t k = 0; k < mnistinputNormalized.test_images.size(); k++ )
    {
        Eigen::MatrixXd xInNormalized; uint8_t lable;
        loadMNISTSample( mnistinputNormalized.test_images, mnistinputNormalized.test_labels, k, xInNormalized, lable );
        addTestSample( xInNormalized, static_cast<int>(lable));
    }

    generateFromLables();


    // get the test images as images
    auto mnistPixelImages = mnist::read_dataset<std::vector, std::vector, double, uint8_t>(MNIST_DATA_LOCATION);
    for( size_t k = 0; k < mnistPixelImages.test_images.size(); k++ )
    {
        Eigen::MatrixXd xImg; uint8_t lable;
        loadMNISTSample( mnistPixelImages.test_images, mnistPixelImages.test_labels, k, xImg, lable );
        DataElement de;
        de.input = xImg;
        de.lable = lable;
        m_testImages.push_back(de);
    }
}

DataElement MnistDataInput::getTestImageAsPixelValues( size_t idx ) const
{
    return m_testImages.at(idx);
}

Eigen::MatrixXd MnistDataInput::representation( const Eigen::MatrixXd& input, bool* representationAvailable  ) const
{
    size_t imgW = 28;
    size_t imgH = 28;

    // check that vector length equal to number of required pixels
    if( imgH * imgW != input.rows() )
    {
        *representationAvailable = false;
        return input;
    }

    Eigen::MatrixXd rep = Eigen::MatrixXd(imgH,imgW);
    for( size_t h = 0; h < imgH; h++ )
    {
        for( size_t w = 0; w < imgW; w++ )
        {
            rep(h,w) = input(imgW*h+w,0);
        }
    }
    
    *representationAvailable = true;
    return rep;
}
