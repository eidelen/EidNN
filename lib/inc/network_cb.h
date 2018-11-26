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

#ifndef NETWORKCALLBACKHEADER
#define NETWORKCALLBACKHEADER

#include <vector>

class NetworkOperationCallback
{
public:

    enum NetworkOperationId
    {
        OpStochasticGradientDescent = 0x00,
        OpTestNetwork
    };

    enum NetworkOperationStatus
    {
        OpResultOk = 0x00,
        OpResultErr,
        OpInProgress
    };

public:

    virtual ~NetworkOperationCallback(){}

    /**
     * Callback function which informs the user about a running neural network opertation.
     * @param opId Operation ID.
     * @param opStatus Operation status.
     * @param progress Opration progress. This is a number between 0.0 and 1.0.
     * @param userId User given id.
     */
    virtual void networkOperationProgress( const NetworkOperationId& opId, const NetworkOperationStatus& opStatus, const double& progress, const int& userId ) = 0;

    /**
     * Callback function which informs about the network test results.
     * @param successRateEuclidean Success rate in terms of the Euclidean distance between expected and actual output.
     * @param successRateMaxIdx Success rate in terms of identical maximum element.
     * @param averageCost Average test cost
     * @param failedSamplesIdx List of failed samples
     * @param userId User given id
     */
    virtual void networkTestResults( const double& successRateEuclidean, const double& successRateMaxIdx,
                                     const double& averageCost,
                                     const std::vector<std::size_t>& failedSamplesIdx, const int& userId) = 0;

};

#endif // NETWORKCALLBACKHEADER
