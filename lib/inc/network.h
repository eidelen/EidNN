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

#ifndef NETWORKHEADER
#define NETWORKHEADER

#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <Eigen/Dense>

#include "network_cb.h"
#include "regularization.h"

class Layer;

class Network
{
public:
    enum ECostFunction
    {
        Quadratic,
        CrossEntropy
    };

public:

    /**
     * Constructor of a neural network.
     * @param networkStructure A vector holding the number of neurons for each layer,
     *                         where the first element of the vector is the number of
     *                         neurons in the first layer, and the last vector item the
     *                         number of neurons in the last layer, the output layer.
     */
    Network( const std::vector<unsigned int> networkStructure );

    /**
     * Copy-Constructor
     * @param n
     */
    Network( const Network& n );

    ~Network();

    /**
     * Compute the neural network output signal based on the input signal x_in.
     * The output signal can be accessed with the function getOutputActivation().
     * @param x_in Input signal.
     * @return true if successful.
     */
    bool feedForward(const Eigen::MatrixXd &x_in );

    /**
     * Get the output activation of this neural network. This function is usually
     * called after feedForward() is executed.
     * @return Output activation vector.
     */
    const Eigen::MatrixXd& getOutputActivation() const { return m_activation_out; }

    /**
     * Returns the number of layers.
     */
    unsigned int getNumberOfLayer() const;

    /**
     * Return a handle to the layer at index layerIdx.
     * @param layerIdx Layer index.
     * @return Null, if layer does not exist. Otherwise handle to layer.
     */
    std::shared_ptr<Layer> getLayer(const unsigned int &layerIdx );
    std::shared_ptr<const Layer> getLayer(const unsigned int &layerIdx ) const;

    /**
     * Returns the last layer, also called output layer.
     * @return Last layer of the network.
     */
    std::shared_ptr<Layer> getOutputLayer();
    std::shared_ptr<const Layer> getOutputLayer() const;

    /**
     * Feedforward, backpropagate and update weigths and biases in each layer corresponding
     * to the computed partial derivatives and the gradient descent method.
     * @see setCostFunction
     * @param x_in Input signal.
     * @param y_out Desired output signal.
     * @param eta Learning rate.
     * @return true if successful.
     */
    bool gradientDescent(const Eigen::MatrixXd &x_in, const Eigen::MatrixXd &y_out, const double& eta);

    /**
     * Feedforward, backpropagate and update weigths and biases in each layer corresponding
     * to the computed partial derivatives and the stochastic gradient descent method.
     * The stochastic gradient descent methods updates the weights and biases by the averaged
     * partial derivatives of a randomly chosen batch of samples. In total, nbrOfSamples / batchsize
     * batches are executed -> this is called an epoch.
     * @see setCostFunction
     * @param samples Input signals.
     * @param lables Desired output signals.
     * @param batchsize Number of samples in the batch.
     * @param eta
     * @return true if successful.
     */
    bool stochasticGradientDescent(const std::vector<Eigen::MatrixXd> &samples, const std::vector<Eigen::MatrixXd> &lables,
                                   const unsigned int& batchsize, const double& eta);

    /**
     * Feedforward, backpropagate and update weigths and biases in each layer corresponding
     * to the computed partial derivatives and the stochastic gradient descent method.
     * The stochastic gradient descent methods updates the weights and biases by the averaged
     * partial derivatives of a randomly chosen batch of samples. In total, nbrOfSamples / batchsize
     * batches are executed -> this is called an epoch. The computation is performed in another
     * thread. The user gets informed over the NetworkOperationCallback interface.
     * @see setCostFunction
     * @param samples Input signals.
     * @param lables Desired output signals.
     * @param batchsize Number of samples in the batch.
     * @param eta Learning rate.
     * @return true if successful.
     */
    bool stochasticGradientDescentAsync(const std::vector<Eigen::MatrixXd> &samples, const std::vector<Eigen::MatrixXd> &lables,
                                        const unsigned int& batchsize, const double& eta );

    /**
     * Tests the network with given samples and lables.
     * @param samples Input sample.
     * @param lables Expected output.
     * @param euclideanDistanceThreshold The threshold when compareing the Euclidean distance between expected output and actual output signal.
     * @param successRateEuclideanDistance Success rate of testing the Euclidean distance.
     * @param successRateIdenticalMax Success rate when testing that the maximum elements are identical.
     * @param averageCost Average cost
     * @param failedSamplesIdx Vector of sample indices which were NOT successful.
     * @return True if successful. Otherwise false.
     */
    bool testNetwork( const std::vector<Eigen::MatrixXd>& samples, const std::vector<Eigen::MatrixXd>& lables,
                      const double& euclideanDistanceThreshold, bool doCallback, double& successRateEuclideanDistance,
                      double& successRateIdenticalMax, double& averageCost, std::vector<size_t>& failedSamplesIdx );

    /**
     * Tests the network with given samples and lables. The computation is performed in another
     * thread. The user gets informed over the NetworkOperationCallback interface.
     * @param samples Input sample.
     * @param lables Expected output.
     * @param euclideanDistanceThreshold The threshold when compareing the Euclidean distance between expected output and actual output signal.
     * @return True if successful. Otherwise false.
     */
    bool testNetworkAsync( const std::vector<Eigen::MatrixXd>& samples, const std::vector<Eigen::MatrixXd>& lables,
                           const double& euclideanDistanceThreshold );

    /**
     * Returns the magnitude of the error vector in the output layer. This error is
     * initialized during the backpropagation.
     * @return error magnitude
     */
    double getNetworkErrorMagnitude() const;

    /**
     * Returns the cost in the output layer. This error is
     * initialized during the backpropagation.
     * @return Cost.
     */
    double getNetworkCost() const;

    /**
     * Set an observer, which gets informed about operation progress.
     * @param observer A pointer to an observer.
     */
    void setObserver( NetworkOperationCallback* observer ) { m_oberserver = observer; }

    /**
    * Return a handle to the current NN operation thread.
    * @return Handle to compuation thread.
    */
    std::thread& getCurrentAsyncOperation() { return m_asyncOperation; }

    /**
     * Indicates if an asynchronous operation is ongoing or not.
     * @return True if operation in progress. Otherwise false.
     */
    bool isOperationInProgress() { return m_operationInProgress; }

    /**
     * Returns the structure of the neural network.
     * @return
     */
    std::vector<unsigned int> getNetworkStructure() const { return m_NetworkStructure; }

    /**
     * Sets the applied cost function in the outputlayer.
     * @param function Cost function id.
     */
    void setCostFunction( const ECostFunction& function );

    /**
     * Serialize the network (layers).
     * @return string holding binary representation of the network.
     */
    std::string serialize() const;

    /**
     * Deserialize a binary representation of a network.
     * @param buffer Binaray data.
     * @return Initialized network
     */
    static Network* deserialize(const std::string& buffer );

    /**
     * Save the current neuronal network to a file.
     * @param filePath Path to file.
     * @return True if successful, otherwise false.
     */
    bool save( const std::string& filePath );

    /**
     * Load a neuronal network from a file
     * @param filePath Path to file.
     * @return Initialized network
     */
    static Network* load( const std::string& filePath );

    /**
     * Enable or disable softmax output layer.
     * @param enable True or false.
     */
    void setSoftmaxOutput( const bool& enable );

    /**
     * Is softmax output enabled or disabled.
     * @return True if enabled. False if disabled.
     */
    bool isSoftmaxOutputEnabled() const;

    void print();

    /**
     * Creates a vector of random indices. Every index occures once.
     * @param numberOfElements Number of elements.
     * @return Vector of indices.
     */
    std::vector<size_t> randomIndices(size_t numberOfElements) const;

    /**
     * Sets the applied regularization.
     * @param method
     * @param lamda
     */
    void setRegularizationMethod( Regularization reg );

    /**
     * Gets the applied regularization.
     * @return Regularization method.
     */
    Regularization getRegularizationMethod() const;

    /**
     * Gets the sum of all square weights in the
     * network.
     * @return Sum of square.
     */
    double getSumOfWeighSquares() const;


private:

    void initNetwork();

    // Do feedforward and backprop. but weights and biases are not updated!
    bool doFeedforwardAndBackpropagation(const Eigen::MatrixXd &x_in, const Eigen::MatrixXd &y_out );

    bool doStochasticGradientDescentBatch(const Eigen::MatrixXd& batch_in, const Eigen::MatrixXd& batch_out, const double& eta, double& cost);

    void sendProg2Obs( const NetworkOperationCallback::NetworkOperationId& opId,
                       const NetworkOperationCallback::NetworkOperationStatus& opStatus, const double& progress  );

    bool prepareForNextAsynchronousOperation();

    void doTestAsync( const std::vector<Eigen::MatrixXd>& samples, const std::vector<Eigen::MatrixXd>& lables,
                      const double& euclideanDistanceThreshold );

private:

    const std::vector<unsigned int> m_NetworkStructure;
    std::vector< std::shared_ptr<Layer> > m_Layers;
    Eigen::MatrixXd m_activation_out;

    NetworkOperationCallback* m_oberserver;
    std::thread m_asyncOperation;
    std::atomic<bool> m_operationInProgress;

    Regularization m_regularization;
};

#endif //NETWORKHEADER
