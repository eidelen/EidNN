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
#include <eigen3/Eigen/Dense>

using namespace std;

class Layer;

class Network
{
public:

    /**
     * Constructor of a neural network.
     * @param networkStructure A vector holding the number of neurons for each layer,
     *                         where the first element of the vector is the number of
     *                         neurons in the first layer, and the last vector item the
     *                         number of neurons in the last layer, the output layer.
     */
    Network( const vector<unsigned int> networkStructure );

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
    shared_ptr<Layer> getLayer(const unsigned int &layerIdx );

    /**
     * Returns the last layer, also called output layer.
     * @return Last layer of the network.
     */
    shared_ptr<Layer> getOutputLayer();

    /**
     * Feedforward, backpropagate and update weigths and biases in each layer corresponding
     * to the computed partial derivatives and the gradient descent method.
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
     * @param samples Input signals.
     * @param lables Desired output signals.
     * @param batchsize Number of samples in the batch.
     * @param eta
     * @return true if successful.
     */
    bool stochasticGradientDescent(const std::vector<Eigen::MatrixXd> &samples, const std::vector<Eigen::MatrixXd> &lables,
                                   const unsigned int& batchsize, const double& eta );

    /**
     * Returns the magnitude of the error vector in the output layer. This error is
     * initialized during the backpropagation.
     * @return error magnitude
     */
    double getNetworkErrorMagnitude();


    void print();


private:

    // Do feedforward and backprop. but weights and biases are not updated!
    bool doFeedforwardAndBackpropagation(const Eigen::MatrixXd &x_in, const Eigen::MatrixXd &y_out );

    void initNetwork();

private:

    const vector<unsigned int> m_NetworkStructure;
    vector< shared_ptr<Layer> > m_Layers;
    Eigen::MatrixXd m_activation_out;
};

#endif //NETWORKHEADER
