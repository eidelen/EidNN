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

#ifndef LAYERHEADER
#define LAYERHEADER


#include <vector>
#include <memory>
#include <string>
#include <eigen3/Eigen/Dense>

#include "network.h"

using namespace std;

class Layer
{
    friend class Network;

public:
    /**
     * Constructor of a layer, which consists of many neurons.
     * @param nbr_of_neurons Number of neurons in this layer.
     * @param nbr_of_inputs Number of inputs to each neuron (usually this number is equal to the amount of neurons in the previous layer)
     */
    Layer( const uint& nbr_of_neurons, const uint& nbr_of_inputs );

    /**
     * Constructor of a layer.
     * @param nbr_of_inputs Number of inputs to each neuron.
     * @param weights Vector of neuron weights-vector
     * @param biases Vector of neuron biases.
     */
    Layer( const uint& nbr_of_inputs, const vector<Eigen::VectorXf>& weights, const vector<float>& biases );

    ~Layer();

    /**
     * Compute the neural layer output signal based on the input signal x_in.
     * The output signal can be accessed with the function getOutputActivation().
     * The computation is done for all neuron at once with the weight matrix.
     * @param x_in Input signal.
     * @return true if successful.
     */
    bool feedForward( const Eigen::VectorXf& x_in );

    /**
     * Sets the weights-vector in each neuron of this layer.
     * @param weights Vector of neuron weights-vector.
     * @return true if successful
     */
    bool setWeights( const vector<Eigen::VectorXf>& weights );

    /**
     * Sets the weights in this layer.
     * @param weights The weight matrix
     * @return true if successful
     */
    bool setWeights( const Eigen::MatrixXf& weights );

    /**
     * Sets the same weight for all neurons and all intputs
     * @param weights Weight value.
     */
    void setWeight( const float& weight );

    /**
     * Returns the current weight matrix. It does not reassemble the matrix
     * ( see updateWeightMatrixAndBiasVector() )
     * @return
     */
    const Eigen::MatrixXf& getWeigtMatrix() const { return m_weightMatrix; }

    /**
     * Sets the bias of each neuron in this layer.
     * @param biases Vector of neuron biases.
     * @return true if successful
     */
    bool setBiases( const vector<float>& biases );

    /**
     * Sets the bias of each neuron in this layer.
     * @param biases Vector of neuron biases.
     * @return true if successful
     */
    bool setBiases( const Eigen::VectorXf& biases );

    /**
     * Sets the same bias for all neurons
     * @param bias Bias value.
     */
    void setBias( const float& bias );

    /**
     * Returns the current bias vector. It does not reassemble the vector
     * ( see updateWeightMatrixAndBiasVector() )
     * @return
     */
    const Eigen::VectorXf& getBiasVector() const { return m_biasVector; }

    /**
     * Resets all weights and biases of each neuron in this layer
     * with zero-mean Gaussian noise of standard deviation 1.0
     */
    void resetRandomlyWeightsAndBiases();

    /**
     * Get the output activation of this layer. This function is usually called
     * after executing feedForward().
     * @return Output activation Vector
     */
    const Eigen::VectorXf& getOutputActivation() const { return m_activation_out; }

    /**
     * Get the input activation of this layer. This is set after calling feedForward().
     * @return Input activation.
     */
    const Eigen::VectorXf& getInputActivation() const { return m_activation_in; }

    /**
     * This is an intermediate result of calling feedForward(). It is the weighted input,
     * or one can also think of it as the activation output without performing the sigmoid
     * function.
     * @return weighted input.
     */
    const Eigen::VectorXf& getWeightedInputZ() const { return m_z_weighted_input; }

    /**
     * This function computes the backpropagation error in case this is the output layer.
     * The actual error can then be accessed by getBackpropagationError().
     * @param expectedNetworkOutput The desired network output.
     * @return Return true if operation was successful. Otherwise false
     */
    bool computeBackpropagationOutputLayerError( const Eigen::VectorXf& expectedNetworkOutput );

    /**
     * Computes the backpropagation error in this layer. The backpropagation error can be accessed
     * by the function getBackpropagationError().
     * @param expectedNetworkOutput The desired network output.
     * @return Return true if operation was successful. Otherwise false
     */
    bool computeBackprogationError(const Eigen::VectorXf& errorNextLayer, const Eigen::MatrixXf& weightMatrixNextLayer );

    /**
     * Computes the partial derivatives of the biases and weights. Results can be
     * accessed by getPartialDerivativesBiases() and getPartialDerivativesWeights().
     */
    void computePartialDerivatives();

    /**
     * Updates the biases and weights within this layer based on the computed
     * derivatives and the learning rate.
     * @param eta Learning rate
     */
    void updateWeightsAndBiases( const float& eta );

    /**
     * Returns the computed backprogation error in this layer.
     * @return
     */
    const Eigen::VectorXf getBackpropagationError() const { return m_backpropagationError; } 

    /**
     * Partial derivatives of the biases. This is set after calling computePartialDerivatives();
     * @return
     */
    const Eigen::VectorXf getPartialDerivativesBiases() const { return m_bias_partialDerivatives; }

    /**
     * Partial derivatives of weights. This is set after calling computePartialDerivatives();
     * @return
     */
    const Eigen::MatrixXf getPartialDerivativesWeights() const { return m_weight_partialDerivatives; }

    /**
     * Computes component wise derrivative of the sigmoid function
     * for each component in the vector z.
     * @param z Vector in
     * @return Vector holding the result.
     */
    static const Eigen::VectorXf d_sigmoid( const Eigen::VectorXf& z );

    unsigned int getNbrOfNeurons() const { return m_nbr_of_neurons; }
    unsigned int getNbrOfNeuronInputs() const { return m_nbr_of_inputs; }

    void print() const;

private:
    void initLayer();

    /**
     * Sets directly the activation output of this layer.
     * This function is called by the network for the
     * first layer, the input layer.
     * @param activation_out
     * @return True if successful.
     */
    bool setActivationOutput( const Eigen::VectorXf& activation_out );


private:
    const unsigned int m_nbr_of_neurons;
    const unsigned int m_nbr_of_inputs;

    Eigen::VectorXf m_activation_in;
    Eigen::VectorXf m_activation_out;
    Eigen::VectorXf m_z_weighted_input;

    Eigen::VectorXf m_backpropagationError;
    Eigen::MatrixXf m_weightMatrix;
    Eigen::VectorXf m_biasVector;

    Eigen::VectorXf m_bias_partialDerivatives;
    Eigen::MatrixXf m_weight_partialDerivatives;
};

#endif //LAYERHEADER
