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

class CostFunction;

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
    Layer( const uint& nbr_of_inputs, const std::vector<Eigen::VectorXd>& weights, const std::vector<double>& biases );

    /**
     * Copy-constructor
     * @param l
     */
    Layer( const Layer& l );


    ~Layer();

    /**
     * Compute the neural layer output signal based on the input signal x_in.
     * The output signal can be accessed with the function getOutputActivation().
     * The computation is done for all neuron at once with the weight matrix.
     * @param x_in Input signal.
     * @return true if successful.
     */
    bool feedForward( const Eigen::MatrixXd& x_in );

    /**
     * Sets the weights-vector in each neuron of this layer.
     * @param weights Vector of neuron weights-vector.
     * @return true if successful
     */
    bool setWeights( const std::vector<Eigen::VectorXd>& weights );

    /**
     * Sets the weights in this layer.
     * @param weights The weight matrix
     * @return true if successful
     */
    bool setWeights( const Eigen::MatrixXd& weights );

    /**
     * Sets the same weight for all neurons and all intputs
     * @param weights Weight value.
     */
    void setWeight( const double& weight );

    /**
     * Returns the current weight matrix. It does not reassemble the matrix
     * ( see updateWeightMatrixAndBiasVector() )
     * @return
     */
    const Eigen::MatrixXd& getWeigtMatrix() const { return m_weightMatrix; }

    /**
     * Sets the bias of each neuron in this layer.
     * @param biases Vector of neuron biases.
     * @return true if successful
     */
    bool setBiases( const std::vector<double>& biases );

    /**
     * Sets the bias of each neuron in this layer.
     * @param biases Vector of neuron biases.
     * @return true if successful
     */
    bool setBiases(const Eigen::MatrixXd &biases );

    /**
     * Sets the same bias for all neurons
     * @param bias Bias value.
     */
    void setBias( const double& bias );

    /**
     * Returns the current bias vector. It does not reassemble the vector
     * ( see updateWeightMatrixAndBiasVector() )
     * @return
     */
    const Eigen::MatrixXd& getBiasVector() const { return m_biasVector; }

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
    const Eigen::MatrixXd& getOutputActivation() const { return m_activation_out; }

    /**
     * Get the input activation of this layer. This is set after calling feedForward().
     * @return Input activation.
     */
    const Eigen::MatrixXd& getInputActivation() const { return m_activation_in; }

    /**
     * This is an intermediate result of calling feedForward(). It is the weighted input,
     * or one can also think of it as the activation output without performing the sigmoid
     * function.
     * @return weighted input.
     */
    const Eigen::MatrixXd& getWeightedInputZ() const { return m_z_weighted_input; }

    /**
     * This function computes the backpropagation error in case this is the output layer.
     * The actual error can then be accessed by getBackpropagationError().
     * @param expectedNetworkOutput The desired network output.
     * @return Return true if operation was successful. Otherwise false
     */
    bool computeBackpropagationOutputLayerError( const Eigen::MatrixXd& expectedNetworkOutput );

    /**
     * Computes the backpropagation error in this layer. The backpropagation error can be accessed
     * by the function getBackpropagationError().
     * @param expectedNetworkOutput The desired network output.
     * @return Return true if operation was successful. Otherwise false
     */
    bool computeBackprogationError(const Eigen::MatrixXd& errorNextLayer, const Eigen::MatrixXd& weightMatrixNextLayer );

    /**
     * Computes the partial derivatives of the biases and weights. Results can be
     * accessed by getPartialDerivativesBiases() and getPartialDerivativesWeights().
     */
    void computePartialDerivatives();

    /**
     * Updates the biases and weights within this layer based on the computed
     * derivatives and the learning rate.
     * @param eta Learning rate
     * @param sampleIdx Based on which sample's partial derivatives the weights and biases should be updated
     */
    void updateWeightsAndBiases( const double& eta, const unsigned int& sampleIdx = 0  );

    /**
     * Corrects the biases and weights within this layer by the passed values.
     * @param deltaBias
     * @param deltaWeight
     */
    void updateWeightsAndBiases(const Eigen::MatrixXd &deltaBias, const Eigen::MatrixXd& deltaWeight );

    /**
     * Returns the computed backprogation error in this layer. Each column of the returned
     * matrix corresponds to one input / output sample. If there was only one sample passed,
     * the matrix has the form of m x 1 ( a vector).
     * @return
     */
    const Eigen::MatrixXd getBackpropagationError() const { return m_backpropagationError; }

    /**
     * Partial derivatives of the biases. This is set after calling computePartialDerivatives().
     * The vector holds the derivatives for each passed sample.
     * @return
     */
    const std::vector<Eigen::MatrixXd>& getPartialDerivativesBiases() const { return m_bias_partialDerivatives; }

    /**
     * Partial derivatives of weights. This is set after calling computePartialDerivatives().
     * The vector holds the derivatives for each passed sample.
     * @return
     */
    const std::vector<Eigen::MatrixXd>& getPartialDerivativesWeights() const { return m_weight_partialDerivatives; }

    /**
     * Set the cost function. This is only relevant in the output layer. The default cost function
     * is the quadratic cost function.
     * @param costFunction The new cost function.
     */
    void setCostFunction( const std::shared_ptr<CostFunction>& costFunction ) { m_costFunction = costFunction; }

    /**
     * Return the currently used cost function.
     * @return Cost function.
     */
    const std::shared_ptr<CostFunction>& getCostFunction() { return m_costFunction; }

    /**
     * Serialize the layer (weights, biases).
     * @return string holding binary representation of the layer.
     */
    std::string serialize() const;

    /**
     * Deserialize a binary representation of a layer.
     * @param buffer Binaray data.
     * @return Initialized layer
     */
    static Layer* deserialize(const std::string& buffer );

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
    bool setActivationOutput(const Eigen::MatrixXd &activation_out );


private:
    unsigned int m_nbr_of_neurons;
    unsigned int m_nbr_of_inputs;

    Eigen::MatrixXd m_activation_in;
    Eigen::MatrixXd m_activation_out;
    Eigen::MatrixXd m_z_weighted_input;

    Eigen::MatrixXd m_backpropagationError;
    Eigen::MatrixXd m_weightMatrix;
    Eigen::MatrixXd m_biasVector;

    std::vector<Eigen::MatrixXd> m_bias_partialDerivatives;
    std::vector<Eigen::MatrixXd> m_weight_partialDerivatives;

    std::shared_ptr<CostFunction> m_costFunction;
};

#endif //LAYERHEADER
