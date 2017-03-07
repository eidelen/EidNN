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

#ifndef _LAYER_H_
#define _LAYER_H_


#include <vector>
#include <memory>
#include <string>
#include <eigen3/Eigen/Dense>

using namespace std;

class Neuron;

class Layer
{
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
     * @param weights Vector of neuron weights.
     * @param biases Vector of neuron biases.
     */
    Layer( const uint& nbr_of_inputs, const vector<Eigen::VectorXf>& weights, const vector<float>& biases );

    ~Layer();

    /**
     * Compute the neural layer output signal based on the input signal x_in.
     * The output signal can be accessed with the function getOutputActivation().
     * @param x_in Input signal.
     * @return true if successful.
     */
    bool feedForward( const Eigen::VectorXf& x_in );
    const Eigen::VectorXf& getOutputActivation() const { return m_activation_out; }
    const Eigen::VectorXf& getWeightedInputZ() const { return m_z_weighted_input; }

    unsigned int getNbrOfNeurons() const { return m_nbr_of_neurons; }
    unsigned int getNbrOfNeuronInputs() const { return m_nbr_of_inputs; }

private:
    void initLayer();

private:
    vector< shared_ptr<Neuron> > m_neurons;
    const unsigned int m_nbr_of_neurons;
    const unsigned int m_nbr_of_inputs;

    Eigen::VectorXf m_activation_out;
    Eigen::VectorXf m_z_weighted_input;

};

#endif //_LAYER_H_
