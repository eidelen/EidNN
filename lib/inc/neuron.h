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

#ifndef _NEURON_H_
#define _NEURON_H_

#include <eigen3/Eigen/Dense>

class Neuron
{

public:

    /**
     * Builds a neuron.
     * @param nbr_of_input Number of incoming activations.
     */
    Neuron( const unsigned int& nbr_of_input );
    ~Neuron();

    /**
     * Returns the number of input connections.
     */
    unsigned int getNbrOfInputs() const { return m_nbr_of_inputs; }

    /**
     * Sets the neuron's weight vector
     * @param weights
     * @return true if successful
     */
    bool setWeights( const Eigen::VectorXf& weights );

    /**
     * Fills the weight vector with Gaussian noise.
     * @param mean Mean of the Gaussian noise.
     * @param deviation Standard deviation of the Gaussian noise.
     */
    void setRandomWeights( const float& mean, const float& deviation );

    /**
     * Returns the weights vector of this neuron.
     * @return
     */
    const Eigen::VectorXf getWeights( ) const { return m_weights; }

    /**
     * Sets a random bias using Gaussian noise.
     * @param mean Mean of the Gaussian noise.
     * @param deviation Standard deviation of the Gaussian noise.
     */
    void setRandomBias( const float& mean, const float& deviation );

    /**
     * Sets the neuron's bias.
     * @param bias
     */
    void setBias( const float& bias );

    /**
     * Returns the bias of this neuron.
     * @return Bias as float
     */
    float getBias() const {return m_bias;}


    /**
     * Computes the activation of this neuron with the passed
     * input (activations from former layer's neurons). Further,
     * the intermediate product z is passed back.
     * @return true if successful
     */
    bool feedForward(const Eigen::VectorXf& x_in, float &z, float& activation );

    /**
    * Sigmoid function value of a given input z.
    */
    static float sigmoid( const float& z );

    /**
    * Sigmoid derivation function value of a given input z.
    */
    static float d_sigmoid( const float& z ); 

private:

    const unsigned int m_nbr_of_inputs;

    Eigen::VectorXf m_x_in;
    Eigen::VectorXf m_weights;
    float m_bias;
    float m_z;
    float a_out;

};

#endif // _NEURON_H_
