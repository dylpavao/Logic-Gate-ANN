#include <iostream>
#include <array>
#include <cmath>
#include "net.h"
using namespace std;

NeuralNet::NeuralNet(array<double, 2> input, array<array<double, 2>, 5> hidden1, array<double, 5> hidden2, double target, array<double, 2> bias)
{
  this->input = input;
  this->hidden1 = hidden1;
  this->hidden2 = hidden2;
  this->target = target;
  this->bias = bias;
}

void NeuralNet::setInputs(array<double, 2> input)
{
  this->input = input;
}

void NeuralNet::setHidden1(array<array<double, 2>, 5> hidden1)
{
  this->hidden1 = hidden1;
}

void NeuralNet::setHidden2(array<double, 5> hidden2)
{
  this->hidden2 = hidden2;
}

void NeuralNet::setTarget(double target)
{
  this->target = target;
}

void NeuralNet::setBias(array<double, 2> bias)
{
  this->bias = bias;
}

double NeuralNet::activationFunction(array<double, 2> input, array<double, 2> weight, double bias)
{
  //calculate net weight for neuron
  for (int i = 0; i < 2; i++)
  {
    bias += input.at(i) * weight.at(i);
  }
  bias = -1 * bias;
  bias = 1 / (1 + exp(bias)); //sigmoid function
  return bias;
}

double NeuralNet::activationFunction(array<double, 5> input, array<double, 5> weight, double bias)
{
  //calculate net weight for neuron
  for (int i = 0; i < 5; i++)
  {
    bias += input.at(i) * weight.at(i);
  }
  bias = -1 * bias;
  bias = 1 / (1 + exp(bias)); //sigmoid function
  return bias;
}

double NeuralNet::calcError(double output)
{
  return pow(target - output, 2);
}

void NeuralNet::updateWeights(array<array<double, 2>, 5> newWeightsH1, array<double, 5> newWeightsH2)
{
  //Update first set of weights
  for (int i = 0; i < 5; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      hidden1.at(i).at(j) = hidden1.at(i).at(j) - LEARNING_RATE * newWeightsH1.at(i).at(j);
    }
  }

  //Update second set of weights
  for (int i = 0; i < 5; i++)
  {
    hidden2.at(i) = hidden2.at(i) - LEARNING_RATE * newWeightsH2.at(i);
  }
}

void NeuralNet::train()
{
  array<double, 5> outs;
  double output;
  //Forward Pass
  for (int i = 0; i < 5; i++)
  {
    outs.at(i) = activationFunction(input, hidden1.at(i), bias.at(0));
  }
  output = activationFunction(outs, hidden2, bias.at(1));

  //Back Propagation
  //Calculate new weights for paths connected to output
  array<double, 5> newWeightsH2;
  for (int i = 0; i < 5; i++)
  {
    newWeightsH2.at(i) = -(target - output) * (output * (1 - output)) * outs.at(i);
  }
  //Calculate new weights for paths connected to hidden layer
  array<array<double, 2>, 5> newWeightsH1;
  for (int i = 0; i < 5; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      newWeightsH1.at(i).at(j) = -(target - output) * (output * (1 - output)) * (hidden2.at(i)) * (outs.at(i) * (1 - outs.at(i))) * input.at(j);
    }
  }
  updateWeights(newWeightsH1, newWeightsH2);
}

double NeuralNet::testWithOutput()
{
  array<double, 5> outs;
  double output;
  double error;
  for (int i = 0; i < 5; i++)
  {
    outs.at(i) = activationFunction(input, hidden1.at(i), bias.at(0));
  }
  output = activationFunction(outs, hidden2, bias.at(1));
  error = calcError(output);
  cout << "Actual Output: " << output << endl;
  cout << "Error: " << error << endl
       << endl;
  return error;
}

double NeuralNet::testWithoutOutput()
{
  array<double, 5> outs;
  double output;
  double error;
  for (int i = 0; i < 5; i++)
  {
    outs.at(i) = activationFunction(input, hidden1.at(i), bias.at(0));
  }
  output = activationFunction(outs, hidden2, bias.at(1));
  error = calcError(output);
  return error;
}

double NeuralNet::getAvgError()
{
  array<array<double, 2>, 4> inputs = {{{0, 0}, {0, 1}, {1, 0}, {1, 1}}}; //set of inputs
  array<double, 4> targets = {1, 1, 1, 0};                                //set of target output values
  double avgError = 0;

  setInputs(inputs.at(0));
  setTarget(targets.at(0));
  avgError += testWithoutOutput();
  setInputs(inputs.at(1));
  setTarget(targets.at(1));
  avgError += testWithoutOutput();
  setInputs(inputs.at(2));
  setTarget(targets.at(2));
  avgError += testWithoutOutput();
  setInputs(inputs.at(3));
  setTarget(targets.at(3));
  avgError += testWithoutOutput();

  return avgError / 4;
}
