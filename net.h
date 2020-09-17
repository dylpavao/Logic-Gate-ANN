#include <array>
using namespace std;
/** This class represents the structure that is the ANN
 * It houses all the calculations for back-propagation and forward passing
 */
class NeuralNet
{
private:
  array<double, 2> input;             //set of input values
  array<array<double, 2>, 5> hidden1; //weights from input layer to hidden
  array<double, 5> hidden2;           //weights from hidden layer to output
  double target;                      //target output value
  array<double, 2> bias;              //bias for each set of weights
  const double LEARNING_RATE = 0.5;   //learning rate to speed up back-propagation
public:
  //Initializes network to have set inputs, weights and target output based on params
  NeuralNet(array<double, 2> input, array<array<double, 2>, 5> hidden1, array<double, 5> hidden2, double target, array<double, 2> bias);

  //Sets the input values to parameter "input"
  void setInputs(array<double, 2> input);

  //Sets the weights from input layer to hidden to parameter "hidden1"
  void setHidden1(array<array<double, 2>, 5> hidden1);

  //Sets the weights from hidden layer to output to parameter "hidden2"
  void setHidden2(array<double, 5> hidden2);

  //Sets the target output value to parameter "target"
  void setTarget(double target);

  //Sets the bias values to parameter "bias"
  void setBias(array<double, 2> bias);

  //Activates the neurons in hidden layer to produce an output for forward pass
  double activationFunction(array<double, 2> input, array<double, 2> weight, double bias);

  //Activates the output neuron to produce a final output for forward pass
  double activationFunction(array<double, 5> input, array<double, 5> weight, double bias);

  //Calculates the error based on parameter "output" and the target value
  double calcError(double output);

  //Updates the weights in the network during the back propagation
  void updateWeights(array<array<double, 2>, 5> newWeightsH1, array<double, 5> newWeightsH2);

  //Executes a single forward pass and back-prop to update weights and train the network
  void train();

  //Executes a single forward pass to test how the netork is performing with output
  double testWithOutput();

  //Executes a single forward pass to test how the netork is performing without output
  double testWithoutOutput();

  //Executes multiple tests on all test data to determine average margin of error
  double getAvgError();
};
