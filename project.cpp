#include <iostream>
#include <chrono>
#include <array>
#include <cstdlib>
#include "net.h"

using namespace std;

/** This is an artificial neural network that is trained to act like a NAND gate.
 * It uses a weighted sum function as the input function and the Sigmoid function
 * as the activation function. The squared error function is used to calculate
 * the error and back-propagation is executed with a learning rate of 0.5.
 * The structure of the ANN is 2x5x1.
 *
 * @author Dylan Pavao
 * @course COSC 3P93
 */
int main(int argc,char* argv[]) {  

  int thread_count = 1;
  if(argc>1) thread_count = stoi(argv[1]);    //get number of threads from execution call arguments 
  if(thread_count > 8) thread_count = 8;

  const int numInput = 2;         //number neurons in input layer
  const int numHidden = 5;        //number neurons in hidden layer
  const int numOutput = 1;        //number neurons in output layer
  const int numberNetWork = 5000; //number of Neural Networks to train  
  double bestAvgError = 1000000;  //used to calculate average error of best performing network   
  double weight;                  //used to calculate weights 
  int bestNetwork;                //index of the best performing network

  array<array<double,numInput>,4> inputs = {{{0,0},{0,1},{1,0},{1,1}}};  //set of inputs that correspond to target outputs (NAND Gate)
  array<double,4> targets = {1,1,1,0};                                   //set of target output values (NAND Gate)
  array<array<double,numInput>,numHidden> hidden1;                       //weights between input layer and hidden layer
  array<double,numHidden> hidden2;                                       //weights between hidden layer and output layer
  array<double,2> bias;                                                  //bias for each set of weights
  array<NeuralNet*, numberNetWork> networks;                             //collection of neural networks to train
  

  auto start = chrono::system_clock::now();   //Start Timer
  srand(310);                                 //seed for random number generation

  //Initialize each Network with random weights [-0.5, 0.5] and Bias  
  for(int i=0; i<numberNetWork; i++){    
    //Weights between Input and Hidden layer 1
    for(int j=0; j<numHidden; j++){ 
      for(int k=0; k<numInput; k++){
        weight = rand() % 100000000;
        weight = (weight / 100000000) - 0.5; 
        hidden1.at(j).at(k) = weight;
      }
    }
    //Weights between Input and Hidden layer 2 
    for(int j=0; j<numHidden; j++){
      weight = rand() % 100000000;
      weight = (weight / 100000000) - 0.5;
      hidden2.at(j) = weight;
    }
    //Bias for each Layer 
    for(int j=0; j<2; j++){
      weight = rand() % 100000000;
      weight = (weight / 100000000) - 0.5;
      bias.at(j) = weight;
    }    
    networks.at(i) = new NeuralNet(inputs.at(0), hidden1, hidden2, targets.at(0), bias);    
  }

  //Train the Neural Networks using OpenMP multi-threading
  double avgErr;
  int testSet = rand() % 4;
  #pragma omp parallel for num_threads(thread_count)
  for(int i=0; i<numberNetWork; i++){
    for(int j=0; j<5000; j++){                          //Each network undergoes 5000 rounds of Back-propagation
      networks.at(i)->setInputs(inputs.at(testSet));
      networks.at(i)->setTarget(targets.at(testSet));
      networks.at(i)->train();
      testSet = rand() % 4;
    }
    avgErr = networks.at(i)->getAvgError();
    #pragma omp critical                                //Mutual Exclusion on shared resource 
    if(avgErr < bestAvgError){
      bestAvgError = avgErr;
      bestNetwork = i;
    }
  }

  //Test the Best Neural Network
  double avgError = 0;
  cout<<"\nNumber of Threads Used: "<<thread_count<<endl;
  cout<<"Number of Networks Trained: "<<numberNetWork<<endl;
  cout<<"\nBest Network: "<<bestNetwork<<endl;
  cout<<"Inputs (0,0)"<<endl;
  cout<<"Expected Output: 1"<<endl;
  networks.at(bestNetwork)->setInputs(inputs.at(0));
  networks.at(bestNetwork)->setTarget(targets.at(0));
  avgError += networks.at(bestNetwork)->testWithOutput();
  cout<<"Inputs (0,1)"<<endl;
  cout<<"Expected Output: 1"<<endl;
  networks.at(bestNetwork)->setInputs(inputs.at(1));
  networks.at(bestNetwork)->setTarget(targets.at(1));
  avgError += networks.at(bestNetwork)->testWithOutput();
  cout<<"Inputs (1,0)"<<endl;
  cout<<"Expected Output: 1"<<endl;
  networks.at(bestNetwork)->setInputs(inputs.at(2));
  networks.at(bestNetwork)->setTarget(targets.at(2));
  avgError += networks.at(bestNetwork)->testWithOutput();
  cout<<"Inputs (1,1)"<<endl;
  cout<<"Expected Output: 0"<<endl;
  networks.at(bestNetwork)->setInputs(inputs.at(3));
  networks.at(bestNetwork)->setTarget(targets.at(3));
  avgError += networks.at(bestNetwork)->testWithOutput();
  avgError = avgError/4;
  cout<<"Average Error: "<<avgError<<endl;

  auto end = chrono::system_clock::now();
  chrono::duration<double> runtime = end-start;   //End Timer 
  cout<<"Elapsed Time: "<<runtime.count()<<"s\n";
}
