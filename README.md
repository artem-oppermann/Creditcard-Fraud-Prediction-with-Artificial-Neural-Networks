# Creditcard-Fraud-Prediction-with-Artificial-Neural-Networks
An artifical neural network that can identify creditcard fraud. The network is trained on the dataset that contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where there are 492 frauds out of 284,807 transactions. The dataset can be found at https://www.kaggle.com/dalpozz/creditcardfraud.

The neural network is not static. The number of hidden layers, the number of neurons and the activation functions for hidden and output neurons can be changed any time. For the hidden neurons the user can choose between sigmoid, relu and tanh, for the output neurons between sigmoid, relu, tanh and linear function as activation.

A network with only one hidden layer consisting of 200 neurons, sigmoid as activation function for hidden neurons and linear as fuction for the output the network achieves after 50 epochs an accuracy of 99,96% on the validation set.


The final performance measurement on the test set yields provides the following confusion matrix:

[[71065     7]

 [29     101]]

The matrix corresponds to the following performance values:

 Precision: 0.935

 Recall: 0.777

 F1 Score: 0.849
 
The network could identify 77.7 % of overall commited frauds that were in the dataset.
93.5 % of the time the network identifies a commited fraud correctly. In that particular case out of 108 predictions for the commited frauds 101 were correct.
