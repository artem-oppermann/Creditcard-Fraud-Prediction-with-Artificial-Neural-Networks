# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:35:25 2017

@author: Artem Oppermann
"""

from model import Model
from preprocess import Preprocess




def main():
    '''Training of the model on the preprocessed data. '''
    
    preprocess=Preprocess()
    data=preprocess.getData(path="creditcard.csv",          # path of the csv file
                            feature_incides=[0,29],         # column indices of the features
                            label_indices=[30],             # column indices of the labels
                            training_size=0.5,              # size for the training set 
                            standardize=True,               # apply standardization?
                            eval_set=True                   # create evaluation set?
                            )
    
    model=Model(batch_size=10,                  # size of the training batch  
                epochs=50,                      # number of training epochs  
                nodes=[29, 200, 2],             # List of neurons, first entry is the number of input, last entry 
                                                # the number of output neurons. The values in between are the hidden neurons.                      
                learning_rate=0.0001,           # learning rate for the training
                hidden_activation="sigmoid",    # activation function for the hidden nodes, choose between "tanh", "sigmoid" and "relu"
                output_activation="linear",     # activation function for the output nodes, choose between "tanh", "sigmoid" and "linear" 
                data=data,                      # the loaded and preprocessed data form the csv file
                do_eval=True                    # measure accuracy of the evaluation set?
                ) 
    
    model.train()
    
    
if __name__ == "__main__":
    
    main()
