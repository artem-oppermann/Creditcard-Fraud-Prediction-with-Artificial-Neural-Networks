# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:28:52 2017

@author: Artem Oppermann
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split




class Preprocess:
    '''Preprocessing of the csv file. Loads, shapes and splits 
    the data from the csv file.
    '''
    
    
    def __init__(self):
        	pass
    
    
    def loadData(self, path):
        ''' Loads the csf file from the given path and returns the file. '''
        
        return pd.read_csv(path)
    
    
    def shape_labels(self, labels):
        '''Shapes the label column (1d-array) into two new label columns (2d-Matrix). 
        Each entry in the old label column with value k becomes a row in the new label matrix.
        All elements in that row are zero, except for column k, which has the entry of 1.
        '''
        
        temp=[]
        for i in labels:
            if i==0:
                temp.append([1,0])
            elif i==1:
                temp.append([0,1])
                
        return np.array(temp)
    
    
    
    def standardizeData(self, data_set):
        ''' Standardize features by removing the mean and scaling to unit variance.'''
        
        sc = StandardScaler()
        
        return sc.fit_transform(data_set)
        
        

    def splitData(self,features, labels, training_size, standardize, eval_set):
        ''' Splits the data into training, testing and validation sets. 
        Furthermore applies standardization and shapes the label array into 
        a label matrix.'''
        
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = \
                                                    (1-training_size), random_state = 0)
        
        if eval_set:
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = 0)
        
        if standardize:
            x_train=self.standardizeData(x_train)
            x_test=self.standardizeData(x_test)
            if eval_set:
                x_val=self.standardizeData(x_val)

        if eval_set:
            shaped_y_val=self.shape_labels(y_val)
            
        shaped_y_train=self.shape_labels(y_train)
        shaped_y_test=self.shape_labels(y_test)
            
        if eval_set:
            return [x_train, shaped_y_train, x_test, shaped_y_test, x_val, shaped_y_val, y_test]
        else: 
            return [x_train, shaped_y_train, x_test, shaped_y_test, y_test]
            
        

    def getData(self, path, feature_incides, label_indices,training_size, standardize, eval_set):
        '''Main function of preprocessing. Loads, encodes and splits the data from the csv-file.'''
        
        data=self.loadData(path)
        features=data.iloc[:,feature_incides[0]:feature_incides[1]].values
        labels=data.iloc[:,label_indices[0]].values
        final_data=self.splitData(features, labels, training_size, standardize, eval_set)
        
        return final_data