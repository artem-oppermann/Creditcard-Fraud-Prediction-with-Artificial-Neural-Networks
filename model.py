# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:31:54 2017

@author: Artem Oppermann
"""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score
import tensorflow as tf
import sys



class Model:
    ''' Artificial neural network model for fraud prediciton'''
    
    def __init__(self, batch_size, epochs, nodes, learning_rate, hidden_activation, output_activation, data, do_eval):
        
        self.batch_size=batch_size
        self.epochs=epochs
        self.nodes=nodes
        self.learning_rate=learning_rate
        
        self.hidden_activation=hidden_activation,
        self.output_activation=output_activation,
        
        self.weights=[]
        self.bias=[]
        
        self.do_eval=do_eval
        
        for i in range(0, len(self.nodes)-1):
            self.weights.append(self.weight_matrix([self.nodes[i],self.nodes[i+1]]))
            
        for i in range(0, len(self.nodes)-2):
            self.bias.append(self.bias_vector([self.nodes[1+i]]))
        
        if self.do_eval:
            self.X_train=data[0]
            self.Y_train=data[1]
            self.X_test=data[2]
            self.Y_test=data[3]
            self.X_val=data[4]
            self.Y_val=data[5]
            self.y_test=data[6]
        else:
            self.X_train=data[0]
            self.Y_train=data[1]
            self.X_test=data[2]
            self.Y_test=data[3]
            self.y_test=data[4]
        
  
  
    def weight_matrix(self, shape):
        ''' Initializes a weight matrix. The weight values are normal
        distributed with a standard deviation of 0.1'''
        
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
        
        return tf.Variable(initial)   
    
    
    
    def bias_vector(self, bias_shape):
        ''' A bias vector is initialized. The bias values are normal
        distributed with a standard deviation of 0.1'''
        
        initial = tf.truncated_normal(bias_shape, stddev=0.1, dtype=tf.float32)
        
        return tf.Variable(initial)
    
    
 
    def get_batch(self, current_index, id):
        ''' Returns the shaped (based on tf.placeholders) batches of features 
        and labels. The batch size for validation and testing is always 1.'''
        
        lower_bound=self.batch_size*current_index
        upper_bound=self.batch_size*(current_index+1)
        index_array=np.array([i for i in range(lower_bound, upper_bound)])
        
        if id=="training":
            return np.array([self.X_train[i] for i in index_array]).reshape(self.batch_size,len(self.X_train[0])),\
                    np.array([self.Y_train[i] for i in index_array]).reshape(self.batch_size,self.nodes[-1])
            
        if id=="validation":
            return np.array(self.X_val[current_index]).reshape(1,len(self.X_val[0])),\
                     np.array(self.Y_val[current_index]).reshape(1,self.nodes[-1])
            
        if id=="testing":
            return np.array(self.X_test[current_index]).reshape(1,len(self.X_test[0])),\
            np.array(self.Y_test[current_index]).reshape(1,self.nodes[-1])
    
           
    
    def predict(self, x):
        ''' Predicts the final values of output nodes, based on chosen hidden
        and output activation functions.'''
        
        layer_values=[x]
        
        if "sigmoid" in self.hidden_activation:
            for i in range(0, len(self.nodes)-2):
                layer_values.append(tf.nn.sigmoid(tf.matmul(layer_values[i], self.weights[i]))+self.bias[i])
        elif "relu" in self.hidden_activation:
            for i in range(0, len(self.nodes)-2):
                layer_values.append(tf.nn.relu(tf.matmul(layer_values[i], self.weights[i]))+self.bias[i])
        elif "tanh" in self.hidden_activation:
            for i in range(0, len(self.nodes)-2):
                layer_values.append(tf.nn.tanh(tf.matmul(layer_values[i], self.weights[i]))+self.bias[i])
                
        if "sigmoid" in self.output_activation:
            return tf.nn.sigmoid(tf.matmul(layer_values[-1], self.weights[-1]))
        elif "relu" in self.output_activation:
            return tf.nn.relu(tf.matmul(layer_values[-1], self.weights[-1]))
        elif "linear" in self.output_activation:
            return tf.matmul(layer_values[-1], self.weights[-1])
                
        
    
    def validation_acc(self, sess, accuracy, prediction, X, Y):
        '''Calculates the accuracy on validation set. Accuracity as defined as
        devision of true predictions by all predictions'''
        
        val_acc=[]
        
        for i in range(0, len(self.X_val)):
            x_val, y_val=self.get_batch(i, "validation")
            val_acc.append(sess.run(accuracy, feed_dict={X: x_val, Y: y_val}))
            
        val_acc=np.array(val_acc)
        print("val_acc: %.7f" % (len(val_acc[val_acc==1])/len(val_acc))) 
        
    
    
    def test_performance(self, sess, accuracy, prediction, X, Y):
        '''Calculates the precision, recall, f1 score and average
        precision score of the  testing set.'''
        
        predictions=[]
        
        for i in range(0, len(self.X_test)):
            x_test, y_test=self.get_batch(i, "testing")
            predictions.append(np.argmax(sess.run(prediction, feed_dict={X: x_test, Y: y_test})))
            
        predictions=np.array(predictions)
        
        print("\n Confusion matrix: \n")
        print(confusion_matrix(self.y_test,predictions))
        print("\n Precision: %.3f" %precision_score(self.y_test,predictions))
        print("\n Recall: %.3f" %recall_score(self.y_test,predictions))
        print("\n F1 Score: %.3f" %f1_score(self.y_test,predictions))
        print("\n Average Precision Score: %.3f" %average_precision_score(self.y_test,predictions))

    
    def train(self):
        '''Training the model.'''
        
        X = tf.placeholder(tf.float32, shape=(None, self.nodes[0]))
        Y = tf.placeholder(tf.float32, shape=(None, self.nodes[-1]))
        
        y_ = self.predict(X)
        
        loss =tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_)
        optimizer=tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        
        prediction=tf.nn.softmax(y_)
        correct_prediction= tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy=tf.cast(correct_prediction, tf.float32)
        
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
           
            for epoch in range(1, self.epochs+1):
                
                for i in range(0, int(len(self.X_train)/self.batch_size)):
                    
                    sys.stdout.write("\r epoch nr. "+str(epoch)  +", training on batch "+str(i)+"/"+str(int(len(self.X_train)/self.batch_size-1)))
                    sys.stdout.flush()
                    
                    x_train, y_train=self.get_batch(i, "training")
                    sess.run(optimizer, feed_dict={X: x_train, Y: y_train})
                    
                if self.do_eval: 
                    self.validation_acc(sess, accuracy, prediction, X, Y)
                
            self.test_performance(sess, accuracy, prediction, X, Y) 
