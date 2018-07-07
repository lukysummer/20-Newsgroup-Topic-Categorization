# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:59:00 2018

@author: Luky
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:39:43 2017
@author: Luky
"""
'''
Question 1 Skeleton Code: NEWSGROUP PREDICTIONS
Users read and post messages (called articles or posts, and collectively termed news) 
to one or more categories (called newsgroups)
----> when a user writes a post, want to automatically categorize it to a newsgroup
'''
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn import tree
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


np.set_printoptions(threshold=np.inf)

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    no_meaning_words = {"the", "then", "Thanks", "Thank", "she", "but", "let", "for", "out", "whether", "not", "through", "couldnt", "cant", "wouldnt", "wont", "shouldnt", "how", "have", "has", "had", "havent", "hasnt", "hadnt", "are", "all", "was", "were", "anyone", "someone", "maybe", "probably", "please", "they", "you", "this", "that", "these", "those", "from", "with", "soon", "within", "sometimes", "often", "always", "why", "how", "whom", "who", "because", "also", "there", "here", "therefore", "since", "from", "until", "what", "hence", "and", "can", "could", "will", "would", "where", "there", "here", "should", "must", "might", "may", "shall"}
    
    new_train = []; new_test = []
    word_length = 0
    new_word = ""
    for i in newsgroups_train.data:
        new_point = ""     
        for j in range(len(i)):
            if((i[j] == " ") | (i[j] == "\n") | (j==len(i)-1)):
                if((word_length > 20) | (word_length < 3) | (new_word in no_meaning_words)): 
                #if((word_length > 20) | (word_length < 3)): 
                    new_word = ""
                    word_length = 0
                else:
                    if(word_length >= 6):
                        # eliminating "ed" - past tense
                        if(new_word[len(new_word)-2:len(new_word)] == "ed"):
                            new_word = new_word[0:len(new_word)-1]
                        # eliminating "ing"
                        elif(new_word[len(new_word)-3:len(new_word)] == "ing"):
                            new_word = new_word[0:len(new_word)-3]
                            
                    new_point = new_point + " " + new_word
                    new_word = ""
                    word_length = 0
            
            elif(i[j].isalpha()):
                new_word += i[j]
                word_length += 1
                
        new_train.append(new_point)
        
                  
    word_length = 0
    new_word = ""         
    for i in newsgroups_test.data:
        new_point = ""    
        for j in range(len(i)):
            if((i[j] == " ") | (i[j] == "\n")):
                if((word_length > 20) | (word_length < 3) | (new_word in no_meaning_words)): 
                #if((word_length > 20) | (word_length < 3)): 
                    new_word = ""
                    word_length = 0
                else:
                    if(word_length >= 6):
                        # eliminating "ed" - past tense
                        if(new_word[len(new_word)-2:len(new_word)] == "ed"):
                            new_word = new_word[0:len(new_word)-1]
                        # eliminating "ing"
                        if(new_word[len(new_word)-3:len(new_word)] == "ing"):
                            new_word = new_word[0:len(new_word)-3]
                    new_point = new_point + " " + new_word
                    new_word = ""
                    word_length = 0
            
            elif(i[j].isalpha()):
                new_word += i[j]
                word_length += 1

        new_test.append(new_point)

    return new_train, new_test, newsgroups_train, newsgroups_test



def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    print("")
   
    bow_train = bow_vectorize.fit_transform(train_data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0])) # 11,314 - train data points
    print('{} feature dimension.'.format(shape[1])) 

    ''' bow_train[i,j] = 1 means : train data point i contains jth word out of 101631 words'''

    print("")
    return bow_train, bow_test, feature_names



def tf_idf_features(train_data, test_data):
    tf_idf_vectorize = TfidfVectorizer()
    
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data)
    
    shape = tf_idf_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
 
    return tf_idf_train, tf_idf_test, feature_names



# 1.(baseline) Bernoulli Naive Bayes Model with binary features
def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int) # if bow_train[i]>0, binary_train[i]=1
    binary_test = (bow_test>0).astype(int)   # if bow_train[i]<0, binary_train[i]=0

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model



# 2. Decision Trees
def decision_tree(bow_train, train_labels, bow_test, test_labels, maximum_depth):
    print("max_depth: ", maximum_depth)
    model = tree.DecisionTreeClassifier(splitter = 'random', max_depth = maximum_depth)
    model.fit(bow_train, train_labels)

    train_pred = model.predict(bow_train)
    print('Decision Tree training accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test)
    print('Decision Tree test accuracy = {}'.format((test_pred == test_labels).mean()))
    print(test_labels.shape)
    print(test_pred.shape)
    print(test_pred)
    print("")
    
    return (train_pred == train_labels).mean(), (test_pred == test_labels).mean(), test_pred



# 3. Random Forest
def random_forest(bow_train, train_labels, bow_test, test_labels, num_trees):
    print("number of trees: ", num_trees)
    model = RandomForestClassifier(n_estimators = num_trees)
    model.fit(bow_train, train_labels)
    
    train_pred = model.predict(bow_train)
    print('Decision Tree training accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test)
    print('Decision Tree test accuracy = {}'.format((test_pred == test_labels).mean()))
    print("")
    
    return (train_pred == train_labels).mean(), (test_pred == test_labels).mean(), test_pred



# 4. Neural Network MLP
def neural_net(bow_train, train_labels, bow_test, test_labels, num_hidden_layer):
    print("number of hidden layers: ", num_hidden_layer)
    model = MLPClassifier(hidden_layer_sizes=(num_hidden_layer), solver='sgd', learning_rate_init=0.05, max_iter=80)
    model.fit(bow_train, train_labels)
 
    train_pred = model.predict(bow_train)
    print('Neural Network training accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test)
    print('Neural Network test accuracy = {}'.format((test_pred == test_labels).mean()))
    print("")
    
    return (train_pred == train_labels).mean(), (test_pred == test_labels).mean(), test_pred

    

def cross_validation(train_data, train_labels, model, param_range):      
    kf = KFold(n_splits = 10)
    train_valid_accuracy = []
    test_valid_accuracy = []
    
    for k in param_range:
        train_valid_accuracy_each_param = []
        test_valid_accuracy_each_param = []
        for train_index, valid_index in kf.split(train_data): 
            # Splitting training data into training & validation sets
            train, valid = train_data[train_index], train_data[valid_index]
            label_train, label_valid = train_labels[train_index], train_labels[valid_index]
            
            if(model == 1):
                train_acc, test_acc, pred = decision_tree(train, label_train, valid, label_valid, k)
            elif(model == 2):
                train_acc, test_acc, pred = random_forest(train, label_train, valid, label_valid, k)
            else:   
                train_acc, test_acc, pred = neural_net(train, label_train, valid, label_valid, k)
                
            train_valid_accuracy_each_param.append(train_acc)
            test_valid_accuracy_each_param.append(test_acc)
                
        train_valid_accuracy.append(np.average(train_valid_accuracy_each_param))
        test_valid_accuracy.append(np.average(test_valid_accuracy_each_param))
        print(test_valid_accuracy)

    print("Average accuracy over different parameter values: ")
    print("param range: ", param_range)
    print("train acc:   ", train_valid_accuracy)
    print("test acc:    ", test_valid_accuracy)
    
    max_valid_accuracy = np.amax(test_valid_accuracy)
    max_valid_accuracy_index = [i for i, x in enumerate(test_valid_accuracy) if x == max_valid_accuracy]
    
    optimal_param = param_range[max_valid_accuracy_index[0]]
    print("Optimal Value of the Hyperparameter: ", optimal_param)
    
    return optimal_param



def confusion_matrix(true_labels, predicted_labels):
    # confusion_matrix, C
    C = np.zeros((20,20))
    
    for i in range(0,len(true_labels)): 
        k = int(true_labels[i])    
        C[predicted_labels[i]][k] += 1 
        
    print(C)
    
    # True-Positive Matrix
    TP = []
    for i in range(20):
        TP.append(C[i][i])

    false_prediction = []
    for i in range(20):
        false_prediction_each_newsgroup = 0
        for j in range(20):
            if(j != i):
                false_prediction_each_newsgroup += C[j][i]
        false_prediction.append(false_prediction_each_newsgroup)
    
    confusion_ratio = []
    for i in range(20):
        confusion_ratio.append(TP[i]/false_prediction[i])
        
    print("Classifier was most confused with following classes: ", np.argsort(confusion_ratio)[:2])

    return confusion_ratio



if __name__ == '__main__':
    # <Loading Data>
    startTime = datetime.now()
    train_data, test_data, train, test = load_data()

    train_tf_idf, test_tf_idf, feat_names_tf_idf = tf_idf_features(train_data, test_data)
    print(""); print("")
    
    
    # <10-fold Cross Validation>
    
    print("1. Decision Tree")
    depths = [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    best_depth = cross_validation(train_tf_idf, train.target, 1, depths)  
    print("Best depth: ", best_depth)
   
    print("2. Random Forest")
    num_trees = [100, 110, 120, 130, 140, 150, 160]
    best_num_of_trees = cross_validation(train_tf_idf, train.target, 2, num_trees) 
    print("Best number of trees: ", best_num_of_trees)
    
    print("3. Neural Network")
    num_hidden_layers = [10, 20, 30, 40, 50, 60]
    best_num_layers = cross_validation(train_tf_idf, train.target, 3, num_hidden_layers)  
    print("Best number of hidden layers: ", best_num_layers)
    
    
    # <Training & Testing>
    
    # 1. Bernoulli Naive Bayes Model
    print("1. Bernoulli Naive Bayes Model/ntf_idf features:")
    bnb_model_tfidf = bnb_baseline(train_tf_idf, train.target, test_tf_idf, test.target)
    print(""); print("")
    
        
    # 2. Decision Tree Model
    print("2. Decision Tree Model/ntf_idf features:")   
    train_acc_DT, test_acc_DT, predictions_DT = decision_tree(train_tf_idf, train.target, test_tf_idf, test.target, best_depth)
    print(""); print("")
    
    
    # 3. Random Forest Model   
    print("2. Random Forest Model/ntf_idf features:")  
    train_acc_RF, test_acc_RF, pred_RF = random_forest(train_tf_idf, train.target, test_tf_idf, test.target, best_num_of_trees)
    print(""); print("")
    
    
    # 4. Neural Network Model
    print("2. Neural Network Model/ntf_idf features:")
    train_acc_MLP, test_acc_MLP, pred_MLP = neural_net(train_tf_idf, train.target, test_tf_idf, test.target, best_num_layers)
    print(""); print("")
    confusion_matrix(test.target, pred_MLP)

    print("")
    print("Runtime: ", datetime.now() - startTime)