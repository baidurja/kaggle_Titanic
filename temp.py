# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print("Classification test!")
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

def preprocess( dataset, offset ):
    X = dataset.iloc[ :, [ 2 + offset, 4 + offset, 5 + offset, 6 + offset, 7 + offset, 9 + offset, 11 + offset ] ].values
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X = LabelEncoder()
    X[ :, 1 ] = labelencoder_X.fit_transform( X[ :, 1 ] )
    X[ :, 6 ] = labelencoder_X.fit_transform( X[ :, 6 ] )
    from sklearn.preprocessing import Imputer
    imputer = Imputer( missing_values = 'NaN', strategy = 'mean', axis = 0 )
    imputer = imputer.fit( X )
    X = imputer.transform( X )
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_toscale = X[ :, [2, 3, 4, 5] ]
    #    X_toscale = X_toscale[ :, None ]
    X_toscale = sc_X.fit_transform( X_toscale )
    #X_toscale.shape = X[ :, [2, 3, 4, 5] ].shape
    X[ :, [2, 3, 4, 5] ] = X_toscale
    onehotencoder = OneHotEncoder( categorical_features = [0, 6] )
    X = onehotencoder.fit_transform(X).toarray()
    
    return X

def visualize( data ):
    XX = dataset.values
    from sklearn.preprocessing import Imputer
    imputer = Imputer( missing_values = 'NaN', strategy = 'mean', axis = 0 )
    X_toscale = XX[ :, 5 ]
    X_toscale = X_toscale[ :, None ]
    X_toscale = imputer.fit_transform( X_toscale )
    X_toscale.shape = XX[ :, 5 ].shape
    XX[ :, 5 ] = X_toscale
#    imputer = imputer.fit( XX )
#    XX = imputer.transform( XX )
    pos_indx = [];
    neg_indx = [];
    pos_cnt = 0;
    neg_cnt = 0;
    for k in range( 0, XX.shape[ 0 ] - 1 ):
        if ( y[ k ] == 0 ):
            neg_indx.append( k )
            neg_cnt = neg_cnt + 1
        else:
            pos_indx.append( k )
            pos_cnt = pos_cnt + 1
                
    col_indx = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ]
    pos_X = XX[ py.ix_( pos_indx, col_indx )]   
    neg_X = XX[ py.ix_( neg_indx, col_indx )]
    
    plt.hist( [ pos_X[ :, 8 ].astype( float ), neg_X[ :, 8 ].astype( float ) ], color = [ 'g', 'r' ] )
    plt.title( 'Fare' )
    plt.show()
    plt.hist( [ pos_X[ :, 6 ].astype( float ), neg_X[ :, 6 ].astype( float ) ], color = [ 'g', 'r' ] )
    plt.title( 'Num of parent or children' )
    plt.show()
    plt.hist( [ pos_X[ :, 5 ].astype( float ), neg_X[ :, 5 ].astype( float ) ], color = [ 'g', 'r' ] )
    plt.title( 'Num of siblings or spouses' )
    plt.show()
    plt.hist( [ pos_X[ :, 4 ].astype( float ), neg_X[ :, 4 ].astype( float ) ], color = [ 'g', 'r' ] )
    plt.title( 'Age' )
    plt.show()
#    plt.hist( [ pos_X[ :, 3 ].astype( str ), neg_X[ :, 3 ].astype( str ) ], color = [ 'g', 'r' ] )
#    plt.title( 'Sex' )
#    plt.show()
    plt.hist( [ pos_X[ :, 1 ].astype( float ), neg_X[ :, 1 ].astype( float ) ], color = [ 'g', 'r' ] )
    plt.title( 'Class' )
    plt.show()
    
    plt.hist( pos_X[ :, 3 ].astype( str ), color = 'green' )
    plt.title( 'Sex' )
    plt.show()
    plt.hist( neg_X[ :, 3 ].astype( str ), color = 'red' )
    plt.title( 'Sex' )
    plt.show()
    plt.hist( pos_X[ :, 10 ].astype( str ), color = 'green' )
    plt.title( 'Embarked' )
    plt.show()
    plt.hist( neg_X[ :, 10 ].astype( str ), color = 'red' )
    plt.title( 'Embarked' )
    plt.show()

dataset = pd.read_csv('train.csv')
dataset.Embarked = dataset.Embarked.fillna( 'S' )

#offset = 0
#X = dataset.iloc[ :, [ 2 + offset, 4 + offset, 5 + offset, 6 + offset, 7 + offset, 9 + offset, 11 + offset ] ].values
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[ :, 1 ] = labelencoder_X.fit_transform( X[ :, 1 ] )
#X[ :, 6 ] = labelencoder_X.fit_transform( X[ :, 6 ] )


X = preprocess( dataset, 0 )
dev_dataset = pd.read_csv( 'test.csv' )
dev_X = preprocess( dev_dataset, -1 )
#X = dataset.iloc[ :, [ 2, 4, 5 ] ].values
y = dataset.iloc[ :, 1 ].values

visualize( dataset )


#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[ :, 1 ] = labelencoder_X.fit_transform( X[ :, 1 ] )
#
#from sklearn.preprocessing import Imputer
#imputer = Imputer( missing_values = 'NaN', strategy = 'mean', axis = 0 )
#imputer = imputer.fit( X )
#X = imputer.transform( X )
#
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_toscale = X[ :, 2 ]
#X_toscale = X_toscale[ :, None ]
#X_toscale = sc_X.fit_transform( X_toscale )
#X_toscale.shape = X[ :, 2 ].shape
#X[ :, 2 ] = X_toscale

#for k in range( 0, X.shape[ 0 ] - 1 ):
#    if ( y[k] == 0 ):
#        plt.scatter( X[k,0],X[k,2],c='r')
#    else:
#        plt.scatter( X[k,0],X[k,2],c='g')  
#    
#plt.show()

#y = labelencoder_X.fit_transform(y)
#onehotencoder = OneHotEncoder( categorical_features = [0] )
#X = onehotencoder.fit_transform(X).toarray()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 100 )

algo = "SVM";

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

if (algo == "SVM"):
    classifier = SVC( C=1, kernel = 'rbf', gamma = 0.143, random_state=0)
elif ( algo == "NaiveBayes"):
    classifier = GaussianNB()    
elif ( algo == "DecisionTree"):
    classifier = DecisionTreeClassifier( criterion = "entropy", random_state = 0 )
elif ( algo == "RandomForest" ):
    classifier = RandomForestClassifier( n_estimators = 20, criterion = "entropy", random_state = 0 )
else:
    classifier = LogisticRegression(random_state=0)

from sklearn.model_selection import cross_val_score
scores = cross_val_score( classifier, X_train, y_train, cv = 10 )
print("CrossValidation score:")
print( scores )
print("Mean accuracy = ", scores.mean())
print("SD accuracy = ", scores.std())

from sklearn.model_selection import GridSearchCV
if ( algo == "SVM" ):
    parameters = [ {'C': [ 1.0, 2.0, 4.0, 0.5, 0.75, 10, 11, 12 ], 'kernel': ['rbf'], 'gamma': [ 1/5, 1/6, 1/6.5, 1/7, 1/8 ] }]
    grid_search = GridSearchCV( estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10 )
    grid_search = grid_search.fit( X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_params = grid_search.best_params_
    print( best_params )

classifier.fit(X_train, y_train)
y_train_pred = classifier.predict( X_train )

y_test_pred = classifier.predict( X_test )

print( classifier.score(X_test, y_test) )


from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix( y_test, y_test_pred )
cm_train = confusion_matrix( y_train, y_train_pred )

cError_train = 100 * ( cm_train[0,1] + cm_train[1,0] ) / ( cm_train[0,0] + cm_train[1,1] +cm_train[0,1] + cm_train[1,0])
cError_test = 100 * ( cm_test[0,1] + cm_test[1,0] ) / ( cm_test[0,0] + cm_test[1,1]+cm_test[0,1] + cm_test[1,0])

from sklearn.metrics import accuracy_score
print("Training accuracy score:")
print( accuracy_score(y_train,y_train_pred))
print("Test accuracy score:")
print( accuracy_score(y_test,y_test_pred))

from sklearn.metrics import classification_report
print( classification_report( y_train, y_train_pred))
print( classification_report( y_test, y_test_pred))

y_dev_pred = classifier.predict( dev_X )

sub = py.column_stack( ( dev_dataset.iloc[ :, 0 ].values, y_dev_pred ) )
py.savetxt( "gender_submission.csv", sub.astype( int ), fmt = "%i", delimiter=",", header = "PassengerId,Survived", comments = '' )

#plt.scatter( X_train, y_train, color = 'red' )
#plt.plot( X_train, y_train_pred, color = 'blue' )
#plt.show()