import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from Data_Prep import get_data, corr_analysis, standardize
from Model import fault_classifier
from Visualize import viewAcc, viewLoss, corr_visual


######## Data Preparation, extraction and visualization ################3

#Extract data from data sets
dataSet = get_data('Faults.train')

X_train = dataSet.iloc[:,:-7]
y_train = dataSet.iloc[:,-7:].values

#Observe data correlation to perform correlation analysis:
corr_visual(X_train,'Before Correlation Analysis')

#Do some correlation analysis to avoide redundant features
X_train = corr_analysis(X_train, corrCriteria = 0.9)

#And now ...
corr_visual(X_train,'After Correlation Analysis')

#Standardize data:
X_train = standardize(X_train)

################### Classificator: Neural Network ############################

#Create the model
model = fault_classifier()

#Fit the model and save the progress visualize it
history = model.fit(X_train,y_train,epochs = 200, validation_split = 0.2)


#Accuracy Visualization
viewAcc(history)

#Loss visualization
viewLoss(history)

################################################################################

## Assesment: Typical overfitting situation. We can add some droputs (or many other techniques) 
# and see if it works.



