import pandas as pd
import torch
import numpy 



import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder        
from sklearn.model_selection import train_test_split


def data_preprocessing(task_1a_dataframe):

	''' 

	This function will be used to load your csv dataset and preprocess it.
	Preprocessing involves cleaning the dataset by removing unwanted features,
	decision about what needs to be done with missing values etc. Note that 
	there are features in the csv file whose values are textual (eg: Industry, 
	Education Level etc)These features might be required for training the model
	but can not be given directly as strings for training. Hence this function 
	should return encoded dataframe in which all the textual features are 
	numerically labeled.
	
	
	'''

	
	label_encoder = LabelEncoder()
	task_1a_dataframe['Education'] = label_encoder.fit_transform(task_1a_dataframe['Education'])
	task_1a_dataframe['City'] = label_encoder.fit_transform(task_1a_dataframe['City'])
	task_1a_dataframe['Gender'] = label_encoder.fit_transform(task_1a_dataframe['Gender'])
	task_1a_dataframe['EverBenched'] = label_encoder.fit_transform(task_1a_dataframe['EverBenched'])
	encoded_dataframe=task_1a_dataframe
	

	return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):
	'''
	
	The purpose of this function is to define the features and
	the required target labels. The function returns a python list
	in which the first item is the selected features and second 
	item is the target label

	
	'''

	
	features = encoded_dataframe.drop(['LeaveOrNot','Gender'], axis=1)   #leaves LeaveOrNot','Gender and takes all others as features
	target = encoded_dataframe['LeaveOrNot']
	features_and_targets=[features, target]    #returns target and features as a list
	

	return features_and_targets


def load_as_tensors(features_and_targets):

	''' 
	Purpose:
	---
	This function aims at loading your data (both training and validation)
	as PyTorch tensors. Here you will have to split the dataset for training 
	and validation, and then load them as as tensors. 
	Training of the model requires iterating over the training tensors. 
	Hence the training sensors need to be converted to iterable dataset
	object.
	
	
	'''

	
	features, target = features_and_targets
	X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=10) #split test and train data

	X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32) #converts to tensor
	X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
	y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
	y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

	train_dataset = TensorDataset(X_train_tensor, y_train_tensor)   #creates dataset
	train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
	tensors_and_iterable_training_data=[X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader]
	return tensors_and_iterable_training_data

class Salary_Predictor():
	'''
	Purpose:
	---
	The architecture and behavior of your neural network model will be
	defined within this class that inherits from nn.Module. Here you
	also need to specify how the input data is processed through the layers. 
	It defines the sequence of operations that transform the input data into 
	the predicted output. When an instance of this class is created and data
	is passed through it, the `forward` method is automatically called, and 
	the output is the prediction of the model based on the input data.
	
	
	'''
	def __init__(self):
		super(Salary_Predictor, self).__init__()
		
		
		self.fc1 = nn.Linear(7, 256)    # 2 hidden layer used to increase accuracy
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, 1)

		self.sigmoid=nn.Sigmoid()
		self.tanh=nn.Tanh()
		self.leaky_relu=nn.LeakyReLU(negative_slope=0.1)

		

	def forward(self, x):
		'''
		Define the activation functions
		'''
		
		x = self.sigmoid(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = torch.relu(self.fc3(x))
		x = self.sigmoid(self.fc4(x))
		predicted_output=x
		

		return predicted_output

def model_loss_function():
	'''
	
	To define the loss function for the model. Loss function measures 
	how well the predictions of a model match the actual target values 
	in training data.
	
	Input Arguments:
	---
	None

	Returns:
	---
	`loss_function`: This can be a pre-defined loss function in PyTorch
					or can be user-defined

	Example call:
	---
	loss_function = model_loss_function()
	'''
	
	loss_function=nn.BCELoss()
	
	
	return loss_function

def model_optimizer(model):
	'''

	To define the optimizer for the model. Optimizer is responsible 
	for updating the parameters (weights and biases) in a way that 
	minimizes the loss function.
	
	'''
	
	optimizer=optim.Adam(model.parameters(), lr=0.0001)
	

	return optimizer

def model_number_of_epochs():
	'''
	To define the number of epochs for training the model

	'''

	number_of_epochs=1500
	return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
	
	
	for epoch in range(number_of_epochs):
		model.train()
		total_loss = 0
		for batch_x, batch_y in tensors_and_iterable_training_data:
			optimizer.zero_grad()
			output = model(batch_x)
			output=torch.squeeze(output,dim=1)
			loss = loss_function(output, batch_y)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		print(f"Epoch {epoch + 1}/{number_of_epochs}, Loss: {total_loss:.4f}")
	

	return model

def validation_function(trained_model, tensors_and_iterable_training_data):
	'''
	
	This function will utilise the trained model to do predictions on the
	validation dataset. This will enable us to understand the accuracy of
	the model.

	
	'''	
	
	X_test_tensor, y_test_tensor = tensors_and_iterable_training_data[1], tensors_and_iterable_training_data[3]
	trained_model.eval()
	with torch.no_grad():
		predictions = trained_model(X_test_tensor)
        
		predicted_labels = (predictions >= 0.5).float()  # Threshold for binary classification
		correct_predictions = (predicted_labels == y_test_tensor.unsqueeze(1)).sum().item()
		total_samples = len(y_test_tensor)
		accuracy = correct_predictions / total_samples
	model_accuracy=accuracy*100
	

	return model_accuracy

########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########	
'''
	Purpose:

	The following is the main function combining all the functions
	mentioned above.

'''
if __name__ == "__main__":

	# reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe
	task_1a_dataframe = pd.read_csv('data.csv')

	# data preprocessing and obtaining encoded data
	encoded_dataframe = data_preprocessing(task_1a_dataframe)

	# selecting required features and targets
	features_and_targets = identify_features_and_targets(encoded_dataframe)

	# obtaining training and validation data tensors and the iterable
	# training data object
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	
	# model is an instance of the class that defines the architecture of the model
	model = Salary_Predictor()

	# obtaining loss function, optimizer and the number of training epochs
	loss_function = model_loss_function()
	optimizer = model_optimizer(model)
	number_of_epochs = model_number_of_epochs()

	# training the model
	trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, 
					loss_function, optimizer)

	# validating and obtaining accuracy
	model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
	print(f"Accuracy on the test set = {model_accuracy}")

	X_train_tensor = tensors_and_iterable_training_data[0]
	x = X_train_tensor[0]
	jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "model.pth")