# Setting up folders which are not present 
import os
try:
	os.mkdir('./Models')
except Exception as e:
	print(e)

try:	
	os.mkdir('./TrainingData')
except Exception as e:
	print(e)