import numpy as np
from tensorflow import keras

# Load the previously trained neural network model
loaded_model = keras.models.load_model('model_15may2023.h5')


#for i in range(8):
#    x_value = float(input("Enter value for X: "))
#    y_value = float(input("Enter value for Y: "))
#    input_values.append([x_value, y_value])


input_values = []
x_value = float(input("Enter antenna number: "))
y_value = float(input("Enter pad number: "))
input_values.append([x_value, y_value])

# Convert the input values to a numpy array and make predictions using the model
X_test = np.array(input_values)
predictions = loaded_model.predict(X_test)

print("Predicted coarse pointing model parameters:")
print(f"A={10.0*predictions[0][0]}, B={predictions[0][1]}, C={predictions[0][2]}")
