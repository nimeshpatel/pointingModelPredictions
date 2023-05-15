import numpy as np
from tensorflow import keras

# Load the previously trained model
loaded_model = keras.models.load_model('model_may2023.h5')

# Compile the model with appropriate loss and optimizer functions
#model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.001))

# Generate some example training data
#X_train = np.random.rand(100, 2)
#y_train = np.random.rand(100, 3)

# Train the neural network on the training data
#model.fit(X_train, y_train, epochs=3000, batch_size=16)
#model.save('model_may2023.h5')

# Read in the input values from the user
#input_values = [[1,7],[2,23],[3,12],[4,5],[5,9],[6,1],[7,8],[8,4]]

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
print(f"A={predictions[0][0]}, B={predictions[0][1]}, C={predictions[0][2]}")


# Print the predicted values
#print("Predictions:")
#for i in range(8):
#    print(f"Prediction {i+1}: A={predictions[i][0]}, B={predictions[i][1]}, C={predictions[i][2]}")
#

# Use the trained neural network to make predictions on the test data
#predictions = model.predict(X_test)

# Print the predicted values for A, B, and C for each test input (X, Y) pair
#for i in range(len(X_test)):
#    print(f"For input values X={X_test[i][0]:.2f} and Y={X_test[i][1]:.2f}, predicted values are: A={predictions[i][0]:.2f}, B={predictions[i][1]:.2f}, C={predictions[i][2]:.2f}")
