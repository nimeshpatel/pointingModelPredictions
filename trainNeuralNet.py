import numpy as np
from tensorflow import keras

# Load the training data from the ASCII file
data = np.loadtxt('./TrainingDataSets/finalTrainingSetNumbered')

#In the input data file the format is as follows:
# serial No., Ant No (X), Pad No. (Y), AzDC (A), AzTltSin (B), AzTltCos # (C), Unixtime
# We divide the AzDC values by 100, to make the numbers comparable to
# the tilt coefficients.

data[:, 3] /= 100.0

# Split the data into input features (X, Y) and target values (A, B, C)
X_train = data[:, 1:3]   # columns 2 and 3 correspond to X and Y
y_train = data[:, 3:6]   # columns 4,5, 6 correspond to A, B, and C
# column 1 is serial number and column 7 is unixtime

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(2,)),
    keras.layers.Dense(128, activation='relu', input_shape=(2,)),
    keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3)
])

# Compile the model with appropriate loss and optimizer functions
#model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.001))

model.compile(optimizer='adam',loss='mse')

# Generate some example training data
#X_train = np.random.rand(100, 2)
#y_train = np.random.rand(100, 3)

# Train the neural network on the training data
model.fit(X_train, y_train, epochs=3000, batch_size=16)
model.save('model_15may2023.h5')

# Generate some example test data
#X_test = np.random.rand(10, 2)

# Read in the input values from the user
input_values = [[1,7],[2,23],[3,12],[4,5],[5,9],[6,1],[7,8],[8,4]]
#for i in range(8):
#    x_value = float(input("Enter value for X: "))
#    y_value = float(input("Enter value for Y: "))
#    input_values.append([x_value, y_value])

# Convert the input values to a numpy array and make predictions using the model
X_test = np.array(input_values)
predictions = model.predict(X_test)

# Print the predicted values
print("Predictions:")
for i in range(8):
    print(f"Prediction {i+1}: A={predictions[i][0]}, B={predictions[i][1]}, C={predictions[i][2]}")


# Use the trained neural network to make predictions on the test data
#predictions = model.predict(X_test)

# Print the predicted values for A, B, and C for each test input (X, Y) pair
#for i in range(len(X_test)):
#    print(f"For input values X={X_test[i][0]:.2f} and Y={X_test[i][1]:.2f}, predicted values are: A={predictions[i][0]:.2f}, B={predictions[i][1]:.2f}, C={predictions[i][2]:.2f}")
