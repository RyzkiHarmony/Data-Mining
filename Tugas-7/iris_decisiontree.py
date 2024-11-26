import numpy as np
import pandas as pd
from sklearn import tree

# Load dataset
irisDataset = pd.read_csv('Dataset Iris.csv', delimiter=';', header=0)
irisDataset["Species"] = pd.factorize(irisDataset.Species)[0]
irisDataset = irisDataset.drop(labels="Id", axis=1)

# Split dataset into training and testing data
dataTraining = np.concatenate(
    (irisDataset.iloc[0:40, :].to_numpy(),
     irisDataset.iloc[50:90, :].to_numpy()), axis=0
)
dataTesting = np.concatenate(
    (irisDataset.iloc[40:50, :].to_numpy(),
     irisDataset.iloc[90:100, :].to_numpy()), axis=0
)

inputTraining = dataTraining[:, 0:4]
inputTesting = dataTesting[:, 0:4]
labelTraining = dataTraining[:, 4]
labelTesting = dataTesting[:, 4]

# Define and train the Decision Tree Classifier
model = tree.DecisionTreeClassifier()
model = model.fit(inputTraining, labelTraining)

# Predict using the testing data
hasilPrediksi = model.predict(inputTesting)

# Print actual and predicted labels
print("Label sebenarnya: ", labelTesting)
print("Hasil prediksi: ", hasilPrediksi)

# Calculate accuracy
prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()

print("Prediksi benar: ", prediksiBenar, "data")
print("Prediksi salah: ", prediksiSalah, "data")
print("Akurasi: ", prediksiBenar / (prediksiBenar + prediksiSalah) * 100, "%")
