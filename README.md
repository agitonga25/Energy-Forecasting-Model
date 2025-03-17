# ⚡ Energy Forecasting Model  

This project predicts **energy demand** using machine learning and deep learning models.  

## 📌 Project Overview  
This **energy demand forecasting model** is built using **Keras and TensorFlow**, incorporating **callbacks for optimization**.  
The current implementation is the final iteration after multiple refinements.  

## 🔄 Model Development Process  

1. **Data Preprocessing**  
   - Converts datetime features into **cyclical representations** using sin/cos transformations  
   - Scales input features using **MinMaxScaler**  
   - Splits data into **training and test sets**  

2. **Baseline Models**  
   - Initial trials with **Linear Regression** and **MLP** models for benchmarking  

3. **Neural Network Training**  
   - Fully connected **feedforward neural network (FNN)** using ReLU activations  
   - Implemented **callbacks**: EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau  
   - Optimized using **Mean Squared Error (MSE) loss** and **Adam optimizer**  

4. **Performance Evaluation**  
   - Evaluates predictions using **Mean Absolute Error (MAE)**  
   - Visualizes **actual vs predicted values**  

## 🔍 Future Enhancements  
- Compare with **LSTM, RNN**, and other deep learning models  
- Integrate **weather data (temperature, solar radiation, humidity)**  
- Optimize **feature selection & hyperparameters**  

## 🚀 How to Use  

1. **Open the Google Colab Notebook**  
   - Click the link below to access the code:  
     👉 [Run on Google Colab](https://colab.research.google.com/drive/17pmYebfvCltdHjCP1W3Ka6iCXb191DwG?usp=sharing)  

2. **Replace the Dataset**  
   - The model expects a CSV file with two columns:  
     - `timeperiod`: Datetime values  
     - `demand`: Energy demand values  
   - Replace **`Historic data.csv`** with **your own dataset**.  

## 📦 Dependencies  
- `numpy`  
- `pandas`  
- `matplotlib`  
- `sklearn`  
- `tensorflow`  

## 🛠 Technologies Used  
- **Python** (Pandas, NumPy, Matplotlib for visualization)  
- **TensorFlow & Keras**  
- **Scikit-learn (sklearn) for preprocessing**  

---

## 🔗 Run the Notebook  
Click here to open the notebook in Google Colab:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17pmYebfvCltdHjCP1W3Ka6iCXb191DwG?usp=sharing)  

**Note**: Install dependencies before running the code.  

## 📜 License  
This project is licensed under the **MIT License**.  

---
