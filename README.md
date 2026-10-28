# ❤️ Cardio-Monitor

### Heart Disease Prediction & Cardiovascular Risk Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git"/>
</p>

<p align="center">
  A machine learning-based web application designed to analyze patient health parameters and predict the likelihood of heart disease.
</p>

---

## 📌 Project Overview

**Cardio-Monitor** is a machine learning-powered web application that uses patient health information to estimate the likelihood of heart disease.

The application combines a predictive machine learning model with a web-based interface, allowing users to enter cardiovascular health parameters and receive a prediction based on the trained model.

The project demonstrates the practical application of machine learning, data processing, model integration, and web development in a healthcare-oriented use case.

## ✨ Features

* **Heart Disease Prediction:** Uses a trained machine learning model to classify patient records.
* **Patient Data Input:** Accepts cardiovascular health parameters for prediction.
* **Interactive Web Interface:** Provides a browser-based interface for entering patient information.
* **Prediction Results:** Displays the model's output for submitted patient data.
* **Data Visualization:** Includes visualization functionality for exploring relevant health data.
* **Model Integration:** Uses serialized machine learning artifacts to perform predictions.
* **Database Integration:** Includes database-related functionality through the project's database module.
* **Modular Architecture:** Separates application logic, prediction functionality, visualization, and supporting components.

## 🛠️ Technology Stack

| Technology                      | Purpose                                      |
| ------------------------------- | -------------------------------------------- |
| Python                          | Core programming language                    |
| Flask                           | Web application framework                    |
| Scikit-learn                    | Machine learning functionality               |
| Pandas                          | Data manipulation and processing             |
| NumPy                           | Numerical computations                       |
| Matplotlib / Plotting Libraries | Data visualization                           |
| HTML5                           | Web page structure                           |
| CSS3                            | User interface styling                       |
| JavaScript                      | Client-side interactivity, where implemented |
| Pickle                          | Loading serialized model artifacts           |
| Git & GitHub                    | Version control and source code management   |

*The exact libraries used for training, visualization, and database operations depend on the implementation and the dependencies listed in `requirements.txt`.*

## 🧠 How It Works

The application follows a machine learning prediction workflow:

1. **Patient Input:** The user enters the required health parameters through the web interface.
2. **Data Processing:** The application prepares the input in the format expected by the trained model.
3. **Model Inference:** The trained machine learning model processes the patient data.
4. **Prediction Generation:** The model generates a classification result.
5. **Result Presentation:** The application displays the prediction through the web interface.

```text
        Patient Health Parameters
                   |
                   v
          Web Application (Flask)
                   |
                   v
          Input Preprocessing
                   |
                   v
       Trained Machine Learning Model
                   |
                   v
          Prediction Generation
                   |
                   v
         Result Display / Analysis
```

## 📂 Project Structure

The repository contains the following principal files and directories:

```text
Cardio-Monitor/
│
├── app.py                  # Main web application
├── prediction.py            # Prediction-related functionality
├── database.py              # Database operations
├── modelbuild.py            # Model training/building functionality
├── visualization.py         # Data visualization
│
├── heart.csv                # Heart disease dataset
├── heartmodel.pkl           # Serialized machine learning model
├── Heart_model1.pkl         # Additional serialized model artifact
├── scaler.pkl               # Serialized feature scaler
│
├── static/                  # Static assets, CSS, JS, and images
├── templates/               # HTML templates
│
├── requirements.txt         # Python dependencies
├── Procfile                 # Deployment process configuration, if used
├── .gitignore               # Git exclusion rules
└── README.md                # Project documentation
```

*The structure above describes the expected project components. Keep only the files and directories that actually exist in your repository.*

## ⚙️ Installation and Setup

Follow these steps to run Cardio-Monitor locally on Windows.

### Prerequisites

Install the following:

* Python 3.10 or another version compatible with the project's dependencies.
* Git.
* A terminal such as Command Prompt or PowerShell.

Check your Python installation:

```bash
python --version
```

Check Git:

```bash
git --version
```

### 1. Clone the Repository

```bash
git clone https://github.com/Poseidon271/Cardio-Monitor.git
```

Navigate into the project directory:

```bash
cd Cardio-Monitor
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it using Command Prompt:

```cmd
venv\Scripts\activate
```

If you're using PowerShell instead:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project's dependencies:

```bash
pip install -r requirements.txt
```

If installation fails, check that your Python version is compatible with the packages specified in `requirements.txt`.

### 4. Verify Required Files

Ensure the model files expected by the application are present.

For example:

```text
heartmodel.pkl
Heart_model1.pkl
scaler.pkl
```

The actual files required depend on which model and scaler `app.py` loads.

Do not rename or remove model files unless you also update their corresponding references in the source code.

### 5. Run the Application

Start the Flask application:

```bash
python app.py
```

If the application starts successfully, the terminal should display its local address.

Open the address shown in the terminal, commonly:

```text
http://127.0.0.1:5000
```

If your application uses a different port, follow the address printed by Flask.

**Note:** If `app.py` does not start the Flask development server directly, follow the startup configuration defined in that file.

## 📊 Dataset and Machine Learning

The project includes a heart disease dataset and serialized machine learning artifacts.

The model uses the input features expected by its training pipeline to generate predictions.

Depending on the implementation, the pipeline may involve:

* Data cleaning and preprocessing.
* Feature selection and transformation.
* Feature scaling.
* Machine learning model training.
* Model serialization.
* Loading the trained model for inference.

The dataset and model must use compatible feature names, ordering, categorical encodings, and preprocessing steps for predictions to be meaningful.

### Model Evaluation

Model performance should be assessed using an independent test dataset.

Recommended evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Actual performance figures should be added only after evaluating the saved model against an appropriate test set.

## 🔍 Application Usage

1. Start the application locally.
2. Open the application in your browser.
3. Enter the requested patient health parameters.
4. Submit the form.
5. Review the prediction generated by the model.
6. Explore any available visualization or database-related functionality.

The exact fields and available features depend on the current application implementation.

## 🚀 Deployment

The application includes a `requirements.txt` file and may include a `Procfile` for deployment.

For a compatible hosting platform:

1. Connect the repository to your hosting provider.
2. Configure the Python environment.
3. Install dependencies from `requirements.txt`.
4. Configure the appropriate application startup command.
5. Ensure the required model and data files are available.
6. Configure environment variables or database settings if needed.
7. Deploy and test the application.

A typical Gunicorn startup command for a Flask application is:

```bash
gunicorn app:app
```

This command assumes that `app.py` exposes a Flask application object named `app`. Adjust it if your implementation uses a different object name.

## 👨‍💻 Author

**Aditya Pratap Sisodia**

Computer Science Engineering

Project: Cardio-Monitor — Heart Disease Prediction System

GitHub: [@Poseidon271](https://github.com/Poseidon271)

## 🎯 Project Objectives

* Apply machine learning techniques to a healthcare-related classification problem.
* Integrate a trained model into a functional web application.
* Provide a straightforward interface for entering patient health information.
* Demonstrate the integration of Python, machine learning, data processing, and web technologies.
* Explore the use of data visualization and database functionality in a health-focused application.

## ⚠️ Medical Disclaimer

Cardio-Monitor is an educational and software development project. Its predictions are model-generated estimates and must not be treated as a medical diagnosis.

The application is not a substitute for professional medical assessment, clinical testing, or advice from a qualified healthcare professional.

##