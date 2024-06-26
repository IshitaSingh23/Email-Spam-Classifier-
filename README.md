# Email Spam Classifier

## Overview

This project aims to automate the process of identifying and filtering unwanted emails by classifying them as spam or not spam using Machine Learning and Natural Language Processing (NLP) techniques.

## Features

- **Exploratory Data Analysis (EDA)**: In-depth analysis of the dataset to understand the distribution and characteristics of the data.
- **Text Preprocessing**: 
  - Stop words removal
  - Stemming
  - Lemmatization
- **Feature Extraction**: 
  - Term Frequency-Inverse Document Frequency (TF-IDF)
  - Count Vectorizer
- **Machine Learning Models**:
  - Naive Bayes
  - Decision Tree
  - Random Forest
  - AdaBoost
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Gradient Boosting Classifier
  - Logistic Regression

The Random Forest classifier achieved the highest accuracy of 97%.

## Deployment

The project is deployed using:
- **Streamlit**: A fast way to build and share data apps.
- **Heroku**: A platform as a service (PaaS) that enables developers to build, run, and operate applications entirely in the cloud.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https:https://github.com/IshitaSingh23/Email-Spam-Classifier-
    cd email-spam-classifier
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

4. **View the application**:
    Open your web browser and navigate to `http://localhost:8501`.

## Usage

1. **Upload a CSV file** containing email data.
2. **Preprocess the data** using the provided text preprocessing steps.
3. **Extract features** using TF-IDF or Count Vectorizer.
4. **Train the model** using any of the provided classifiers.
5. **Classify new emails** and identify whether they are spam or not.

## File Structure

- `app.py`: The main file to run the Streamlit application.
- `data/`: Directory containing dataset files.
- `models/`: Directory containing saved models.
- `notebooks/`: Jupyter notebooks for EDA and model training.
- `requirements.txt`: List of required Python packages.
- `README.md`: This readme file.

## Dependencies

- pandas
- numpy
- scikit-learn
- nltk
- streamlit
- joblib
- heroku

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Thanks to the developers of the libraries used in this project.
- Special thanks to the contributors and maintainers of Streamlit and Heroku.
