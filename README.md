# credit-card-approval-prediction

This project implements a Credit Card Approval Prediction System using a Random Forest Classifier. The system preprocesses input data, trains a model, and evaluates its performance on predicting whether a credit card application should be approved or denied.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Evaluation](#model-evaluation)
- [Example Prediction](#example-prediction)
- [Contributing](#contributing)
- [License](#license)

## Features

- Data preprocessing: Handles missing values, encodes categorical variables, and scales numerical features.
- Model training: Utilizes a Random Forest Classifier with class weight balancing to handle imbalanced datasets.
- Model evaluation: Provides accuracy score and classification report.
- Feature importance visualization: Displays the importance of each feature in the prediction.

## Installation

To run this project, you need to have Python installed along with the required libraries. You can install the necessary libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/credit-card-approval-prediction.git
cd credit-card-approval-prediction
```

2. Run the script:

```bash
python credit_card_approval_system.py
```

3. The script will preprocess a sample dataset, train the model, evaluate its performance, and make a prediction for a new application.

## Data

The example dataset used in this project is generated randomly for demonstration purposes. You can replace the sample dataset with your actual credit card approval dataset by loading it in the `main()` function.

```python
data = pd.read_csv('your_credit_card_approval_dataset.csv')
```

## Model Evaluation

After training the model, the system evaluates its performance on a test set and prints the accuracy and classification report. It also visualizes the feature importance.

## Example Prediction

The system allows you to make predictions for new credit card applications. You can modify the `new_application` DataFrame in the `main()` function to test different scenarios.

```python
new_application = pd.DataFrame({
    'income': [75000],
    'age': [35],
    'employment_status': ['Employed'],
    'credit_score': [720]
})
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
