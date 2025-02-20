# Steam Game Review Classification

## Overview
This project aims to classify Steam game reviews into different categories based on their price and the number of reviews. The dataset used is a cleaned version of Steam game data, which includes features such as game price, number of reviews, and review type.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run this project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- seaborn
- scikit-learn
- tensorflow

You can install the required libraries using pip:

```bash
pip install pandas numpy seaborn scikit-learn tensorflow
Usage
Clone the repository:

bash
Run
Copy code
git clone https://github.com/yourusername/steam-review-classification.git
cd steam-review-classification
Place the steam_cleaned.csv dataset in the project directory.

Run the script:

bash
Run
Copy code
python your_script_name.py
Data Preprocessing
The data preprocessing steps include:

Loading the dataset and checking for null values.
Removing rows with null values.
Converting the Release_date to a consistent datetime format.
Filtering out free games (games with a price of 0).
Removing outliers based on Z-scores for price and number of reviews.
Model Training
A neural network model is built using TensorFlow's Keras API. The model architecture consists of:

Input layer with 4 neurons
Hidden layers with 32, 64, and 128 neurons
Output layer with 7 neurons (for 7 review types)
The model is compiled with the sparse_categorical_crossentropy loss function and the Adam optimizer. It is trained for 100 epochs.

Results
After training, the model predicts the review types for the test dataset. The predictions are compared with the actual values to evaluate the model's performance.

Example output:

Run
Copy code
Y Prediction:  ['Positive', 'Negative', ...]
Y Real:        ['Positive', 'Negative', ...]
Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Run
