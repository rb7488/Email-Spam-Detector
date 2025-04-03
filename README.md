
# Email Spam Detector

A simple Python-based Email Spam Detector using **Natural Language Processing (NLP)** and **Naïve Bayes classification**.

## Features
- **Spam Detection**: Identifies whether an email is spam or not.
- **Text Preprocessing**: Cleans email text by removing special characters.
- **Machine Learning Model**: Uses TF-IDF and Naïve Bayes for classification.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/email-spam-detector.git
   cd email-spam-detector
   ```

2. Install dependencies:
   ```bash
   pip install pandas scikit-learn
   ```

3. Ensure the dataset `spam.csv` is in the project directory.

## Usage

Run the script to detect spam emails:

```bash
python spam_detector.py
```

### Example

```
Enter your email message: "Congratulations! You've won a free iPhone. Click here!"
Classification: Spam 🚨
```

## Dataset
- **Source**: [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Format**: CSV (`spam.csv` with labeled emails)

## How It Works
1. Loads and preprocesses the dataset.
2. Converts text into numeric form using TF-IDF.
3. Trains a **Naïve Bayes Classifier**.
4. Evaluates accuracy and allows user input for classification.

