# Employee Retention Prediction Project

## Overview
The **Employee Retention Prediction** project is designed to build and evaluate machine learning models that predict whether an employee is likely to stay at or leave a company. This is achieved through data analysis and training of multiple classification algorithms to understand key factors that contribute to employee turnover. The project can also be deployed locally using Docker.

## Features
- Exploratory data analysis (EDA) to identify trends and potential drivers of employee turnover.
- Implementation of various machine learning models including Logistic Regression, Random Forest, Naive Bayes, SVM, and Decision Tree.
- Feature engineering and preprocessing for better model training.
- Comparative analysis of model performance based on metrics like accuracy, precision, recall, and F1-score.
- Docker containerization for easy deployment.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `pandas`, `numpy` for data manipulation
  - `matplotlib`, `seaborn` for data visualization
  - `scikit-learn` for machine learning model development
- **Deployment**: Docker

## Data Description
The dataset used in this project (`HR_comma_sep.csv`) contains the following features:

| Feature                 | Description                                              |
|-------------------------|----------------------------------------------------------|
| `satisfaction_level`    | Employee's satisfaction level                           |
| `last_evaluation`       | Last evaluation score                                   |
| `number_project`        | Number of projects assigned to the employee             |
| `average_montly_hours`  | Average monthly working hours                           |
| `time_spend_company`    | Number of years spent at the company                    |
| `Work_accident`         | Whether the employee had a work accident (binary)       |
| `left`                  | Whether the employee has left the company (target)      |
| `promotion_last_5years` | Whether the employee was promoted in the last 5 years   |
| `Department`            | Department where the employee works                     |
| `salary`                | Salary level (categorical: 'low', 'medium', 'high')     |

## Key Steps
1. **Exploratory Data Analysis (EDA)**:
   - Analyzed the distribution of features and identified patterns.
   - Examined the relationship between key features and employee turnover.

2. **Data Preprocessing**:
   - Handled categorical data using `pd.get_dummies`.
   - Standardized numerical features using `StandardScaler`.
   - Removed less relevant features like `Work_accident`.

3. **Model Development**:
   - Trained five machine learning models: Logistic Regression, Random Forest, Naive Bayes, SVM, and Decision Tree.
   - Evaluated model performance using confusion matrices and classification reports.

4. **Model Evaluation**:
   - The models were evaluated based on their precision, recall, F1-score, and training speed.
   - **Random Forest** provided robust results, while **GaussianNB** and **Decision Tree** excelled in training speed.

## Docker Deployment
The project is ready for deployment using Docker. Follow the steps below to deploy the application:

1. **Ensure Docker is installed** on your machine.
2. **Navigate to the project directory**:
   ```bash
   cd /path/to/your/project
   ```

3. **Build the Docker image**:
   ```bash
   docker build -t employee_turnover_app .
   ```

4. **Run the Docker container**:
   ```bash
   docker run -d -p 5000:5000 employee_turnover_app
   ```

5. **Access the application**:
   - Open your web browser and navigate to `http://localhost:5000`.

## Results
### Model Performance Summary:
- **Logistic Regression**: Underperformed with low precision and recall.
- **Random Forest**: Strong accuracy and balanced metrics, suitable for reliable predictions.
- **GaussianNB**: Fastest training and prediction, recommended for quick insights.
- **Decision Tree**: Good balance of performance and training speed.
- **SVM**: High accuracy but slightly slower in training.

## Conclusion
The project successfully demonstrated the use of machine learning models to predict employee retention based on various features. The insights gained can help businesses take proactive measures to improve employee satisfaction and reduce turnover rates.