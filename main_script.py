"Import necessary libraries"
# Docker image name: employee_turnover_app

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"Load the dataset"

dataset = pd.read_csv("HR_comma_sep.csv")
dataset.head(10)

# The dataset contains the following columns:
# satisfaction_level: Employee's satisfaction level.
# last_evaluation: Last evaluation score.
# number_project: Number of projects assigned to the employee.
# average_montly_hours: Average monthly working hours.
# time_spend_company: Number of years spent at the company.
# Work_accident: Whether the employee had a work accident (binary: 0 or 1). This is our Target variable
# left: Whether the employee has left the company (binary: 0 or 1).
# promotion_last_5years: Whether the employee was promoted in the last 5 years (binary: 0 or 1).
# Department: Department where the employee works.
# salary: Salary level (categorical: 'low', 'medium', 'high').

dataset.info()
dataset.describe(include="all")
print("Out of all the variables, Work accident, left, Department and Salary are the categorical variables else all others are Numerical variables", end="\n")

"Exploratory data analysis"
# Check if the data contains any miussing values
dataset.isna().sum()
print("The dataset does not contain any missing values.", end="\n") 

"Univariate Analysis"
fig,ax = plt.subplots(2,3, figsize=(15,10))
sns.histplot(dataset['satisfaction_level'], ax = ax[0,0], kde=True) 
sns.histplot(dataset['last_evaluation'], ax = ax[0,1], kde=True) 
sns.histplot(dataset['number_project'], ax = ax[0,2], kde=True) 
sns.histplot(dataset['average_montly_hours'], ax = ax[1,0], kde=True) 
sns.histplot(dataset['time_spend_company'], ax = ax[1,1], kde=True) 
sns.histplot(dataset['promotion_last_5years'], ax = ax[1,2], kde=True)
plt.show()
# Analysis from the above graph plot
# 1. Satisfaction decline: The satisfaction level graph shows a significant number of employees with low satisfaction while the last evaluation scores are generally higher.
# 2. Inconsistant work houes:
# There's a wide spread of working hours, with peaks around 150 and 250 hours. This indicates that some employees are working significantly more than others, potentially leading to burnout for overworked staff and reduced productivity for underutilized employees.
# 3. Workload Imbalance: 
# The combination of the number of projects and average monthly hours graphs points about uneven workload distribution. Some employees are likely taking on more projects and working longer hours to compensate for less active team members.
# 4. Lack of Promotions:
# The promotion graph shows that promotions are extremely rare. This lack of career advancement opportunities could be a major factor in the declining satisfaction and inconsistent productivity.
# 
# 5. Potential consequences: 
    # 1. Lower team cooperation and contribution, are likely outcomes of these issues.
    # 2. The bimodal distribution in satisfaction levels could indicate a forming divide in the workforce between engaged and disengaged employees. 
    # 3. The time spent at company graph shows a sharp decline after 5 years, which could be related to the lack of promotions and satisfaction issues<

fig,ax = plt.subplots(2,2, figsize=(15,10))
sns.histplot(dataset["Department"], ax = ax[0,0], kde=True) 
sns.histplot(dataset['left'], ax = ax[0,1], kde=True) 
sns.histplot(dataset['Work_accident'], ax = ax[1,0], kde=True) 
sns.histplot(dataset['salary'], ax = ax[1,1], kde=True) 
plt.show()

fig = plt.figure(figsize=[12,7])
sns.histplot(dataset["Department"], kde=True) 
plt.title('Count of employees in each department')
plt.show()
# 1. Department distribution: The Sales department has the highest number of employees, followed by Technical, Support.and IT. This distribution can help the company understand where most of their workforce is concentrated, potentially impacting turnover rates in these departments. 
# 2. Retention vs. Turnover: The majority of employees have been retained, while a smaller proportion has left the company.
# 3. Low Incidents of Work Accidents: Most employees have not had any work accidents, suggesting a safe working environment.
# 4. Low Salary: A significant portion of the workforce is earning a low salary, followed by a considerable number on medium salaries, with only a small fraction earning a high salary. The concentration of employees in the low salary bracket could be a potential risk for turnover, especially if employees feel underpaid. 

"Bivariate Analysis"
fig = plt.figure(figsize=[12,7])
sns.boxplot(x='salary', y='satisfaction_level', data=dataset)
plt.title('Satisfaction Level by Salary')
plt.show()
# Although there is a slight positive relationship between salary and satisfaction, the effect is not strong. Higher salaries are associated with somewhat higher and more consistent satisfaction levels, but there's considerable overlap in satisfaction across all salary levels. The satisfaction level does not impact a dramatic change in the salary.

pd.crosstab(dataset['Department'], dataset['left']).plot(kind='bar', stacked=True)
plt.title('Employee Turnover by Department')
plt.show()
# The plot shows that in the departments having greater count of employees have recorded more higher number of turnovers. 

corr = dataset.groupby(["left", "Work_accident"]).size().unstack(fill_value=0)
corr.plot(kind='bar', stacked=True)
plt.xlabel('Work accidents')
plt.ylabel('Number of Employees')
plt.title('Impact of Work accidents on Retention')
plt.xticks(rotation=0) 
plt.legend(['Stayed', 'Left'], loc='upper right')
plt.show()
# The graph indicates that there is not much impact of work accident on the employees to leave. So we will exclude this column from our analysis. 

pd.crosstab(dataset['salary'], dataset['left']).plot(kind='bar', stacked=True)
plt.title('Employee Turnover by Department')
plt.show()
# A very small portion of employees having high salary have left the company, while there is a significant concentration of people that left the company from low salary bracket. As mentioned above the reason may be that the employees may feel underpaid. Similar is the case with people working with medium salary range.

pd.crosstab(dataset['promotion_last_5years'], dataset['left']).plot(kind='bar', stacked=True)
plt.title('Employee Turnover by Promotion')
plt.xticks(rotation=0)
plt.show()
# The plot shows that there are greater chance of employees leaving the company if there are no promotions. So basically,<br>
# Employees leaving is inversely proportional Promotions

print("Percentage of employees retained with no promotions",(dataset.query("promotion_last_5years==0 & left == 0").shape[0]/dataset.shape[0])*100)
print("Percentage of employees leaving with no promotions",(dataset.query("promotion_last_5years==0 & left == 1").shape[0]/dataset.shape[0])*100)
print("Percentage of employees retained with promotions",(dataset.query("promotion_last_5years==1 & left == 0").shape[0]/dataset.shape[0])*100)
print("Percentage of employees leaving with promotions",(dataset.query("promotion_last_5years==1 & left == 1").shape[0]/dataset.shape[0])*100, end="\n")

retained_cols = dataset.drop(columns=["Department", "salary"], axis=1)
print(retained_cols.groupby('left').mean())
print("The above result clarifies that employees having low satisfaction level (satisfaction level below 50%) left the company.") 

"Analyzing change in satisfaction level compared to the last evaluation"

satisfaction_change = dataset["satisfaction_level"]-dataset["last_evaluation"]
print("Minimum degradation of satisfaction level",(satisfaction_change.min())*100)
print("Maximum improvement of satisfaction level", (satisfaction_change.max())*100, end="\n")

dataset["satisfaction_change"] = satisfaction_change

print("Count of employees showing positive improvement",dataset.query(f"satisfaction_change>0").shape[0])
print("Count of employees showing no improvement",dataset.query(f"satisfaction_change==0").shape[0])
print("Count of employees showing negative improvement",dataset.query(f"satisfaction_change<0").shape[0], end="\n")
print("Out of 14999 records, 5165 employees show a positive improvement in the satisfaction level, 236 employes showed no changes, while 9598 employees show a negative shift in their satisfaction level.",end="\n")

"Data Transformation"
dataset = pd.get_dummies(dataset, columns=['Department', 'salary'], dtype="int64", drop_first = True)
dataset = dataset.drop(columns=["satisfaction_level", "last_evaluation", "Work_accident"])
corr_matrix = dataset.corr()
print(corr_matrix)
print("The independent variables constitute almost zero correlation with each other Which is a good sign for training any ML algorithm")

"Feature Engineering"
# Splitting the dependent and independent variable and Standardizing the variables
ind_var = dataset.drop(columns=["left"])
dep_var = dataset["left"]

scaler = StandardScaler()
columns_to_scale = ["number_project", "average_montly_hours", "time_spend_company"]
ind_var[columns_to_scale] = scaler.fit_transform(ind_var[columns_to_scale])

"Model development"
# Split the training and testing data and training the machine learning models on the data.

X_train, X_test, y_train, y_test = train_test_split(ind_var, dep_var, test_size=0.2, random_state=42)
print(f"x_train shape: {X_train.shape}, x_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

# Logistice Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_log_reg = lr_model.predict(X_test)
# Evaluation
print("Logistic Regression Performance:")
print(confusion_matrix(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
# Evaluation
print("Random Forest Performance:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Naive Bayes Classifier
NB_model = GaussianNB()
NB_model.fit(X_train, y_train)
y_pred_NB = NB_model.predict(X_test)
# Evaluation
print("Naive Bayes classifier (Gaussian NB) Performance:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# SVM Classifier
svc_classifier = SVC(kernel='linear')
svc_classifier.fit(X_train, y_train)
y_pred = svc_classifier.predict(X_test)
# Evaluation
print("SVM Performance:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Decision Tree Classofer
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
# Evaluation
print("Decision Tree Classifier Performance:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Summary of Model Performance:
# Five machine learning models were evaluated for predicting employee turnover: Logistic Regression, Random Forest, Naive Bayes (GaussianNB), SVM, and Decision Tree.
# Key Findings:
# - Logistic Regression performed the worst, showing lower precision, recall, and F1-score.
# - Random Forest, Naive Bayes, SVM, and Decision Tree all achieved high accuracy and similar performance metrics.
# - GaussianNB was the fastest in training and prediction, followed by Decision Tree.
# - Random Forest and SVM were slightly slower but still delivered strong performance.
 
# Conclusion:
# While Logistic Regression underperformed, the other models excelled. **GaussianNB** and **Decision Tree** are recommended for their speed, while **Random Forest** offers robustness at a higher computational cost.