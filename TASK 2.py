# Import libraries
import pandas as pd
import seaborn as sns

# Load the Titanic dataset
TITANIC = pd.read_csv('train.csv')

# Checking for missing values
print(TITANIC.isnull().sum())

# Dropping columns that are unlikely to be useful in analysis
TITANIC.drop(columns=["Cabin", "Name", "Ticket"], axis=1, inplace=True)

# Filling missing values
TITANIC["Embarked"] = TITANIC["Embarked"].fillna(TITANIC["Embarked"].mode()[0])  # Mode for 'Embarked'
TITANIC['Age'].fillna(TITANIC["Age"].median(), inplace=True)  # Median for 'Age'

# Confirming that there are no null values
print(TITANIC.isnull().sum())


# Detecting outliers using IQR (Interquartile Range) for Age
Q1_age = TITANIC['Age'].quantile(0.25)
Q3_age = TITANIC['Age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
lower_bound_age = Q1_age - 1.5 * IQR_age
upper_bound_age = Q3_age + 1.5 * IQR_age

# Identifying outliers in Age
outliers_age = TITANIC[(TITANIC['Age'] < lower_bound_age) | (TITANIC['Age'] > upper_bound_age)]
print(f'Outliers in Age:\n{outliers_age}')

# Detecting outliers using IQR for Fare
Q1_fare = TITANIC['Fare'].quantile(0.25)
Q3_fare = TITANIC['Fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare
lower_bound_fare = Q1_fare - 1.5 * IQR_fare
upper_bound_fare = Q3_fare + 1.5 * IQR_fare

# Identifying outliers in Fare
outliers_fare = TITANIC[(TITANIC['Fare'] < lower_bound_fare) | (TITANIC['Fare'] > upper_bound_fare)]
print(f'Outliers in Fare:\n{outliers_fare}')

# Summary statistics after handling missing values
print(TITANIC.describe())
