# task-1
Features :
     * Import & Explore Data: Load dataset and analyze data types and missing values.
     * Missing Values Handling: Impute missing values using median and mode.
     * Categorical Encoding: Convert categorical columns into numerical format using one-hot encoding.
     * Feature Scaling: Normalize numerical features with StandardScaler.
     * Outlier Detection and Removal: Visualize outliers with boxplots and remove them using the IQR method.  
data sets:
     * The dataset is loaded from a CSV file ("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv").
     * Contains passenger details including age, fare, class, sex, and survival status.
requirement:
     * Python 3.x
     * pandas
     * numpy
     * scikit-learn
     * matplotlib
     * seaborn
usage:     
     * Load the dataset.
     * Run each step sequentially to clean and prepare data.
     * After preprocessing, the data will be ready for machine learning modeling.
code:
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('titanic.csv')
df.drop(columns=['Cabin'], inplace=True, errors='ignore')
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
scaler = StandardScaler()
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
sns.boxplot(data=df[numeric_cols])
plt.show()




    
     
     

