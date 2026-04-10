import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#Load Dataset

df = pd.read_csv("titanic.csv")

df.head()

#Shape of dataset

df.shape

#Column information

df.info()

#Statistical Summary

df.describe()

#Mean (Average Age)

np.mean(df['Age'])

#Median

np.median(df['Age'])

#Standard Deviation

np.std(df['Age'])

#Age Distribution (Histogram)

plt.hist(df['Age'].dropna())
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

#Survival Count (Bar Chart)

sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

#Gender Distribution

sns.countplot(x='Sex', data=df)
plt.title("Gender Count")
plt.show()

#Survival vs Gender

sns.countplot(x='Survived', hue='Sex', data=df)
plt.title("Survival by Gender")
plt.show()

#Survival vs Passenger Class

sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title("Survival by Class")
plt.show()

#Age vs Survival (Boxplot)

sns.boxplot(x='Survived', y='Age', data=df)
plt.show()

#Correlation (Important for ML)

corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#Pairplot (Very Powerful)

sns.pairplot(df, hue='Survived')
plt.show()

#Survival Pie Chart

px.pie(df, names='Survived', title='Survival Distribution').show()

#Age Histogram (Interactive)

px.histogram(df, x='Age', color='Survived').show()

#Scatter Plot

px.scatter(df, x='Age', y='Fare', color='Survived').show()
