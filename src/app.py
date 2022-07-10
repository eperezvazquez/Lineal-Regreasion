import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statsmodels.api as sm

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv')
df.info()
df.isnull().sum()
df.head()
df.describe()
len(df)
df['sex'].value_counts()
df['smoker'].value_counts()
figure = px.histogram(df, x = "sex", color = "smoker", title= "Number of Smokers")
figure.show()
df["sex"] = df["sex"].map({"female": 0, "male": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
print(df.head())
df['region'].value_counts()
pie = df["region"].value_counts()
regions = pie.index
population = pie.values
fig = px.pie(df, values=population, names=regions)
fig.show()
print(df.corr())
df['charges'].hist()
fig = px.box(df, y="charges")
fig.show()
fig = px.box(df, x="sex", y="charges")
fig.show()
fig = px.box(df, x="region", y="charges")
fig.show()
fig = px.box(df, x="smoker", y="charges")
fig.show()
fig = px.box(df, x="children", y="charges")
fig.show()
ax1 = df[df['smoker'] == 'no'].plot(kind='scatter', x='age', y='charges', color='green', alpha=0.5, figsize=(8,6))
df[df['smoker'] == 'yes'].plot(kind='scatter', x='age', y='charges', color='red', alpha=0.5, figsize=(8,6), ax=ax1)
plt.legend(labels=['no', 'yes'])
plt.title('Relationship between Age and Charges', size=18)
plt.xlabel('Age', size=12)
plt.ylabel('Charges', size=12);
ax1 = df[df['smoker'] == 'no'].plot(kind='scatter', x='bmi', y='charges', color='green', alpha=0.5, figsize=(8,6))
df[df['smoker'] == 'yes'].plot(kind='scatter', x='bmi', y='charges', color='red', alpha=0.5, figsize=(8,6), ax=ax1)
plt.legend(labels=['no', 'yes'])
plt.title('Relationship between BMI and Charges', size=18)
plt.xlabel('BMI', size=12)
plt.ylabel('Charges', size=12);
ax1 = df[df['smoker'] == 'no'].plot(kind='scatter', x='children', y='charges', color='green', alpha=0.5, figsize=(8,6))
df[df['smoker'] == 'yes'].plot(kind='scatter', x='children', y='charges', color='red', alpha=0.5, figsize=(8,6), ax=ax1)
plt.legend(labels=['no', 'yes'])
plt.title('Relationship between Children and Charges', size=18)
plt.xlabel('Children', size=12)
plt.ylabel('Charges', size=12)
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0


def conv_region(region_name):
    if region_name == 'southwest':
        return 1
    elif region_name == 'southeast':
        return 2
    elif region_name == 'northwest':
        return 3
    elif region_name == 'northeast':
        return 4
    else:
        return 'región sin determinar'
df['region'] = df.apply(lambda x: conv_region(x['region']), axis=1)
df.head()
df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
df.head()
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True,cmap='viridis', vmax=1, vmin=-1, center=0)
### modelo regresión lineal
X = df.drop(['charges'], axis=1)
y = df['charges']
# separo en muestras de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# estimo modelo
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
# veo los coeficientes
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
X_train
print('Predicted prima : \n', regr.predict([[edad,sex,bm,children,smoker,region]]))
### modelo 2 regresión lineal
x = np.array(df[["age", "sex", "bmi", "smoker"]])
y = np.array(df["charges"])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

forest = RandomForestRegressor()
forest.fit(xtrain, ytrain)

ypred = forest.predict(xtest)
df= pd.DataFrame(data={"Predicted Premium Amount": ypred})
print(df.head())

