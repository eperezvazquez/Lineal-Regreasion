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

# función para convertir región a numérico
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

# re categorizo "smoker" en 0 y 1
df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)

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

 veo los coeficientes
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# ejemplo
edad = 33
sex = 1
bm = 22
children = 0
smoker = 1
region = 3

# predigo target (charges) según datos de ejemplo
print('Predicted prima : \n', regr.predict([[edad,sex,bm,children,smoker,region]]))

## training a ml model
x = np.array(df[["age", "sex", "bmi", "smoker"]])
y = np.array(df["charges"])
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(xtrain, ytrain)

ypred = forest.predict(xtest)
df= pd.DataFrame(data={"Predicted Premium Amount": ypred})
print(df.head())

