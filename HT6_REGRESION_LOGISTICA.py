# %% [markdown]
# # HOJA DE TRABAJO 6 REGRESION LOGISTICA

# Raul Jimenez 19017

# Donaldo Garcia 19683

# Oscar Saravia 19322

# link al repo: https://github.com/raulangelj/HT6_REGRESION_LOGISTICA

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression
import sklearn.preprocessing
import scipy.cluster.hierarchy as sch
import seaborn as sns
import skfuzzy as fuzz
from sklearn.metrics import confusion_matrix
import pylab
import sklearn.mixture as mixture
import random
import math
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.datasets import load_nfl
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn import datasets
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from sklearn.metrics import confusion_matrix as Confusion_Matrix
from sklearn.model_selection import train_test_split

# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# %% [markdown]
# ## Use los mismos conjuntos de entrenamiento y prueba que utilizó en las dos hojas anteriores.

# %%
train = pd.read_csv('./train.csv', encoding='latin1')
train.head()

# %%
usefullAttr = ['SalePrice', 'LotArea', 'OverallCond', 'YearBuilt', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF',
               '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'PoolArea', 'Neighborhood', 'OverallQual']


# %%
data = train[usefullAttr]
data.head()


# %%
sns.pairplot(data[['SalePrice', 'LotArea', 'TotalBsmtSF',
             'GrLivArea', 'TotRmsAbvGrd', 'OverallQual']], hue='SalePrice')
plt.show()

# %%
plt.subplots(figsize=(8, 8))
sns.heatmap(data[['SalePrice', 'LotArea', 'TotalBsmtSF',
                  'GrLivArea', 'TotRmsAbvGrd', 'OverallQual']].corr(), annot=True, fmt="f").set_title("Correlación de las variables numéricas de Iris")

# %% [markdown]
# ### GRAFICAS DE VARIABLES

# %%


def get_histogram_qq(variable):
    plt.hist(x=data[variable] .dropna(), color='#F2AB6D', rwidth=1)
    plt.title(f'Histograma de la variable{variable}')
    plt.xlabel(variable)
    plt.ylabel('frencuencias')
    plt.rcParams['figure.figsize'] = (30, 30)
    plt.show()

    distribucion_generada = data[variable].dropna()
    # Represento el Q-Q plot
    qqplot(distribucion_generada, line='s')
    plt.show()

# %% [markdown]
# #### SalePricee
# Se puede determinar que la variable SalePrice no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.


# %%
get_histogram_qq('SalePrice')

# %% [markdown]
# #### LotArea
# Se puede determinar que la variable LotArea no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('LotArea')

# %% [markdown]
# #### OverallCond
# Se puede determinar que la variable OverallCond no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('OverallCond')

# %% [markdown]
# #### YearBuilt
# Se puede determinar que la variable YearBuilt no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('YearBuilt')

# %% [markdown]
# #### MasVnrArea
# Se puede determinar que la variable MasVnrArea no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('MasVnrArea')

# %% [markdown]
# #### TotalBsmtSF
# Se puede determinar que la variable TotalBsmtSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('TotalBsmtSF')

# %% [markdown]
# #### 1stFlrSF
# Se puede determinar que la variable 1stFlrSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('1stFlrSF')

# %% [markdown]
# #### 2ndFlrSF
# Se puede determinar que la variable 2ndFlrSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('2ndFlrSF')

# %% [markdown]
# #### GrLivArea
# Se puede determinar que la variable GrLivArea no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('GrLivArea')

# %% [markdown]
# #### TotRmsAbvGrd
# Se puede determinar que la variable TotRmsAbvGrd no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('TotRmsAbvGrd')

# %% [markdown]
# #### GarageCars
# Se puede determinar que la variable GarageCars no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('GarageCars')

# %% [markdown]
# #### WoodDeckSF
# Se puede determinar que la variable WoodDeckSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('WoodDeckSF')

# %% [markdown]
# #### OpenPorchSF
# Se puede determinar que la variable OpenPorchSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('OpenPorchSF')

# %% [markdown]
# #### EnclosedPorch
# Se puede determinar que la variable EnclosedPorch no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('EnclosedPorch')

# %% [markdown]
# #### PoolArea
# Se puede determinar que la variable PoolArea no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('PoolArea')

# %% [markdown]
# #### Neighborhood
# Se puede determinar que la variable Neighborhood no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
eje_x = np.array(pd.value_counts(data['Neighborhood']).keys())
eje_y = pd.value_counts(data['Neighborhood'])

plt.bar(eje_x, eje_y)
plt.rcParams['figure.figsize'] = (10, 10)
plt.ylabel('Frecuencia de la variable Neighborhood')
plt.xlabel('Años')
plt.title('Grafico de barras para la variable Neighborhood')
plt.show()

# %% [markdown]
# ## 3. Incluya un análisis de grupos en el análisis exploratorio. Explique las características de los grupos.
# Se puede concluir que los datos normalizados son viables para el uso de clusters o grupos. Se logra llegar a esta conclucion debido a que nuestro test de hopkins sale de 0.08 junto con la grafica VAT. Con la grafica del codo se puede determinar que se pueden utilizar dos clusters debido a que es en ese dato donde se encuentra mas marcado el codo. Pero tambien se podria usar 7 debido a que tambien se encuenatra marcada ahi una punta.

# %%
data.hist()
plt.show()

# %%
# NORMALIZAMOS DATOS
if 'Neighborhood' in data.columns:
    usefullAttr.remove('Neighborhood')
data = train[usefullAttr]
X = []
for column in data.columns:
    try:
        column
        data[column] = (data[column]-data[column].mean()) / \
            data[column].std()
        X.append(data[column])
    except Exception:
        continue
data_clean = data.dropna(subset=usefullAttr, inplace=True)
X_Scale = np.array(data)
X_Scale

# %%
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(X_Scale)
kmeans_result = kmeans.predict(X_Scale)
kmeans_clusters = np.unique(kmeans_result)
for kmeans_cluster in kmeans_clusters:
    # get data points that fall in this cluster
    index = np.where(kmeans_result == kmeans_cluster)
    # make the plot
    plt.scatter(X_Scale[index, 0], X_Scale[index, 1])
plt.show()

# %%
data['cluster'] = kmeans.labels_
print(data[data['cluster'] == 0].describe().transpose())
print(data[data['cluster'] == 1].describe().transpose())
print(data[data['cluster'] == 2].describe().transpose())
# ## Variable clasificacion
# %%
# Clasificacion de casas en: Economias, Intermedias o Caras.
data.fillna(0)

minPrice = data['SalePrice'].min()
maxPrice = data['SalePrice'].max()
division = (maxPrice - minPrice) / 3
data['Clasificacion'] = data['LotArea']

data['Clasificacion'][data['SalePrice'] < minPrice + division] = 'Economica'
data['Clasificacion'][data['SalePrice'] >= minPrice + division] = 'Intermedia'
data['Clasificacion'][data['SalePrice'] >= minPrice + division * 2] = 'Caras'

# %% [markdown]
# #### Contamos la cantidad de casas por clasificacion

# %%
# Obtener cuantos datos hay por cada clasificacion
print(data['Clasificacion'].value_counts())

# %% [markdown]
# ## 1. Cree  una  variable  dicotómica  por  cada  una  de  las  categorías de  la  variable  respuesta categórica que creó en hojas anteriores. Debería tener 3 variables dicotómicas (valores 0 y 1) una que diga si la vivienda es cara o no, media o no, económica o no.

# %%
data["CARA"] = np.where(data["cluster"] == 2, 1, 0)
data["INTERMEDIA"] = np.where(data["cluster"] == 1, 1, 0)
data["ECONOMICA"] = np.where(data["cluster"] == 0, 1, 0)
data

# %% [markdown]
# ## 2. Use los mismos conjuntos de entrenamiento y prueba que utilizó en las hojas anteriores.

# %% [markdown]
# ## Dividmos en entrenamiento y prueba

# %% [markdown]
# # Estableciendo los conjuntos de Entrenamiento y Prueba

# %%
y = data['Clasificacion']
X = data[['SalePrice', 'LotArea', 'TotalBsmtSF',
          'GrLivArea', 'TotRmsAbvGrd', 'OverallQual']]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, train_size=0.7)
y_train

# %% [markdown]
# 70% de entrenamiento y 30% prueba

# %%
X_train.info()

# %%
X_test.info()

# %% [markdown]
# ## 3. Elabore  un  modelo  de  regresión  logística  para  conocer  si  una  vivienda  es  cara  o  no, utilizando el conjunto de entrenamiento y explique los resultados a los que llega. Muestre el modelo gráficamente. El experimento debe ser reproducible por lo que debe fijar que los conjuntos de entrenamiento y prueba sean los mismos siempre que se ejecute el código.

# ### R/Analizando los datos que se obtuvieron se puede determinar que la eficiencia del algoritmo de regresion logistica es bastante bueno, se puede llegar a esta conclucion debido a que la matriz de confucion es de 97.4%. Lo cual se puede observar en la matriz que se imprime a acontinuacion.

# %%
logReg = LogisticRegression(solver='liblinear')
logReg.fit(X_train, y_train)
y_pred = logReg.predict(X_test)
y_proba = logReg.predict_proba(X)[:, 1]
cm = confusion_matrix(y_test, y_pred)

# %% [markdown]
# ## 4. Analice el modelo. Determine si hay multicolinealidad en las variables, y cuáles son las que aportan  al  modelo,  por  su  valor  de  significación.  Haga  un  análisis  de  correlación  de  las variables del modelo y especifique si el modelo se adapta bien a los datos. Explique si hay sobreajuste (overfitting) o no.

# ### R/ Analizando las tablas de VIF y Tolerancia, se logra determinar que las variables de caras, intermedias y economicas si estan relacionadas con las variables de precio, lot area, overallqual y total rooms above ground, de igual forma con nuestro heatmap de relacion logramos ver que estas variables si tienen relacion. Ahora analizando nuestros datos de accuracy y precision, 0.98 y 0.97 respectivamente, logramos determinar que si existe un overfitting, el modelo no es capaz de ajustar bien los datos.

# %%
# hm = sns.heatmap(data.corr(), annot=True, mask=np.triu(
#     np.ones_like(data.corr(), dtype=bool)), vmin=-1, vmax=1)
hm = sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1)
plt.show()

# %%
# Extraido de: https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f


def calculate_vif(df, features):
    vif, tolerance = {}, {}
    # all the features that you want to examine
    for feature in features:
        # extract all the other features you will regress against
        X = [f for f in features if f != feature]
        X, y = df[X], df[feature]
        # extract r-squared from the fit
        r2 = LinearRegression().fit(X, y).score(X, y)

        # calculate tolerance
        tolerance[feature] = 1 - r2
        # calculate VIF
        vif[feature] = 1/(tolerance[feature])
    # return VIF DataFrame
    return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})


# %%
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')
print('Matriz de confusión para regresion lineal\n', cm)
print('Accuracy: ', accuracy)
print("Precision:", metrics.precision_score(
    y_test, y_pred, average='weighted'))

# %%
calculate_vif(df=data, features=['SalePrice',
              'GrLivArea', 'LotArea', 'OverallQual', 'TotRmsAbvGrd', 'CARA'])
# %%
calculate_vif(df=data, features=['SalePrice',
              'GrLivArea', 'LotArea', 'OverallQual', 'TotRmsAbvGrd', 'INTERMEDIA'])
# %%
calculate_vif(df=data, features=['SalePrice',
              'GrLivArea', 'LotArea', 'OverallQual', 'TotRmsAbvGrd', 'ECONOMICA'])

# %%
