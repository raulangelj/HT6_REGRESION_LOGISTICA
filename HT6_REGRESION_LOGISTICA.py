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
# import skfuzzy as fuzz
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
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, RepeatedKFold, cross_validate, cross_val_score

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
dataOG = data.copy()
data

# %% [markdown]
# ## 2. Use los mismos conjuntos de entrenamiento y prueba que utilizó en las hojas anteriores.

# %% [markdown]
# ## Dividmos en entrenamiento y prueba

# %% [markdown]
# # Estableciendo los conjuntos de Entrenamiento y Prueba

# %%
cara = data.pop('CARA')
intermedia = data.pop('INTERMEDIA')
economica = data.pop('ECONOMICA')
print('para variable CARA')
y = cara
X = data[['LotArea', 'TotalBsmtSF',
          'GrLivArea', 'TotRmsAbvGrd', 'OverallQual']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, train_size=0.7)
y_train

# 70% de entrenamiento y 30% prueba

print(X_train.info())
print(X_test.info())

print('para variable intermedia')
yi = intermedia
Xi = data[['LotArea', 'TotalBsmtSF',
          'GrLivArea', 'TotRmsAbvGrd', 'OverallQual']]

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    Xi, yi, test_size=0.3, train_size=0.7)
y_train_i

# 70% de entrenamiento y 30% prueba

print(X_train_i.info())
print(X_test_i.info())

print('para variable economica')
ye = intermedia
Xe = data[['LotArea', 'TotalBsmtSF',
          'GrLivArea', 'TotRmsAbvGrd', 'OverallQual']]

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    Xe, ye, test_size=0.3, train_size=0.7)
y_train_e

# 70% de entrenamiento y 30% prueba

print(X_train_e.info())
print(X_test_e.info())

# %% [markdown]
# ## 3. Elabore  un  modelo  de  regresión  logística  para  conocer  si  una  vivienda  es  cara  o  no, utilizando el conjunto de entrenamiento y explique los resultados a los que llega. Muestre el modelo gráficamente. El experimento debe ser reproducible por lo que debe fijar que los conjuntos de entrenamiento y prueba sean los mismos siempre que se ejecute el código.

# ### R/Analizando los datos que se obtuvieron se puede determinar que la eficiencia del algoritmo de regresion logistica es bastante bueno, se puede llegar a esta conclucion debido a que la matriz de confucion es de 97.4%. Lo cual se puede observar en la matriz que se imprime a acontinuacion.

# %%
# Variable CARA
logReg = LogisticRegression(solver='liblinear')
logReg.fit(X_train, y_train)
y_pred = logReg.predict(X_test)
y_proba = logReg.predict_proba(X)[:, 1]
cm = confusion_matrix(y_test, y_pred)

# variable INTERMEDIA
logReg_i = LogisticRegression(solver='liblinear')
logReg_i.fit(X_train_i, y_train_i)
y_pred_i = logReg_i.predict(X_test_i)
y_proba_i = logReg_i.predict_proba(Xi)[:, 1]
cm_i = confusion_matrix(y_test_i, y_pred_i)

# variable ECONOMICA
logReg_e = LogisticRegression(solver='liblinear')
logReg_e.fit(X_train_e, y_train_e)
y_pred_e = logReg_e.predict(X_test_e)
y_proba_e = logReg_e.predict_proba(Xe)[:, 1]
cm_e = confusion_matrix(y_test_e, y_pred_e)

# %% [markdown]
# ## 4. Analice el modelo. Determine si hay multicolinealidad en las variables, y cuáles son las que aportan  al  modelo,  por  su  valor  de  significación.  Haga  un  análisis  de  correlación  de  las variables del modelo y especifique si el modelo se adapta bien a los datos. Explique si hay sobreajuste (overfitting) o no.

# ### R/ Analizando las tablas de VIF y Tolerancia, se logra determinar que las variables de caras, intermedias y economicas si estan relacionadas con las variables de precio, lot area, overallqual y total rooms above ground, de igual forma con nuestro heatmap de relacion logramos ver que estas variables si tienen relacion. Ahora analizando nuestros datos de accuracy y precision, se obtuvieron los siguientes valores 0.96 de accuracy y de precision para las casas caras, 0.88 de accuracy y 0.88 de precision para las casas intermedias y 0.89 de accuracy y 0.89 de precision para las economicas. Por lo tanto si hay overfitting, ya que los valores obtenidos en accuracy y precision son cercanos a 1
# %%
# hm = sns.heatmap(data.corr(), annot=True, mask=np.triu(
#     np.ones_like(data.corr(), dtype=bool)), vmin=-1, vmax=1)
heatmapmatriz = data.copy()
hm = sns.heatmap(heatmapmatriz[['LotArea', 'TotalBsmtSF',
                                'GrLivArea', 'TotRmsAbvGrd', 'OverallQual']].corr(), annot=True, vmin=-1, vmax=1)
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
print('\nDATOS DE CARA')
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')
print('Matriz de confusión para regresion lineal\n', cm)
print('Accuracy: ', accuracy)
print("Precision:", metrics.precision_score(
    y_test, y_pred, average='weighted'))
print("Recall:", recall)
print("F1:", f1)

print('\nDATOS DE INTERMEDIA')
accuracy_i = accuracy_score(y_test_i, y_pred_i)
precision_i = precision_score(y_test_i, y_pred_i, average='micro')
recall_i = recall_score(y_test_i, y_pred_i, average='micro')
f1_i = f1_score(y_test_i, y_pred_i, average='micro')
print('Matriz de confusión para regresion lineal\n', cm_i)
print('Accuracy: ', accuracy_i)
print("Precision:", metrics.precision_score(
    y_test_i, y_pred_i, average='weighted'))
print("Recall:", recall_i)
print("F1:", f1_i)

print('\nDATOS DE ECONOMICA')
accuracy_e = accuracy_score(y_test_e, y_pred_e)
precision_e = precision_score(y_test_e, y_pred_e, average='micro')
recall_e = recall_score(y_test_e, y_pred_e, average='micro')
f1_e = f1_score(y_test_e, y_pred_e, average='micro')
print('Matriz de confusión para regresion lineal\n', cm_e)
print('Accuracy: ', accuracy_e)
print("Precision:", metrics.precision_score(
    y_test_e, y_pred_e, average='weighted'))
print("Recall:", recall_e)
print("F1:", f1_e)


# %%
calculate_vif(df=dataOG, features=['SalePrice',
              'GrLivArea', 'LotArea', 'OverallQual', 'TotRmsAbvGrd', 'CARA'])
# %%
calculate_vif(df=dataOG, features=['SalePrice',
              'GrLivArea', 'LotArea', 'OverallQual', 'TotRmsAbvGrd', 'INTERMEDIA'])
# %%
calculate_vif(df=dataOG, features=['SalePrice',
              'GrLivArea', 'LotArea', 'OverallQual', 'TotRmsAbvGrd', 'ECONOMICA'])

# %% 
# ## 5. Utilice  el  modelo  con  el  conjunto  de  prueba  y  determine  la  eficiencia  del  algoritmo  para clasificar o predecir, en dependencia de las características de la variable respuesta. 
print('\nDATOS DE CARA')
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
print('Accuracy: ', accuracy)
print("Precision:", metrics.precision_score(
    y_test, y_pred, average='weighted'))

print('\nDATOS DE INTERMEDIA')
accuracy_i = accuracy_score(y_test_i, y_pred_i)
precision_i = precision_score(y_test_i, y_pred_i, average='micro')
print('Accuracy: ', accuracy_i)
print("Precision:", metrics.precision_score(
    y_test_i, y_pred_i, average='weighted'))

print('\nDATOS DE ECONOMICA')
accuracy_e = accuracy_score(y_test_e, y_pred_e)
precision_e = precision_score(y_test_e, y_pred_e, average='micro')
print('Accuracy: ', accuracy_e)
print("Precision:", metrics.precision_score(
    y_test_e, y_pred_e, average='weighted'))

print('\nR// Despues de calcular la exactitud y la precision de los modelos de las casas caras, intermedias y economicas, podemos determinar que la eficiencia del algoritmo es alta, se obtuvieron valores muy cercanos a 1, mostrando que los modelos de conjunto de prueba tienen un alto nivel de eficiencia.')

# %% 
# ## 6. Haga  un  análisis  de  la  eficiencia  del  algoritmo  usando  una  matriz  de  confusión.  Tenga  en cuenta la efectividad, donde el algoritmo se equivocó más, donde se equivocó menos y la importancia que tienen los errores.

# evaluate on the train dataset
cross_validate(logReg, X_train, y_train, return_train_score=True)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluar modelo
scores = cross_val_score(logReg, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = np.absolute(scores)
print(f'Mean MAE: {np.mean(scores)} ({np.std(scores)})')
print(f'Este algoritmo tuvo una media de score de {np.mean(scores)} con ({np.std(scores)}) lo cual lo hace un algoritmo con error leve lo cual se pudo haber dado a datos atípicos en la data.')

# %%
# ## 7. Cree otros dos modelos que determinen si una casa es barata o no, o intrmedia o no. Repita para cada modelo, los pasos del 1- 6.
print('En los puntos anteriores, se realizaron todas las pruebas para los 3 modelos: barata, intermedia o cara')

# %%
# ## 8. Compare la eficiencia de los 3 modelos que creó (uno para barata, otro para media y otro para  cara)  ¿Cuál  se  demoró  más  en  procesar?¿Cuál  se  equivocó  más?¿Cuál  se  equivocó menos?¿por qué?
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
accuracy_i = accuracy_score(y_test_i, y_pred_i)
precision_i = precision_score(y_test_i, y_pred_i, average='micro')
accuracy_e = accuracy_score(y_test_e, y_pred_e)
precision_e = precision_score(y_test_e, y_pred_e, average='micro')
print('Comparando la eficiencia entre los tres modelos, el que se deboro mas en procesar fue')
print('=====================================================================================')
print('||           || Modelo Caro          || Modelo Intermedio          || Modelo Economico ||')
print('=====================================================================================')
print('|| Exactitud || ', accuracy, ' || ', accuracy_i, ' || ', accuracy_e, ' ||')
print('=====================================================================================')
print('|| Precision || ', precision, ' || ', precision_i, ' || ', precision_e, ' ||')
print('=====================================================================================')

print('\nRespondiendo la pregunta de cual fue el modelo que tardo mas en procesar, se determino que tardan lo mismo, debido a que el tamaño de los conjuntos de prueba es el mismo, ademas de que los tiempos obtenidos en la ejecucion de cada modelo es el mismo.')
print('\nEl que se equivoco mas fue el modelo intermedio, ya que obtuvo un accuracy y precision de 0.88 y el que menos se equivoco fue el modelo Caro, porque el accuracy y la precision obtenidos se acercan mas a 1, siendo 0.96')
# %%
