#!/usr/bin/env python
# coding: utf-8

# # U2 - Implementing a Predictor from scratch
# 

# Christian Adriel Rodriguez Narvaez

# ## Objective:  
# 
# Create a model of prediction (Regression) that is able to predict the average of people in extreme poverty for each federative entity.

# ## DATA PREPROCESING 

# In[3]:


import pandas as pd 


# In[4]:


df = pd.read_csv ('indicadores.csv',encoding='ISO-8859-1')


# In[5]:


df.replace(['NA', 'NULL'], pd.NA, inplace = True)

print("Number of NaN values in each column:")
print (df.isnull().sum())

df = df.dropna()
df = df.dropna(how='any')

print("El numero de (filas, columnas) es:", df.shape)
df.to_csv('indicadores_new.csv', index=False)


# In[6]:


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False) 

pd.read_csv('indicadores_new.csv')


# In[7]:


df_new = 'indicadores_new.csv'
columnas_categoricas = ['ent', 'nom_ent', 'mun', 'clave_mun','nom_mun', ]

#Eliminar
df_new = df.drop(columnas_categoricas, axis = 1)
df_new.to_csv('indicadores_final.csv', index=False)


# In[8]:


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False) 

pd.read_csv('indicadores_final.csv')


# In[9]:


df_f = pd.read_csv('indicadores_final.csv')


# In[10]:


df_f['Bajo_00'] = (df["gdo_rezsoc00"] == "Bajo_00").astype(int)


# In[11]:


import csv
archivo_csv = 'indicadores_final.csv'


columnas_a_eliminar = ['gdo_rezsoc00', 'gdo_rezsoc05', 'gdo_rezsoc10']

with open(archivo_csv, 'r', newline='') as file:
    csv_reader = csv.DictReader(file)
    
   
    columnas = csv_reader.fieldnames
    
    
    columnas_restantes = [col for col in columnas if col not in columnas_a_eliminar]
    print("Columnas restantes:", columnas_restantes)
     # Crear un nuevo archivo CSV sin las columnas especificadas
    with open('dataset_sin_columnas.csv', 'w', newline='') as new_file:
        csv_writer = csv.DictWriter(new_file, fieldnames=columnas_restantes)
        csv_writer.writeheader()
        
        for row in csv_reader:
            # Eliminar las columnas no deseadas
            for col in columnas_a_eliminar:
                del row[col]
            # Escribir la fila en el nuevo archivo
            csv_writer.writerow(row)


# In[12]:


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False) 

pd.read_csv('dataset_sin_columnas.csv')


# ## DIVISION DE DATOS

# In[14]:


data = pd.read_csv('dataset_sin_columnas.csv')


# In[49]:


train = 0.8
test = 1 - train

# Número de filas para cada conjunto
num_filas = len(data)
num_filas_entrenamiento = int(train * num_filas)
num_filas_prueba = num_filas - num_filas_entrenamiento

# Dividir los datos
datos_entrenamiento = data[:num_filas_entrenamiento]
datos_prueba = data[num_filas_entrenamiento:]


# In[50]:


num_filas_entrenamiento = len(datos_entrenamiento)
print("Número de datos en el conjunto de entrenamiento:", num_filas_entrenamiento)
num_filas_prueba = len(datos_prueba)
print("Número de datos en el conjunto de prueba:", num_filas_prueba)


# In[51]:


# Divide los datos de entrenamiento en características (X_train) y etiquetas (y_train)
X_train = datos_entrenamiento.drop('pobreza_e', axis=1)
Y_train = datos_entrenamiento['pobreza_e']

# Divide los datos de prueba en características (X_test) y etiquetas (y_test)
X_test = datos_prueba.drop('pobreza_e', axis=1)
y_test = datos_prueba['pobreza_e']


# ## Simple Regretion Model

# In[54]:


import numpy as np 
# Definir una función para entrenar el modelo de regresión lineal
def entrenar_regresion_lineal(X, Y, num_iteraciones, tasa_aprendizaje):
    m, n = X.shape  # m: número de ejemplos, n: número de características
    theta = np.zeros(n)  # Inicializar los parámetros del modelo a cero

    for i in range(num_iteraciones):
        # Calcular las predicciones
        y_pred = np.dot(X, theta)
        
        # Calcular el error
        error = y_pred - Y
        
        # Actualizar los parámetros theta
        gradient = (1/m) * np.dot(X.T, error)
        theta -= tasa_aprendizaje * gradient

    return theta

# Entrenar el modelo
num_iteraciones = 100000
tasa_aprendizaje = 0.0000000000001
theta_entrenado = entrenar_regresion_lineal(X_train, Y_train, num_iteraciones, tasa_aprendizaje)

# Función para predecir valores
def predecir(X, theta):
    return np.dot(X, theta)

# Realizar predicciones en los datos de prueba
y_pred = predecir(X_test, theta_entrenado)

# Calcular el error de la regresión (por ejemplo, el error cuadrático medio)
error = np.mean((y_pred - y_test) ** 2)
print(f"Error cuadrático medio en datos de prueba: {error}")


# ## Regretion Model 

# In[64]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Cargar tus datos en un DataFrame de pandas
datos = pd.read_csv('dataset_sin_columnas.csv')

# Definir las características y etiquetas
X = datos.drop('pobreza_e', axis=1)
y = datos['pobreza_e']

# Normalización de características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear una lista de valores de alpha en una escala logarítmica
alphas = np.logspace(-5, 5, num=100)

# Almacenar resultados en una lista de diccionarios
resultados = []

for alpha in alphas:
    # Crear un modelo de regresión Ridge
    modelo = Ridge(alpha=alpha)
    
    modelo.fit(X_train, y_train)
   
    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    resultados.append({'Alpha': alpha, 'MSE': mse})

# Convertir la lista de resultados en un DataFrame
resultados_df = pd.DataFrame(resultados)

# Encontrar el mejor valor de alpha basado en el menor MSE
mejor_fila = resultados_df.loc[resultados_df['MSE'].idxmin()]
mejor_alpha = mejor_fila['Alpha']
mejor_mse = mejor_fila['MSE']

# Imprimir el mejor valor de alpha y su MSE correspondiente
print(f"Mejor valor de alpha: {mejor_alpha}")
print(f"Error cuadrático medio (MSE) correspondiente: {mejor_mse}")

# Graficar la evolución del error en función de los valores de alpha en una escala logarítmica
plt.plot(resultados_df['Alpha'], resultados_df['MSE'], marker='o')
plt.xscale('log')
plt.xlabel('Valor de alpha (log scale)')
plt.ylabel('Error cuadrático medio (MSE)')
plt.title('Evolución del Error con Regularización Ridge (Escala Logarítmica)')
plt.show()


# In[ ]:




