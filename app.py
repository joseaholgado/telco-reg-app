import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Cargar el modelo guardado
@st.cache_resource
def cargar_modelo():
    with open('churn-model.pck', 'rb') as file:
        contenido = pickle.load(file)
        if isinstance(contenido, tuple) or isinstance(contenido, list):
            return contenido[1]  # Asume que el modelo está en la segunda posición
        return contenido

# Cargar el modelo
modelo_regresion = cargar_modelo()

# Título de la app
st.title("Predicción de Telco - Modelo de Regresión Lineal")

st.write("""
Esta aplicación utiliza un modelo de regresión lineal entrenado sobre el dataset Telco para predecir características relacionadas con clientes de telecomunicaciones.
Introduce los valores de las variables para hacer una predicción.
""")

# Entradas del usuario
st.sidebar.header("Introduce las características del cliente")

gender = st.sidebar.selectbox("Género", ['Femenino', 'Masculino'])
seniorcitizen = st.sidebar.selectbox("Senior Citizen (1=Sí, 0=No)", [1, 0])
partner = st.sidebar.selectbox("Tiene pareja", ['Sí', 'No'])
dependents = st.sidebar.selectbox("Tiene dependientes", ['Sí', 'No'])
tenure = st.sidebar.number_input("Años de permanencia", min_value=0, max_value=72, value=12)
phoneservice = st.sidebar.selectbox("Tiene servicio telefónico", ['Sí', 'No'])
multiplelines = st.sidebar.selectbox("Tiene múltiples líneas", ['Sí', 'No', 'No phone service'])
internetservice = st.sidebar.selectbox("Servicio de Internet", ['Fibra óptica', 'DSL', 'No internet service'])
onlinesecurity = st.sidebar.selectbox("Seguridad online", ['Sí', 'No', 'No internet service'])
onlinebackup = st.sidebar.selectbox("Copia de seguridad online", ['Sí', 'No', 'No internet service'])
deviceprotection = st.sidebar.selectbox("Protección de dispositivo", ['Sí', 'No', 'No internet service'])
techsupport = st.sidebar.selectbox("Soporte técnico", ['Sí', 'No', 'No internet service'])
streamingtv = st.sidebar.selectbox("Streaming TV", ['Sí', 'No', 'No internet service'])
streamingmovies = st.sidebar.selectbox("Streaming Movies", ['Sí', 'No', 'No internet service'])
contract = st.sidebar.selectbox("Tipo de contrato", ['Mes a mes', 'Un año', 'Dos años'])
paperlessbilling = st.sidebar.selectbox("Facturación sin papel", ['Sí', 'No'])
paymentmethod = st.sidebar.selectbox("Método de pago", ['Banco', 'Cheque electrónico', 'Transferencia bancaria', 'Crédito automático'])
monthlycharges = st.sidebar.number_input("Cargo mensual", min_value=0, value=70)
totalcharges = st.sidebar.number_input("Cargo total", min_value=0, value=200)

# Preprocesar las variables categóricas
def preprocesar_datos(input_data):
    data = input_data.copy()
    data['gender'] = np.where(data['gender'] == 'Masculino', 1, 0)
    data['partner'] = np.where(data['partner'] == 'Sí', 1, 0)
    data['dependents'] = np.where(data['dependents'] == 'Sí', 1, 0)
    data['phoneservice'] = np.where(data['phoneservice'] == 'Sí', 1, 0)
    data['multiplelines'] = np.where(data['multiplelines'] == 'Sí', 1, 
                                     np.where(data['multiplelines'] == 'No phone service', 0, -1))
    data['internetservice'] = np.where(data['internetservice'] == 'No internet service', 0, 
                                       np.where(data['internetservice'] == 'DSL', 1, 2))
    data['onlinesecurity'] = np.where(data['onlinesecurity'] == 'Sí', 1, 
                                      np.where(data['onlinesecurity'] == 'No internet service', 0, -1))
    data['onlinebackup'] = np.where(data['onlinebackup'] == 'Sí', 1, 
                                    np.where(data['onlinebackup'] == 'No internet service', 0, -1))
    data['deviceprotection'] = np.where(data['deviceprotection'] == 'Sí', 1, 
                                        np.where(data['deviceprotection'] == 'No internet service', 0, -1))
    data['techsupport'] = np.where(data['techsupport'] == 'Sí', 1, 
                                   np.where(data['techsupport'] == 'No internet service', 0, -1))
    data['streamingtv'] = np.where(data['streamingtv'] == 'Sí', 1, 
                                   np.where(data['streamingtv'] == 'No internet service', 0, -1))
    data['streamingmovies'] = np.where(data['streamingmovies'] == 'Sí', 1, 
                                       np.where(data['streamingmovies'] == 'No internet service', 0, -1))
    data['contract'] = np.where(data['contract'] == 'Mes a mes', 0, 
                                np.where(data['contract'] == 'Un año', 1, 2))
    data['paperlessbilling'] = np.where(data['paperlessbilling'] == 'Sí', 1, 0)
    data['paymentmethod'] = np.where(data['paymentmethod'] == 'Banco', 0, 
                                     np.where(data['paymentmethod'] == 'Cheque electrónico', 1, 
                                              np.where(data['paymentmethod'] == 'Transferencia bancaria', 2, 3)))
    return data

# Crear un DataFrame con los datos introducidos
nuevos_datos = pd.DataFrame({
    'gender': [gender],
    'seniorcitizen': [seniorcitizen],
    'partner': [partner],
    'dependents': [dependents],
    'tenure': [tenure],
    'phoneservice': [phoneservice],
    'multiplelines': [multiplelines],
    'internetservice': [internetservice],
    'onlinesecurity': [onlinesecurity],
    'onlinebackup': [onlinebackup],
    'deviceprotection': [deviceprotection],
    'techsupport': [techsupport],
    'streamingtv': [streamingtv],
    'streamingmovies': [streamingmovies],
    'contract': [contract],
    'paperlessbilling': [paperlessbilling],
    'paymentmethod': [paymentmethod],
    'monthlycharges': [monthlycharges],
    'totalcharges': [totalcharges]
})

# Preprocesar los datos antes de hacer la predicción
nuevos_datos_procesados = preprocesar_datos(nuevos_datos)

# Validar las columnas del modelo y reorganizar
if hasattr(modelo_regresion, 'feature_names_in_'):
    columnas_esperadas = modelo_regresion.feature_names_in_
    nuevos_datos_procesados = nuevos_datos_procesados[columnas_esperadas]

# Realizar la predicción con el modelo cargado
if st.sidebar.button('Predecir'):
    try:
        prediccion = modelo_regresion.predict(nuevos_datos_procesados)
        st.write(f"La predicción del modelo es: {prediccion[0]:.2f}")
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
