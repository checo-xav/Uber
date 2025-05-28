import pandas as pd
import numpy as np

def importar_datos():
    """
    Importa los datos (se puede modificar para que extraiga de algun servidor)
    
    """
    #Importe de datos
    data = pd.read_csv('/Users/checo_xav/Documents/Uber/data/BC_A&A_with_ATD.csv',na_values='\\N') # Carga de archivo CSV
    return data

def remove_univariate_outliers(df, columns):
    """
    Elimina outliers para las columnas numericas
    
    """
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.75 * IQR
        upper_bound = Q3 + 1.75 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def data_transformation(data):
    """
    Limpia & Transforma la data 
    
    """
    data = data.copy()
    #Procesamiento de los datos
    data['territory'] = data['territory'].str.replace('Long Tail - Region', 'Long Tail') # Corrección de territorio
    data['driver_uuid'] = data['driver_uuid'].fillna('Uknown') # Tratamiento a los missing values como "Desconocido"
    data['courier_flow'] = data['courier_flow'].fillna('Uknown') # Tratamiento de missing values como "Desconocidos" y  "Logistics" acumulan el 98.6 %.
    data = data[data['courier_flow'].isin(['Motorbike', 'UberEats', 'Logistics', 'Uknown'])]  #Entre "Motorbike", "UberEats", "Desconocidos" y  "Logistics" acumulan el 98.6 %. Ademas el restante no suena muy uber eats
    data['geo_archetype'] = np.where(data['geo_archetype'].isin(['Drive momentum', 'Defend CP', 'Play offense']), data['geo_archetype'], 'Others') # Validacion: geo_archetype: Drive momentum, Defend CP, Play offense acumulan el 93.8 %, puedo categorizar el resto como otros    
    data = data[data['pickup_distance'].notna()] # La info faltante en pickup_distance son las mimsas filas que dropoff_distance,dropear ya que solo representan el 1.28%

    #Procesamiento de delivery_trip_uuid y workflow_uuid por su cualidad de identifcadores no repetidos
    data['workflow_uuid'] = (
        data.groupby('workflow_uuid')
        .cumcount()
        .astype(str)
        .radd(data['workflow_uuid'] + '_'))
    data['delivery_trip_uuid'] = data['delivery_trip_uuid'].str.replace('_0$', '', regex=True)
    data['delivery_trip_uuid'] = (
        data.groupby('delivery_trip_uuid')
        .cumcount()
        .astype(str)
        .radd(data['delivery_trip_uuid'] + '_'))
    data['delivery_trip_uuid'] = data['delivery_trip_uuid'].str.replace('_0$', '', regex=True)

    #Tipos de variables
    tipos_columnas = {
                        'region': 'string',
                        'territory': 'string',
                        'country_name': 'string',
                        'workflow_uuid': 'string',
                        'driver_uuid': 'string',
                        'delivery_trip_uuid': 'string',
                        'courier_flow': 'string',
                        'restaurant_offered_timestamp_utc': 'string',
                        'order_final_state_timestamp_local': 'string',
                        'eater_request_timestamp_local': 'string',
                        'geo_archetype': 'string',
                        'merchant_surface': 'string',
                        'pickup_distance': 'float',
                        'dropoff_distance': 'float',
                        'ATD': 'float'}
    data = data.astype(tipos_columnas)
    data['restaurant_offered_timestamp_utc'] = pd.to_datetime(data['restaurant_offered_timestamp_utc'], errors='coerce')
    data['order_final_state_timestamp_local'] = pd.to_datetime(data['order_final_state_timestamp_local'], errors='coerce')
    data['eater_request_timestamp_local'] = pd.to_datetime(data['eater_request_timestamp_local'], errors='coerce')

    #Construccion de nuevas variables
    data['restaurant_offered_timestamp_local'] = pd.to_datetime(data['restaurant_offered_timestamp_utc'], errors='coerce')\
                                                   .dt.tz_localize('UTC')\
                                                   .dt.tz_convert('America/Mexico_City')
                                                   
    data['hora'] = data['eater_request_timestamp_local'].dt.hour.astype(str)
    data['weekday'] = data['eater_request_timestamp_local'].dt.day_name(locale='es_ES')
    data['week_number'] = data['eater_request_timestamp_local'].dt.isocalendar().week
    data['date'] = data['eater_request_timestamp_local'].dt.date.astype(str)
    data['tiempo_total'] = (data['order_final_state_timestamp_local'] - data['eater_request_timestamp_local']).dt.total_seconds() / 60
    data['distancia_total'] = data['pickup_distance'] + data['dropoff_distance']
    data = data.drop(['region', 'country_name'], axis=1)
    #1. Target Definition:  (Validacion ATD) 
    data['order_final_state_timestamp_local'] = data['order_final_state_timestamp_local'].dt.tz_localize('America/Mexico_City')
    data['Est_ATD'] = (data['order_final_state_timestamp_local'] - data['restaurant_offered_timestamp_local']).dt.total_seconds() / 60
    data = data[data['Est_ATD'] >= 0]
    data['ATD'] = data['Est_ATD']
    data = data.drop(['Est_ATD'], axis=1)

    #Tratamiento de Outliers a variables numericas
    data = remove_univariate_outliers(data, data.select_dtypes(include='number').columns )
    return data

