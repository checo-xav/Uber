#"Library"
import os
import pandas as pd # Para manejo de datos
import numpy as np #  Para manejo de datos

os.getcwd()


def entregas_semana(df):
    """
    Genera una tabla resumen de entregas por semana.

    Parámetros:
        df (DataFrame): DataFrame con las columnas 'week_number' y 'delivery_trip_uuid'

    Retorna:
        DataFrame con columnas ['week_number', 'total_entregas']
    """
    
    resumen = (
    df.groupby(['territory', 'week_number'])
    .agg(total_entregas=('delivery_trip_uuid', 'count'))
    .reset_index()
    .sort_values(by=['territory', 'week_number'])
)   
    return resumen

def resumen_heatmap(df):
    """
    Genera una tabla resumen para un heatmap de entregas por territorio, día de la semana y hora.

    Parámetros:
        df (DataFrame): DataFrame con las columnas 'territory', 'weekday', 'hora' y 'delivery_trip_uuid'

    Retorna:
        DataFrame con columnas ['territory', 'weekday', 'hora', 'total_entregas']
    """
    resumen = (
        df.groupby(['territory', 'weekday', 'hora'])
        .agg(total_entregas=('delivery_trip_uuid', 'count'))
        .reset_index()
    )
    return resumen

def dist_courier_flow(df):
    """
    Genera una tabla resumen para un heatmap de entregas por territorio, día de la semana y hora.

    Parámetros:
        df (DataFrame): DataFrame con las columnas 'territory', 'weekday', 'hora' y 'delivery_trip_uuid'

    Retorna:
        DataFrame con columnas ['territory', 'weekday', 'hora', 'total_entregas']
    """
    resumen = (
        df.groupby(['territory', 'courier_flow'])
        .agg(trip_count=('delivery_trip_uuid', 'count'))
        .reset_index()
    )
    # Calcular total por territory
    resumen['total_by_territory'] = resumen.groupby('territory')['trip_count'].transform('sum')

    # Calcular porcentaje
    resumen['percentage'] = (resumen['trip_count'] / resumen['total_by_territory']) * 100
    return resumen

def hist_merchant_surface(df):
    """
    Genera una tabla resumen de merchant_surface
    Parámetros:
        df (DataFrame): DataFrame con las columnas 'week_number' y 'delivery_trip_uuid'

    Retorna:
        DataFrame con columnas ['week_number', 'total_entregas']
    """
    
    resumen = (
    df.groupby(['territory', 'week_number', 'merchant_surface'])
    .agg(total_entregas=('delivery_trip_uuid', 'count'))
    .reset_index()
    .sort_values(by=['territory', 'week_number'])
)   
    return resumen