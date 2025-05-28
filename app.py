#"Library"
import os #Directorio de trabajo en mac
import pandas as pd # Para manejo de datos
import numpy as np #  Para manejo de datos
import matplotlib.pyplot as plt # Para graficos
import streamlit as st #Para la app web
import seaborn as sns  # Para mejorar la estética de los gráficos
import plotly.express as px #para graficos 

#"Funciones"
from src import data_processing, Calculated_Tables
#from scr.data_processing import importar_datos, remove_univariate_outliers, data_transformation


#"Directorio Mac"
#os.getcwd()
#os.chdir('/Users/checo_xav/Documents/Uber')



#"Carga de datos"
@st.cache_data
def cargar_datos():
    return data_processing.importar_datos()
data = cargar_datos()



#"Transformacion y Limpieza"
@st.cache_data
def transformar_datos(data):
    return data_processing.data_transformation(data)
df = transformar_datos(data)



# "Tablas Calculadas"
@st.cache_data
def entregas_semana(df):
    return Calculated_Tables.entregas_semana(df)
resumen_semanal = entregas_semana(df)

@st.cache_data
def resumen_heatmap(df):
    return Calculated_Tables.resumen_heatmap(df)
resumen_heat = resumen_heatmap(df)

@st.cache_data
def dist_courier_flow(df):
    return Calculated_Tables.dist_courier_flow(df)
resumen_courier_flow = dist_courier_flow(df)

@st.cache_data
def hist_merchant_surface(df):
    return Calculated_Tables.hist_merchant_surface(df)
resumen_merchant_surface = hist_merchant_surface(df)



#Formato Dashboard
st.markdown(
    """
    <style>
    /* Caja de texto gris */
    .gray-box {
        background-color: #F0F2F6;
        padding: 15px;
        border-radius: 10px;
        color: #333333;
        font-size: 16px;
        margin-bottom: 10px;
    }

    /* Color de fondo de la app */
    .stApp {
        background-color: #06C167;
    }

    /* Color de los títulos */
    h1, h2, h3 {
        color: #000000;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# " IV. Estructura Dashboard"
tab1, tab2 = st.tabs(["Dashboard Info", "Descriptive Statistics"])

# Tab 1: General Info
with tab1:
    st.title("Automation & Analytics: Deliverys Dashboard")

    # Gray-box
    st.markdown(
        """
        <div class="gray-box">
        <strong> At Uber Eats, delivering food at the right time isn’t just about logistics — it’s about trust. </strong> 
        <br><br>
        Dashboard info: 
        This dashboard aims to explore, analyze delivery patterns and support operational forecasting for Uber Eats different teams. 
        <br>
        The table that feeds this model comes from a workflow which updates on a weekly basis a SQL quey and stores the result in AA_tables. 
        The tables that generate this query are: delivery_matching.eats_dispatch_metrics_job_message ( metrics for delivery trips ), tmp.lea_trips_scope_atd_consolidation_v2 ( consolidated information about delivery trips ) ,dwh.dim_city ( information about cities in Latin America ), kirby_external_data.cities_strategy_region ( additional information about cities ).
        <br><br>
        For more information contact Monika Fellmer or Sergio Chavez.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Nota pequeña
    st.caption("This Dashbaord get updated each week")

# Tab 2: Análisis Descriptivo
with tab2:
    
    st.subheader("Descriptive Statistics")
    st.markdown(
        """
        <div class="gray-box">
        This tab presents an overview of delivery data with several interactive visualizations. Users can filter the analysis by selecting one or more territories.
        <br>
        - Key Metrics: Displays average pickup distance, dropoff distance, and average delivery time (ATD) based on the filtered data.
        <br>
        - Weekly Delivery Trends: A line chart showing the total number of deliveries per week throughout 2025, either by territory or as an overall trend.
        <br>
        - Heatmap: Highlights the hours and weekdays with the highest delivery volumes, helping to identify peak times during the week.
        <br>
        - Courier Flow Distribution: A bar chart illustrating the number of delivery trips by different courier flows, broken down by territory when filtered.""",
        unsafe_allow_html=True)

        
    # Sidebar filters solo si estoy en tab2
    #selected_territories = st.sidebar.multiselect(
    #    "Selecciona territorios:",
    #    options=sorted(df['territory'].unique()),
    #    default=sorted(df['territory'].unique())
    #)
    
    selected_territories = st.multiselect(
        "Select Territorys:",
        options=sorted(df['territory'].unique())
    )
    
    #####----------------Gráfico de Tarjeta ---------------------------------
    if selected_territories:
        df_filtrado = df[df['territory'].isin(selected_territories)]
    else:
        df_filtrado = df.copy()
    
    # Calcular promedios
    pickup_avg = df_filtrado['pickup_distance'].mean()
    dropoff_avg = df_filtrado['dropoff_distance'].mean()
    ATD_avg = df_filtrado['ATD'].mean()
    
    # Mostrar métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Pickup Distance (km)", f"{pickup_avg:.2f}")
    col2.metric("Avg Dropoff Distance (km)", f"{dropoff_avg:.2f}")
    col3.metric("Avg ATD (min)", f"{ATD_avg:.2f}")
    
    
    st.subheader("2025 Weekly Delivery Analysis Series")
    
    ##### ------------------ Gráfico de lineas -----------------
    if selected_territories:
        resumen_filtrado = resumen_semanal[
            resumen_semanal['territory'].isin(selected_territories)
        ]
    
        # Gráfico por territorio
        fig = px.line(
            resumen_filtrado,
            x='week_number',
            y='total_entregas',
            color='territory',
            title="2025 Weekly Delivery Analysis Series",
            markers=True
        )
    
    else:
        # Si no hay selección → total acumulado por semana
        resumen_total = (
            resumen_semanal
            .groupby('week_number')
            .agg(total_entregas=('total_entregas', 'sum'))
            .reset_index()
        )
    
        # Gráfico acumulado sin color
        fig = px.line(
            resumen_total,
            x='week_number',
            y='total_entregas',
            title="2025 Weekly Delivery Analysis Series",
            markers=True
        )
    
    fig.update_layout(
        xaxis_title="2025 Week Number",
        yaxis_title="Total Deliverys",
        template="simple_white",
        title_x=0
        
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("HeatMap")
    ##### -------------------------- HeatMap -----------------------------------
    if selected_territories:
        resumen_filtrado_heat = resumen_heat[
            resumen_heat['territory'].isin(selected_territories)
    ]
    else:
        resumen_filtrado_heat = resumen_heat.copy()

    # Orden explícito para los días de la semana en español
    dias_semana = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
    
    # Convierte weekday a categoría con orden para controlar el eje y
    resumen_filtrado_heat['weekday'] = resumen_filtrado_heat['weekday'].str.lower()  # asegúrate que está en minúscula para emparejar
    resumen_filtrado_heat['weekday'] = pd.Categorical(
        resumen_filtrado_heat['weekday'],
        categories=dias_semana,
        ordered=True
    )
    horas_unicas = sorted(resumen_filtrado_heat['hora'].unique(), key=lambda x: int(x))

    
    # Pivot table con fill_value=0
    heatmap_data = resumen_filtrado_heat.pivot_table(
        index='weekday',
        columns='hora',
        values='total_entregas',
        aggfunc='sum',
        fill_value=0
    )
    
    # Reindexar columnas para que estén en el orden correcto de horas
    heatmap_data = heatmap_data.reindex(columns=horas_unicas)
    
    # Ahora creamos el heatmap con plotly
    fig_heat = px.imshow(
    heatmap_data,
    aspect="auto",
    color_continuous_scale='Viridis',
    labels=dict(x="Hour of the day", y="Weekday", color="Total Deliverys"),
    title="HeatMap"
    )   
    
    # Configurar eje y con orden de días
    dias_semana = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
    fig_heat.update_yaxes(categoryorder='array', categoryarray=dias_semana)
    st.plotly_chart(fig_heat, use_container_width=True)

    #####----------------Gráfico de barras---------------------------------
    st.subheader("Distribution of the trips by the different courier flows")    
    # Filtrar df base por territorios seleccionados
    # Filtrar según territorios seleccionados (si hay alguno)
    if selected_territories:
        resumen_courier_flow_filtrado = resumen_courier_flow[
            resumen_courier_flow['territory'].isin(selected_territories)
        ]
    else:
        resumen_courier_flow_filtrado = resumen_courier_flow.copy()
    
    # Creamos el gráfico de barras
    if selected_territories:
        resumen_filtrado_flow = resumen_courier_flow[
            resumen_courier_flow['territory'].isin(selected_territories)
        ]
        # Si seleccionaron todos → mostrar separados
        if set(selected_territories) == set(df['territory'].unique()):
            fig_bar = px.bar(
                resumen_filtrado_flow,
                x='courier_flow',
                y='trip_count',
                color='territory',
                barmode='group',
                title="Courier Flow Distribution by Territory"
            )
        else:
            # Solo los seleccionados, pero sin color por territory
            resumen_agg = (
                resumen_filtrado_flow
                .groupby('courier_flow')
                .agg(trip_count=('trip_count', 'sum'))
                .reset_index()
            )
            fig_bar = px.bar(
                resumen_agg,
                x='courier_flow',
                y='trip_count',
                title="Courier Flow Distribution (Selected Territories)"
            )
    else:
        # Sin selección → todo acumulado
        resumen_total_flow = (
            resumen_courier_flow
            .groupby('courier_flow')
            .agg(trip_count=('trip_count', 'sum'))
            .reset_index()
        )
        fig_bar = px.bar(
            resumen_total_flow,
            x='courier_flow',
            y='trip_count',
            title="Courier Flow Distribution (All Territories)"
        )
    
    # Layout común
    fig_bar.update_layout(
        xaxis_title="Courier Flow",
        yaxis_title="Number of Delivery Trips",
        template="simple_white",
        title_x=0
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
#Se corre desde la terminal como: 
# cd /Users/checo_xav/Documents/Uber
# streamlit run app.py   
