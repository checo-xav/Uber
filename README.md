
# Uber: Streamlit Dashboard

At Uber Eats, delivering food at the right time isn’t just about logistics — it’s about trust. 
    
Dashboard info: This dashboard aims to explore, analyze delivery patterns and support operational forecasting for Uber Eats different teams. 

The table that feeds this model comes from a workflow which updates on a weekly basis a SQL quey and stores the result in AA_tables. 

The tables that generate this query are: delivery_matching.eats_dispatch_metrics_job_message ( metrics for delivery trips ), tmp.lea_trips_scope_atd_consolidation_v2 ( consolidated information about delivery trips ) ,dwh.dim_city ( information about cities in Latin America ), kirby_external_data.cities_strategy_region ( additional information about cities ).
        
app.py is a python script which allow you to reproduce the
Automation & Analytics: Deliverys Dashboard. 

For more information contact Monika Fellmer or Sergio Chavez.


Lets walkthrough it: 

## Clone Repository

```bash
cd initial_route ( Eg: cd route/Users/checo_xav/Documents)
git clone https://github.com/checo-xav/Uber.git
cd final_route (Eg: cd route/Users/checo_xav/Documents/Uber)
```

## Install requirements

```bash
pip install -r requirements.txt
```

Do not foregt to upload the "BC_A&A_with_ATD.csv" file into Data 

## Run App

```bash
streamlit run app.py   
```
