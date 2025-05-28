CREATE OR REPLACE TABLE AA_tables.tu_tabla_destino AS
SELECT  region.territory,
		city.country_name ,
		
		con.workflow_uuid,
		con.driver_uuid,
		con.delivery_trip_uuid,
		con.courier_flow,
		
		con.restaurant_offered_timestamp_utc, ##Debe homologarse para ser compatible con las horas locales
		con.order_final_state_timestamp_local, 
		con.eater_request_timestamp_local,
		
		con.geo_archetype,
		con.merchant_surface,
		
		ROUND(dispatch.pickupdistance / 1000, 3) , #Distancia de rest a objetivo
		ROUND(dispatch.traveldistance/ 1000, 3)  AS dropoff_distance , #del objetivo a entrefa, 
		
		TIMESTAMPDIFF(SECOND, con.restaurant_offered_timestamp_utc,
		    CONVERT_TZ(con.order_final_state_timestamp_local, 'America/Mexico_City', 'UTC')
		) / 60.0 AS ATD
		
		FROM 
		    delivery_matching.eats_dispatch_metrics_job_message AS dispatch
		JOIN 
		    dwh.dim_city AS city
		    ON dispatch.cityid = city.city_id
		JOIN 
		     kirby_external_data.cities_strategy_region AS region
		    ON city.city_id = region.city_id
		JOIN 
		    tmp.lea_trips_scope_atd_consolidation_v2 AS con
		    ON dispatch.jobuuid = con.delivery_trip_uuid
		
		WHERE 
		    city.country_name = 'Mexico'
		AND 
DATE(dispatch.datestr) BETWEEN DATE_SUB(DATE('{{ds}}'), INTERVAL 7 DAY) AND DATE_SUB(DATE('{{ds}}'), INTERVAL 1 DAY)
