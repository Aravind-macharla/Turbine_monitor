# -*- coding: utf-8 -*-
"""
Steam Turbine Condition Monitoring System
Main Module

Available functions:
- Real-time performance monitoring
- Anomaly detection
- Fault diagnosis

@author: Your Name
"""

import sys
sys.path.append('path/to/monitoring_system') 
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from _function.database_operation import query_data, write_data, timecircle
import _function.parameter_calculation as pc
from _function.anormal_detection import anomaly_detection
from _function.fault_detection import feature_extraction

# =============================================================================
# Data Acquisition
# =============================================================================
start_time = '2016-06-01 00:15:00'  # Input from main function: sys.argv[1]
end_time = timecircle(start_time, 1)

# Query operational data from database
data = query_data(
    "SELECT * FROM TB_ST_REAL_RUN WHERE TURID = 21 AND CYTIME BETWEEN '%s' AND '%s'" 
    % (end_time, start_time)
)

# =============================================================================
# Data Classification and Organization
# =============================================================================

# Ambient and boundary conditions
# Ambient humidity: V167, Ambient temperature: V168, Ambient pressure: V169
# Steam turbine power output: V122
boundary = pd.DataFrame({
    'Power': data['V122'],
    'T': data['V168'],
    'H': data['V167'],
    'P': data['V169']
})

# High Pressure (HP) Turbine Section
# HP inlet steam pressure: V125, HP inlet steam temperature: V126
# HP exhaust pressure: V127, HP exhaust temperature: V128
hp_turbine = pd.DataFrame({
    'p_in': data['V125'],
    't_in': data['V126'],
    'p_out': data['V127'],
    't_out': data['V128']
})

# Intermediate Pressure (IP) Turbine Section
# IP inlet steam pressure: V129, IP inlet steam temperature: V130
# IP exhaust pressure: V131, IP exhaust temperature: V132
ip_turbine = pd.DataFrame({
    'p_in': data['V129'],
    't_in': data['V130'],
    'p_out': data['V131'],
    't_out': data['V132']
})

# Low Pressure (LP) Turbine Section
# LP inlet steam pressure: V133, LP inlet steam temperature: V134
# LP exhaust temperature: V135, Condenser vacuum: V136
lp_turbine = pd.DataFrame({
    'p_in': data['V133'],
    't_in': data['V134'],
    't_out': data['V135'],
    'vacuum': data['V136']
})

# Condenser and Cooling System
# Cooling water inlet temperature: V137, Cooling water outlet temperature: V138
# Cooling water flow rate: V139
condenser = pd.DataFrame({
    'cw_t_in': data['V137'],
    'cw_t_out': data['V138'],
    'cw_flow': data['V139']
})

# Steam Supply System (from HRSG or boiler)
# Main steam flow: V106, Main steam temperature: V107, Main steam pressure: V108
# Reheat steam flow: V118, Reheat steam temperature: V119, Reheat steam pressure: V120
steam_supply = pd.DataFrame({
    'main_flow': data['V106'],
    'main_temp': data['V107'],
    'main_press': data['V108'],
    'reheat_flow': data['V118'],
    'reheat_temp': data['V119'],
    'reheat_press': data['V120']
})

# Feedwater System
# Feedwater temperature: V121, Feedwater pressure: V140, Feedwater flow: V141
feedwater = pd.DataFrame({
    'fw_temp': data['V121'],
    'fw_press': data['V140'],
    'fw_flow': data['V141']
})

# Overall turbine heat rate
heat_rate = data['V166']

# =============================================================================
# Data Preprocessing and Validation
# =============================================================================
# Add data validation and cleaning steps here as needed

# =============================================================================
# Performance Indicators Calculation
# =============================================================================

st_power = boundary['Power']
ambient_pressure = boundary['P'] / 10000  # Convert to bar

# HP Turbine Section
hp_p_in = hp_turbine['p_in']
hp_t_in = hp_turbine['t_in'] + 273.15  # Convert to Kelvin
hp_p_out = hp_turbine['p_out']
hp_t_out = hp_turbine['t_out'] + 273.15

# IP Turbine Section
ip_p_in = ip_turbine['p_in']
ip_t_in = ip_turbine['t_in'] + 273.15
ip_p_out = ip_turbine['p_out']
ip_t_out = ip_turbine['t_out'] + 273.15

# LP Turbine Section
lp_p_in = lp_turbine['p_in']
lp_t_in = lp_turbine['t_in'] + 273.15
lp_t_out = lp_turbine['t_out'] + 273.15
lp_vacuum = lp_turbine['vacuum']

# Steam turbine performance calculations
steamturbine = pc.SteamTurbine(
    st_power.iloc[-1],
    hp_p_in.iloc[-1], hp_t_in.iloc[-1], hp_p_out.iloc[-1], hp_t_out.iloc[-1],
    ip_p_in.iloc[-1], ip_t_in.iloc[-1], ip_p_out.iloc[-1], ip_t_out.iloc[-1],
    lp_p_in.iloc[-1], lp_t_in.iloc[-1], lp_t_out.iloc[-1], lp_vacuum.iloc[-1],
    steam_supply['main_flow'].iloc[-1], steam_supply['main_temp'].iloc[-1],
    condenser['cw_t_in'].iloc[-1], condenser['cw_flow'].iloc[-1]
)

# Calculate performance indicators
hp_efficiency = steamturbine.hp_section.isentropic_efficiency()
hp_stage_efficiency = steamturbine.hp_section.stage_efficiency()
hp_pressure_ratio = steamturbine.hp_section.pressure_ratio()

ip_efficiency = steamturbine.ip_section.isentropic_efficiency()
ip_stage_efficiency = steamturbine.ip_section.stage_efficiency()
ip_pressure_ratio = steamturbine.ip_section.pressure_ratio()

lp_efficiency = steamturbine.lp_section.isentropic_efficiency()
lp_stage_efficiency = steamturbine.lp_section.stage_efficiency()

overall_efficiency = steamturbine.overall_efficiency()
turbine_heat_rate = steamturbine.heat_rate()
condenser_performance = steamturbine.condenser.performance_index()

# Organize real values for monitoring
st_real_value = pd.DataFrame({
    'hp_p_in': hp_turbine['p_in'],
    'hp_t_in': hp_turbine['t_in'],
    'hp_p_out': hp_turbine['p_out'],
    'hp_t_out': hp_turbine['t_out'],
    'ip_p_in': ip_turbine['p_in'],
    'ip_t_in': ip_turbine['t_in'],
    'ip_p_out': ip_turbine['p_out'],
    'ip_t_out': ip_turbine['t_out'],
    'lp_p_in': lp_turbine['p_in'],
    'lp_t_in': lp_turbine['t_in'],
    'lp_t_out': lp_turbine['t_out'],
    'lp_vacuum': lp_turbine['vacuum'],
    'hp_efficiency': hp_efficiency,
    'ip_efficiency': ip_efficiency,
    'lp_efficiency': lp_efficiency,
    'overall_efficiency': overall_efficiency,
    'heat_rate': turbine_heat_rate,
    'condenser_perf': condenser_performance
})

# =============================================================================
# Load Reference Models (Baseline Performance Models)
# =============================================================================

# Define variables to monitor
variables = [
    'hp_p_in', 'hp_t_in', 'hp_p_out', 'hp_t_out',
    'ip_p_in', 'ip_t_in', 'ip_p_out', 'ip_t_out',
    'lp_p_in', 'lp_t_in', 'lp_t_out', 'lp_vacuum',
    'hp_efficiency', 'ip_efficiency', 'lp_efficiency',
    'overall_efficiency', 'heat_rate', 'condenser_perf'
]

# Initialize model dictionaries
reference_model = {v: None for v in variables}
std_model = {v: None for v in variables}

# Load pre-trained models
for v in variables:
    reference_model[v] = joblib.load(
        'path/to/models/ST_GLM_ref_{0}.pkl'.format(v)
    )
    std_model[v] = joblib.load(
        'path/to/models/ST_GLM_std_{0}.pkl'.format(v)
    )

# =============================================================================
# Calculate Reference Values and Standard Deviations
# =============================================================================

reference = {v: None for v in variables}
std = {v: None for v in variables}

# Predict reference values based on boundary conditions (Power, Temperature)
for v in variables:
    boundary_input = np.array(boundary[['Power', 'T']].iloc[-1]).reshape(1, -1)
    reference[v] = reference_model[v].predict(boundary_input)
    std[v] = std_model[v].predict(boundary_input)

# =============================================================================
# Anomaly Detection (Based on 3-sigma rule)
# =============================================================================

lower_limit = {v: None for v in variables}
upper_limit = {v: None for v in variables}
indicator = {v: None for v in variables}

# Calculate control limits and detect anomalies
for v in variables:
    lower_limit[v] = reference[v] - 3 * std[v]
    upper_limit[v] = reference[v] + 3 * std[v]
    indicator[v] = anomaly_detection(
        st_real_value[v], st_power, lower_limit[v], upper_limit[v]
    )

# =============================================================================
# Feature Extraction for Fault Diagnosis
# =============================================================================

feature = feature_extraction(indicator)

# =============================================================================
# Fault Diagnosis
# =============================================================================

fault_model = joblib.load('path/to/models/ST_fault_model.pkl')
fault = fault_model.predict(feature)

# =============================================================================
# Write Results to Database
# =============================================================================

# Performance Indicators
# HP efficiency: V190, IP efficiency: V191, LP efficiency: V192
# Overall efficiency: V193, Heat rate: V194, Condenser performance: V195
sql_st_1 = """INSERT INTO TB_ST_REAL_RUN
(ID, TURID, CYTIME, V190, V191, V192, V193, V194, V195)
VALUES (seq_st_common.nextval, 21, '%s', '%f', '%f', '%f', '%f', '%f', '%f')""" % (
    start_time, hp_efficiency, ip_efficiency, lp_efficiency,
    overall_efficiency, turbine_heat_rate, condenser_performance
)
write_data(sql_st_1)

# Reference Values
sql_st_2_reference = """INSERT INTO TB_ST_REAL_REFERENCE
(ID, TURID, CYTIME, V125, V126, V127, V128, V129, V130, V131, V132, 
V133, V134, V135, V136, V190, V191, V192, V193, V194, V195)
VALUES (seq_st_common.nextval, 21, '%s', '%f', '%f', '%f', '%f', '%f', '%f', 
'%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f')""" % (
    start_time,
    reference['hp_p_in'], reference['hp_t_in'], reference['hp_p_out'], reference['hp_t_out'],
    reference['ip_p_in'], reference['ip_t_in'], reference['ip_p_out'], reference['ip_t_out'],
    reference['lp_p_in'], reference['lp_t_in'], reference['lp_t_out'], reference['lp_vacuum'],
    reference['hp_efficiency'], reference['ip_efficiency'], reference['lp_efficiency'],
    reference['overall_efficiency'], reference['heat_rate'], reference['condenser_perf']
)
write_data(sql_st_2_reference)

# Lower Limits
sql_st_3_lower = """INSERT INTO TB_ST_REAL_REFERENCE_LOWER
(ID, TURID, CYTIME, V125, V126, V127, V128, V129, V130, V131, V132,
V133, V134, V135, V136, V190, V191, V192, V193, V194, V195)
VALUES (seq_st_common.nextval, 21, '%s', '%f', '%f', '%f', '%f', '%f', '%f',
'%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f')""" % (
    start_time,
    lower_limit['hp_p_in'], lower_limit['hp_t_in'], lower_limit['hp_p_out'], lower_limit['hp_t_out'],
    lower_limit['ip_p_in'], lower_limit['ip_t_in'], lower_limit['ip_p_out'], lower_limit['ip_t_out'],
    lower_limit['lp_p_in'], lower_limit['lp_t_in'], lower_limit['lp_t_out'], lower_limit['lp_vacuum'],
    lower_limit['hp_efficiency'], lower_limit['ip_efficiency'], lower_limit['lp_efficiency'],
    lower_limit['overall_efficiency'], lower_limit['heat_rate'], lower_limit['condenser_perf']
)
write_data(sql_st_3_lower)

# Upper Limits
sql_st_4_upper = """INSERT INTO TB_ST_REAL_REFERENCE_UPPER
(ID, TURID, CYTIME, V125, V126, V127, V128, V129, V130, V131, V132,
V133, V134, V135, V136, V190, V191, V192, V193, V194, V195)
VALUES (seq_st_common.nextval, 21, '%s', '%f', '%f', '%f', '%f', '%f', '%f',
'%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f')""" % (
    start_time,
    upper_limit['hp_p_in'], upper_limit['hp_t_in'], upper_limit['hp_p_out'], upper_limit['hp_t_out'],
    upper_limit['ip_p_in'], upper_limit['ip_t_in'], upper_limit['ip_p_out'], upper_limit['ip_t_out'],
    upper_limit['lp_p_in'], upper_limit['lp_t_in'], upper_limit['lp_t_out'], upper_limit['lp_vacuum'],
    upper_limit['hp_efficiency'], upper_limit['ip_efficiency'], upper_limit['lp_efficiency'],
    upper_limit['overall_efficiency'], upper_limit['heat_rate'], upper_limit['condenser_perf']
)
write_data(sql_st_4_upper)

# Anomaly Detection Results
sql_st_5_anomaly = """INSERT INTO TB_ST_REAL_ANOMALY
(ID, TURID, CYTIME, V125, V126, V127, V128, V129, V130, V131, V132,
V133, V134, V135, V136, V190, V191, V192, V193, V194, V195)
VALUES (seq_st_common.nextval, 21, '%s', '%f', '%f', '%f', '%f', '%f', '%f',
'%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f')""" % (
    start_time,
    indicator['hp_p_in'], indicator['hp_t_in'], indicator['hp_p_out'], indicator['hp_t_out'],
    indicator['ip_p_in'], indicator['ip_t_in'], indicator['ip_p_out'], indicator['ip_t_out'],
    indicator['lp_p_in'], indicator['lp_t_in'], indicator['lp_t_out'], indicator['lp_vacuum'],
    indicator['hp_efficiency'], indicator['ip_efficiency'], indicator['lp_efficiency'],
    indicator['overall_efficiency'], indicator['heat_rate'], indicator['condenser_perf']
)
write_data(sql_st_5_anomaly)

# Feature Extraction Results
sql_st_6_features = """INSERT INTO TB_ST_REAL_AUTOFEATURE
(ID, TURID, CYTIME, V1, V2, V3, V4, V5, V6, V7)
VALUES (seq_st_common.nextval, 21, '%s', '%f', '%f', '%f', '%f', '%f', '%f', '%f')""" % (
    start_time, feature['hp_eff'], feature['ip_eff'], feature['lp_eff'],
    feature['pressure_drop'], feature['temp_drop'], feature['vacuum_dev'], feature['heat_rate_dev']
)
write_data(sql_st_6_features)

# Fault Diagnosis Results
# Fault types: HP Fouling (HF), IP Fouling (IF), LP Fouling (LF),
# HP Blade Damage (HB), IP Blade Damage (IB), LP Blade Damage (LB),
# Condenser Fouling (CF), Seal Leakage (SL), Bearing Issue (BI)
sql_st_7_faults = """INSERT INTO TB_ST_REAL_AUTOFAULT
(ID, TURID, CYTIME, V1, V2, V3, V4, V5, V6, V7, V8, V9)
VALUES (seq_st_common.nextval, 21, '%s', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f')""" % (
    start_time, fault['HF'], fault['IF'], fault['LF'], fault['HB'],
    fault['IB'], fault['LB'], fault['CF'], fault['SL'], fault['BI']
)
write_data(sql_st_7_faults)

print("Steam turbine condition monitoring cycle completed successfully.")
