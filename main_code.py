# -*- coding: utf-8 -*-
"""
Steam Turbine Condition Monitoring System - Main Module

Real-time monitoring, anomaly detection, and fault diagnosis
"""

import sys
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from _function.database_operation import query_data, write_data, timecircle
import _function.parameter_calculation as pc
from _function.anormal_detection import anomaly_detection
from _function.fault_detection import feature_extraction

# =============================================================================
# CONFIGURATION
# =============================================================================
TURBINE_ID = 21  # Steam turbine unit ID
INTERVAL_MINUTES = 1  # Data acquisition interval

# =============================================================================
# DATA ACQUISITION
# =============================================================================
start_time = '2016-06-01 00:15:00'  # Could be sys.argv[1] in production
end_time = timecircle(start_time, INTERVAL_MINUTES)

# Query real-time data from database
query_sql = """
    SELECT * FROM TB_ST_REAL_RUN 
    WHERE TURID = %d 
    AND CYTIME BETWEEN '%s' AND '%s'
""" % (TURBINE_ID, end_time, start_time)

data = query_data(query_sql)

# =============================================================================
# DATA ORGANIZATION
# =============================================================================

# Boundary conditions
boundary = pd.DataFrame({
    'Power': data['V122'],      # Steam turbine power (MW)
    'T': data['V168'],          # Ambient temperature (°C)
    'H': data['V167'],          # Ambient humidity (%)
    'P': data['V169']           # Ambient pressure (Pa)
})

# HP Section parameters
hp_params = pd.DataFrame({
    'p_in': data['V125'],       # HP inlet pressure (MPa)
    't_in': data['V126'],       # HP inlet temperature (°C)
    'p_out': data['V127'],      # HP outlet pressure (MPa)
    't_out': data['V128']       # HP outlet temperature (°C)
})

# IP Section parameters
ip_params = pd.DataFrame({
    'p_in': data['V129'],       # IP inlet pressure (MPa)
    't_in': data['V130'],       # IP inlet temperature (°C)
    'p_out': data['V131'],      # IP outlet pressure (MPa)
    't_out': data['V132']       # IP outlet temperature (°C)
})

# LP Section parameters
lp_params = pd.DataFrame({
    'p_in': data['V133'],       # LP inlet pressure (MPa)
    't_in': data['V134'],       # LP inlet temperature (°C)
    't_out': data['V135'],      # LP outlet temperature (°C)
    'vacuum': data['V136']      # Condenser vacuum (kPa)
})

# Condenser parameters
condenser_params = pd.DataFrame({
    'cw_t_in': data['V137'],    # Cooling water inlet temp (°C)
    'cw_t_out': data['V138'],   # Cooling water outlet temp (°C)
    'cw_flow': data['V139']     # Cooling water flow (kg/s)
})

# Steam flow parameters
steam_flows = pd.DataFrame({
    'main_flow': data['V106'],      # Main steam flow (kg/s)
    'reheat_flow': data['V118']     # Reheat steam flow (kg/s)
})

# Heat rate
heat_rate_data = data['V166']

# =============================================================================
# PERFORMANCE CALCULATION
# =============================================================================

st_power = boundary['Power'].iloc[-1]
ambient_pressure = boundary['P'].iloc[-1] / 10000  # Convert to MPa

# Create turbine section objects
hp_section = pc.SteamTurbineSection(
    p_in=hp_params['p_in'].iloc[-1],
    t_in=hp_params['t_in'].iloc[-1],
    p_out=hp_params['p_out'].iloc[-1],
    t_out=hp_params['t_out'].iloc[-1],
    m_steam=steam_flows['main_flow'].iloc[-1]
)

ip_section = pc.SteamTurbineSection(
    p_in=ip_params['p_in'].iloc[-1],
    t_in=ip_params['t_in'].iloc[-1],
    p_out=ip_params['p_out'].iloc[-1],
    t_out=ip_params['t_out'].iloc[-1],
    m_steam=steam_flows['reheat_flow'].iloc[-1]
)

lp_section = pc.SteamTurbineSection(
    p_in=lp_params['p_in'].iloc[-1],
    t_in=lp_params['t_in'].iloc[-1],
    p_out=lp_params['vacuum'].iloc[-1] / 1000,  # Convert kPa to MPa
    t_out=lp_params['t_out'].iloc[-1],
    m_steam=steam_flows['reheat_flow'].iloc[-1] * 0.95  # Account for extraction
)

condenser = pc.Condenser(
    vacuum=lp_params['vacuum'].iloc[-1],
    t_exhaust=lp_params['t_out'].iloc[-1],
    cw_t_in=condenser_params['cw_t_in'].iloc[-1],
    cw_t_out=condenser_params['cw_t_out'].iloc[-1],
    cw_flow=condenser_params['cw_flow'].iloc[-1]
)

# Create complete turbine object
steamturbine = pc.SteamTurbine(
    power=st_power,
    hp_section=hp_section,
    ip_section=ip_section,
    lp_section=lp_section,
    condenser=condenser,
    m_main_steam=steam_flows['main_flow'].iloc[-1]
)

# Calculate performance indicators
hp_efficiency = hp_section.isentropic_efficiency()
hp_pressure_ratio = hp_section.pressure_ratio()

ip_efficiency = ip_section.isentropic_efficiency()
ip_pressure_ratio = ip_section.pressure_ratio()

lp_efficiency = lp_section.isentropic_efficiency()
lp_pressure_ratio = lp_section.pressure_ratio()

overall_efficiency = steamturbine.overall_efficiency()
turbine_heat_rate = steamturbine.heat_rate()
condenser_performance = condenser.performance_index()

# Organize calculated values
st_real_value = pd.DataFrame({
    # HP Section
    'hp_p_in': hp_params['p_in'],
    'hp_t_in': hp_params['t_in'],
    'hp_p_out': hp_params['p_out'],
    'hp_t_out': hp_params['t_out'],
    'hp_efficiency': hp_efficiency,
    'hp_pressure_ratio': hp_pressure_ratio,
    
    # IP Section
    'ip_p_in': ip_params['p_in'],
    'ip_t_in': ip_params['t_in'],
    'ip_p_out': ip_params['p_out'],
    'ip_t_out': ip_params['t_out'],
    'ip_efficiency': ip_efficiency,
    'ip_pressure_ratio': ip_pressure_ratio,
    
    # LP Section
    'lp_p_in': lp_params['p_in'],
    'lp_t_in': lp_params['t_in'],
    'lp_t_out': lp_params['t_out'],
    'lp_vacuum': lp_params['vacuum'],
    'lp_efficiency': lp_efficiency,
    'lp_pressure_ratio': lp_pressure_ratio,
    
    # Overall
    'overall_efficiency': overall_efficiency,
    'heat_rate': turbine_heat_rate,
    'condenser_perf': condenser_performance
})

# =============================================================================
# LOAD BASELINE MODELS
# =============================================================================

variables = [
    'hp_p_in', 'hp_t_in', 'hp_p_out', 'hp_t_out', 'hp_efficiency', 'hp_pressure_ratio',
    'ip_p_in', 'ip_t_in', 'ip_p_out', 'ip_t_out', 'ip_efficiency', 'ip_pressure_ratio',
    'lp_p_in', 'lp_t_in', 'lp_t_out', 'lp_vacuum', 'lp_efficiency', 'lp_pressure_ratio',
    'overall_efficiency', 'heat_rate', 'condenser_perf'
]

reference_model = {v: None for v in variables}
std_model = {v: None for v in variables}

# Load pre-trained GLM models
for v in variables:
    try:
        reference_model[v] = joblib.load(
            f'model/ST_GLM_ref_{v}.pkl'
        )
        std_model[v] = joblib.load(
            f'model/ST_GLM_std_{v}.pkl'
        )
    except FileNotFoundError:
        print(f"Warning: Model for {v} not found. Skipping...")
        continue

# =============================================================================
# CALCULATE REFERENCE VALUES
# =============================================================================

reference = {v: None for v in variables}
std = {v: None for v in variables}

boundary_input = np.array(boundary[['Power', 'T']].iloc[-1]).reshape(1, -1)

for v in variables:
    if reference_model[v] is not None:
        reference[v] = reference_model[v].predict(boundary_input)[0]
        std[v] = std_model[v].predict(boundary_input)[0]

# =============================================================================
# ANOMALY DETECTION
# =============================================================================

lower_limit = {v: None for v in variables}
upper_limit = {v: None for v in variables}
indicator = {v: None for v in variables}

for v in variables:
    if reference[v] is not None and std[v] is not None:
        lower_limit[v] = reference[v] - 3 * std[v]
        upper_limit[v] = reference[v] + 3 * std[v]
        
        # Two-stage anomaly detection with steady-state check
        indicator[v] = anomaly_detection(
            variable=st_real_value[v],
            power=boundary['Power'],
            lower_limit=lower_limit[v],
            upper_limit=upper_limit[v]
        )

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

feature = feature_extraction(indicator)

# =============================================================================
# FAULT DIAGNOSIS
# =============================================================================

fault_model = joblib.load('model/ST_fault_model.pkl')
fault_probabilities = fault_model.predict(feature)

# =============================================================================
# DATABASE WRITING
# =============================================================================

# 1. Write calculated performance indicators
sql_st_performance = """
INSERT INTO TB_ST_REAL_RUN
(ID, TURID, CYTIME, 
 V190, V191, V192, V193, V194, V195, V196, V197, V198, V199)
VALUES (seq_st_common.nextval, %d, '%s',
        %f, %f, %f, %f, %f, %f, %f, %f, %f, %f)
""" % (
    TURBINE_ID, start_time,
    hp_efficiency, hp_pressure_ratio,
    ip_efficiency, ip_pressure_ratio,
    lp_efficiency, lp_pressure_ratio,
    overall_efficiency, turbine_heat_rate, condenser_performance,
    st_power
)
write_data(sql_st_performance)

# 2. Write reference values
reference_values = [reference[v] for v in variables if reference[v] is not None]
sql_st_reference = """
INSERT INTO TB_ST_REAL_REFERENCE
(ID, TURID, CYTIME, V125, V126, V127, V128, V129, V130, V131, V132,
 V133, V134, V135, V136, V190, V191, V192, V193, V194, V195, V196)
VALUES (seq_st_common.nextval, %d, '%s', %s)
""" % (
    TURBINE_ID, start_time,
    ', '.join([f"'{v}'" for v in reference_values])
)
write_data(sql_st_reference)

# 3. Write lower control limits
lower_values = [lower_limit[v] for v in variables if lower_limit[v] is not None]
sql_st_lower = """
INSERT INTO TB_ST_REAL_REFERENCE_LOWER
(ID, TURID, CYTIME, V125, V126, V127, V128, V129, V130, V131, V132,
 V133, V134, V135, V136, V190, V191, V192, V193, V194, V195, V196)
VALUES (seq_st_common.nextval, %d, '%s', %s)
""" % (
    TURBINE_ID, start_time,
    ', '.join([f"'{v}'" for v in lower_values])
)
write_data(sql_st_lower)

# 4. Write upper control limits
upper_values = [upper_limit[v] for v in variables if upper_limit[v] is not None]
sql_st_upper = """
INSERT INTO TB_ST_REAL_REFERENCE_UPPER
(ID, TURID, CYTIME, V125, V126, V127, V128, V129, V130, V131, V132,
 V133, V134, V135, V136, V190, V191, V192, V193, V194, V195, V196)
VALUES (seq_st_common.nextval, %d, '%s', %s)
""" % (
    TURBINE_ID, start_time,
    ', '.join([f"'{v}'" for v in upper_values])
)
write_data(sql_st_upper)

# 5. Write anomaly indicators
indicator_values = [indicator[v] for v in variables if indicator[v] is not None]
sql_st_anomaly = """
INSERT INTO TB_ST_REAL_ANOMALY
(ID, TURID, CYTIME, V125, V126, V127, V128, V129, V130, V131, V132,
 V133, V134, V135, V136, V190, V191, V192, V193, V194, V195, V196)
VALUES (seq_st_common.nextval, %d, '%s', %s)
""" % (
    TURBINE_ID, start_time,
    ', '.join([f"'{v}'" for v in indicator_values])
)
write_data(sql_st_anomaly)

# 6. Write extracted features
sql_st_features = """
INSERT INTO TB_ST_REAL_AUTOFEATURE
(ID, TURID, CYTIME, V1, V2, V3, V4, V5, V6, V7)
VALUES (seq_st_common.nextval, %d, '%s',
        %f, %f, %f, %f, %f, %f, %f)
""" % (
    TURBINE_ID, start_time,
    feature['hp_eff'], feature['ip_eff'], feature['lp_eff'],
    feature['pressure_drop'], feature['temp_drop'],
    feature['vacuum_dev'], feature['heat_rate_dev']
)
write_data(sql_st_features)

# 7. Write fault diagnosis results
sql_st_faults = """
INSERT INTO TB_ST_REAL_AUTOFAULT
(ID, TURID, CYTIME, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13)
VALUES (seq_st_common.nextval, %d, '%s',
        %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f)
""" % (
    TURBINE_ID, start_time,
    fault_probabilities['HF'], fault_probabilities['HC'], fault_probabilities['HS'],
    fault_probabilities['IF'], fault_probabilities['IC'], fault_probabilities['IS'],
    fault_probabilities['LF'], fault_probabilities['LC'], fault_probabilities['LS'],
    fault_probabilities['CF'], fault_probabilities['CA'],
    fault_probabilities['BI'], fault_probabilities['VB']
)
write_data(sql_st_faults)

print(f"[{start_time}] Steam turbine monitoring cycle completed successfully.")

# Print diagnostic summary if anomalies detected
anomaly_count = sum(1 for v in indicator.values() if v == 1)
if anomaly_count > 0:
    print(f"⚠️  {anomaly_count} anomalies detected!")
    for param, ind in indicator.items():
        if ind == 1:
            print(f"  - {param}: {st_real_value[param].iloc[-1]:.2f} "
                  f"(expected: {reference[param]:.2f} ± {3*std[param]:.2f})")
    
    # Print most likely faults
    print("\n🔍 Fault Diagnosis:")
    fault_list = sorted(fault_probabilities.items(), key=lambda x: x[1], reverse=True)
    for fault_name, prob in fault_list[:3]:
        if prob > 0.3:
            print(f"  - {fault_name}: {prob*100:.1f}% probability")
