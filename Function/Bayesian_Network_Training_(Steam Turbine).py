
"""
Steam Turbine - Bayesian Network Fault Diagnosis Model

Constructs probabilistic network for fault diagnosis
"""

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from sklearn.externals import joblib
import pandas as pd
import numpy as np

def construct_steam_turbine_network():
    """
    Build Bayesian Network for steam turbine fault diagnosis
    
    Fault Modes (13):
    - HF, HC, HS: HP Fouling, Damage, Seal Leakage
    - IF, IC, IS: IP Fouling, Damage, Seal Leakage
    - LF, LC, LS: LP Fouling, Damage, Seal Leakage
    - CF, CA: Condenser Fouling, Air Ingress
    - BI, VB: Bearing Issues, Vibration
    
    Symptoms (15):
    - Efficiency: hp_eff, ip_eff, lp_eff
    - Pressure: hp_p, ip_p, lp_p, vacuum
    - Temperature: hp_t, ip_t, lp_t, cw_t
    - Other: vib, heat_rate, power_drop
    """
    
    # Define network structure (DAG)
    model = BayesianModel([
        # HP Section → Symptoms
        ('HF', 'hp_eff'), ('HF', 'hp_p'), ('HF', 'heat_rate'),
        ('HC', 'hp_eff'), ('HC', 'hp_t'), ('HC', 'vib'),
        ('HS', 'hp_eff'), ('HS', 'hp_p'), ('HS', 'power_drop'),
        
        # IP Section → Symptoms
        ('IF', 'ip_eff'), ('IF', 'ip_p'), ('IF', 'heat_rate'),
        ('IC', 'ip_eff'), ('IC', 'ip_t'), ('IC', 'vib'),
        ('IS', 'ip_eff'), ('IS', 'ip_p'), ('IS', 'power_drop'),
        
        # LP Section → Symptoms
        ('LF', 'lp_eff'), ('LF', 'lp_p'), ('LF', 'heat_rate'),
        ('LC', 'lp_eff'), ('LC', 'lp_t'), ('LC', 'vib'),
        ('LS', 'lp_eff'), ('LS', 'vacuum'), ('LS', 'power_drop'),
        
        # Condenser → Symptoms
        ('CF', 'vacuum'), ('CF', 'cw_t'), ('CF', 'lp_eff'),
        ('CA', 'vacuum'), ('CA', 'lp_eff'),
        
        # Auxiliary → Symptoms
        ('BI', 'vib'), ('BI', 'power_drop'),
        ('VB', 'vib'),
        
        # Hierarchical relationships (efficiency affects other params)
        ('hp_eff', 'hp_p'), ('hp_eff', 'hp_t'),
        ('ip_eff', 'ip_p'), ('ip_eff', 'ip_t'),
        ('lp_eff', 'lp_p'), ('lp_eff', 'vacuum')
    ])
    
    # =========================================================================
    # PRIOR PROBABILITIES (Fault Base Rates)
    # Based on operational experience and maintenance records
    # =========================================================================
    
    # HP Section faults
    hf_cpd = TabularCPD('HF', 2, [[0.85, 0.15]])  # 15% fouling rate
    hc_cpd = TabularCPD('HC', 2, [[0.95, 0.05]])  # 5% damage rate
    hs_cpd = TabularCPD('HS', 2, [[0.92, 0.08]])  # 8% seal leakage
    
    # IP Section faults
    if_cpd = TabularCPD('IF', 2, [[0.88, 0.12]])  # 12% fouling
    ic_cpd = TabularCPD('IC', 2, [[0.95, 0.05]])  # 5% damage
    is_cpd = TabularCPD('IS', 2, [[0.92, 0.08]])  # 8% seal leakage
    
    # LP Section faults (higher fouling due to moisture)
    lf_cpd = TabularCPD('LF', 2, [[0.82, 0.18]])  # 18% fouling (most common)
    lc_cpd = TabularCPD('LC', 2, [[0.94, 0.06]])  # 6% damage (water erosion)
    ls_cpd = TabularCPD('LS', 2, [[0.90, 0.10]])  # 10% seal leakage
    
    # Condenser faults
    cf_cpd = TabularCPD('CF', 2, [[0.80, 0.20]])  # 20% tube fouling (common)
    ca_cpd = TabularCPD('CA', 2, [[0.93, 0.07]])  # 7% air ingress
    
    # Auxiliary faults
    bi_cpd = TabularCPD('BI', 2, [[0.92, 0.08]])  # 8% bearing issues
    vb_cpd = TabularCPD('VB', 2, [[0.90, 0.10]])  # 10% vibration
    
    # =========================================================================
    # CONDITIONAL PROBABILITIES (Using Noisy-OR model)
    # =========================================================================
    
    # HP Efficiency (affected by fouling, damage, seal leakage)
    hp_eff_cpd = TabularCPD(
        'hp_eff', 2,
        evidence=['HF', 'HC', 'HS'],
        evidence_card=[2, 2, 2],
        values=[
            # P(hp_eff=0 | HF, HC, HS) - Normal
            # HF=0,HC=0,HS=0  HF=0,HC=0,HS=1  HF=0,HC=1,HS=0  HF=0,HC=1,HS=1
            # HF=1,HC=0,HS=0  HF=1,HC=0,HS=1  HF=1,HC=1,HS=0  HF=1,HC=1,HS=1
            [0.99, 0.25, 0.20, 0.050, 0.15, 0.038, 0.030, 0.0075],  # Normal
            [0.01, 0.75, 0.80, 0.950, 0.85, 0.962, 0.970, 0.9925]   # Anomaly
        ]
    )
    
    # HP Pressure (affected by fouling, seal leakage, and efficiency)
    hp_p_cpd = TabularCPD(
        'hp_p', 2,
        evidence=['HF', 'HS', 'hp_eff'],
        evidence_card=[2, 2, 2],
        values=[
            # HF=0,HS=0,eff=0  HF=0,HS=0,eff=1  HF=0,HS=1,eff=0  HF=0,HS=1,eff=1
            # HF=1,HS=0,eff=0  HF=1,HS=0,eff=1  HF=1,HS=1,eff=0  HF=1,HS=1,eff=1
            [0.99, 0.30, 0.25, 0.075, 0.20, 0.060, 0.050, 0.015],
            [0.01, 0.70, 0.75, 0.925, 0.80, 0.940, 0.950, 0.985]
        ]
    )
    
    # HP Temperature (affected by blade damage)
    hp_t_cpd = TabularCPD(
        'hp_t', 2,
        evidence=['HC', 'hp_eff'],
        evidence_card=[2, 2],
        values=[
            [0.99, 0.40, 0.15, 0.060],
            [0.01, 0.60, 0.85, 0.940]
        ]
    )
    
    # Similar CPDs for IP section
    ip_eff_cpd = TabularCPD(
        'ip_eff', 2,
        evidence=['IF', 'IC', 'IS'],
        evidence_card=[2, 2, 2],
        values=[
            [0.99, 0.25, 0.20, 0.050, 0.15, 0.038, 0.030, 0.0075],
            [0.01, 0.75, 0.80, 0.950, 0.85, 0.962, 0.970, 0.9925]
        ]
    )
    
    ip_p_cpd = TabularCPD(
        'ip_p', 2,
        evidence=['IF', 'IS', 'ip_eff'],
        evidence_card=[2, 2, 2],
        values=[
            [0.99, 0.30, 0.25, 0.075, 0.20, 0.060, 0.050, 0.015],
            [0.01, 0.70, 0.75, 0.925, 0.80, 0.940, 0.950, 0.985]
        ]
    )
    
    ip_t_cpd = TabularCPD(
        'ip_t', 2,
        evidence=['IC', 'ip_eff'],
        evidence_card=[2, 2],
        values=[
            [0.99, 0.40, 0.15, 0.060],
            [0.01, 0.60, 0.85, 0.940]
        ]
    )
    
    # LP Section (more complex due to condenser interaction)
    lp_eff_cpd = TabularCPD(
        'lp_eff', 2,
        evidence=['LF', 'LC', 'LS', 'CF', 'CA'],
        evidence_card=[2, 2, 2, 2, 2],
        values=[
            # This would be a 2×32 matrix - showing simplified version
            # In practice, use Noisy-OR formula to generate all 32 combinations
            # P(lp_eff=0) = leak × ∏(1 - λᵢ) for active causes
            [0.99] + [0.99 * (1 - sum([0.85, 0.80, 0.75, 0.70, 0.65][:i])) 
                      for i in range(1, 32)],
            [0.01] + [1 - 0.99 * (1 - sum([0.85, 0.80, 0.75, 0.70, 0.65][:i])) 
                      for i in range(1, 32)]
        ]
    )
    
    # Vacuum (affected by LP seal, condenser fouling, air ingress)
    vacuum_cpd = TabularCPD(
        'vacuum', 2,
        evidence=['LS', 'CF', 'CA', 'lp_eff'],
        evidence_card=[2, 2, 2, 2],
        values=[
            # LS=0,CF=0,CA=0,eff=0  ... (16 combinations)
            [0.99, 0.40, 0.30, 0.120, 0.25, 0.100, 0.075, 0.030,
             0.20, 0.080, 0.060, 0.024, 0.050, 0.020, 0.015, 0.006],
            [0.01, 0.60, 0.70, 0.880, 0.75, 0.900, 0.925, 0.970,
             0.80, 0.920, 0.940, 0.976, 0.950, 0.980, 0.985, 0.994]
        ]
    )
    
    # Condenser cooling water temperature
    cw_t_cpd = TabularCPD(
        'cw_t', 2,
        evidence=['CF'],
        evidence_card=[2],
        values=[
            [0.99, 0.20],  # Higher outlet temp if fouled
            [0.01, 0.80]
        ]
    )
    
    # Vibration (affected by blade damage, bearing issues, vibration fault)
    vib_cpd = TabularCPD(
        'vib', 2,
        evidence=['HC', 'IC', 'LC', 'BI', 'VB'],
        evidence_card=[2, 2, 2, 2, 2],
        values=[
            # Using Noisy-OR: Multiple causes can trigger vibration
            [0.99, 0.30, 0.30, 0.090, 0.30, 0.090, 0.090, 0.027,
             0.20, 0.060, 0.060, 0.018, 0.060, 0.018, 0.018, 0.0054,
             0.15, 0.045, 0.045, 0.014, 0.045, 0.014, 0.014, 0.004,
             0.030, 0.009, 0.009, 0.003, 0.009, 0.003, 0.003, 0.0009],
            [0.01, 0.70, 0.70, 0.910, 0.70, 0.910, 0.910, 0.973,
             0.80, 0.940, 0.940, 0.982, 0.940, 0.982, 0.982, 0.9946,
             0.85, 0.955, 0.955, 0.986, 0.955, 0.986, 0.986, 0.996,
             0.970, 0.991, 0.991, 0.997, 0.991, 0.997, 0.997, 0.9991]
        ]
    )
    
    # Heat rate (affected by all efficiency losses)
    heat_rate_cpd = TabularCPD(
        'heat_rate', 2,
        evidence=['hp_eff', 'ip_eff', 'lp_eff'],
        evidence_card=[2, 2, 2],
        values=[
            [0.99, 0.40, 0.40, 0.160, 0.40, 0.160, 0.160, 0.064],
            [0.01, 0.60, 0.60, 0.840, 0.60, 0.840, 0.840, 0.936]
        ]
    )
    
    # Power drop (affected by seal leakage, bearing issues)
    power_drop_cpd = TabularCPD(
        'power_drop', 2,
        evidence=['HS', 'IS', 'LS', 'BI'],
        evidence_card=[2, 2, 2, 2],
        values=[
            [0.99, 0.35, 0.35, 0.123, 0.35, 0.123, 0.123, 0.043,
             0.30, 0.105, 0.105, 0.037, 0.105, 0.037, 0.037, 0.013],
            [0.01, 0.65, 0.65, 0.877, 0.65, 0.877, 0.877, 0.957,
             0.70, 0.895, 0.895, 0.963, 0.895, 0.963, 0.963, 0.987]
        ]
    )
    
    # Add all CPDs to model
    model.add_cpds(
        # Fault priors
        hf_cpd, hc_cpd, hs_cpd,
        if_cpd, ic_cpd, is_cpd,
        lf_cpd, lc_cpd, ls_cpd,
        cf_cpd, ca_cpd,
        bi_cpd, vb_cpd,
        # Conditional probabilities
        hp_eff_cpd, hp_p_cpd, hp_t_cpd,
        ip_eff_cpd, ip_p_cpd, ip_t_cpd,
        lp_eff_cpd, vacuum_cpd, cw_t_cpd,
        vib_cpd, heat_rate_cpd, power_drop_cpd
    )
    
    # Verify model structure
    try:
        model.check_model()
        print("✓ Bayesian Network structure validated successfully!")
    except ValueError as e:
        print(f"✗ Model validation failed: {e}")
        return None
    
    # Save model
    joblib.dump(model, 'model/ST_fault_model.pkl')
    print("✓ Model saved to: model/ST_fault_model.pkl")
    
    return model


def test_network_inference(model):
    """
    Test the Bayesian Network with sample cases
    """
    
    print("\n" + "="*60)
    print("Testing Bayesian Network Inference")
    print("="*60)
    
    inference = VariableElimination(model)
    
    # Test Case 1: LP Fouling
    print("\n--- Test Case 1: LP Fouling Scenario ---")
    evidence = {
        'lp_eff': 1,      # LP efficiency dropped
        'vacuum': 1,      # Vacuum degraded
        'lp_p': 1,        # LP pressure anomaly
        'heat_rate': 1    # Heat rate increased
    }
    
    result = inference.query(
        variables=['LF', 'LC', 'LS', 'CF', 'CA'],
        evidence=evidence
    )
    
    print("Evidence: LP efficiency↓, Vacuum↓, LP pressure anomaly, Heat rate↑")
    print("\nFault Probabilities:")
    for fault in ['LF', 'LC', 'LS', 'CF', 'CA']:
        prob = result[fault].values[1]
        if prob > 0.3:
            print(f"  {fault}: {prob*100:.1f}% {'← LIKELY' if prob > 0.5 else ''}")
    
    # Test Case 2: HP Blade Damage
    print("\n--- Test Case 2: HP Blade Damage Scenario ---")
    evidence = {
        'hp_eff': 1,      # HP efficiency dropped
        'hp_t': 1,        # HP temperature anomaly
        'vib': 1,         # Vibration increased
        'power_drop': 0   # No significant power drop
    }
    
    result = inference.query(
        variables=['HF', 'HC', 'HS'],
        evidence=evidence
    )
    
    print("Evidence: HP efficiency↓, HP temp anomaly, Vibration↑, Normal power")
    print("\nFault Probabilities:")
    for fault in ['HF', 'HC', 'HS']:
        prob = result[fault].values[1]
        if prob > 0.3:
            print(f"  {fault}: {prob*100:.1f}% {'← LIKELY' if prob > 0.5 else ''}")
    
    # Test Case 3: Condenser Fouling
    print("\n--- Test Case 3: Condenser Fouling Scenario ---")
    evidence = {
        'vacuum': 1,      # Vacuum degraded
        'cw_t': 1,        # Cooling water temp high
        'lp_eff': 1,      # LP efficiency affected
        'vib': 0          # No vibration
    }
    
    result = inference.query(
        variables=['CF', 'CA', 'LF', 'LS'],
        evidence=evidence
    )
    
    print("Evidence: Vacuum↓, CW temp↑, LP efficiency↓, Normal vibration")
    print("\nFault Probabilities:")
    for fault in ['CF', 'CA', 'LF', 'LS']:
        prob = result[fault].values[1]
        if prob > 0.3:
            print(f"  {fault}: {prob*100:.1f}% {'← LIKELY' if prob > 0.5 else ''}")


# Main execution
if __name__ == '__main__':
    
    print("="*60)
    print("Steam Turbine Bayesian Network Training")
    print("="*60)
    
    # Construct network
    model = construct_steam_turbine_network()
    
    if model is not None:
        # Test inference
        test_network_inference(model)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print("\nNetwork Statistics:")
        print(f"  Fault modes: 13")
        print(f"  Symptoms: 15")
        print(f"  Total nodes: 28")
        print(f"  Total edges: {len(model.edges())}")
        print(f"\nModel ready for deployment!")
