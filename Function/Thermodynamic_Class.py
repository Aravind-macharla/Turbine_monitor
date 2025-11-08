
"""
Steam Turbine Condition Monitoring - Thermodynamic Calculations
"""

from pyXSteam.XSteam import XSteam

class SteamTurbineSection():
    """Base class for HP/IP/LP turbine sections"""
    steamTable = XSteam(XSteam.UNIT_SYSTEM_BARE)
    
    def __init__(self, p_in, t_in, p_out, t_out, m_steam=None):
        self.p_in = p_in      # Inlet pressure (MPa)
        self.t_in = t_in      # Inlet temperature (°C)
        self.p_out = p_out    # Outlet pressure (MPa)
        self.t_out = t_out    # Outlet temperature (°C)
        self.m_steam = m_steam  # Steam flow rate (kg/s)
    
    def isentropic_efficiency(self):
        """Calculate isentropic efficiency"""
        h_in = self.steamTable.h_pt(self.p_in, self.t_in)
        h_out = self.steamTable.h_pt(self.p_out, self.t_out)
        s_in = self.steamTable.s_pt(self.p_in, self.t_in)
        
        # Isentropic outlet state
        t_out_s = self.steamTable.t_ps(self.p_out, s_in)
        h_out_s = self.steamTable.h_pt(self.p_out, t_out_s)
        
        efficiency = (h_in - h_out) / (h_in - h_out_s)
        return efficiency
    
    def pressure_ratio(self):
        """Calculate pressure ratio"""
        return self.p_in / self.p_out
    
    def enthalpy_drop(self):
        """Calculate actual enthalpy drop"""
        h_in = self.steamTable.h_pt(self.p_in, self.t_in)
        h_out = self.steamTable.h_pt(self.p_out, self.t_out)
        return h_in - h_out
    
    def power_output(self):
        """Calculate section power output (MW)"""
        if self.m_steam is None:
            return None
        delta_h = self.enthalpy_drop()  # kJ/kg
        power = self.m_steam * delta_h / 1000  # MW
        return power


class Condenser():
    """Condenser performance calculation"""
    steamTable = XSteam(XSteam.UNIT_SYSTEM_BARE)
    
    def __init__(self, vacuum, t_exhaust, cw_t_in, cw_t_out, cw_flow):
        self.vacuum = vacuum          # Condenser vacuum (kPa absolute)
        self.t_exhaust = t_exhaust    # Exhaust steam temperature (°C)
        self.cw_t_in = cw_t_in        # Cooling water inlet temp (°C)
        self.cw_t_out = cw_t_out      # Cooling water outlet temp (°C)
        self.cw_flow = cw_flow        # Cooling water flow (kg/s)
    
    def performance_index(self):
        """
        Calculate condenser performance index
        Lower is better (closer to ideal)
        """
        # Saturation temperature at condenser pressure
        t_sat = self.steamTable.tsat_p(self.vacuum / 1000)  # Convert to MPa
        
        # Terminal temperature difference (TTD)
        ttd = self.t_exhaust - t_sat
        
        # Cooling water temperature rise
        cw_rise = self.cw_t_out - self.cw_t_in
        
        # Performance index (lower = better)
        # Ideal: TTD ≈ 0, meaning exhaust at saturation
        perf_index = ttd / cw_rise if cw_rise > 0 else 999
        return perf_index
    
    def vacuum_deviation(self, expected_vacuum):
        """Deviation from expected vacuum"""
        return (expected_vacuum - self.vacuum) / expected_vacuum


class SteamTurbine():
    """Complete steam turbine system"""
    
    def __init__(self, power, hp_section, ip_section, lp_section, 
                 condenser, m_main_steam):
        self.power = power  # Total power output (MW)
        self.hp = hp_section
        self.ip = ip_section
        self.lp = lp_section
        self.condenser = condenser
        self.m_main_steam = m_main_steam  # Main steam flow (kg/s)
    
    def overall_efficiency(self):
        """Calculate overall isentropic efficiency"""
        # Weighted by power output of each section
        hp_power = self.hp.power_output() or 0
        ip_power = self.ip.power_output() or 0
        lp_power = self.lp.power_output() or 0
        total_power = hp_power + ip_power + lp_power
        
        if total_power == 0:
            return None
        
        hp_eff = self.hp.isentropic_efficiency()
        ip_eff = self.ip.isentropic_efficiency()
        lp_eff = self.lp.isentropic_efficiency()
        
        overall_eff = (hp_power * hp_eff + ip_power * ip_eff + 
                      lp_power * lp_eff) / total_power
        return overall_eff
    
    def heat_rate(self):
        """Calculate turbine heat rate (kJ/kWh)"""
        if self.power == 0:
            return None
        
        # Total heat input (from steam enthalpy)
        steamTable = XSteam(XSteam.UNIT_SYSTEM_BARE)
        
        # Assume main steam at HP inlet conditions
        h_main = steamTable.h_pt(self.hp.p_in, self.hp.t_in)
        
        # Reference state (condensate at condenser)
        h_ref = steamTable.hL_p(self.condenser.vacuum / 1000)
        
        # Heat input per kg of steam
        q_in = h_main - h_ref  # kJ/kg
        
        # Total heat input
        Q_total = self.m_main_steam * q_in  # kJ/s = kW
        
        # Heat rate
        heat_rate = Q_total / self.power  # kJ/kWh... wait, units!
        heat_rate = (Q_total / 1000) / self.power * 3600  # kJ/kWh
        return heat_rate


# Example usage
if __name__ == "__main__":
    # Create turbine sections
    hp_section = SteamTurbineSection(
        p_in=12.5,    # 12.5 MPa
        t_in=538,     # 538°C
        p_out=2.8,    # 2.8 MPa
        t_out=342,    # 342°C
        m_steam=150   # 150 kg/s
    )
    
    ip_section = SteamTurbineSection(
        p_in=2.5,     # 2.5 MPa (after reheat)
        t_in=538,     # 538°C
        p_out=0.45,   # 0.45 MPa
        t_out=285,    # 285°C
        m_steam=145   # 145 kg/s (some extraction)
    )
    
    lp_section = SteamTurbineSection(
        p_in=0.40,    # 0.40 MPa
        t_in=270,     # 270°C
        p_out=0.0050, # 5 kPa (vacuum)
        t_out=33,     # 33°C
        m_steam=140   # 140 kg/s
    )
    
    condenser = Condenser(
        vacuum=5.0,       # 5 kPa
        t_exhaust=33,     # 33°C
        cw_t_in=20,       # 20°C
        cw_t_out=28,      # 28°C
        cw_flow=5000      # 5000 kg/s
    )
    
    turbine = SteamTurbine(
        power=150,         # 150 MW
        hp_section=hp_section,
        ip_section=ip_section,
        lp_section=lp_section,
        condenser=condenser,
        m_main_steam=150
    )
    
    # Calculate performance indicators
    print("=== Steam Turbine Performance ===")
    print(f"HP Efficiency: {hp_section.isentropic_efficiency()*100:.2f}%")
    print(f"IP Efficiency: {ip_section.isentropic_efficiency()*100:.2f}%")
    print(f"LP Efficiency: {lp_section.isentropic_efficiency()*100:.2f}%")
    print(f"Overall Efficiency: {turbine.overall_efficiency()*100:.2f}%")
    print(f"Heat Rate: {turbine.heat_rate():.1f} kJ/kWh")
    print(f"Condenser Performance: {condenser.performance_index():.3f}")
