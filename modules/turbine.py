import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class HydroTurbine:
    """
    Hydro turbine modeling and control system
    """
    
    def __init__(self, rated_power=50, efficiency=0.85, head=10, flow_rate=100):
        """
        Initialize hydro turbine parameters
        
        Args:
            rated_power (float): Rated power output in kW
            efficiency (float): Turbine efficiency (0-1)
            head (float): Water head in meters
            flow_rate (float): Water flow rate in L/s
        """
        self.rated_power = rated_power
        self.efficiency = efficiency
        self.head = head
        self.flow_rate = flow_rate
        self.current_power = 0
        self.current_rpm = 0
        self.water_level = 100  # Percentage of reservoir
        
    def calculate_power_output(self, flow_rate=None, head=None):
        """
        Calculate power output based on water flow and head
        
        P = ρ * g * Q * H * η
        Where:
        P = Power (W)
        ρ = Water density (1000 kg/m³)
        g = Gravity (9.81 m/s²)
        Q = Flow rate (m³/s)
        H = Head (m)
        η = Efficiency
        """
        if flow_rate is None:
            flow_rate = self.flow_rate
        if head is None:
            head = self.head
            
        # Convert L/s to m³/s
        flow_m3_s = flow_rate / 1000
        
        # Calculate theoretical power
        power_watts = 1000 * 9.81 * flow_m3_s * head * self.efficiency
        power_kw = power_watts / 1000
        
        # Apply constraints
        self.current_power = min(power_kw, self.rated_power)
        return self.current_power
    
    def calculate_rpm(self, power_output=None):
        """Calculate turbine RPM based on power output"""
        if power_output is None:
            power_output = self.current_power
            
        # Simplified RPM calculation based on power
        max_rpm = 1200  # Typical hydro turbine max RPM
        rpm_ratio = power_output / self.rated_power
        self.current_rpm = max_rpm * rpm_ratio
        return self.current_rpm
    
    def update_water_level(self, inflow_rate, outflow_rate, dt=0.25):
        """
        Update reservoir water level
        
        Args:
            inflow_rate (float): Water coming in (L/s)
            outflow_rate (float): Water going out (L/s)
            dt (float): Time step in hours
        """
        reservoir_capacity = 10000  # Liters
        
        # Net flow in liters per time step
        net_flow = (inflow_rate - outflow_rate) * dt * 3600  # Convert to L/hour
        
        # Update water level percentage
        level_change = (net_flow / reservoir_capacity) * 100
        self.water_level += level_change
        self.water_level = np.clip(self.water_level, 0, 100)
        
        return self.water_level
    
    def optimize_flow_rate(self, target_power):
        """Optimize flow rate to achieve target power"""
        def objective(flow_rate):
            power = self.calculate_power_output(flow_rate[0])
            return abs(power - target_power)
        
        # Constraints: flow rate between 10 and 200 L/s
        bounds = [(10, 200)]
        initial_guess = [self.flow_rate]
        
        result = minimize(objective, initial_guess, bounds=bounds)
        optimal_flow = result.x[0]
        
        return optimal_flow
    
    def get_efficiency_curve(self, flow_rates=None):
        """Generate efficiency curve for different flow rates"""
        if flow_rates is None:
            flow_rates = np.linspace(10, 200, 50)
            
        efficiencies = []
        powers = []
        
        for flow in flow_rates:
            power = self.calculate_power_output(flow)
            # Efficiency decreases at very low and very high flows
            normalized_flow = flow / 100  # Optimal at 100 L/s
            eff = self.efficiency * (1 - abs(normalized_flow - 1) * 0.3)
            eff = max(0.3, min(0.95, eff))  # Clamp between 30% and 95%
            
            efficiencies.append(eff)
            powers.append(power)
        
        return pd.DataFrame({
            'flow_rate': flow_rates,
            'efficiency': efficiencies,
            'power_output': powers
        })
    
    def simulate_operation(self, hours=24, inflow_pattern='variable'):
        """
        Simulate turbine operation over time
        
        Args:
            hours (int): Simulation duration in hours
            inflow_pattern (str): 'constant', 'variable', or 'seasonal'
        """
        time_steps = hours * 4  # 15-minute intervals
        time_series = []
        
        for i in range(time_steps):
            hour = (i * 0.25) % 24
            
            # Generate inflow pattern
            if inflow_pattern == 'constant':
                inflow = 120
            elif inflow_pattern == 'variable':
                # Daily pattern with some randomness
                base_flow = 100 + 30 * np.sin(2 * np.pi * hour / 24)
                inflow = base_flow + np.random.normal(0, 10)
            else:  # seasonal
                # Seasonal variation
                day_of_year = (i // 96) % 365
                seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * day_of_year / 365)
                inflow = 100 * seasonal_factor + np.random.normal(0, 15)
            
            inflow = max(20, inflow)  # Minimum inflow
            
            # Calculate outflow (flow rate used by turbine)
            target_power = self.rated_power * (0.7 + 0.3 * np.sin(2 * np.pi * hour / 24))
            optimal_outflow = self.optimize_flow_rate(target_power)
            
            # Update system
            water_level = self.update_water_level(inflow, optimal_outflow, 0.25)
            power_output = self.calculate_power_output(optimal_outflow)
            rpm = self.calculate_rpm(power_output)
            
            # Adjust for low water levels
            if water_level < 20:
                power_output *= (water_level / 20)
                rpm *= (water_level / 20)
            
            time_series.append({
                'hour': hour,
                'time_step': i,
                'inflow_rate': round(inflow, 2),
                'outflow_rate': round(optimal_outflow, 2),
                'water_level': round(water_level, 2),
                'power_output': round(power_output, 2),
                'rpm': round(rpm, 1),
                'efficiency': round(self.efficiency, 3)
            })
        
        return pd.DataFrame(time_series)
    
    def get_status(self):
        """Get current turbine status"""
        return {
            'power_output': self.current_power,
            'rpm': self.current_rpm,
            'water_level': self.water_level,
            'efficiency': self.efficiency,
            'rated_power': self.rated_power,
            'flow_rate': self.flow_rate,
            'head': self.head
        }
    
    def maintenance_check(self):
        """Check if maintenance is needed"""
        issues = []
        
        if self.current_rpm > 1100:
            issues.append("High RPM - check bearings")
        
        if self.water_level < 15:
            issues.append("Low water level - reduce output")
        
        if self.efficiency < 0.7:
            issues.append("Low efficiency - inspect turbine blades")
        
        if self.current_power < 0.5 * self.rated_power and self.flow_rate > 80:
            issues.append("Low power output - check for blockages")
        
        return {
            'status': 'OK' if not issues else 'ATTENTION REQUIRED',
            'issues': issues,
            'next_maintenance': '30 days'  # Simple placeholder
        }
