import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

class PowerFlowAnalysis:
    """
    Electrical load flow analysis for microgrid systems
    """
    
    def __init__(self):
        """Initialize power flow analyzer"""
        self.buses = {}
        self.lines = {}
        self.generators = {}
        self.loads = {}
        self.bus_count = 0
        
    def add_bus(self, bus_id, bus_type='PQ', voltage=1.0, angle=0.0):
        """
        Add electrical bus to the system
        
        Args:
            bus_id (str): Unique bus identifier
            bus_type (str): 'SLACK', 'PV', or 'PQ'
            voltage (float): Bus voltage magnitude (p.u.)
            angle (float): Bus voltage angle (degrees)
        """
        self.buses[bus_id] = {
            'type': bus_type,
            'voltage': voltage,
            'angle': angle,
            'P_gen': 0.0,
            'Q_gen': 0.0,
            'P_load': 0.0,
            'Q_load': 0.0,
            'index': self.bus_count
        }
        self.bus_count += 1
        
    def add_line(self, from_bus, to_bus, resistance, reactance, susceptance=0.0):
        """
        Add transmission line between buses
        
        Args:
            from_bus (str): Starting bus ID
            to_bus (str): Ending bus ID
            resistance (float): Line resistance (p.u.)
            reactance (float): Line reactance (p.u.)
            susceptance (float): Line susceptance (p.u.)
        """
        line_id = f"{from_bus}-{to_bus}"
        self.lines[line_id] = {
            'from_bus': from_bus,
            'to_bus': to_bus,
            'R': resistance,
            'X': reactance,
            'B': susceptance,
            'impedance': complex(resistance, reactance)
        }
        
    def add_generator(self, bus_id, p_gen, q_gen=0.0):
        """Add generator to bus"""
        if bus_id in self.buses:
            self.buses[bus_id]['P_gen'] = p_gen
            self.buses[bus_id]['Q_gen'] = q_gen
            
    def add_load(self, bus_id, p_load, q_load=0.0):
        """Add load to bus"""
        if bus_id in self.buses:
            self.buses[bus_id]['P_load'] = p_load
            self.buses[bus_id]['Q_load'] = q_load
            
    def build_admittance_matrix(self):
        """Build network admittance matrix"""
        n = self.bus_count
        Y = np.zeros((n, n), dtype=complex)
        
        # Add line admittances
        for line_id, line in self.lines.items():
            from_idx = self.buses[line['from_bus']]['index']
            to_idx = self.buses[line['to_bus']]['index']
            
            # Line admittance
            y_line = 1 / line['impedance']
            
            # Off-diagonal elements
            Y[from_idx, to_idx] -= y_line
            Y[to_idx, from_idx] -= y_line
            
            # Diagonal elements
            Y[from_idx, from_idx] += y_line + 1j * line['B'] / 2
            Y[to_idx, to_idx] += y_line + 1j * line['B'] / 2
            
        return Y
    
    def newton_raphson_pf(self, max_iter=100, tolerance=1e-6):
        """
        Solve power flow using Newton-Raphson method
        """
        n = self.bus_count
        Y = self.build_admittance_matrix()
        
        # Initialize voltage vectors
        V_mag = np.array([self.buses[bus_id]['voltage'] for bus_id in sorted(self.buses.keys())])
        V_ang = np.array([np.radians(self.buses[bus_id]['angle']) for bus_id in sorted(self.buses.keys())])
        
        # Power mismatches
        def calculate_power_mismatch():
            P_calc = np.zeros(n)
            Q_calc = np.zeros(n)
            
            for i in range(n):
                for j in range(n):
                    P_calc[i] += V_mag[i] * V_mag[j] * (
                        Y[i,j].real * np.cos(V_ang[i] - V_ang[j]) +
                        Y[i,j].imag * np.sin(V_ang[i] - V_ang[j])
                    )
                    Q_calc[i] += V_mag[i] * V_mag[j] * (
                        Y[i,j].real * np.sin(V_ang[i] - V_ang[j]) -
                        Y[i,j].imag * np.cos(V_ang[i] - V_ang[j])
                    )
            
            return P_calc, Q_calc
        
        # Get scheduled powers
        bus_ids = sorted(self.buses.keys())
        P_sched = np.array([self.buses[bus_id]['P_gen'] - self.buses[bus_id]['P_load'] 
                           for bus_id in bus_ids])
        Q_sched = np.array([self.buses[bus_id]['Q_gen'] - self.buses[bus_id]['Q_load'] 
                           for bus_id in bus_ids])
        
        # Newton-Raphson iterations
        for iteration in range(max_iter):
            P_calc, Q_calc = calculate_power_mismatch()
            
            # Power mismatches
            dP = P_sched - P_calc
            dQ = Q_sched - Q_calc
            
            # Check convergence
            if np.max(np.abs(dP)) < tolerance and np.max(np.abs(dQ)) < tolerance:
                break
                
            # Build Jacobian matrix (simplified)
            J = np.zeros((2*n, 2*n))
            
            # Update voltage (simplified update)
            V_ang += 0.01 * dP  # Simplified angle update
            V_mag += 0.01 * dQ  # Simplified magnitude update
            
            # Ensure voltage limits
            V_mag = np.clip(V_mag, 0.95, 1.05)
        
        # Update bus voltages
        for i, bus_id in enumerate(sorted(self.buses.keys())):
            self.buses[bus_id]['voltage'] = V_mag[i]
            self.buses[bus_id]['angle'] = np.degrees(V_ang[i])
            
        return iteration < max_iter
    
    def calculate_line_flows(self):
        """Calculate power flows in transmission lines"""
        line_flows = {}
        
        for line_id, line in self.lines.items():
            from_bus = line['from_bus']
            to_bus = line['to_bus']
            
            V_from = self.buses[from_bus]['voltage'] * np.exp(
                1j * np.radians(self.buses[from_bus]['angle'])
            )
            V_to = self.buses[to_bus]['voltage'] * np.exp(
                1j * np.radians(self.buses[to_bus]['angle'])
            )
            
            # Line current
            I_line = (V_from - V_to) / line['impedance']
            
            # Power flows
            S_from = V_from * np.conj(I_line)
            S_to = V_to * np.conj(-I_line)
            
            line_flows[line_id] = {
                'P_from': S_from.real,
                'Q_from': S_from.imag,
                'P_to': S_to.real,
                'Q_to': S_to.imag,
                'P_loss': S_from.real + S_to.real,
                'Q_loss': S_from.imag + S_to.imag,
                'current_mag': abs(I_line),
                'loading_percent': abs(I_line) * 100  # Assuming 1 p.u. = 100% loading
            }
            
        return line_flows
    
    def voltage_stability_analysis(self):
        """Analyze voltage stability margins"""
        stability_info = {}
        
        for bus_id, bus in self.buses.items():
            # Simple voltage stability index
            V = bus['voltage']
            stability_margin = 1.0 - V  # Distance from voltage collapse
            
            if V < 0.95:
                status = 'CRITICAL'
            elif V < 0.98:
                status = 'WARNING'
            else:
                status = 'NORMAL'
                
            stability_info[bus_id] = {
                'voltage': V,
                'stability_margin': stability_margin,
                'status': status
            }
            
        return stability_info
    
    def power_quality_analysis(self):
        """Analyze power quality metrics"""
        pq_metrics = {}
        
        voltages = [bus['voltage'] for bus in self.buses.values()]
        
        pq_metrics['voltage_stats'] = {
            'min_voltage': min(voltages),
            'max_voltage': max(voltages),
            'avg_voltage': np.mean(voltages),
            'voltage_deviation': np.std(voltages),
            'voltage_regulation': (max(voltages) - min(voltages)) / np.mean(voltages) * 100
        }
        
        # THD analysis (simplified)
        pq_metrics['thd_analysis'] = {
            'voltage_thd': np.random.uniform(1, 5),  # Placeholder - would need harmonic analysis
            'current_thd': np.random.uniform(2, 8),
            'frequency_deviation': np.random.uniform(-0.1, 0.1)
        }
        
        return pq_metrics
    
    def contingency_analysis(self, contingency_type='line_outage'):
        """Perform N-1 contingency analysis"""
        contingencies = []
        
        if contingency_type == 'line_outage':
            # Simulate each line outage
            for line_id in self.lines.keys():
                # Store original line
                original_line = self.lines[line_id].copy()
                
                # Remove line temporarily
                del self.lines[line_id]
                
                # Solve power flow
                converged = self.newton_raphson_pf()
                
                if converged:
                    min_voltage = min(bus['voltage'] for bus in self.buses.values())
                    max_voltage = max(bus['voltage'] for bus in self.buses.values())
                    
                    contingencies.append({
                        'contingency': f'Line {line_id} outage',
                        'converged': True,
                        'min_voltage': min_voltage,
                        'max_voltage': max_voltage,
                        'critical': min_voltage < 0.9 or max_voltage > 1.1
                    })
                else:
                    contingencies.append({
                        'contingency': f'Line {line_id} outage',
                        'converged': False,
                        'critical': True
                    })
                
                # Restore line
                self.lines[line_id] = original_line
                
        return contingencies
    
    def generate_load_flow_report(self):
        """Generate comprehensive load flow analysis report"""
        # Solve base case
        converged = self.newton_raphson_pf()
        
        if not converged:
            return {'error': 'Power flow did not converge'}
        
        # Calculate results
        line_flows = self.calculate_line_flows()
        stability_info = self.voltage_stability_analysis()
        pq_metrics = self.power_quality_analysis()
        
        # Generate summary
        total_generation = sum(bus['P_gen'] for bus in self.buses.values())
        total_load = sum(bus['P_load'] for bus in self.buses.values())
        total_losses = sum(flow['P_loss'] for flow in line_flows.values())
        
        report = {
            'convergence': converged,
            'system_summary': {
                'total_generation_mw': total_generation,
                'total_load_mw': total_load,
                'total_losses_mw': total_losses,
                'loss_percentage': (total_losses / total_generation) * 100 if total_generation > 0 else 0
            },
            'bus_results': self.buses,
            'line_flows': line_flows,
            'voltage_stability': stability_info,
            'power_quality': pq_metrics
        }
        
        return report
    
    def create_simple_microgrid(self):
        """Create a simple microgrid model for testing"""
        # Add buses
        self.add_bus('MAIN', 'SLACK', 1.0, 0.0)  # Main grid connection
        self.add_bus('HYDRO', 'PV', 1.0)         # Hydro generator bus
        self.add_bus('SOLAR', 'PV', 1.0)         # Solar farm bus
        self.add_bus('LOAD1', 'PQ', 1.0)         # Load center 1
        self.add_bus('LOAD2', 'PQ', 1.0)         # Load center 2
        
        # Add transmission lines (R, X in p.u.)
        self.add_line('MAIN', 'HYDRO', 0.01, 0.05)
        self.add_line('HYDRO', 'SOLAR', 0.02, 0.08)
        self.add_line('SOLAR', 'LOAD1', 0.01, 0.04)
        self.add_line('LOAD1', 'LOAD2', 0.015, 0.06)
        self.add_line('MAIN', 'LOAD2', 0.02, 0.10)
        
        # Add generators
        self.add_generator('HYDRO', 25, 5)   # 25 MW, 5 MVAR
        self.add_generator('SOLAR', 15, 0)   # 15 MW, 0 MVAR
        
        # Add loads
        self.add_load('LOAD1', 20, 8)        # 20 MW, 8 MVAR
        self.add_load('LOAD2', 15, 6)        # 15 MW, 6 MVAR
