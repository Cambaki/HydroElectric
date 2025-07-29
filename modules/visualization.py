import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

class HydroVisualization:
    """
    Visualization tools for hydro generator data analysis
    """
    
    def __init__(self):
        self.color_palette = {
            'hydro': '#1f77b4',
            'solar': '#ff7f0e', 
            'wind': '#2ca02c',
            'diesel': '#d62728',
            'battery': '#9467bd',
            'demand': '#8c564b'
        }
    
    def plot_power_sources(self, df, interactive=True):
        """Plot power generation from different sources"""
        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hydro_kw'], 
                                   name='Hydro', line=dict(color=self.color_palette['hydro'])))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['solar_kw'], 
                                   name='Solar', line=dict(color=self.color_palette['solar'])))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['wind_kw'], 
                                   name='Wind', line=dict(color=self.color_palette['wind'])))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['diesel_used_kw'], 
                                   name='Diesel', line=dict(color=self.color_palette['diesel'])))
            
            fig.update_layout(title="Power Generation by Source", 
                            xaxis_title="Time", yaxis_title="Power (kW)")
            return fig
        else:
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['hydro_kw'], label='Hydro', color=self.color_palette['hydro'])
            plt.plot(df['timestamp'], df['solar_kw'], label='Solar', color=self.color_palette['solar'])
            plt.plot(df['timestamp'], df['wind_kw'], label='Wind', color=self.color_palette['wind'])
            plt.plot(df['timestamp'], df['diesel_used_kw'], label='Diesel', color=self.color_palette['diesel'])
            plt.title("Power Generation by Source")
            plt.xlabel("Time")
            plt.ylabel("Power (kW)")
            plt.legend()
            plt.grid(True)
            return plt.gcf()
    
    def plot_battery_status(self, df):
        """Create battery SOC gauge and flow chart"""
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "indicator"}, {"type": "scatter"}]],
            subplot_titles=("Battery State of Charge", "Battery Power Flow")
        )
        
        # SOC Gauge
        latest_soc = df['battery_soc'].iloc[-1]
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=latest_soc,
            title={'text': "SOC (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "green"},
                   'steps': [
                       {'range': [0, 20], 'color': "red"},
                       {'range': [20, 50], 'color': "yellow"},
                       {'range': [50, 100], 'color': "green"}]}
        ), row=1, col=1)
        
        # Battery Flow
        fig.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['battery_flow_kw'],
            name='Battery Flow',
            line=dict(color=self.color_palette['battery'])
        ), row=1, col=2)
        
        fig.update_layout(height=400, title="Battery System Status")
        return fig
    
    def plot_demand_vs_supply(self, df):
        """Plot demand vs total supply"""
        total_supply = df['hydro_kw'] + df['solar_kw'] + df['wind_kw'] + df['diesel_used_kw']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=total_supply, 
                               name='Total Supply', fill='tonexty'))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['demand_kw'], 
                               name='Demand', line=dict(color=self.color_palette['demand'])))
        
        fig.update_layout(title="Supply vs Demand", 
                        xaxis_title="Time", yaxis_title="Power (kW)")
        return fig
    
    def plot_efficiency_analysis(self, df):
        """Analyze system efficiency"""
        total_gen = df['hydro_kw'] + df['solar_kw'] + df['wind_kw'] + df['diesel_used_kw']
        efficiency = (df['demand_kw'] / total_gen * 100).fillna(0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Efficiency over time
        ax1.plot(df['timestamp'], efficiency, color='purple', linewidth=2)
        ax1.set_title("System Efficiency Over Time")
        ax1.set_ylabel("Efficiency (%)")
        ax1.grid(True)
        
        # Efficiency distribution
        ax2.hist(efficiency, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax2.set_title("Efficiency Distribution")
        ax2.set_xlabel("Efficiency (%)")
        ax2.set_ylabel("Frequency")
        ax2.grid(True)
        
        plt.tight_layout()
        return plt.gcf()
    
    def create_power_dashboard(self, df):
        """Create comprehensive power system dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Power Sources", "Battery SOC", "Supply vs Demand", "Efficiency"),
            specs=[[{"type": "scatter"}, {"type": "indicator"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Power sources (stacked area)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hydro_kw'], 
                               name='Hydro', fill='tonexty', stackgroup='one'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['solar_kw'], 
                               name='Solar', fill='tonexty', stackgroup='one'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['wind_kw'], 
                               name='Wind', fill='tonexty', stackgroup='one'), row=1, col=1)
        
        # Battery SOC
        latest_soc = df['battery_soc'].iloc[-1]
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=latest_soc,
            gauge={'axis': {'range': [0, 100]}}
        ), row=1, col=2)
        
        # Supply vs Demand
        total_supply = df['hydro_kw'] + df['solar_kw'] + df['wind_kw'] + df['diesel_used_kw']
        fig.add_trace(go.Scatter(x=df['timestamp'], y=total_supply, name='Supply'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['demand_kw'], name='Demand'), row=2, col=1)
        
        # Efficiency
        efficiency = (df['demand_kw'] / total_supply * 100).fillna(0)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=efficiency, name='Efficiency %'), row=2, col=2)
        
        fig.update_layout(height=800, title="Hydro Power System Dashboard")
        return fig
