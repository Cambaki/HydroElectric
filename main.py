import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.animation as animation
from io import BytesIO
from hydro_ml_simulation import simulate_microgrid, DemandPredictor
from datetime import timedelta

# Import our custom modules
from modules.visualization import HydroVisualization
from modules.turbine import HydroTurbine
from modules.load_flow import PowerFlowAnalysis

st.set_page_config(layout="wide")
st.title("🔋 Hydroelectric Generator Real-Time Dashboard")

# Initialize our custom modules
viz = HydroVisualization()
turbine = HydroTurbine(rated_power=50, efficiency=0.85)
power_system = PowerFlowAnalysis()

# Create a simple microgrid for analysis
power_system.create_simple_microgrid()

# Sidebar Controls
sim_days = st.sidebar.slider("Simulation Days", 1, 14, 7)
show_charts = st.sidebar.multiselect("Select Visuals", [
    "Source Contribution", "Battery SOC Gauge", "Actual vs Predicted Demand",
    "Energy Flow Over Time", "Correlation Heatmap", "Power Loss Pie Chart",
    "Animated Power Flow", "Turbine Analysis", "Power System Analysis"
], default=["Source Contribution", "Battery SOC Gauge"])

# Advanced analysis options
st.sidebar.subheader("🔧 Advanced Analysis")
show_turbine_sim = st.sidebar.checkbox("Turbine Simulation", value=False)
show_power_flow = st.sidebar.checkbox("Power Flow Analysis", value=False)
turbine_hours = st.sidebar.slider("Turbine Simulation Hours", 1, 48, 24)

# Run simulation and train model
df_hist = simulate_microgrid(sim_days=sim_days)
predictor = DemandPredictor()
predictor.fit(df_hist)
start_time = pd.to_datetime(df_hist['timestamp'].iloc[-1]) + timedelta(minutes=15)
df_new = simulate_microgrid(sim_days=1, start_time=start_time)
preds = predictor.predict(df_new)
df_new["demand_pred"] = preds

# Add cumulative and derived columns
df_new["total_gen"] = df_new['hydro_kw'] + df_new['solar_kw'] + df_new['wind_kw'] + df_new['diesel_used_kw']
df_new["energy_generated"] = df_new["total_gen"].cumsum()
df_new["energy_demand"] = df_new["demand_kw"].cumsum()
df_new["unmet_demand"] = np.clip(df_new["demand_kw"] - df_new["total_gen"], 0, None)
df_new["eff_loss"] = np.abs(df_new["battery_flow_kw"]) * 0.05  # 5% assumed loss

# Layout Columns
col1, col2 = st.columns(2)

# Chart 1: Power Source Contributions
if "Source Contribution" in show_charts:
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_new['timestamp'], y=df_new['hydro_kw'], name='Hydro'))
        fig.add_trace(go.Scatter(x=df_new['timestamp'], y=df_new['solar_kw'], name='Solar'))
        fig.add_trace(go.Scatter(x=df_new['timestamp'], y=df_new['wind_kw'], name='Wind'))
        fig.add_trace(go.Scatter(x=df_new['timestamp'], y=df_new['diesel_used_kw'], name='Diesel'))
        fig.update_layout(title="Power Source Output Over Time", xaxis_title="Time", yaxis_title="kW")
        st.plotly_chart(fig, use_container_width=True)

# Chart 2: Battery SOC Gauge
if "Battery SOC Gauge" in show_charts:
    with col2:
        latest_soc = df_new['battery_soc'].iloc[-1]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_soc,
            title={'text': "Battery State of Charge (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "green"},
                   'steps': [
                       {'range': [0, 50], 'color': "red"},
                       {'range': [50, 80], 'color': "yellow"},
                       {'range': [80, 100], 'color': "green"}]}))
        st.plotly_chart(fig, use_container_width=True)

# Chart 3: Actual vs Predicted Demand
if "Actual vs Predicted Demand" in show_charts:
    with st.expander("📈 Actual vs Predicted Demand (ML Model)", expanded=True):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_new['timestamp'], df_new['demand_kw'], label="Actual")
        ax.plot(df_new['timestamp'], df_new['demand_pred'], label="Predicted", linestyle="--")
        ax.set_title("Actual vs Predicted Demand")
        ax.set_ylabel("kW")
        ax.set_xlabel("Time")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# Chart 4: Cumulative Energy Flow
if "Energy Flow Over Time" in show_charts:
    with st.expander("⚡ Cumulative Energy Over Time"):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(df_new['timestamp'], df_new['energy_generated'], label="Generated", alpha=0.5)
        ax.fill_between(df_new['timestamp'], df_new['energy_demand'], label="Demand", alpha=0.5)
        ax.set_title("Cumulative Energy Comparison")
        ax.set_ylabel("kWh")
        ax.set_xlabel("Time")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# Chart 5: Correlation Heatmap
if "Correlation Heatmap" in show_charts:
    with st.expander("📊 Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_new.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# Chart 6: Power Loss Pie Chart
if "Power Loss Pie Chart" in show_charts:
    with st.expander("📉 Power Distribution and Loss"):
        labels = ['Useful Energy', 'Battery Loss', 'Unmet Demand']
        values = [
            df_new['energy_generated'].iloc[-1] - df_new['eff_loss'].sum() - df_new['unmet_demand'].sum(),
            df_new['eff_loss'].sum(),
            df_new['unmet_demand'].sum()
        ]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(title="Power Utilization Breakdown")
        st.plotly_chart(fig)

# Chart 7: Animated Power Flow
if "Animated Power Flow" in show_charts:
    with st.expander("🎞️ Power Flow Animation"):
        fig, ax = plt.subplots()
        line1, = ax.plot([], [], lw=2, label='Supply')
        line2, = ax.plot([], [], lw=2, label='Demand')
        ax.set_xlim(0, len(df_new))
        ax.set_ylim(0, max(df_new['total_gen'].max(), df_new['demand_kw'].max()) + 1)
        ax.set_title('Power Flow Over Time')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('kW')
        ax.legend()

        def animate(i):
            line1.set_data(range(i), df_new['total_gen'].iloc[:i])
            line2.set_data(range(i), df_new['demand_kw'].iloc[:i])
            return line1, line2

        ani = animation.FuncAnimation(fig, animate, frames=len(df_new), interval=50, blit=True)
        buf = BytesIO()
        ani.save(buf, format="gif")
        st.image(buf.getvalue(), caption="Real-Time Power Flow", use_column_width=True)

# NEW: Turbine Analysis Section
if "Turbine Analysis" in show_charts or show_turbine_sim:
    st.subheader("🌊 Hydro Turbine Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Turbine simulation
        turbine_data = turbine.simulate_operation(hours=turbine_hours)
        
        # Plot turbine performance
        fig_turbine = go.Figure()
        fig_turbine.add_trace(go.Scatter(x=turbine_data['time_step'], y=turbine_data['power_output'], 
                                       name='Power Output (kW)', line=dict(color='blue')))
        fig_turbine.add_trace(go.Scatter(x=turbine_data['time_step'], y=turbine_data['water_level'], 
                                       name='Water Level (%)', yaxis='y2', line=dict(color='cyan')))
        
        fig_turbine.update_layout(
            title="Turbine Performance Over Time",
            xaxis_title="Time Step (15-min intervals)",
            yaxis_title="Power (kW)",
            yaxis2=dict(title="Water Level (%)", overlaying='y', side='right')
        )
        st.plotly_chart(fig_turbine, use_container_width=True)
    
    with col2:
        # Turbine status and efficiency
        status = turbine.get_status()
        maintenance = turbine.maintenance_check()
        
        st.metric("Current Power Output", f"{status['power_output']:.1f} kW")
        st.metric("Current RPM", f"{status['rpm']:.0f}")
        st.metric("Water Level", f"{status['water_level']:.1f}%")
        st.metric("Efficiency", f"{status['efficiency']*100:.1f}%")
        
        # Maintenance status
        st.subheader("🔧 Maintenance Status")
        if maintenance['status'] == 'OK':
            st.success(f"✅ {maintenance['status']}")
        else:
            st.warning(f"⚠️ {maintenance['status']}")
            for issue in maintenance['issues']:
                st.write(f"• {issue}")

# NEW: Power System Analysis Section  
if "Power System Analysis" in show_charts or show_power_flow:
    st.subheader("⚡ Electrical Power System Analysis")
    
    # Update power system with current generation data
    latest_hydro = df_new['hydro_kw'].iloc[-1]
    latest_solar = df_new['solar_kw'].iloc[-1]
    
    power_system.add_generator('HYDRO', latest_hydro/1000, latest_hydro/1000*0.2)  # Convert to MW
    power_system.add_generator('SOLAR', latest_solar/1000, 0)
    
    # Generate power flow report
    report = power_system.generate_load_flow_report()
    
    if 'error' not in report:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 System Summary")
            summary = report['system_summary']
            st.metric("Total Generation", f"{summary['total_generation_mw']:.2f} MW")
            st.metric("Total Load", f"{summary['total_load_mw']:.2f} MW") 
            st.metric("System Losses", f"{summary['total_losses_mw']:.3f} MW")
            st.metric("Loss Percentage", f"{summary['loss_percentage']:.2f}%")
            
        with col2:
            st.subheader("🔌 Bus Voltages")
            bus_data = []
            for bus_id, bus_info in report['bus_results'].items():
                bus_data.append({
                    'Bus': bus_id,
                    'Voltage (p.u.)': f"{bus_info['voltage']:.3f}",
                    'Angle (deg)': f"{bus_info['angle']:.1f}",
                    'Status': '✅ Normal' if 0.95 <= bus_info['voltage'] <= 1.05 else '⚠️ Out of Range'
                })
            st.dataframe(pd.DataFrame(bus_data))
        
        # Power quality metrics
        pq = report['power_quality']
        st.subheader("📈 Power Quality Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Voltage", f"{pq['voltage_stats']['min_voltage']:.3f} p.u.")
        with col2:
            st.metric("Max Voltage", f"{pq['voltage_stats']['max_voltage']:.3f} p.u.")
        with col3:
            st.metric("Voltage Regulation", f"{pq['voltage_stats']['voltage_regulation']:.2f}%")
    else:
        st.error("❌ Power flow analysis failed to converge")

# Enhanced Summary with Module Data
st.subheader("📌 Comprehensive System Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Energy System**")
    st.markdown(f"**Final Battery SOC:** `{df_new['battery_soc'].iloc[-1]:.2f}%`")
    st.markdown(f"**Peak Demand:** `{df_new['demand_kw'].max():.2f} kW`")
    st.markdown(f"**Total Energy Generated:** `{df_new['total_gen'].sum():.2f} kWh`")

with col2:
    if show_turbine_sim:
        st.markdown("**Turbine Performance**")
        avg_power = turbine_data['power_output'].mean()
        avg_efficiency = turbine_data['efficiency'].mean()
        final_water = turbine_data['water_level'].iloc[-1]
        st.markdown(f"**Avg Turbine Output:** `{avg_power:.1f} kW`")
        st.markdown(f"**Avg Efficiency:** `{avg_efficiency*100:.1f}%`") 
        st.markdown(f"**Final Water Level:** `{final_water:.1f}%`")

with col3:
    if show_power_flow and 'error' not in report:
        st.markdown("**Power System**")
        st.markdown(f"**System Generation:** `{report['system_summary']['total_generation_mw']:.2f} MW`")
        st.markdown(f"**System Losses:** `{report['system_summary']['loss_percentage']:.2f}%`")
        voltage_range = f"{report['power_quality']['voltage_stats']['min_voltage']:.3f} - {report['power_quality']['voltage_stats']['max_voltage']:.3f}"
        st.markdown(f"**Voltage Range:** `{voltage_range} p.u.`")