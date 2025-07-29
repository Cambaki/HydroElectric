# ⚡ HydroPower Pro - Advanced Energy Management System

**Powered by Cambaki Energy Solutions**

A comprehensive hydroelectric power generation simulation and monitoring system built with Python and Streamlit. Features real-time data visualization, machine learning demand prediction, turbine physics modeling, and electrical power system analysis.

## 🌟 Features

### 📊 **Real-Time Dashboard**
- Interactive power source visualization (Hydro, Solar, Wind, Diesel)
- Battery state-of-charge monitoring with gauges
- Supply vs demand analysis with ML predictions
- Energy flow animations and correlations
- Power loss and efficiency tracking

### 🤖 **Machine Learning**
- Demand prediction using Random Forest algorithm
- Historical data training with time-series features
- Real-time prediction accuracy monitoring
- Automated model retraining capabilities

### 🌊 **Turbine Physics Simulation**
- Realistic hydro turbine modeling (P = ρ × g × Q × H × η)
- Water level and flow rate optimization
- RPM monitoring and efficiency curves
- Maintenance alerts and performance tracking
- Seasonal and variable inflow patterns

### ⚡ **Electrical Power System Analysis**
- Load flow analysis using Newton-Raphson method
- Bus voltage monitoring and stability analysis
- Power quality metrics (THD, voltage regulation)
- N-1 contingency analysis
- Microgrid modeling and optimization

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Windows OS (batch files included)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/hydro-generator-project.git
   cd hydro-generator-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run main.py
   ```
   *Or double-click `run_dashboard.bat`*

## 📁 Project Structure

```
hydro-generator-project/
├── main.py                    # Main Streamlit dashboard
├── hydro_ml_simulation.py     # ML models and simulation engine
├── dashboard.py               # Alternative dashboard (optional)
├── requirements.txt           # Python dependencies
├── run_dashboard.bat          # Windows batch file to run app
├── setup_git.bat             # Git configuration helper
├── modules/
│   ├── visualization.py      # Advanced plotting and charts
│   ├── turbine.py            # Hydro turbine physics modeling
│   └── load_flow.py          # Electrical power system analysis
├── data/                     # Data storage (if any)
└── outputs/                  # Generated reports and exports
```

## 🎯 Usage

### Basic Dashboard
1. **Launch the app** using `run_dashboard.bat` or `streamlit run main.py`
2. **Configure simulation** using the sidebar controls:
   - Adjust simulation days (1-14)
   - Select visualizations to display
   - Enable advanced analysis options

### Advanced Features
- **Turbine Analysis**: Enable "Turbine Simulation" for detailed physics modeling
- **Power System Analysis**: Enable "Power Flow Analysis" for electrical engineering calculations
- **ML Predictions**: Automatic demand forecasting with accuracy metrics

### Key Visualizations
- **Power Sources**: Real-time generation by source type
- **Battery Management**: SOC gauges and flow monitoring  
- **Demand Prediction**: ML model accuracy and forecasting
- **System Efficiency**: Performance metrics and loss analysis
- **Turbine Performance**: RPM, water levels, and maintenance alerts
- **Electrical Analysis**: Voltage stability and power quality

## 🔧 Configuration

### Turbine Parameters
Modify in `modules/turbine.py`:
```python
turbine = HydroTurbine(
    rated_power=50,      # kW
    efficiency=0.85,     # 85%
    head=10,            # meters
    flow_rate=100       # L/s
)
```

### Power System Setup
Configure in `modules/load_flow.py`:
- Bus definitions and connections
- Generator and load specifications
- Transmission line parameters

## 📈 Technical Details

### Machine Learning Model
- **Algorithm**: Random Forest Regressor
- **Features**: Time patterns, weather data, historical demand
- **Training**: Continuous learning from simulation data
- **Accuracy**: Real-time RMSE and correlation tracking

### Turbine Physics
- **Power Calculation**: P = ρ × g × Q × H × η
- **Efficiency Modeling**: Variable efficiency curves
- **Water Management**: Reservoir level simulation
- **Optimization**: Flow rate optimization for target power

### Electrical Analysis
- **Load Flow**: Newton-Raphson iterative solver
- **Voltage Stability**: Real-time stability margins
- **Power Quality**: THD analysis and voltage regulation
- **Contingency**: N-1 outage analysis

## 🛠️ Development

### Adding New Features
1. **Visualizations**: Extend `modules/visualization.py`
2. **Turbine Models**: Modify `modules/turbine.py`
3. **Power System**: Update `modules/load_flow.py`
4. **ML Models**: Enhance `hydro_ml_simulation.py`

### Testing
Run the dashboard with different configurations:
- Various simulation days and time periods
- Different turbine parameters
- Multiple power system scenarios

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
plotly>=5.15.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎉 Acknowledgments

- Built with Streamlit for rapid dashboard development
- Uses scikit-learn for machine learning capabilities
- Plotly and Matplotlib for interactive visualizations
- Scipy for advanced engineering calculations

## 📞 Contact

**Cambaki** - Clambak874@gmail.com

Project Link: [https://github.com/Cambaki/HydroElectric](https://github.com/Cambaki/HydroElectric)

---
*Built with ❤️ for sustainable energy monitoring and analysis*
