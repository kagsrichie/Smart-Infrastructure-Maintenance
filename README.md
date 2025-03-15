# Smart-infrastructure-maintenance
A machine learning system for predictive maintenance of industrial infrastructure
# Smart Infrastructure Predictive Maintenance

A comprehensive machine learning system for predicting equipment failures and optimizing maintenance schedules in industrial and infrastructure settings.

![Predictive Maintenance Dashboard](images/dashboard_preview.png)

## Overview

This project implements a complete predictive maintenance solution that helps infrastructure operators detect potential equipment failures before they occur, reducing downtime and maintenance costs while improving safety. The system uses machine learning to analyze sensor data and operational parameters, identifying patterns that precede failures.

### Key Features

- **Multi-Model Comparison**: Automatically trains and compares Random Forest, Gradient Boosting, and XGBoost models
- **Time-Series Analysis**: Evaluates model performance over time to ensure stability
- **Intelligent Maintenance Scheduling**: Generates optimized maintenance schedules based on risk assessment
- **Feature Importance Analysis**: Identifies which sensors and parameters best predict failures
- **Interactive Dashboards**: HTML dashboards for monitoring model performance and maintenance needs
- **Full Data Pipeline**: From data preprocessing to model deployment and maintenance scheduling

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/smart-infrastructure-maintenance.git
cd smart-infrastructure-maintenance

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from smart_infrastructure.predictive_maintenance import PredictiveMaintenanceSystem

# Initialize the system
maintenance_system = PredictiveMaintenanceSystem(
    data_path="data/maintenance_data.csv", 
    model_save_path="models", 
    results_path="results"
)

# Explore and preprocess data
maintenance_system.explore_data()
maintenance_system.preprocess_data(target_col="failure")

# Train models
maintenance_system.train_models()

# Analyze feature importance
feature_importance = maintenance_system.analyze_feature_importance()

# Create maintenance schedule
schedule = maintenance_system.create_maintenance_schedule(
    new_data=test_data,
    time_col="timestamp",
    id_col="equipment_id",
    risk_threshold=0.7
)

# Generate deployment dashboard
maintenance_system.deployment_dashboard()
```

## Data Requirements

The system works with time-series data containing:

- **Equipment identifiers**: Unique IDs for each piece of equipment
- **Timestamps**: When measurements were taken
- **Sensor readings**: Measurements from various sensors (temperature, pressure, vibration, etc.)
- **Operational parameters**: Usage hours, age, maintenance history, etc.
- **Failure indicators**: Binary target variable indicating equipment failure

Example dataset structure:

| timestamp | equipment_id | sensor_temp | sensor_pressure | maintenance_since_days | failure |
|-----------|--------------|-------------|-----------------|------------------------|---------|
| 2023-01-01 08:00 | EQ001 | 65.2 | 102.3 | 45 | 0 |
| 2023-01-01 09:00 | EQ002 | 82.7 | 89.5 | 120 | 1 |

## Comprehensive Documentation

### Data Preprocessing

The system handles:
- Missing value imputation
- Feature scaling
- Categorical encoding
- Datetime feature extraction
- Feature selection

```python
maintenance_system.preprocess_data(
    target_col='failure',
    categorical_cols=['equipment_id', 'equipment_type', 'location'],
    numerical_cols=['sensor_temp', 'sensor_pressure', 'sensor_vibration'],
    drop_cols=['notes']
)
```

### Model Training

Train and compare multiple models:

```python
# Full parameter grid search
maintenance_system.train_models(
    models_to_train=['rf', 'gb', 'xgb'],
    param_grids={
        'rf': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }
)
```

### Model Evaluation

Evaluate model performance with detailed metrics:

```python
# Evaluate over time
maintenance_system.evaluate_model_over_time(
    time_col='timestamp',
    freq='M',  # Monthly
    rolling_window=3
)
```

### Maintenance Scheduling

Generate risk-based maintenance schedules:

```python
maintenance_system.create_maintenance_schedule(
    new_data=current_readings,
    time_col='timestamp',
    id_col='equipment_id',
    risk_threshold=0.6,
    lead_time_days=7
)
```

## Example Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| RF    | 0.92     | 0.89      | 0.86   | 0.88     | 0.96    |
| XGB   | 0.93     | 0.91      | 0.87   | 0.89     | 0.97    |

### Feature Importance

![Feature Importance](images/feature_importance.png)

### Maintenance Schedule

![Maintenance Schedule](images/maintenance_schedule.png)

## Use Cases

- **Industrial Equipment**: Predicting failures in pumps, motors, compressors, and valves
- **Building Systems**: HVAC, elevators, electrical systems, and water management
- **Infrastructure**: Bridges, tunnels, power grids, and transportation systems
- **Manufacturing**: Production lines, CNC machines, and quality control systems
- **Energy**: Wind turbines, solar inverters, transformers, and power plants

## Customization

The system is designed to be modular and adaptable. You can:

- Add new models by extending the `train_models` method
- Implement custom preprocessing steps for specific data types
- Integrate with existing maintenance management systems
- Add specialized visualizations for domain-specific insights

## Project Structure

```
smart-infrastructure-maintenance/
├── data/                  # Data storage
├── models/                # Saved models
├── results/               # Analysis results and visualizations
├── smart_infrastructure/  # Main package
│   ├── __init__.py
│   ├── predictive_maintenance.py  # Core system
│   └── utils/
│       ├── preprocessing.py
│       ├── visualization.py
│       └── evaluation.py
├── examples/              # Example scripts
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your work or research, please cite:

```
@software{Smart_Infrastructure_Predictive_Maintenance,
  author = {Your Name},
  title = {Smart Infrastructure Predictive Maintenance},
  year = {2025},
  url = {https://github.com/your-username/smart-infrastructure-maintenance}
}
```
