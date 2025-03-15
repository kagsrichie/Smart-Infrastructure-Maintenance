import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from smart_infrastructure.predictive_maintenance import PredictiveMaintenanceSystem

def generate_synthetic_data(n_samples=1000, n_equipment=50, start_date='2022-01-01', end_date='2023-01-01'):
    """
    Generate synthetic data for predictive maintenance example.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_equipment : int
        Number of unique equipment IDs
    start_date : str
        Start date for data generation
    end_date : str
        End date for data generation
        
    Returns:
    --------
    pandas.DataFrame
        Synthetic dataset
    """
    # Generate equipment IDs
    equipment_ids = [f"EQ{i:03d}" for i in range(1, n_equipment + 1)]
    
    # Generate timestamps
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, periods=n_samples)
    
    # Initialize data dictionary
    data = {
        'timestamp': np.random.choice(dates, n_samples),
        'equipment_id': np.random.choice(equipment_ids, n_samples),
        'sensor_temp': np.random.normal(65, 15, n_samples),
        'sensor_pressure': np.random.normal(100, 20, n_samples),
        'sensor_vibration': np.random.normal(0.5, 0.2, n_samples),
        'sensor_rotation': np.random.normal(1000, 200, n_samples),
        'sensor_voltage': np.random.normal(220, 30, n_samples),
        'sensor_current': np.random.normal(50, 10, n_samples),
        'operational_hours': np.random.randint(0, 24, n_samples),
        'maintenance_since_days': np.random.randint(0, 365, n_samples),
        'equipment_age_years': np.random.uniform(0, 10, n_samples)
    }
    
    # Create the DataFrame
    df = pd.DataFrame(data)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Generate the target variable (failure)
    # Higher probability of failure when:
    # - Temperature is high
    # - Vibration is high
    # - Equipment age is high
    # - Days since maintenance is high
    
    # Normalize features for the failure probability calculation
    temp_norm = (df['sensor_temp'] - df['sensor_temp'].min()) / (df['sensor_temp'].max() - df['sensor_temp'].min())
    vib_norm = (df['sensor_vibration'] - df['sensor_vibration'].min()) / (df['sensor_vibration'].max() - df['sensor_vibration'].min())
    age_norm = df['equipment_age_years'] / 10
    maint_norm = df['maintenance_since_days'] / 365
    
    # Calculate failure probability
    failure_prob = 0.3 * temp_norm + 0.3 * vib_norm + 0.2 * age_norm + 0.2 * maint_norm
    
    # Add some randomness
    failure_prob = failure_prob + np.random.normal(0, 0.1, n_samples)
    failure_prob = np.clip(failure_prob, 0, 1)
    
    # Generate binary failure indicator (1 = failure, 0 = no failure)
    df['failure'] = (failure_prob > 0.7).astype(int)
    
    # Add equipment type
    equipment_types = ['Pump', 'Motor', 'Valve', 'Compressor', 'Generator']
    df['equipment_type'] = np.random.choice(equipment_types, n_samples)
    
    # Add location
    locations = ['Building A', 'Building B', 'Building C', 'Outdoor', 'Basement']
    df['location'] = np.random.choice(locations, n_samples)
    
    # Add weather condition
    weather_conditions = ['Normal', 'Hot', 'Cold', 'Humid', 'Dry']
    df['weather_condition'] = np.random.choice(weather_conditions, n_samples)
    
    return df

def main():
    """
    Main function to demonstrate the Predictive Maintenance System.
    """
    print("Generating synthetic data for predictive maintenance example...\n")
    
    # Generate synthetic data
    data = generate_synthetic_data(n_samples=10000, n_equipment=100)
    
    # Save data to CSV for later use
    data.to_csv("data/maintenance_data.csv", index=False)
    print(f"Data generated and saved to data/maintenance_data.csv")
    print(f"Shape: {data.shape}")
    print(f"Failure rate: {data['failure'].mean():.2%}\n")
    
    # Initialize the predictive maintenance system
    print("Initializing the predictive maintenance system...\n")
    maintenance_system = PredictiveMaintenanceSystem(data_path=data, 
                                                    model_save_path='models', 
                                                    results_path='results')
    
    # Explore the data
    print("Exploring the data...\n")
    data_info = maintenance_system.explore_data(save_plots=True)
    
    # Preprocess the data
    print("\nPreprocessing the data...\n")
    categorical_cols = ['equipment_id', 'equipment_type', 'location', 'weather_condition']
    numerical_cols = ['sensor_temp', 'sensor_pressure', 'sensor_vibration', 'sensor_rotation',
                      'sensor_voltage', 'sensor_current', 'operational_hours', 
                      'maintenance_since_days', 'equipment_age_years']
    
    X_train, X_test, y_train, y_test = maintenance_system.preprocess_data(
        target_col='failure',
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        drop_cols=['timestamp']  # We'll extract time features
    )
    
    # Train models
    print("\nTraining models...\n")
    
    # Define smaller parameter grids for demonstration
    param_grids = {
        'rf': {
            'n_estimators': [100],
            'max_depth': [10],
            'min_samples_split': [5]
        },
        'xgb': {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [5]
        }
    }
    
    models = maintenance_system.train_models(models_to_train=['rf', 'xgb'], param_grids=param_grids)
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...\n")
    feature_imp = maintenance_system.analyze_feature_importance(top_n=15)
    print("\nTop 10 features:")
    print(feature_imp.head(10))
    
    # Evaluate model over time
    print("\nEvaluating model performance over time...\n")
    time_metrics = maintenance_system.evaluate_model_over_time(time_col='timestamp', freq='M')
    
    # Generate some new data for prediction
    print("\nGenerating new data for predictions...\n")
    new_data = generate_synthetic_data(n_samples=500, n_equipment=20, 
                                       start_date='2023-01-01', end_date='2023-02-01')
    
    # Create maintenance schedule
    print("\nCreating maintenance schedule...\n")
    maintenance_schedule = maintenance_system.create_maintenance_schedule(
        new_data=new_data,
        time_col='timestamp',
        id_col='equipment_id',
        risk_threshold=0.6,
        lead_time_days=5
    )
    
    print("\nMaintenance schedule (top 10 highest risk):")
    print(maintenance_schedule[['equipment_id', 'failure_probability', 
                               'risk_category', 'recommended_maintenance_date']].head(10))
    
    # Save the system for later use
    print("\nSaving the predictive maintenance system...\n")
    system_path = maintenance_system.save_system()
    
    # Create deployment dashboard
    print("\nCreating deployment dashboard...\n")
    dashboard_path = maintenance_system.deployment_dashboard()
    print(f"Dashboard created at: {dashboard_path}")
    
    print("\nDone! The predictive maintenance system is ready for use.")
    
if __name__ == "__main__":
    # Create directories if they don't exist
    import os
    for directory in ['data', 'models', 'results']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    main()
