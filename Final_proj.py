import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates

# Import necessary modules from scikit-learn for data preprocessing and modeling
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

# Import regression models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Import utilities for model evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import FitFailedWarning

# Import necessary modules for multi-objective optimization
from desdeo_problem import Variable, ScalarObjective, MOProblem
from desdeo_emo.EAs.RVEA import RVEA

import joblib
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Load preprocessed datasets
data_elem_path = '/Users/sepuliini/Desktop/DDO_kurssi/cleaned_data_elem.xlsx'
data_ys_path = '/Users/sepuliini/Desktop/DDO_kurssi/cleaned_data_ys.xlsx'
data_uts_path = '/Users/sepuliini/Desktop/DDO_kurssi/cleaned_data_uts.xlsx'
data_el_path = '/Users/sepuliini/Desktop/DDO_kurssi/cleaned_data_el.xlsx'

data_elem = pd.read_excel(data_elem_path)
data_ys = pd.read_excel(data_ys_path)
data_uts = pd.read_excel(data_uts_path)
data_el = pd.read_excel(data_el_path)

# Function to evaluate models with hyperparameter tuning using GridSearchCV
def evaluate_models_with_hyperparameters(data, targets, model_dict, param_grid):
    results = {}
    for target in targets:
        results[target] = {}
        X = data.drop(target, axis=1)  # Features
        y = data[target]  # Target variable
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for model_name, model in model_dict.items():
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=FitFailedWarning)
                try:
                    # Perform hyperparameter tuning using GridSearchCV
                    grid_search = GridSearchCV(estimator=model, param_grid=param_grid[model_name], 
                                               cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                except FitFailedWarning:
                    # Handle cases where model fitting fails
                    print(f"Fit failed for {model_name} on target {target}. Setting score to NaN.")
                    results[target][model_name] = {'MSE': float('nan'), 'R2': float('nan')}
                    continue
            
                # Best model from GridSearchCV
                best_model = grid_search.best_estimator_
                
                # Predict on the test set using the best model
                y_pred = best_model.predict(X_test)
                
                # Calculate MSE and R2 and store the results
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results[target][model_name] = {'MSE': mse, 'R2': r2, 'best_model': best_model, 'best_params': grid_search.best_params_}
                print(f'MSE for {target} using {model_name}: {mse}')
                print(f'R2 for {target} using {model_name}: {r2}')
    
    return results

# Define targets (output variables)
targets = ['YS', 'UTS', 'EL']

# Dictionary to hold models
model_dict = {
    "Ada": AdaBoostRegressor(random_state=42),
    "KNR": KNeighborsRegressor(),
    "DTR": DecisionTreeRegressor(random_state=42),
    "RFR": RandomForestRegressor(n_estimators=100, random_state=42),
    "GBR": GradientBoostingRegressor(random_state=42),
    "XGB": XGBRegressor(random_state=42)
}

# Hyperparameter grid for each model
param_grid = {
    'Ada': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
    'KNR': {'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance']},
    'DTR': {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 10, 20]},
    'RFR': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
    'GBR': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
    'XGB': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
}

# Combine the datasets for modeling
combined_data = {
    'YS': data_ys,
    'UTS': data_uts,
    'EL': data_el
}

# Run evaluation for each model and target
model_performance = {}
for target, data in combined_data.items():
    model_performance[target] = evaluate_models_with_hyperparameters(data, [target], model_dict, param_grid)

# Store best models and their parameters for each objective
best_models = {}
for target in targets:
    best_models[target] = {}
    best_mse = float('inf')
    best_model_info = None
    for model_name, model_info in model_performance[target][target].items():
        if model_info['MSE'] < best_mse:
            best_mse = model_info['MSE']
            best_model_info = model_info
    best_models[target] = best_model_info

# Print the best performing surrogate technique and its parameters for each objective
for target in targets:
    print(f"\nBest performing surrogate technique and parameters for '{target}':")
    print(f"Model: {best_models[target]['best_model']}")
    print(f"Parameters: {best_models[target]['best_params']}")
    print(f"MSE: {best_models[target]['MSE']}")
    print(f"R2: {best_models[target]['R2']}")

# Train final models and perform optimization using DESDEO RVEA
models = {}

for target in targets:
    X = combined_data[target].drop(target, axis=1)
    y = combined_data[target][target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train final model for the target using the best model and parameters
    best_model = best_models[target]['best_model']
    best_params = best_models[target]['best_params']
    
    final_model = GridSearchCV(best_model, param_grid={k: [v] for k, v in best_params.items()}, cv=5, n_jobs=-1)
    final_model.fit(X_train, y_train)
    models[target] = final_model

print("Training done, proceeding to optimization")

# Define objectives for the optimization problem
def objective_1(x):
    return models['YS'].predict(x)

def objective_2(x):
    return -models['UTS'].predict(x)

def objective_3(x):
    return -models['EL'].predict(x)

# Define scalar objectives
scalar_obj_1 = ScalarObjective("Objective 1", objective_1)  # Minimize YS
scalar_obj_2 = ScalarObjective("Objective 2", objective_2)  # Maximize UTS
scalar_obj_3 = ScalarObjective("Objective 3", objective_3)  # Maximize EL
objectives = [scalar_obj_1, scalar_obj_2, scalar_obj_3]

# Define decision variables
variables = []
for col in X.columns:
    variable = Variable(col, lower_bound=0.3, upper_bound=1.0, initial_value=0.5)
    variables.append(variable)

# Initialize the multi-objective problem
problem = MOProblem(variables=variables, objectives=objectives)

# Initialize the RVEA evolver
evolver = RVEA(problem, 
               interact=False,
               n_iterations=10,
               n_gen_per_iter=100,
               population_size=100)

# Run the optimization process
while evolver.continue_evolution():
    evolver.iterate()
print("Optimization done")

# Retrieve and print the solutions
individuals, solutions, archive = evolver.end() 
print(solutions)

### Visualization ###

# Convert solutions to DataFrame for visualization
solutions_df = pd.DataFrame(solutions, columns=['YS', 'UTS', 'EL'])

# Normalize the data for parallel coordinates plot
solutions_df_normalized = (solutions_df - solutions_df.min()) / (solutions_df.max() - solutions_df.min())

# Add an identifier column for parallel coordinates plot
solutions_df_normalized['ID'] = range(len(solutions_df_normalized))

# Create parallel coordinates plot
plt.figure(figsize=(12, 6))
parallel_coordinates(solutions_df_normalized, 'ID', cols=['YS', 'UTS', 'EL'], color=plt.cm.tab10(range(len(solutions_df_normalized))))
plt.title('Parallel Coordinates Plot of Pareto Front Solutions')
plt.xlabel('Objectives')
plt.ylabel('Normalized Value')
plt.legend([], frameon=False)  # Remove the legend
plt.show()

# Visualize the Pareto front in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the Pareto front
ax.scatter(solutions[:, 0], solutions[:, 1], solutions[:, 2], c='r', marker='o')

# Set labels
ax.set_xlabel('YS')
ax.set_ylabel('UTS')
ax.set_zlabel('EL')

# Set title
ax.set_title('Pareto Front')

# Show plot
plt.show()