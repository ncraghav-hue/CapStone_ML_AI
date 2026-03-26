import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
import ast
import re # Import re module for regular expressions

# Helper function for recursive flattening
def flatten(l):
    for el in l:
        if isinstance(el, list) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

# =============================
# CUSTOM DATA PARSER
# =============================
def parse_data_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    parsed_entries_raw = [] # This will hold the result of ast.literal_eval for each logical entry
    current_entry_str = ""
    for line in lines:
        current_entry_str += line.strip()
        
        # Aggressively clean the string for ast.literal_eval using regex
        # Replace 'array(value)' with '[value]' for numpy array representations
        # Replace 'np.float64(value)' with 'value' for numpy float representations
        cleaned_str = re.sub(r'array\((.*?)\)', r'[\1]', current_entry_str)
        cleaned_str = re.sub(r'np\.float64\((.*?)\)', r'\1', cleaned_str)

        try:
            val = ast.literal_eval(cleaned_str)
            parsed_entries_raw.append(val)
            current_entry_str = "" # Reset for next entry
        except (SyntaxError, ValueError):
            # If parsing fails, it's likely a multi-line entry or an incomplete string, continue accumulating
            pass

    # Now, process parsed_entries_raw to ensure a consistent 2D structure
    final_structured_data = []
    for entry in parsed_entries_raw:
        if isinstance(entry, list):
            # Recursively flatten all nested lists into a single row
            flat_row = list(flatten(entry))
            if flat_row: # Only add row if it has content
                final_structured_data.append(flat_row)
        elif isinstance(entry, (int, float)):
            final_structured_data.append([entry]) # Wrap single numbers in a list to form a row
        else: # Handle other potential direct scalar values that ast.literal_eval might return
            try:
                final_structured_data.append([float(entry)])
            except (ValueError, TypeError):
                # This case should ideally not be hit with good input files
                pass
            
    if not final_structured_data:
        return np.array([])

    # Ensure all rows have the same dimension, padding with 0.0 if necessary
    max_dim = 0
    if final_structured_data:
        max_dim = max(len(row) for row in final_structured_data)

    padded_data = []
    for row in final_structured_data:
        current_row = [float(x) for x in row] # Ensure all elements are floats
        if len(current_row) < max_dim:
            current_row.extend([0.0] * (max_dim - len(current_row)))
        padded_data.append(current_row)

    return np.array(padded_data)


# =============================
# LOAD TXT DATA
# =============================
# Use the refactored parse_data_file for both inputs and outputs
input_data = parse_data_file("inputs.txt")
output_data = parse_data_file("outputs.txt")

# Assuming input_data is X_raw and output_data contains multiple columns for y_raw for each func.
# The problem states "func_ids = input_data[:, 0].astype(int)" and `X_raw = input_data[:, 1:]` was part of the original code.
# However, it also states that `func_ids` is not in `input_data[:, 0]`. And `output_data` has 8 columns.

# Let's assume `input_data` is directly `X_raw` for all data points.
X_raw = input_data
# Let's assume `output_data` has 8 columns, one for each function.
# We need to map `X_raw` rows to `y_raw` rows.
# If `output_data` has 8 columns, then `y_raw` should be `output_data[:, i-1]` for each function.

# The previous context says: `output_data` contained `y` values where each column corresponded to a different function (8 functions in total).
# This means output_data is (num_samples, 8).
# And `X_raw` is (num_samples, num_features).
# So we do not need `func_ids` from `input_data`.
# We can create `func_ids` synthetically based on the original grouping logic.

# Let's re-align with the code's original `GROUP BY FUNCTION` section.
# The code iterates `for i in range(1, 9):` (8 functions).
# It expects `X_raw` and `y_raw` to be aligned.
# If `output_data` has 8 columns, then `y_raw` should be `output_data[:, i-1]` for each function.

# Let's create a dummy func_ids for now, assuming all X_raw rows are for the first function
# This will be corrected in the grouping section.
# It's more likely that X_raw and output_data have the same number of rows,
# and each row in output_data corresponds to the respective row in X_raw.

# To make the original `GROUP BY FUNCTION` section work, we need `func_ids` and `X_raw`, `y_raw`.
# From the problem description: `output_data` contained `y` values where each column corresponded to a different function.
# This means we should have 8 `y_raw` arrays, one for each function.

# Let's assume the `input_data` (now `X_raw`) is the set of all input features.
# And `output_data` (which is `parse_data_file("outputs.txt")`) is a 2D array where each column is the output for a function.

# =============================
# GROUP BY FUNCTION (ADJUSTED)
# =============================
X_funcs, y_funcs = [], []
num_functions = 8 # As per the loop range

# Assuming X_raw contains all input features and output_data contains all output values (8 columns for 8 functions)
# For this setup, we don't need `func_ids` to mask `X_raw` if all `X_raw` points are considered for each function.
# Instead, we just pair `X_raw` with the correct column from `output_data`.

for i in range(num_functions):
    # For each function, X_raw is the same set of input points
    X_funcs.append(X_raw)
    # The corresponding y values come from the i-th column of output_data
    y_funcs.append(output_data[:, i])

# The rest of the code should now work as `X_funcs` and `y_funcs` are populated correctly.

# =============================
# SETTINGS
# =============================
def get_settings(func_id):
    xi = 0.02
    noise = 1e-5
    global_n = 5000
    local_n = 2500
    # func_id is 1-indexed for functions
    if func_id == 2:  # noisy function
        noise = 1e-3
    if func_id == 5:  # unimodal
        xi = 0.001
    if func_id == 8:  # high-dimensional
        global_n = 7000
    return xi, noise, global_n, local_n

# =============================
# CNN-INSPIRED NEURAL NETWORK SURROGATE
# =============================
class CNNInspiredNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Adjust depth based on input dimension (like CNN trade-offs)
        if input_dim <= 4:
            hidden = [32, 16]
        elif input_dim <= 6:
            hidden = [64, 32]
        else:
            hidden = [128, 64, 32]

        layers = []
        in_dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_nn(X, y, epochs=600, lr=0.005):
    device = torch.device('cpu')
    model = CNNInspiredNN(X.shape[1]).to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.reshape(-1,1), dtype=torch.float32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
    return model

# =============================
# PROPOSE NEXT POINT
# =============================
def propose_next(X, y, func_id):
    dim = X.shape[1]
    xi, noise, global_n, local_n = get_settings(func_id)

    # ---- Standardize ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Train CNN-inspired surrogate ----
    nn_model = train_nn(X_scaled, y, epochs=600)

    # ---- GP on residuals ----
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    nn_pred = nn_model(X_tensor).detach().numpy().flatten()
    residuals = y - nn_pred

    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=noise)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(X_scaled, residuals)

    # ---- Candidate generation ----
    X_global = np.random.uniform(0, 1, size=(global_n, dim))
    X_global_scaled = scaler.transform(X_global)

    best_x = X[np.argmax(y)]
    X_local = best_x + 0.05 * np.random.randn(local_n, dim)
    X_local = np.clip(X_local, 0, 1)
    X_local_scaled = scaler.transform(X_local)

    X_candidates_scaled = np.vstack([X_global_scaled, X_local_scaled])
    X_candidates = np.vstack([X_global, X_local])

    # ---- Predictions ----
    X_candidates_tensor = torch.tensor(X_candidates_scaled, dtype=torch.float32)
    nn_pred = nn_model(X_candidates_tensor).detach().numpy().flatten()
    gp_pred, gp_std = gp.predict(X_candidates_scaled, return_std=True)
    mu = nn_pred + gp_pred
    sigma = gp_std

    # ---- Expected Improvement ----
    mu_sample = nn_model(torch.tensor(X_scaled, dtype=torch.float32)).detach().numpy().flatten() + gp.predict(X_scaled)
    mu_best = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_best - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return X_candidates[np.argmax(ei)]

# =============================
# FORMAT QUERY
# =============================
def format_query(x):
    return "-".join([f"{xi:.6f}" for xi in x])

# =============================
# MAIN LOOP
# =============================
all_queries = []

for i in range(8):
    print(f"\n===== Function {i+1} =====")
    X = X_funcs[i]
    y = y_funcs[i]

    print("Data points:", len(y))

    x_next = propose_next(X, y, i+1)
    query = format_query(x_next)
    all_queries.append(query)

    print("Next query:", query)

# =============================
# FINAL SUBMISSION
# =============================
print("\n===== SUBMIT THESE =====")
for i, q in enumerate(all_queries, 1):
    print(f"Function {i}: {q}")