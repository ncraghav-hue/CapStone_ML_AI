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
# Replace np.loadtxt with the custom parser
input_data = parse_data_file("inputs.txt")
output_data = parse_data_file("outputs.txt")

# The problem description indicates:
# `input_data` represents the raw features (`X_raw`) for all observations.
# `output_data` contains `y` values where each column corresponds to a different function (8 functions in total).

X_raw = input_data
# We assume output_data is a 2D array where each column is the output for a function.
# The original code's `func_ids = input_data[:, 0]` and `y_raw = output_data[:, 1]`
# are no longer directly applicable given the structure of `input_data` and `output_data` after custom parsing.
# The `func_ids` were previously used to group `X_raw` and `y_raw`.

# =============================
# GROUP BY FUNCTION (ADJUSTED)
# =============================
X_funcs, y_funcs = [], []
num_functions = 8 # As per the loop range in the original code

# For this setup, we don't need `func_ids` from the input file if `output_data` has 8 columns
# corresponding to the 8 functions. Each function will use the same `X_raw`.
for i in range(num_functions):
    # For each function, X_raw is the same set of input points
    X_funcs.append(X_raw)
    # The corresponding y values come from the i-th column of output_data
    y_funcs.append(output_data[:, i])

# =============================
# FUNCTION-SPECIFIC HYPERPARAMETERS (Decoding-inspired)
# =============================
def get_hyperparams(func_id, input_dim):
    if input_dim <= 4:
        hidden = [32,16]
        lr = 0.004
        epochs = 500
        noise_level = 1e-5
    elif input_dim <= 6:
        hidden = [64,32]
        lr = 0.0035
        epochs = 600
        noise_level = 1e-5
    else:
        hidden = [128,64,32]
        lr = 0.003
        epochs = 700
        noise_level = 1e-5
    xi = 0.02
    global_n, local_n = 5000, 2500
    if func_id == 2: noise_level = 1e-3
    if func_id == 5: xi = 0.001
    if func_id == 8: global_n = 7000
    return hidden, lr, epochs, xi, noise_level, global_n, local_n

# =============================
# SURROGATE MODEL
# =============================
class DecodingNN(nn.Module):
    def __init__(self,input_dim,hidden_layers):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim,h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim,1))
        self.net = nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)

def train_nn(X,y,hidden_layers,lr,epochs):
    model = DecodingNN(X.shape[1],hidden_layers)
    X_tensor = torch.tensor(X,dtype=torch.float32)
    y_tensor = torch.tensor(y.reshape(-1,1),dtype=torch.float32)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred,y_tensor)
        loss.backward()
        optimizer.step()
    return model

# =============================
# PROPOSE NEXT POINT
# =============================
def propose_next(X,y,func_id):
    dim = X.shape[1]
    hidden, lr, epochs, xi, noise_level, global_n, local_n = get_hyperparams(func_id,dim)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train NN surrogate
    nn_model = train_nn(X_scaled,y,hidden,lr,epochs)

    # GP on residuals
    nn_pred = nn_model(torch.tensor(X_scaled,dtype=torch.float32)).detach().numpy().flatten()
    residuals = y - nn_pred
    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=noise_level)
    gp = GaussianProcessRegressor(kernel=kernel,normalize_y=True)
    gp.fit(X_scaled,residuals)

    # Candidate generation
    X_global = np.random.uniform(0,1,(global_n,dim))
    X_local = X[np.argmax(y)] + 0.05*np.random.randn(local_n,dim)
    X_local = np.clip(X_local,0,1)
    X_candidates = np.vstack([X_global,X_local])
    X_candidates_scaled = scaler.transform(X_candidates)

    # Predictions
    nn_pred = nn_model(torch.tensor(X_candidates_scaled,dtype=torch.float32)).detach().numpy().flatten()
    gp_pred,gp_std = gp.predict(X_candidates_scaled,return_std=True)
    mu = nn_pred + gp_pred
    sigma = gp_std

    # Expected Improvement
    mu_sample = nn_model(torch.tensor(X_scaled,dtype=torch.float32)).detach().numpy().flatten() + gp.predict(X_scaled)
    mu_best = np.max(mu_sample)
    with np.errstate(divide='warn'):
        imp = mu - mu_best - xi
        Z = imp / sigma
        ei = imp*norm.cdf(Z) + sigma*norm.pdf(Z)
        ei[sigma==0.0] = 0.0

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
    X = X_funcs[i]
    y = y_funcs[i]
    x_next = propose_next(X,y,i+1)
    query = format_query(x_next)
    all_queries.append(query)
    print(f"Function {i+1} next query: {query}")

# =============================
# FINAL SUBMISSION
# =============================
print("\n===== SUBMIT THESE =====")
for i,q in enumerate(all_queries,1):
    print(f"Function {i}: {q}")