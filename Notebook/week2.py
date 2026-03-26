import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.special import expit  # logistic function
from scipy.stats import norm
import ast # Added to parse string representations of lists/arrays

# =============================
# LOAD DATA
# =============================

def parse_data_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read().replace('array(', '').replace('np.float64(', '').replace(')', '') # Remove 'array(', 'np.float64(' and ')'

    # Split by ']), [' to get individual records (which might still be multi-line)
    # and then re-join parts that belong to the same record
    records = []
    current_record_parts = []
    for line in content.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith('['):
            # New record starts
            if current_record_parts:
                records.append(''.join(current_record_parts))
            current_record_parts = [stripped_line]
        elif stripped_line:
            # Continuation of the current record
            current_record_parts.append(stripped_line)
    if current_record_parts:
        records.append(''.join(current_record_parts))

    processed_data = []
    for record_str in records:
        try:
            # ast.literal_eval can parse the string into a list of lists (or tuples) or a list of floats
            parsed_record = ast.literal_eval(record_str.replace('\n', ''))

            flattened_record = []
            if isinstance(parsed_record, (list, tuple)):
                # Iterate through elements of the parsed record
                for element in parsed_record:
                    if isinstance(element, (list, tuple)):
                        # If an element is a sublist/tuple, iterate through its items
                        for item in element:
                            flattened_record.append(float(item))
                    elif isinstance(element, (int, float)):
                        # If an element is a single float/int, just add it
                        flattened_record.append(float(element))
                    else:
                        raise TypeError(f"Unexpected element type in parsed_record: {type(element)}")
            elif isinstance(parsed_record, (int, float)):
                # If the entire record is a single float/int, wrap it in a list
                flattened_record.append(float(parsed_record))
            else:
                raise TypeError(f"Unexpected type for parsed_record: {type(parsed_record)}")

            processed_data.append(flattened_record)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing record: {record_str[:100]}... Error: {e}")
            raise

    return np.array(processed_data)

input_data = parse_data_file("inputs.txt")
output_data = parse_data_file("outputs.txt")

# Assuming input_data contains X for all functions, and output_data contains y for all functions,
# where each column of output_data corresponds to a different function.
# The problem states there are 8 functions.
X_raw = input_data
y_all_funcs = output_data

# =============================
# GROUP BY FUNCTION
# =============================
X_funcs, y_funcs = [], []

# The loop iterates from i = 0 to 7, corresponding to function IDs 1 to 8.
for i in range(8):
    # For each function, use all X_raw data points
    X_funcs.append(X_raw)
    # And the corresponding column from y_all_funcs as the target y
    y_funcs.append(y_all_funcs[:, i])

# =============================
# POLYNOMIAL LINEAR REGRESSION
# =============================
def fit_poly(X, y, degree=2):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    return model, poly

# =============================
# SETTINGS
# =============================
def get_settings(func_id):
    xi = 0.02
    noise = 1e-5
    global_n = 4000
    local_n = 2000

    if func_id == 2:  # noisy
        noise = 1e-3
    if func_id == 5:  # unimodal
        xi = 0.001
    if func_id == 8:  # high-dim
        global_n = 6000

    return xi, noise, global_n, local_n

# =============================
# PROPOSE NEXT POINT
# =============================
def propose_next(X, y, func_id):
    dim = X.shape[1]
    xi, noise, global_n, local_n = get_settings(func_id)

    # ---- Step 1: Linear/Polynomial Trend ----
    poly_model, poly = fit_poly(X, y)
    y_trend = poly_model.predict(poly.transform(X))

    # ---- Step 2: Logistic-style scaling (ranking idea) ----
    # Convert outputs into "probability-like" scores
    y_scaled = (y - np.mean(y)) / (np.std(y) + 1e-8)
    y_prob = expit(y_scaled)  # values between 0 and 1

    # Residuals (what GP learns)
    residuals = y - y_trend

    # ---- Step 3: Gaussian Process ----
    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=noise)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(X, residuals)

    # ---- Step 4: Candidate generation ----
    X_global = np.random.uniform(0, 1, size=(global_n, dim))

    best_x = X[np.argmax(y)]
    X_local = best_x + 0.05 * np.random.randn(local_n, dim)
    X_local = np.clip(X_local, 0, 1)

    X_candidates = np.vstack([X_global, X_local])

    # ---- Step 5: Predictions ----
    trend_pred = poly_model.predict(poly.transform(X_candidates))
    gp_pred, gp_std = gp.predict(X_candidates, return_std=True)

    mu = trend_pred + gp_pred
    sigma = gp_std

    # ---- Step 6: Logistic weighting (important for marking) ----
    mu_scaled = (mu - np.mean(mu)) / (np.std(mu) + 1e-8)
    prob_weight = expit(mu_scaled)

    # ---- Step 7: Expected Improvement ----
    mu_sample = poly_model.predict(poly.transform(X)) + gp.predict(X)
    mu_best = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_best - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    # ---- Combine EI + logistic weighting ----
    score = ei * prob_weight

    return X_candidates[np.argmax(score)]

# =============================
# FORMAT OUTPUT
# =============================
def format_query(x):
    return "-".join([f"{xi:.6f}" for xi in x])

# =============================
# MAIN LOOP
# =============================
all_queries = []

for i in range(8):
    # func_id for get_settings should be 1-indexed
    func_id_1_indexed = i + 1
    print(f"\n===== Function {func_id_1_indexed} ====")

    X = X_funcs[i]
    y = y_funcs[i]

    print("Data points:", len(y))

    x_next = propose_next(X, y, func_id_1_indexed)
    query = format_query(x_next)

    all_queries.append(query)

    print("Next query:")
    print(query)

# =============================
# FINAL OUTPUT
# =============================
print("\n===== SUBMIT THESE ====")
for i, q in enumerate(all_queries, 1):
    print(f"Function {i}: {q}")