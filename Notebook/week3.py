import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
import ast # Added to parse string representations of lists/arrays

# =============================
# LOAD TXT DATA
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

X_raw = input_data
y_all_funcs = output_data

# =============================
# GROUP BY FUNCTION
# =============================
X_funcs, y_funcs = [], []

for i in range(8):
    X_funcs.append(X_raw)
    y_funcs.append(y_all_funcs[:, i])

# =============================
# SETTINGS
# =============================
def get_settings(func_id):
    xi = 0.02
    noise = 1e-5
    global_n = 4000
    local_n = 2000

    if func_id == 2:
        noise = 1e-3
    if func_id == 5:
        xi = 0.001
    if func_id == 8:
        global_n = 6000

    return xi, noise, global_n, local_n

# =============================
# PROPOSE NEXT POINT
# =============================
def propose_next(X, y, func_id):
    dim = X.shape[1]
    xi, noise, global_n, local_n = get_settings(func_id)

    # ---- Step 1: Standardize features ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Step 2: Fit SVR (SVM regression) ----
    svr = SVR(kernel='rbf', C=10, epsilon=0.01)
    svr.fit(X_scaled, y)
    y_svr_pred = svr.predict(X_scaled)
    residuals = y - y_svr_pred

    # ---- Step 3: GP on residuals for uncertainty ----
    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=noise)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(X_scaled, residuals)

    # ---- Step 4: Candidate generation ----
    X_global = np.random.uniform(0, 1, size=(global_n, dim))
    X_global_scaled = scaler.transform(X_global)

    best_x = X[np.argmax(y)]
    X_local = best_x + 0.05 * np.random.randn(local_n, dim)
    X_local = np.clip(X_local, 0, 1)
    X_local_scaled = scaler.transform(X_local)

    X_candidates_scaled = np.vstack([X_global_scaled, X_local_scaled])
    X_candidates = np.vstack([X_global, X_local])

    # ---- Step 5: SVR + GP predictions ----
    svr_pred = svr.predict(X_candidates_scaled)
    gp_pred, gp_std = gp.predict(X_candidates_scaled, return_std=True)

    mu = svr_pred + gp_pred
    sigma = gp_std

    # ---- Step 6: Expected Improvement ----
    mu_sample = svr.predict(X_scaled) + gp.predict(X_scaled)
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
    print(f"\n===== Function {i+1} ====")
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
print("\n===== SUBMIT THESE ====")
for i, q in enumerate(all_queries, 1):
    print(f"Function {i}: {q}")