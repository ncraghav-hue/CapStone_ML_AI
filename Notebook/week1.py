import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm

# =============================
# Expected Improvement
# =============================
def expected_improvement(X, X_sample, Y_sample, model, xi):
    mu, sigma = model.predict(X, return_std=True)
    mu_sample = model.predict(X_sample)

    sigma = sigma.reshape(-1, 1)
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu.reshape(-1, 1) - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei.ravel()

# =============================
# Function-specific tuning
# =============================
def get_settings(func_id):
    # Defaults
    xi = 0.01
    noise = 1e-5
    n_candidates = 3000

    if func_id == 1:
        xi = 0.1
    elif func_id == 2:
        xi = 0.05
        noise = 1e-3
    elif func_id == 4:
        xi = 0.05
    elif func_id == 5:
        xi = 0.001
    elif func_id == 7:
        xi = 0.05
        n_candidates = 5000
    elif func_id == 8:
        xi = 0.1
        n_candidates = 8000

    return xi, noise, n_candidates

# =============================
# Propose next query
# =============================
def propose_next(X, y, func_id):
    dim = X.shape[1]
    xi, noise, n_candidates = get_settings(func_id)

    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=noise)
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    model.fit(X, y)

    bounds = np.array([[0.0, 1.0]] * dim)

    # Random candidate search (robust for all dims)
    X_rand = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, dim))
    ei = expected_improvement(X_rand, X, y, model, xi)

    return X_rand[np.argmax(ei)]

# =============================
# Format output
# =============================
def format_query(x):
    return "-".join([f"{xi:.6f}" for xi in x])

# =============================
# MAIN LOOP (ALL 8 FUNCTIONS)
# =============================
all_queries = []

for i in range(1, 9):
    print(f"\n===== Function {i} =====")

    # Load data
    X = np.load(f"function_{i}_inputs.npy")
    y = np.load(f"function_{i}_outputs.npy")

    print("Shape:", X.shape)

    # Generate query
    x_next = propose_next(X, y, i)
    query = format_query(x_next)

    all_queries.append(query)

    print("Next query:")
    print(query)

# =============================
# FINAL OUTPUT (SUBMIT THIS)
# =============================
print("\n===== FINAL SUBMISSION =====")
for i, q in enumerate(all_queries, 1):
    print(f"Function {i}: {q}")