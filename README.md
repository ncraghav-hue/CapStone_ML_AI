Black-Box Optimization (BBO) Capstone Project

Project Overview
The goal of this capstone project is to explore and optimize black-box functions, where the underlying system is unknown or expensive to evaluate. In this challenge, we aim to intelligently select query points in high-dimensional input spaces to maximize performance signals while minimizing the number of function evaluations.

Black-box optimization is highly relevant in real-world machine learning applications, such as:
Hyperparameter tuning
Experimental design
Robotics and control systems

This project strengthens skills in surrogate modelling, iterative optimization, and decision-making under uncertainty, which are directly transferable to careers in data science, machine learning engineering, and applied research.
Inputs and Outputs

Inputs:
Multi-dimensional vectors representing candidate solutions
Each query can have 4–8 dimensions depending on the iteration
Values are normalized, typically in the range [0, 1]
Queries are submitted in a dash-separated format, for example:
0.472138-0.581247-0.362849-0.491283-0.623178-0.472138

Outputs:
A scalar performance signal (response value) corresponding to the submitted input vector
Higher values indicate better-performing configurations
No additional information about the underlying function is provided
Challenge Objectives
Objective: Maximize the unknown black-box function using as few queries as possible

Constraints:
Limited number of queries
Function evaluations are costly and potentially noisy
The function structure is unknown and may be non-linear or high-dimensional
Success is measured by the ability to discover high-performing input configurations efficiently, while demonstrating a systematic approach to exploration and exploitation.

Technical Approach
Across the first three iterations, my strategy evolved as follows:
Iteration 1 – Broad Exploration:
Random and structured points to cover the input space
Gather baseline information about high- and low-performing regions
Iteration 2 – Focused Exploitation:
Concentrated queries around promising regions from iteration 1
Occasional extreme points to maintain exploration
Heuristics guided point selection based on observed patterns
Iteration 3 – Model-Informed Querying:
Structured query generation informed by surrogate modeling logic
Exploit high-value clusters while keeping some points for exploration
Surrogate models guide selection but heuristics still play a role

Machine Learning and Modeling Considerations:
Surrogate Models:
SVR (Support Vector Regression) or SVMs to approximate the black-box function
Soft-margin SVMs can classify high vs. low performance regions
Kernel SVMs can model non-linear response surfaces to guide query selection
Balancing Exploration vs. Exploitation:
Exploit known high-performing regions
Include occasional sampling from untested or extreme regions to avoid local optima
Iterative Refinement:
Each round of queries informs the next
Surrogate models and heuristics improve with additional data

Project Structure
BBO-Capstone/
│
├── data/
│   ├── initial_queries.csv       # Original query points submitted
│   ├── iteration_1_results.csv  # Response values from first round
│   ├── iteration_2_results.csv  # Response values from second round
│   └── iteration_3_results.csv  # Response values from third round
│
|
├── notebooks/
│   ├── ipynb                     # Exploratory analysis of query results
│   ├── ipynb                    # Building and evaluating surrogate models
│   └── ipynb      # Iterative query selection workflow
│
├── README.md                     # Project overview and documentation
└── requirements.txt              # Python dependencies


This structure keeps data, models, queries, and analysis notebooks organized, making it easy to reproduce experiments and track iterative progress.

