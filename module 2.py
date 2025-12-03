from math import exp
from itertools import product
from collections import deque


# BN structure (DAG)
nodes = ["Humidity","NDVI","Pheromone","CropMaturity","PestPop","Outbreak"]
parents = {
    "Humidity": [],
    "NDVI": [],
    "Pheromone": [],
    "CropMaturity": [],
    "PestPop": ["Humidity","NDVI","Pheromone","CropMaturity"],
    "Outbreak": ["PestPop","CropMaturity"]
}

# State spaces (ordered lists)
states = {
    "Humidity": ["Low","Medium","High"],
    "NDVI": ["Good","Moderate","Poor"],
    "Pheromone": ["Low","Medium","High"],
    "CropMaturity": ["Early","Mid","Late"],
    "PestPop": ["Low","High"],
    "Outbreak": ["No","Yes"]
}


# Root priors (example realistic priors)
priors = {
    "Humidity": {"Low":0.30, "Medium":0.45, "High":0.25},
    "NDVI": {"Good":0.50, "Moderate":0.35, "Poor":0.15},
    "Pheromone": {"Low":0.60, "Medium":0.30, "High":0.10},
    "CropMaturity": {"Early":0.25, "Mid":0.50, "Late":0.25}
}

# Logistic CPT parameters for PestPop
def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

logistic_params = {
    "intercept": -2.0,
    "w_humidity_high": 1.2,
    "w_humidity_med": 0.6,
    "w_ndvi_poor": 1.0,
    "w_ndvi_moderate": 0.3,
    "w_pheromone_high": 2.0,
    "w_pheromone_med": 0.8,
    "w_crop_late": 1.5,
    "w_crop_mid": 0.6
}


def p_pest_high_given_parents(humidity, ndvi, pheromone, crop):
    z = logistic_params["intercept"]
    if humidity == "High":
        z += logistic_params["w_humidity_high"]
    elif humidity == "Medium":
        z += logistic_params["w_humidity_med"]
    if ndvi == "Poor":
        z += logistic_params["w_ndvi_poor"]
    elif ndvi == "Moderate":
        z += logistic_params["w_ndvi_moderate"]
    if pheromone == "High":
        z += logistic_params["w_pheromone_high"]
    elif pheromone == "Medium":
        z += logistic_params["w_pheromone_med"]
    if crop == "Late":
        z += logistic_params["w_crop_late"]
    elif crop == "Mid":
        z += logistic_params["w_crop_mid"]
    return sigmoid(z)

# Outbreak CPT
p_outbreak = {
    ("High","Late"): 0.92,
    ("High","Mid"): 0.85,
    ("High","Early"): 0.65,
    ("Low","Late"): 0.15,
    ("Low","Mid"): 0.05,
    ("Low","Early"): 0.02
}

def joint_probability(assignment):
    p = 1.0
    # root priors
    for r in ["Humidity","NDVI","Pheromone","CropMaturity"]:
        p *= priors[r][assignment[r]]
    # PestPop
    p_high = p_pest_high_given_parents(assignment["Humidity"], assignment["NDVI"],
                                        assignment["Pheromone"], assignment["CropMaturity"])
    if assignment["PestPop"] == "High":
        p *= p_high
    else:
        p *= (1.0 - p_high)
    # Outbreak
    if assignment["Outbreak"] == "Yes":
        p *= p_outbreak[(assignment["PestPop"], assignment["CropMaturity"]) ]
    else:
        p *= (1.0 - p_outbreak[(assignment["PestPop"], assignment["CropMaturity"]) ])
    return p

# Exact inference by enumeration
def enumerate_posterior(query_var, query_state, evidence):
    hidden_vars = [v for v in nodes if v not in evidence and v != query_var]
    total = 0.0
    denom = 0.0
    for prod in product(*[states[h] for h in hidden_vars]):
        assign = {}
        assign.update(evidence)
        assign[query_var] = query_state
        for h, val in zip(hidden_vars, prod):
            assign[h] = val
        total += joint_probability(assign)
    for q_state in states[query_var]:
        s = 0.0
        for prod in product(*[states[h] for h in hidden_vars]):
            assign = {}
            assign.update(evidence)
            assign[query_var] = q_state
            for h, val in zip(hidden_vars, prod):
                assign[h] = val
            s += joint_probability(assign)
        denom += s
    if denom == 0:
        return 0.0
    return total / denom

def compute_outbreak_posterior(evidence):
    return enumerate_posterior("Outbreak","Yes", evidence)

# Bayes-ball (d-separation) implementation
children = {n: [] for n in nodes}
for child, pars in parents.items():
    for p in pars:
        children[p].append(child)


def is_d_separated(X, Y, Z):
    visited = set()
    queue = deque()
    queue.append((X, None, 'up'))
    queue.append((X, None, 'down'))
    while queue:
        node, prev, direction = queue.popleft()
        key = (node, prev, direction)
        if key in visited:
            continue
        visited.add(key)
        if node == Y:
            return False
        if node in Z:
            if direction == 'up':
                for par in parents[node]:
                    queue.append((par, node, 'up'))
            else:
                # observed node with down-travel: stop (do not pass to children)
                pass
        else:
            if direction == 'up':
                for par in parents[node]:
                    queue.append((par, node, 'up'))
                for ch in children[node]:
                    queue.append((ch, node, 'down'))
            else:
                for ch in children[node]:
                    queue.append((ch, node, 'down'))
                for par in parents[node]:
                    queue.append((par, node, 'up'))
    return True


# Risk mapping
def risk_level_from_prob(p):
    if p < 0.20:
        return "Low"
    elif p <= 0.60:
        return "Medium"
    else:
        return "High"
    
if __name__ == '__main__':
    test_cases = {
        "A_strong": {"Humidity":"High","NDVI":"Poor","Pheromone":"High","CropMaturity":"Late"},
        "B_moderate": {"Humidity":"High","NDVI":"Moderate","Pheromone":"Medium","CropMaturity":"Mid"}
    }

    print('\n=== Bayesian Network: Pest Outbreak Risk ===')
    print('Nodes:', nodes)
    print('\nLogistic parameters (P(PestPop=High|parents)):', logistic_params)
    print('\nOutbreak CPT (P(Outbreak=Yes | PestPop, CropMaturity)):', p_outbreak)

    results = {}
    for name, evidence in test_cases.items():
        p = compute_outbreak_posterior(evidence)
        results[name] = {"evidence": evidence, "P(Outbreak=Yes)": p, "Risk": risk_level_from_prob(p)}

    for name, info in results.items():
        print(f"\n{name}:")
        for k,v in info["evidence"].items():
            print(f"  {k} = {v}")
        print(f"  P(Outbreak=Yes | evidence) = {info['P(Outbreak=Yes)']:.6f}")
        print(f"  Risk level = {info['Risk']}")

    # d-separation checks
    dsep_q1 = is_d_separated("Humidity", "Outbreak", {"PestPop"})
    dsep_q2_uncond = is_d_separated("NDVI","Pheromone", set())
    dsep_q2_given_pest = is_d_separated("NDVI","Pheromone", {"PestPop"})

    print('\n--- d-separation checks (Bayes-ball) ---')
    print("Are Humidity and Outbreak d-separated given {PestPop}? ->", dsep_q1, "(True means d-separated)")
    print("Are NDVI and Pheromone d-separated (unconditioned)? ->", dsep_q2_uncond)
    print("Are NDVI and Pheromone d-separated given {PestPop}? ->", dsep_q2_given_pest)
    def pest_pop_posterior(evidence):
        return enumerate_posterior("PestPop","High", evidence)

    print('\nP(PestPop=High | evidence) for strong case A:', pest_pop_posterior(test_cases['A_strong']))
