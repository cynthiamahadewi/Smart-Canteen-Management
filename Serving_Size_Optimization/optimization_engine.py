# optimisation_engine.py
import numpy as np
from scipy.optimize import minimize

def run_serving_optimization(dishes_dict):
    """
    Existing logic from Serving_Size_Optimization.ipynb
    """
    names = list(dishes_dict.keys())
    data_list = [dishes_dict[n] for n in names]
    C_budget = 25000
    
    def objective(S):
        total_profit = 0
        for idx, d in enumerate(data_list):
            # Demand model: D_base * (S/S_base)^alpha
            Q = d['D_base'] * (S[idx] / d['S_base'])**d['alpha']
            profit = (d['P'] - d['C'] * S[idx]) * Q
            total_profit -= profit  
        return total_profit

    cons = []
    for idx, d in enumerate(data_list):
        # Lower bound with 15% safety buffer
        lower_bound = d['S_base'] * d['R_i'] * 1.15 
        upper_bound = d['S_base'] 
        cons.append({'type': 'ineq', 'fun': lambda S, lb=lower_bound, i=idx: S[i] - lb})
        cons.append({'type': 'ineq', 'fun': lambda S, ub=upper_bound, i=idx: ub - S[i]})

    S0 = [d['S_base'] for d in data_list]
    res = minimize(objective, S0, method='SLSQP', constraints=cons)
    return dict(zip(names, res.x))