import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import (
    Rates,
    Liabilities,
    TransitionMatrix, 
    CreditRiskSimulation,
    Issuers,
    Bonds,
    CDISimulation,
    CDIParameters
)

val_date = '28-02-2025'

# Rates
yields = pd.read_csv('data/rates.csv',  index_col = 'date', parse_dates = True, dayfirst = True, usecols = ['date', 'yield'])
rates = Rates(data=yields, val_date=val_date)

# Liabilities
liab_cfs = pd.read_csv('data/liabs.csv', parse_dates = True, dayfirst = True, usecols = ['date', 'cashflow'], index_col = 'date')
liabilities = Liabilities(data = liab_cfs)

# Issuers
issuers = Issuers(data = pd.read_csv('data/issuers.csv'))

# Bonds
bonds_df = pd.read_csv("data/bonds.csv", dtype={"id": str})
first_cf_date = '31-12-2025'
spread_map = {"A": 0.02, "BBB": 0.03, "BB": 0.04, "B": 0.05, "CCC": 0.06}
allocation = 400e06
bonds = Bonds(
    data = bonds_df,
    issuers = issuers,
    first_cf_date = first_cf_date,
    spread_map = spread_map,
    allocation = allocation,
    val_date = val_date,
    rates = rates,
)

# Simulate Economy
np.random.seed(42)
n_years = 20
n_sim = 250
rho_e = .1
rho_s = np.array([.2, .3])
rating_labels = ['A', 'BBB', 'BB', 'B', 'CCC', 'Default']
transition_matrix = TransitionMatrix(
    tmatrix = np.array([
    [0.85, 0.05, 0.04, 0.03, 0.02, 0.01],
    [0.05, 0.81, 0.05, 0.04, 0.03, 0.02],
    [0.00, 0.05, 0.83, 0.05, 0.04, 0.03],
    [0.00, 0.00, 0.05, 0.86, 0.05, 0.04],
    [0.00, 0.00, 0.00, 0.05, 0.90, 0.05],
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]
    ]),
    labels = rating_labels
)
sim_obj = CreditRiskSimulation(
    issuers = issuers,
    n_sim = n_sim,
    n_years = n_years,
    rho_e = rho_e,
    rho_s = rho_s,
    transition_matrix = transition_matrix
)

sim_results = sim_obj.run()
expected = bonds.expected_results().aggregate()
simulated = bonds.simulated_results(sim_results).aggregate()

# CDI
parameters = CDIParameters(
    asset_buffer = np.array([
        12987500, 
        13494013, 
        14020279, 
        14567070, 
        15135186, 
        15725458, 
        16338751,
        16975962, 
        17638024, 
        18325907
    ]),
    year_11_pmt = 18e06,
    gaap_int = 0.0201,
    mortality_risk_buffer = 12.8078e06,
    day_0_hgb_gap_buffer = 12.5e06
)

bond_sim = bonds.simulated_results(sim_results)
total_assets = 400e06
cdi_obj = CDISimulation(
    val_date = val_date,
    total_assets = total_assets,
    bond_sim = bond_sim,
    liabilities = liabilities,
    rates = rates,
    parameters = parameters
)
cdi_results = cdi_obj.run()

cdi_results.data
