import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cdi_model_1802 import *

# Data path
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
project_root = Path.cwd()
data_path = project_root / "data"

# German Goverment Bond (Bund) Yields
bunds_df = pd.read_csv(data_path / 'bund_yields.csv', parse_dates=['date'])
rates = Rates(
    yields = bunds_df['yield'].values,
    dates =  bunds_df['date'].values
)

# Spreads
spreads_df = pd.read_csv(data_path / 'credit_spreads.csv', index_col='rating')
spread_map = spreads_df['spread'].to_dict()

# Issuers
issuers_df = pd.read_csv(data_path / 'issuers.csv', dtype={'id': 'str'}, index_col='id')
issuers = Issuers(
    ids=issuers_df.index.to_list(),
    ratings=issuers_df['rating'],
    sectors=issuers_df['sector'],
    names=issuers_df['name']
)

# Bonds
bond_df = pd.read_csv(data_path / 'bonds.csv', dtype={'id': 'str', 'issuer_id': 'str'}, index_col='id')
cashflow_ladder = pd.read_csv(data_path / 'cashflow_ladder.csv', parse_dates=['date'], dayfirst = True).set_index('date')
bonds = Bonds(
    ids = bond_df.index.to_list(),
    issuer_ids = bond_df['issuer_id'],
    recoveries = bond_df['recovery'],
    cashflows = cashflow_ladder,
    issuers = issuers
)

# Transition matrix
tmatrix = pd.read_csv(data_path / 'transition_matrix.csv')
transition_matrix = TransitionMatrix(
    tmatrix = tmatrix.values,
    labels = tmatrix.columns
)

# Economy and Sector correlations
rho_e = .2
rho_s = np.array([rho_e])

# Model object
cr_model = CreditRiskModel(
    transition_matrix = transition_matrix,
    rho_e = rho_e,
    rho_s = rho_s,
    issuer_ids = issuers.ids,
    sector_map = issuers.sectors,
    ratings_map = issuers.ratings
)

# Run Simulation
np.random.seed(123)
n_sim = 5000
n_years = 25
sim_results = cr_model.run(n_sim, n_years)

# Simulate CDI Solution
bond_sim = bonds.run_sim(val_date, rates, spread_map, sim_results)