import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

# --- Data --- #

# Data path
project_root = Path.cwd()
data_path = project_root / "data"

# Liabilities
liabilities_df = pd.read_csv(data_path / 'liabs.csv', parse_dates=['date'], dayfirst=True, index_col = 'date')
liabilities = Liabilities(
    dates=liabilities_df.index.to_list(),
    cashflows=liabilities_df['cashflow']
)

# German Goverment Bond (Bund) Yields
bund_yields = pd.read_csv(data_path / 'bund_yields.csv', parse_dates=['date'], index_col = 'date')
rates = Rates(bund_yields['yield'], bund_yields.index.to_list())

# Spreads
spreads_df = pd.read_csv(data_path / 'credit_spreads.csv', index_col='rating')
spread_map = spreads_df['spread'].to_dict()

# Issuers
issuers_df = pd.read_csv(data_path / 'issuers.csv', dtype={'id': 'str'}, index_col='id')
issuers = Issuers(
    ids=issuers_df.index.to_list(),
    ratings=issuers_df['rating'],
    sectors=issuers_df['sector'],
    names = issuers_df['name']
)

# Bonds
bond_data = pd.read_csv(data_path / 'bonds.csv', dtype={'id': 'str', 'issuer_id': 'str'}, index_col='id')
cashflow_ladder = pd.read_csv(data_path / 'cashflow_ladder.csv', parse_dates=['date'], dayfirst=True).set_index('date')
bonds = Bonds(
    ids = bond_data.index.to_list(),
    issuer_ids = bond_data['issuer_id'],
    recoveries = bond_data['recovery'],
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
rho_e = .24
rho_s = np.array([rho_e])   # Match economy correlation to remove sector impact

# Model object
cr_model = CreditRiskModel(
    transition_matrix = transition_matrix,
    rho_e = rho_e,
    rho_s = rho_s,
    issuer_ids = issuers.ids,
    sector_map = issuers.sectors,
    ratings_map = issuers.ratings
)

# --- Mandate Config --- #

# Valuation Date
val_date = pd.to_datetime('01-01-2026', dayfirst = True)

# Bond Allocations (Notionals per Bond ID)
cdi_allocation = pd.read_csv(data_path / 'cdi_allocation.csv', index_col = 'id').squeeze()
cmbp_allocation = pd.read_csv(data_path / 'cmbp_allocation.csv', index_col = 'id').squeeze()

# Starting Cash Value
cash = 5e06

# Mandate Parameters
heubeck_liabilities = 297_033_196
r_gaap = 0.0201
r_ifrs = 0.039
mortality_buffer = 1.0559
cmbp_margin = 0.001
fee = 0.003
asset_buffer = 12.5e06
performance_cap = 50e06

# CDI Object
cdi_fox = CDIMandate_Fox(
    liabilities=liabilities,
    cash=cash,
    bonds=bonds,
    cdi_allocation=cdi_allocation,
    cmbp_allocation=cmbp_allocation,
    heubeck_liabilities=heubeck_liabilities,
    r_gaap=r_gaap,
    r_ifrs=r_ifrs,
    cmbp_margin=cmbp_margin,
    mortality_buffer=mortality_buffer,
    fee=fee,
    asset_buffer=asset_buffer,
    performance_cap=performance_cap
)

# --- Run --- #

# Simulate Economy
np.random.seed(123)
n_sim = 10000
n_years = 25
sim_results = cr_model.run(n_sim, n_years)

# Simulate CDI Solution
cdi_sim_results = cdi_fox.run(val_date, rates, spread_map, sim_results)