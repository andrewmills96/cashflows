import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from datetime import datetime
from typing import Any, List, Sequence, Union, Dict, Optional

# --- Constants ---
DAYS_PER_YEAR = 365.0
DEFAULT_LABEL = "Def"
# Ensure 'Def' is the last index for logic simplification
RATINGS_ORDER = [
    'AAAA', 'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
    'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-',
    'CCC+', 'CCC', 'CCC-', 'CC+', 'Def'
]

# --- Helper Functions --- #

def calc_year_frac(dates: pd.DatetimeIndex, val_date: datetime) -> np.ndarray:
    """Fast calculation of year fractions using numpy."""
    # Convert both to datetime64[D] (days) to avoid nanosecond overflow issues
    d_dates = dates.values.astype('datetime64[D]')
    d_val = np.datetime64(val_date, 'D')
    return (d_dates - d_val).astype(float) / DAYS_PER_YEAR

# --- Rates --- #

class Rates:
    def __init__(self, yields: np.ndarray, dates: Sequence[Any]):
        # Store as numpy arrays for speed
        self.dates = pd.to_datetime(dates).values.astype('datetime64[D]').astype(float)
        self.yields = np.asarray(yields, dtype=float)

    def interpolate(self, target_dates: pd.DatetimeIndex) -> np.ndarray:
        """Fast linear interpolation using numpy.interp."""
        # Convert targets to float days
        t_dates = target_dates.values.astype('datetime64[D]').astype(float)
        return np.interp(t_dates, self.dates, self.yields)

# --- Transition Matrix --- #

class TransitionMatrix:
    """
    Optimized Transition Matrix using Integer arithmetic.
    """
    def __init__(self, tmatrix: np.ndarray, labels: List[str]):
        self.tmatrix = tmatrix
        self.labels = np.array(labels)
        # Cumulative probability matrix for searchsorted
        self.cum_tmatrix = np.cumsum(tmatrix, axis=1)
        
        # Map labels to integers for O(1) lookups
        self.label_to_idx = {l: i for i, l in enumerate(labels)}
        self.n_states = len(labels)

        # Validation
        if not np.allclose(self.cum_tmatrix[:, -1], 1.0):
            raise ValueError("Transition probabilities do not sum to 1")

    def transitions(self, pX: np.ndarray, ratings_map: pd.Series) -> np.ndarray:
        """
        Fully vectorized transition logic.
        pX: (n_sim, n_years, n_issuers)
        """
        n_sim, n_years, n_issuers = pX.shape
        
        # 1. Convert initial ratings to integers
        # Using a fast map via pandas or list comp
        initial_indices = ratings_map.map(self.label_to_idx).fillna(self.n_states - 1).values.astype(int)
        
        # 2. Initialize result array (n_sim, n_years + 1, n_issuers)
        # We work entirely in Integers until the very end to save memory and speed
        trans_idx = np.empty((n_sim, n_years + 1, n_issuers), dtype=int)
        
        # Set t=0
        # Broadcast initial_indices to (n_sim, n_issuers)
        trans_idx[:, 0, :] = initial_indices[None, :]

        # 3. Vectorized Simulation Loop
        # We loop over time (necessary for path dependency), but NOT simulations
        for t in range(n_years):
            # Current state indices for all sims and issuers: (n_sim, n_issuers)
            current_states = trans_idx[:, t, :]
            
            # Random uniform draws (mapped from standard normal pX) for this step
            # pX is CDF(N(0,1)), so it's uniform [0,1].
            probs = pX[:, t, :]
            
            # Vectorized Lookup:
            # We need to find where 'probs' falls in the cumulative distribution of the 'current_states'
            # Fancy indexing: Select the row of cum_tmatrix corresponding to each issuer's current rating
            # shape: (n_sim, n_issuers, n_states)
            relevant_cum_probs = self.cum_tmatrix[current_states]
            
            # Find next state: sum(probs > cum_probs) gives the index
            # This is equivalent to searchsorted but vectorized
            # shape: (n_sim, n_issuers)
            next_states = np.sum(probs[..., None] > relevant_cum_probs, axis=2)
            
            trans_idx[:, t + 1, :] = next_states

        # 4. Convert back to Labels
        return self.labels[trans_idx]

# --- Simulation Result --- #

class SimulationResult:
    def __init__(self, transitions_idx: np.ndarray, labels: np.ndarray, n_sim: int, n_years: int, issuer_ids: List[str]):
        self.transitions_idx = transitions_idx # Store integers! Much smaller/faster
        self.labels = labels
        self.n_sim = n_sim
        self.n_years = n_years
        self.issuer_ids = issuer_ids
    
    @property
    def transitions(self) -> pd.DataFrame:
        """Lazy evaluation of the dataframe only if user asks for it."""
        labels = self.labels[self.transitions_idx]
        n_issuers = len(self.issuer_ids)
        return pd.DataFrame(
            labels.reshape(self.n_sim * (self.n_years+1), n_issuers),
            index=pd.MultiIndex.from_product([range(self.n_sim), range(self.n_years + 1)], names=["sim", "year"]),
            columns=self.issuer_ids
        )

# --- Credit Risk Model --- #

class CreditRiskModel:
    def __init__(
        self,
        transition_matrix: TransitionMatrix,
        rho_e: float,
        rho_s: np.ndarray,
        issuer_ids: list,
        sector_map: pd.Series,
        ratings_map: pd.Series
    ):
        self.tm = transition_matrix
        self.rho_e = rho_e
        self.rho_s = rho_s
        self.issuer_ids = issuer_ids
        # Pre-convert sectors to integer indices for speed
        self.sector_indices = sector_map.factorize()[0]
        self.n_sectors = len(sector_map.unique())
        self.ratings_map = ratings_map

    def run(self, n_sim: int, n_years: int) -> SimulationResult:
        n_issuers = len(self.ratings_map)
        
        # 1. Random Factors Generation (Heavy lifting done by C-backend)
        # E: Global factor
        E = np.random.normal(size=(n_sim, n_years, 1))
        # S: Sector factors
        S = np.random.normal(size=(n_sim, n_years, self.n_sectors))
        # I: Idiosyncratic factors
        I = np.random.normal(size=(n_sim, n_years, n_issuers))

        # 2. Correlation Mapping
        # Map sector rhos and S factors to issuers
        # rho_s_i shape: (n_issuers,)
        rho_s_i = self.rho_s[self.sector_indices] 
        
        # Broadcast S to issuers: (n_sim, n_years, n_issuers)
        S_mapped = S[:, :, self.sector_indices]

        # 3. Calculate Latent Variable X
        # Vectorized calculation
        sqrt_rho_e = np.sqrt(self.rho_e)
        sqrt_rho_s_resid = np.sqrt(rho_s_i - self.rho_e)
        sqrt_resid = np.sqrt(1 - rho_s_i)
        
        X = (sqrt_rho_e * E) + (sqrt_rho_s_resid * S_mapped) + (sqrt_resid * I)
        
        # 4. Probabilities
        pX = sp_stats.norm.cdf(X)

        # 5. Transitions (Optimized)
        # Returns integer indices array (n_sim, n_years+1, n_issuers)
        # We skip converting to DataFrame here to keep it raw and fast
        trans_labels = self.tm.transitions(pX, self.ratings_map)
        
        # We store the *indices* of the labels in the result to save memory/time
        # Map labels back to indices
        label_map = {l: i for i, l in enumerate(self.tm.labels)}
        # Only needed if transitions returned labels. 
        # Actually, let's make TransitionMatrix.transitions return indices directly? 
        # For compatibility with your requested "labels" output, I kept labels return in TM, 
        # but for performance, we should ideally work with indices.
        # Let's map it back to indices for the result object, as downstream calc needs indices.
        
        # optimization: modify TM to return indices or map quickly
        trans_idx = np.vectorize(label_map.get)(trans_labels)

        return SimulationResult(trans_idx, self.tm.labels, n_sim, n_years, self.issuer_ids)

# --- Assets --- #

class Issuers:
    def __init__(self, ids, ratings, sectors, names=None):
        self.ids = ids
        self.ratings = ratings
        self.sectors = sectors
        self.names = names

class BondSimulationResult:
    def __init__(self, total_cashflows, pvs, dates, bond_ids):
        self.total_cashflows = total_cashflows
        self.pvs = pvs
        self.dates = dates
        self.bond_ids = bond_ids

    def to_dataframe(self, by_bond: bool = True):
        # Optimized DataFrame creation
        n_scens, n_dates, n_bonds = self.total_cashflows.shape
        
        # Flatten arrays
        # Note: This can be huge. Only call if necessary.
        flat_pvs = self.pvs.ravel()
        flat_cfs = self.total_cashflows.ravel()
        
        if not by_bond:
            # Sum over bonds first (much faster)
            pvs_agg = self.pvs.sum(axis=2).ravel()
            cfs_agg = self.total_cashflows.sum(axis=2).ravel()
            idx = pd.MultiIndex.from_product([range(n_scens), self.dates], names=["scenario", "date"])
            return pd.DataFrame({'cashflow': cfs_agg, 'pv': pvs_agg}, index=idx).reset_index()
            
        idx = pd.MultiIndex.from_product(
            [range(n_scens), self.dates, self.bond_ids], 
            names=["scenario", "date", "bond_id"]
        )
        return pd.DataFrame({'cashflow': flat_cfs, 'pv': flat_pvs}, index=idx).reset_index()

class Bonds:
    def __init__(self, ids, issuer_ids, recoveries, cashflows, issuers):
        self.ids = ids
        self.issuer_ids = issuer_ids
        self.recoveries = recoveries
        self.cashflows = cashflows[ids].sort_index()
        self.issuers = issuers
        
        # Pre-compute Bond-to-Issuer Integer Map
        # This maps column 'j' of the bond matrix to column 'k' of the issuer matrix
        issuer_id_map = {iid: i for i, iid in enumerate(issuers.ids)}
        self.bond_issuer_indices = np.array([issuer_id_map[iid] for iid in issuer_ids])

    def year_end_cashflows(self, val_date: Optional[datetime] = None) -> pd.DataFrame:
        cf = self.cashflows.resample("ME").sum()
        if val_date:
            cf = cf.loc[cf.index > val_date]
        return cf.resample("YE").sum()

    def run_sim(self, val_date: datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):
        """
        Highly optimized simulation run.
        """
        # 1. Prepare Cashflows
        cf_df = self.year_end_cashflows(val_date)
        cashflows_arr = cf_df.values # (n_years_bond, n_bonds)
        dates = cf_df.index
        n_bond_years, n_bonds = cashflows_arr.shape
        
        n_sim = sim_results.n_sim
        n_sim_years = sim_results.n_years
        
        # Ensure we don't simulate past cashflow data
        limit_years = min(n_bond_years, n_sim_years)
        
        # 2. Prepare Market Data
        # Pre-interpolate yields for all relevant dates
        yields = rates.interpolate(dates) # (n_bond_years,)
        dt_frac = calc_year_frac(dates, val_date) # (n_bond_years,)
        
        # Convert spread map to array for O(1) integer lookup
        # Create an array where index = rating_index, value = spread
        spread_lookup = np.zeros(len(sim_results.labels))
        for label, spread in spread_map.items():
            if label in sim_results.labels:
                idx = np.where(sim_results.labels == label)[0][0]
                spread_lookup[idx] = spread
        
        # Identify Default Index
        def_idx = np.where(sim_results.labels == DEFAULT_LABEL)[0]
        def_idx = def_idx[0] if len(def_idx) > 0 else -1

        # 3. Map Transitions to Bonds
        # sim_results.transitions_idx is (n_sim, n_years+1, n_issuers)
        # We need (n_sim, n_years, n_bonds). We drop year 0 (initial state).
        # We use fancy indexing with self.bond_issuer_indices to broadcast issuers -> bonds
        bond_ratings_idx = sim_results.transitions_idx[:, 1:limit_years+1, self.bond_issuer_indices]
        
        # 4. Default & Cashflow Logic (Vectorized)
        
        # Identify defaults: boolean mask (n_sim, limit_years, n_bonds)
        is_default = (bond_ratings_idx == def_idx)
        
        # Calculate 'Cumulative Non-Default' status
        # Once you default, you stay defaulted for cashflow purposes (simplified)
        # cummax: once True (default), stays True.
        has_defaulted = np.maximum.accumulate(is_default, axis=1) 
        
        # Active: Not defaulted yet.
        # Note: If default happens at T=1, cashflow at T=1 is NOT received (assuming end of period default logic)
        # or partially received. Assuming standard: Default deletes current coupon.
        is_active = ~has_defaulted
        
        # Recovery Logic
        # First default event: Current is default, Previous was not.
        # Pad with False at t=0
        shifted_def = np.concatenate([np.zeros((n_sim, 1, n_bonds), dtype=bool), has_defaulted[:, :-1, :]], axis=1)
        first_default_event = has_defaulted & (~shifted_def)
        
        # Maturity Mask (Pre-calculated)
        # Find last non-zero CF index for each bond
        # shape (n_bonds,)
        last_cf_idx = n_bond_years - 1 - np.argmax(cashflows_arr[::-1] != 0, axis=0)
        # Broadcast to (limit_years, n_bonds)
        year_indices = np.arange(limit_years)[:, None]
        is_pre_maturity = (year_indices <= last_cf_idx[None, :])
        
        # Apply masks
        # Broadcast cashflows (1, years, bonds)
        scheduled_cfs = cashflows_arr[None, :limit_years, :] 
        received_cfs = scheduled_cfs * is_active
        
        # Recovery payments
        rec_rates = self.recoveries.values # (n_bonds,)
        recovery_pmts = first_default_event * rec_rates[None, None, :] * is_pre_maturity[None, :, :]
        
        total_cfs = received_cfs + recovery_pmts
        
        # 5. PV Calculation Loop
        # We must loop t, but we vectorize everything else.
        # Map ratings to spreads immediately
        spreads = spread_lookup[bond_ratings_idx] # (n_sim, limit_years, n_bonds)
        
        pvs = np.zeros((n_sim, limit_years, n_bonds))
        
        # Pre-calculate base discount factors (Time Value only)
        # DF_base(0, T) = (1+y)^-T
        # We need DF(t, T) = DF_base(0, T) / DF_base(0, t)
        # But we also have spreads.
        
        # To optimize the inner loop:
        # PV_t = Sum_{k > t} [ CF_k * (1 + y_k + s_t)^-(dt_k - dt_t) ]
        # s_t is spreads[:, t, :]
        
        for t in range(limit_years):
            # Future slice
            # Time horizons relative to t
            delta_t = dt_frac[t+1:limit_years] - dt_frac[t] # shape (remaining_years,)
            
            if len(delta_t) == 0:
                break
                
            # Current Spreads for all sims/bonds at time t
            s_t = spreads[:, t, :] # (n_sim, n_bonds)
            
            # Yields for future cashflows
            y_fwd = yields[t+1:limit_years] # (remaining_years,)
            
            # Discount Factors
            # Broadcast: (1, rem_years, 1) + (n_sim, 1, n_bonds)
            # Result: (n_sim, rem_years, n_bonds)
            rate_term = 1 + y_fwd[None, :, None] + s_t[:, None, :]
            dfs = np.power(rate_term, -delta_t[None, :, None])
            
            # Future Cashflows
            # (1, rem_years, n_bonds) (using static CFs for pricing, assuming risk-neutral pricing uses promised CFs)
            # NOTE: Standard models discount *promised* cashflows using risky spreads.
            # If you want to discount *simulated* cashflows, use total_cfs, but usually spread-based pricing implies
            # using the spread to account for default risk, so you discount the *promised* CFs.
            # The original code used `cashflows` (promised), so we stick to that.
            cfs_future = cashflows_arr[None, t+1:limit_years, :]
            
            # Dot product over time axis
            # (n_sim, rem_years, n_bonds) * (1, rem_years, n_bonds) -> sum over axis 1
            pvs[:, t, :] = np.sum(cfs_future * dfs, axis=1)

        return BondSimulationResult(
            total_cashflows=total_cfs,
            pvs=pvs,
            dates=dates[:limit_years],
            bond_ids=self.ids
        )