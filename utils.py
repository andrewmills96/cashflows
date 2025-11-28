import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import datetime
from dataclasses import dataclass
from typing import List, Sequence, Mapping, Optional, Union, Any, Dict, Tuple

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
DAYS_PER_YEAR = 365.0

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def calc_dt(dates: Union[List[datetime.datetime], pd.DatetimeIndex], val_date: datetime.datetime) -> np.ndarray:
    """Calculates time difference in years between a list of dates and a valuation date."""
    if isinstance(dates, list):
        dates = pd.DatetimeIndex(dates)
    # Ensure numpy array for slicing
    return np.array(((dates - val_date).days / DAYS_PER_YEAR))[:, None]

# ------------------------------------------------------------------------------
# Base Data Classes
# ------------------------------------------------------------------------------
class BaseDataClass:
    """
    Base class for data containers with strict schema validation.
    """
    typeChecks = {
        'date': pd.api.types.is_datetime64_any_dtype,
        'numeric': pd.api.types.is_numeric_dtype,
        'string': pd.api.types.is_string_dtype
    }

    REQUIRED_COLUMNS: Dict[str, str] = {}
    data: pd.DataFrame

    def validate_inputs(self):
        """Confirm that the DataFrame contains exactly the correct columns and types."""
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        # Check required columns exist
        missing_cols = set(self.REQUIRED_COLUMNS.keys()) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing required column(s): {missing_cols}")

        # Check for unexpected columns (Strict Schema)
        extra_cols = set(self.data.columns) - set(self.REQUIRED_COLUMNS.keys())
        if extra_cols:
            raise ValueError(f"Unexpected column(s) detected: {extra_cols}. Expected only: {list(self.REQUIRED_COLUMNS.keys())}")

        # Check types
        bad_types = []
        for col, req_type in self.REQUIRED_COLUMNS.items():
            if req_type is not None:
                # If expecting date, try to convert if not already
                if req_type == 'date' and not self.typeChecks['date'](self.data[col]):
                    try:
                        self.data[col] = pd.to_datetime(self.data[col], dayfirst=True)
                    except Exception:
                        bad_types.append(f"Column '{col}' could not be converted to datetime.")
                        continue
                
                # Re-check type
                if not self.typeChecks[req_type](self.data[col]):
                    bad_types.append(f"Column '{col}': expected {req_type}, found {self.data[col].dtype}")

        if bad_types:
            raise TypeError(f"Column type mismatch: {bad_types}")

# ------------------------------------------------------------------------------
# Market Data Classes
# ------------------------------------------------------------------------------
class Rates(BaseDataClass):
    """
    Class to hold a discount rate curve.
    """
    REQUIRED_COLUMNS = {"date": 'date', "yield": 'numeric'}

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.validate_inputs()
        
        # yields and fwds as a date-indexed Series
        self.yields = self.data.set_index('date')['yield']

    def calc_fwds(self, val_date: datetime.datetime) -> pd.Series:
        """Calculate implied forward rates from the specified yield curve."""

        # ensure you are only using yields after val date
        y = self.yields.loc[val_date:]
        
        # Explicitly cast to DatetimeIndex to satisfy static analysis
        dates = pd.DatetimeIndex(y.index)

        # Use global constant for consistency
        t = np.array((dates - val_date).days / DAYS_PER_YEAR) 
        
        # Avoid division by zero at t=0
        with np.errstate(divide='ignore', invalid='ignore'):
            acc = (1 + y) ** t
        
        # forwards calculated as the percentage change in total accumulation factors
        # shift() moves data down, so acc / acc.shift() is A(t) / A(t-1)
        fwds = (acc / acc.shift()) ** (1 / (t - np.roll(t, 1))) - 1

        # first forward set to first yield
        fwds.iloc[0] = y.iloc[0]
        fwds.name = 'fwds'
        return fwds

    def interpolate(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Linearly interpolates yields to the specified dates."""
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)

        # Combine and sort index
        combined_index = self.yields.index.union(dates).sort_values()

        # Interpolate
        interpolated = (
            self.yields
            .reindex(combined_index)
            .interpolate(method="time")
            .reindex(dates)
        )
        return interpolated


class Liabilities(BaseDataClass):
    """
    Class to hold liability cashflows.
    """
    REQUIRED_COLUMNS = {"date": 'date', "cashflow": 'numeric'}

    def __init__(self, data: pd.DataFrame, name='liabilities'):
        self.name = name
        self.data = data.copy()
        # Validation will handle datetime conversion
        self.validate_inputs() 
        self.data.set_index("date", inplace=True)
        self.cashflows = self.data["cashflow"]

    def to_dataframe(self):
        return self.cashflows.to_frame()

    def to_series(self):
        return self.cashflows

    def pv(self, rates: Union["Rates", float], val_date: datetime.datetime, shift=0.0) -> float:
        """
        Calculate PV of liabilities for a specified date and discount rate.
        """
        # Subset of Cashflows and dates beyond valuation date
        cashflows = self.cashflows.loc[val_date:]
        if cashflows.empty:
            return 0.0
            
        dates = pd.DatetimeIndex(cashflows.index)
        val_date_ts = pd.to_datetime(val_date)
        t = (dates - val_date_ts).days / DAYS_PER_YEAR

        # Allow for rates to be defined as a curve (Rates) object, or a single flat rate.
        if isinstance(rates, (int, float)):
            net_yield = pd.Series(rates + shift, index=cashflows.index)
        else:
            yields_int = rates.interpolate(dates)
            net_yield = yields_int + shift
            
        df = (1 + net_yield) ** -t
        pv = (cashflows * df).sum()
        return pv

    def __str__(self):
        return str(self.data)


class Issuers(BaseDataClass):
    """
    Class for holding data on bond issuers.
    """
    REQUIRED_COLUMNS = {"id": 'string', "sector": 'string', "rating": 'string'}

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        # Ensure all required string columns are cast to string before validation
        # This handles cases where 'sector' or 'rating' might be inferred as int/object
        for col in ['id', 'sector', 'rating']:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype(str)
        
        self.validate_inputs()
        
        self.n_sectors = self.data['sector'].nunique()
        self.n_issuers = self.data['id'].nunique()
        self.ids = np.array(self.data['id'])
        
        # Fast lookup for IDs
        self.id_to_idx = {id_: i for i, id_ in enumerate(self.ids)}

        # Check that ids are unique
        if self.data['id'].duplicated().any():
            duplicates = self.data.loc[self.data['id'].duplicated(), 'id'].unique()
            raise KeyError(f"Duplicated IDs detected: {list(duplicates)}")

    def __str__(self):
        return str(self.data)

# ------------------------------------------------------------------------------
# Simulation Classes
# ------------------------------------------------------------------------------
class TransitionMatrix:
    """
    Class for holding a credit rating transition matrix.
    Optimized for vectorized state transitions.
    """
    def __init__(self, tmatrix: np.ndarray, labels: list[str]):
        if tmatrix.ndim != 2:
            raise IndexError("2-d transition matrix expected")
        if tmatrix.shape[0] != tmatrix.shape[1]:
            raise IndexError("Square transition matrix expected")
        if len(labels) != tmatrix.shape[0]:
            raise IndexError("labels is not the same length as one side of the transition matrix")

        self.labels = np.array(labels)
        self.label_to_idx = {l: i for i, l in enumerate(labels)}
        
        self.tmatrix = pd.DataFrame(tmatrix, index=labels, columns=labels)
        
        # Pre-calculate cumulative sum for faster vectorized lookup
        # Shape: (n_states, n_states)
        self.cum_probs = np.cumsum(tmatrix, axis=1)
        # Ensure last is 1.0 explicitly to avoid float errors
        self.cum_probs[:, -1] = 1.0

        # Validate probabilities
        bad_rows = [labels[i] for i in range(len(labels)) if not np.isclose(self.cum_probs[i, -1], 1.0)]
        if bad_rows:
            raise ValueError(f"Transition probabilities for the following states do not sum to 1.0: {bad_rows}")

    def get_next_state(self, current_idxs: np.ndarray, unif_draws: np.ndarray) -> np.ndarray:
        """
        Vectorized state transition.
        Args:
            current_idxs: (N,) int array of current rating indices
            unif_draws: (N,) float array (0-1)
        Returns:
            (N,) int array of next rating indices
        """
        # Get thresholds for each current state: (N, n_states)
        # Advanced indexing: selects the row from cum_probs corresponding to each current_idx
        thresholds = self.cum_probs[current_idxs]
        
        # Find insertion points: (N,)
        # argmax on boolean gives index of first True
        return (unif_draws[:, None] < thresholds).argmax(axis=1)

    def __str__(self):
        return str(self.tmatrix)


@dataclass
class SimulationResult:
    """Container for results of a CreditRiskModel run."""
    E: np.ndarray  # Economic Factors
    S: np.ndarray  # Sector Factors
    I: np.ndarray  # Idiosyncratic Factors
    X: np.ndarray  # Latent Variable
    pX: np.ndarray # Probability Map
    transitions: np.ndarray # (n_sim, n_years, n_issuers) Int Array of rating indices
    rating_labels: np.ndarray # Array of label strings corresponding to indices
    transitions_df: pd.DataFrame  # Formatted Transitions (Lazy or computed)
    n_sim: int
    n_years: int


class CreditRiskModel:
    """
    Monte Carlo simulation of issuer credit ratings using a Gaussian Copula model.
    """

    def __init__(
            self,
            issuers: Issuers,
            rho_e: float,
            rho_s: np.ndarray,
            transition_matrix: TransitionMatrix
    ):
        self.rho_e = rho_e
        self.rho_s = rho_s
        self.transition_matrix = transition_matrix
        self.issuers = issuers
        
        self.n_sectors = issuers.n_sectors
        self.n_issuers = issuers.n_issuers
        
        # Maps to index into sector arrays
        self.sector_map = pd.Categorical(issuers.data['sector']).codes
        
        # Map initial ratings to integers
        self.initial_rating_indices = np.array([
            self.transition_matrix.label_to_idx[r] for r in issuers.data['rating']
        ])

        if len(rho_s) < issuers.n_sectors:
            raise ValueError("One or more sectors in the Issuers object do not have a sector correlation defined.")

    def run(self, n_sim: int, n_years: int) -> SimulationResult:
        """
        Executes the simulation using fully vectorized numpy operations.
        """
        # Generate Factors: (Simulations, Years, Dimensions)
        E = np.random.normal(size=[n_sim, n_years])
        S = np.random.normal(size=[n_sim, n_years, self.n_sectors])
        I = np.random.normal(size=[n_sim, n_years, self.n_issuers])

        # Broadcast Factors
        X = np.sqrt(self.rho_e) * E[:, :, np.newaxis] + \
            np.sqrt(self.rho_s[self.sector_map] - self.rho_e)[np.newaxis, np.newaxis, :] * S[:, :, self.sector_map] + \
            np.sqrt(1 - self.rho_s[self.sector_map])[np.newaxis, np.newaxis, :] * I

        # Map latent variable X to probability space [0, 1]
        pX = sp_stats.norm.cdf(X)

        # Initialize Transitions Array: (n_sim, n_years, n_issuers)
        transitions = np.zeros((n_sim, n_years, self.n_issuers), dtype=int)
        
        # Set Initial State (broadcasted)
        current_ratings = np.tile(self.initial_rating_indices, (n_sim, 1))

        # Time Loop
        for t in range(n_years):
            # pX slice for this year: (n_sim, n_issuers)
            draws = pX[:, t, :]
            
            # Vectorized update: flatten to (N,) for the helper, then reshape back
            shape = current_ratings.shape # (n_sim, n_issuers)
            
            next_r = self.transition_matrix.get_next_state(
                current_ratings.ravel(), 
                draws.ravel()
            )
            
            current_ratings = next_r.reshape(shape)
            transitions[:, t, :] = current_ratings

        # Create DataFrame for backward compatibility / inspection
        # This is the slowest part, can be skipped if not plotting
        # Flatten transitions to (Total_Steps, I)
        flat_transitions = transitions.reshape(-1, self.n_issuers)
        # Map indices to labels
        flat_labels = self.transition_matrix.labels[flat_transitions]
        
        df = pd.DataFrame(flat_labels, columns=self.issuers.ids)
        # Add index columns
        # Repeat year sequence n_sim times
        df['year'] = np.tile(np.arange(n_years), n_sim)
        # Repeat scenario index, each repeated n_years times
        df['scenario'] = np.repeat(np.arange(n_sim), n_years)
        
        transitions_df = df.melt(id_vars=['scenario', 'year'], var_name='issuer_id', value_name='rating')

        return SimulationResult(
            E=E, S=S, I=I, X=X, pX=pX, 
            transitions=transitions,
            rating_labels=self.transition_matrix.labels,
            transitions_df=transitions_df, 
            n_sim=n_sim, n_years=n_years
        )

# ------------------------------------------------------------------------------
# Asset Classes
# ------------------------------------------------------------------------------
class Asset(BaseDataClass):
    """Base class for an asset object."""
    def __init__(self, data):
        self.data = data

    def run_sim(self, allocation, val_date, rates, spread_map, sim_results):
        raise NotImplementedError
    
    def run_sim_arrays(self, allocation, val_date, rates, spread_map, sim_results) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

class Bonds(Asset):
    """
    Class for holding data on bonds in a portfolio.
    Optimized to work with integer rating indices.
    """
    REQUIRED_COLUMNS = {'id': 'string' ,'issuer_id' : 'string', 'notional': 'numeric', 'recovery': 'numeric'}

    def __init__(self, data: pd.DataFrame, cashflow_table: pd.DataFrame, issuers: Issuers):
        self.data = data.copy()
        self.data['id'] = self.data['id'].astype(str)
        self.data['issuer_id'] = self.data['issuer_id'].astype(str)
        
        self.cashflow_table = cashflow_table.copy()
        if 'date' in self.cashflow_table.columns:
             self.cashflow_table['date'] = pd.to_datetime(self.cashflow_table['date'], dayfirst=True)
        self.cashflow_table.set_index('date', inplace=True)
        
        self.issuers = issuers
        self.validate_inputs()

        # Gather data for the bonds
        self.ids = np.array(self.data['id'])
        self.issuer_map = np.array(self.data['issuer_id'])
        self.notional_map = np.array(self.data['notional'])
        self.recovery_map = np.array(self.data['recovery'])
        self.n_bonds = len(self.data)
        
        # Pre-map bonds to issuer indices for fast lookup in simulation results
        self.bond_issuer_indices = np.array([self.issuers.id_to_idx[iid] for iid in self.issuer_map])

    def validate_inputs(self):
        super().validate_inputs()
        # Verify integrity
        cf_table_ids = set(self.cashflow_table.columns)
        bond_data_ids = set(self.data['id'])
        if cf_table_ids != bond_data_ids:
            raise KeyError("Mismatch between Bond Data and Cashflows.")
        
        # Verify issuers exist
        bond_issuers = set(self.data['issuer_id'])
        issuer_ids = set(self.issuers.ids)
        missing_issuers = bond_issuers - issuer_ids
        if missing_issuers:
            raise KeyError(f"Bond issuers not found in Issuers object: {missing_issuers}")

    def _get_spread_table(self, spread_map: dict, rating_labels: np.ndarray) -> np.ndarray:
        """Creates a lookup array where index=rating_int, value=spread."""
        arr = np.zeros(len(rating_labels))
        for i, label in enumerate(rating_labels):
            arr[i] = spread_map.get(label, 0.0)
        return arr

    def _get_default_index(self, rating_labels: np.ndarray) -> int:
        """Finds integer index for 'Default' state."""
        # Assuming label is "Default"
        matches = np.where(rating_labels == "Default")[0]
        if len(matches) > 0:
            return matches[0]
        return -1 # Should not happen if Default is in matrix

    def pv(self, val_date: datetime.datetime, rates: Rates, spread_map: dict) -> float:
        cfs_future = self.cashflow_table.loc[self.cashflow_table.index >= val_date]
        if cfs_future.empty:
            return 0.0

        # Explicitly cast to DatetimeIndex to resolve Pylance ambiguity
        dates = pd.DatetimeIndex(cfs_future.index)
        cashflows = cfs_future.to_numpy()

        yields = rates.interpolate(dates).to_numpy()[:, None]

        # Standard Pandas Merge for single point calculation is fine
        merged = self.data.merge(self.issuers.data[['id', 'rating']], how='left', left_on='issuer_id', right_on='id')
        spreads = merged["rating"].map(spread_map).to_numpy()

        yields_mat = np.broadcast_to(yields, cashflows.shape)
        spreads_mat = np.broadcast_to(spreads, cashflows.shape)

        dt = calc_dt(dates, val_date)
        dfs = (1 + yields_mat + spreads_mat) ** (-dt)

        return np.sum(cashflows * dfs)

    def sim_pv(self, allocation: float, val_date: datetime.datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates PV timeline. Returns raw arrays (pv, dates) for efficiency.
        """
        raw_pv = self.pv(val_date, rates, spread_map)
        ratio = allocation / raw_pv if raw_pv != 0 else 0
        cashflows_adj = self.cashflow_table.loc[val_date:] * ratio
        
        cashflows_np = cashflows_adj.to_numpy()
        
        # Explicitly cast to DatetimeIndex to resolve Pylance ambiguity
        dates = pd.DatetimeIndex(cashflows_adj.index)
        
        n_years_cf = len(dates)
        n_sim = sim_results.n_sim
        n_bonds = self.n_bonds

        # 1. Get Integer Ratings for Bonds (S, T, B)
        # Using pre-calculated issuer indices to slice the 3D transitions array
        bond_ratings = sim_results.transitions[:, :n_years_cf, self.bond_issuer_indices]

        # 2. Map Ratings to Spreads using NumPy lookup (S, T, B)
        spread_table = self._get_spread_table(spread_map, sim_results.rating_labels)
        spreads_3d = spread_table[bond_ratings]

        # 3. Rates
        yields = rates.interpolate(dates).to_numpy()
        dt = ((dates - val_date).days / DAYS_PER_YEAR).to_numpy()

        # 4. Vectorized Discounting
        pvs = np.zeros((n_sim, n_years_cf, n_bonds))
        
        for t in range(n_years_cf - 1):
            future_cfs = cashflows_np[t+1:] # (Remaining, B)
            delta_t = (dt[t+1:] - dt[t])[:, None] # (Remaining, 1)
            
            # Spread at time t applies to future
            spread_at_t = spreads_3d[:, t, :] # (S, B)
            yields_future = yields[t+1:][:, None] # (Remaining, 1)
            
            # (1, Rem, 1) + (S, 1, B) -> (S, Rem, B)
            total_rate = yields_future[None, :, :] + spread_at_t[:, None, :]
            
            dfs = (1 + total_rate) ** -delta_t[None, :, :]
            term_pvs = future_cfs[None, :, :] * dfs
            pvs[:, t, :] = term_pvs.sum(axis=1)

        # 5. Default Logic
        default_idx = self._get_default_index(sim_results.rating_labels)
        if default_idx != -1:
            is_default = (bond_ratings == default_idx)
            pvs *= (~is_default)

        return pvs, dates.to_numpy()

    def sim_cashflows(self, allocation: float, val_date: datetime.datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult) -> np.ndarray:
        """Returns raw cashflow array (S, T, B)."""
        raw_pv = self.pv(val_date, rates, spread_map)
        ratio = allocation / raw_pv if raw_pv != 0 else 0
        cashflows_adj = self.cashflow_table.loc[val_date:] * ratio

        cashflows = cashflows_adj.to_numpy()
        n_years = len(cashflows)
        n_sim = sim_results.n_sim

        # Get Ratings (S, T, B)
        bond_ratings = sim_results.transitions[:, :n_years, self.bond_issuer_indices]

        # Active logic
        years = np.arange(n_years)[:, None]
        maturity_idx = n_years - np.argmax((cashflows != 0)[::-1, :], axis=0)
        active_mask = (years < maturity_idx).astype(int)

        # Recovery
        recovery_val = self.recovery_map * self.notional_map * ratio
        
        # Default
        default_idx = self._get_default_index(sim_results.rating_labels)
        
        if default_idx != -1:
            is_default = (bond_ratings == default_idx).astype(int)
            # First default logic: (S, T, B)
            default_cumsum = np.cumsum(is_default, axis=1)
            first_default_mask = (is_default == 1) & (default_cumsum == 1)
            not_defaulted_mask = (default_cumsum == 0)
        else:
            first_default_mask = np.zeros_like(bond_ratings)
            not_defaulted_mask = np.ones_like(bond_ratings)

        # Calc Flows
        regular_flows = cashflows[None, :, :] * not_defaulted_mask
        recovery_flows = recovery_val[None, None, :] * first_default_mask * active_mask[None, :, :]
        
        return regular_flows + recovery_flows

    def run_sim(self, allocation: float, val_date: datetime.datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):
        """
        Legacy wrapper. Computes PVS/CFS, returns DataFrame for Portfolio aggregation.
        """
        pvs, dates = self.sim_pv(allocation, val_date, rates, spread_map, sim_results)
        cfs = self.sim_cashflows(allocation, val_date, rates, spread_map, sim_results)
        
        n_sim = sim_results.n_sim
        
        # Create output dataframe structure matching expected API
        # Flattening arrays (S, T, B) -> (S*T*B)
        
        # To avoid massive DF creation inside loop, we construct it once here
        results = pd.DataFrame(
            {
                "cashflow": cfs.reshape(-1),
                "pv": pvs.reshape(-1)
            },
            index=pd.MultiIndex.from_product(
                [range(n_sim), dates.tolist(), self.ids.tolist()],
                names=["scenario", "date", "bond_id"]
            )
        ).reset_index()
        
        # Aggregate to Scenario-Date level immediately to save memory
        aggregated = results.groupby(['scenario', 'date'])[['cashflow', 'pv']].sum().reset_index()
        return aggregated
    
    def run_sim_arrays(self, allocation: float, val_date: datetime.datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):
        """
        Fast path: returns aggregated numpy arrays (S, T) for cashflow and PV.
        """
        pvs, dates = self.sim_pv(allocation, val_date, rates, spread_map, sim_results)
        cfs = self.sim_cashflows(allocation, val_date, rates, spread_map, sim_results)
        
        # Sum over bonds (axis 2) -> (S, T)
        total_pvs = pvs.sum(axis=2)
        total_cfs = cfs.sum(axis=2)
        
        return total_cfs, total_pvs, dates


class Portfolio:
    """Class to hold a portfolio of assets."""
    def __init__(self, asset_list: List[Asset], allocations: List[float]):
        self.assets = asset_list
        self.allocations = allocations
        self.total_allocation = np.sum(self.allocations)

    def run_sim(self, val_date: datetime.datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):
        # Legacy DF output
        sims = [
            asset.run_sim(allocation, val_date, rates, spread_map, sim_results)
            for asset, allocation in zip(self.assets, self.allocations)
        ]
        combined = pd.concat(sims).groupby(['date', 'scenario']).sum().reset_index()
        return combined

    def run_sim_arrays(self, val_date: datetime.datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregates arrays from all assets."""
        total_cf = None
        total_pv = None
        dates = None
        
        for asset, allocation in zip(self.assets, self.allocations):
            if hasattr(asset, 'run_sim_arrays'):
                cf, pv, d = asset.run_sim_arrays(allocation, val_date, rates, spread_map, sim_results)
                
                if total_cf is None:
                    total_cf = cf
                    total_pv = pv
                    dates = d
                else:
                    total_cf += cf
                    total_pv += pv
            else:
                raise NotImplementedError("Asset does not support fast array simulation")
        
        if total_cf is None or total_pv is None or dates is None:
             raise ValueError("Portfolio yielded no results (possibly empty).")
             
        return total_cf, total_pv, dates

# ------------------------------------------------------------------------------
# Mandate Classes
# ------------------------------------------------------------------------------
@dataclass
class CDISimulationResult:
    cdi_results: pd.DataFrame
    bond_results: pd.DataFrame
    expected_pv_payment: float

class CDIMandate:
    def __init__(self, liabilities: Liabilities, portfolio: Portfolio, cash: float):
        self.liabilities = liabilities
        self.portfolio = portfolio
        self.cash = cash

    def run(self, val_date: datetime.datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):
        pass

class CDIMandate_Fox(CDIMandate):
    def __init__(
        self, liabilities: Liabilities, portfolio: Portfolio, cash: float,
        asset_buffer: np.ndarray, gaap_int: float, mortality_buffer: float,
        special_payment_year: int = 10
    ):
        super().__init__(liabilities, portfolio, cash)
        self.asset_buffer = asset_buffer
        self.gaap_int = gaap_int
        self.mortality_buffer = mortality_buffer
        self.special_payment_year = special_payment_year

    def _payment_expected_value(self, cdi_results: pd.DataFrame, val_date: datetime.datetime, rates: Rates):
        df = cdi_results.copy()
        dates_idx = pd.DatetimeIndex(pd.to_datetime(df['date']))
        df['t'] = calc_dt(dates_idx, val_date)
        yields = rates.yields.to_frame(name='yield')
        merged = df.merge(yields, how='left', left_on='date', right_index=True)
        
        merged['payment_pv'] = merged['payment'] * (1 + merged['yield'])**(-merged['t'])
        payments = merged.groupby(['scenario'])['payment_pv'].sum()
        return payments.mean()

    def run(self, val_date: datetime.datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult) -> CDISimulationResult:
        
        # 1. Base Calcs
        L0_gaap = self.liabilities.pv(rates=self.gaap_int, val_date=val_date)
        mortality_risk = self.mortality_buffer / L0_gaap if L0_gaap != 0 else 0
        day_0_hgb_gap = (L0_gaap + self.mortality_buffer) - (self.cash + self.portfolio.total_allocation + self.asset_buffer[0])

        # 2. Liability Prep
        liability_cashflows = self.liabilities.cashflows.loc[val_date:]
        dates = pd.DatetimeIndex(liability_cashflows.index)
        T = len(dates)
        val_date_ts = pd.to_datetime(val_date)
        dt = ((dates - val_date_ts).days / DAYS_PER_YEAR).to_numpy()

        liability_pvs = np.zeros(T)
        liab_cf_np = liability_cashflows.to_numpy()
        
        for i in range(T):
            if i < T - 1:
                t_delta = dt[i+1:] - dt[i]
                df = (1 + self.gaap_int) ** -t_delta
                liability_pvs[i] = np.sum(liab_cf_np[i+1:] * df)

        cumulative_cf = liability_cashflows.cumsum()
        meltdown_liabilities = np.maximum(
            (L0_gaap - cumulative_cf) * (1 + mortality_risk) * (1 + self.gaap_int) ** dt, 0
        )
        meltdown_liab_np = np.array(meltdown_liabilities)

        next_2_liabs = (liability_cashflows.shift(-1) + liability_cashflows.shift(-2)).fillna(0)
        next_2_liabs_np = next_2_liabs.to_numpy()

        # 3. Asset Simulation (FAST PATH)
        starting_cash = self.cash
        
        # Use optimized array return (S, T)
        bond_cfs, bond_pvs, bond_dates = self.portfolio.run_sim_arrays(val_date, rates, spread_map, sim_results)
        n_sim = sim_results.n_sim

        # Handle Mismatch in time horizons
        # We need bond arrays to be at least T length
        T_bonds = bond_cfs.shape[1]
        
        if T_bonds < T:
            pad_width = T - T_bonds
            # Pad axis 1 (time) with zeros at the end
            # ((0,0) for sim axis, (0, pad_width) for time axis)
            bond_cfs = np.pad(bond_cfs, ((0, 0), (0, pad_width)), mode='constant')
            bond_pvs = np.pad(bond_pvs, ((0, 0), (0, pad_width)), mode='constant')

        # 4. Vectorized Waterfall (Simulating all scenarios at once per timestep)
        fwds = rates.calc_fwds(val_date).to_numpy()
        
        # State Vectors (S, T) or (S,)
        cash_t = np.full(n_sim, starting_cash)
        hgb_gap_t = np.full(n_sim, day_0_hgb_gap)
        
        # Result collectors
        res_cash = np.zeros((n_sim, T))
        res_assets = np.zeros((n_sim, T))
        res_payment = np.zeros((n_sim, T))
        
        # Track total payment to handle "payment[:t].sum() == 0" logic
        total_payment_so_far = np.zeros(n_sim)

        for t in range(T):
            # Roll Forward
            fwd_rate = fwds[t] if t < len(fwds) else fwds[-1]
            
            # (S,) = (S,) * scalar + (S,) - scalar
            cash_t = cash_t * (1 + fwd_rate) + bond_cfs[:, t] - liab_cf_np[t]
            
            assets_t = cash_t + bond_pvs[:, t]
            
            # Buffer
            buffer_t = self.asset_buffer[t+1] if (t+1) < len(self.asset_buffer) else 0
            
            meltdown_assets_t = assets_t + buffer_t
            
            # Gap
            current_gap = meltdown_liab_np[t] - meltdown_assets_t
            # clip(0, prev)
            hgb_gap_t = np.clip(current_gap, 0, hgb_gap_t)
            
            # Special Payment (t specific)
            if t == self.special_payment_year:
                # Mask of scenarios needing payment
                mask = assets_t < liability_pvs[t]
                shortfall = liability_pvs[t] - assets_t
                extra = np.minimum(shortfall, buffer_t)
                
                # Apply only where mask is True
                cash_t += (extra * mask)
                assets_t += (extra * mask)
            
            # Insolvency Trigger
            # Check condition: Assets < Next2Liabs AND No prior payment
            insolvent_mask = (assets_t < next_2_liabs_np[t]) & (total_payment_so_far == 0)
            
            payment_now = np.zeros(n_sim)
            # If triggered, pay gap
            payment_now[insolvent_mask] = hgb_gap_t[insolvent_mask]
            
            # Injection
            cash_t += payment_now
            assets_t += payment_now
            
            total_payment_so_far += payment_now
            
            # Store
            res_cash[:, t] = cash_t
            res_assets[:, t] = assets_t
            res_payment[:, t] = payment_now

        # 5. Format Output to match legacy DataFrame structure
        # Flatten arrays (S, T) -> S*T
        flat_scenarios = np.repeat(np.arange(n_sim), T)
        flat_dates = np.tile(dates, n_sim)
        
        # Reconstruct Bond DF for legacy result requirement
        bond_df = pd.DataFrame({
            'scenario': flat_scenarios,
            'date': flat_dates,
            'cashflow': bond_cfs[:, :T].ravel(), # Slice to T if bonds were longer
            'pv': bond_pvs[:, :T].ravel()
        })

        cdi_results = pd.DataFrame({
            'date': flat_dates,
            'scenario': flat_scenarios,
            'liability_pvs': np.tile(liability_pvs, n_sim),
            'assets': res_assets.ravel(),
            'payment': res_payment.ravel(),
            'cash': res_cash.ravel()
        })

        expected_pv = self._payment_expected_value(cdi_results, val_date, rates)

        return CDISimulationResult(cdi_results, bond_df, expected_pv)