import numpy as np
import pandas as pd
import scipy as sp
import warnings

from datetime import datetime
from typing import Any, List, Sequence, Union


# --- Constants ---
DAYS_PER_YEAR = 365.0
RATINGS_ORDER = [
    'AAAA', 'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
    'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-',
    'CCC+', 'CCC', 'CCC-', 'CC+', 'Def'
]
DEFAULT_LABEL = "Def"

# --- Helper Functions --- #

def calc_dt(dates: Union[pd.DatetimeIndex, Sequence[datetime]], val_date: datetime) -> np.ndarray:
    """Calculate the time deltas between a valuation date and a list of future dates."""
    if isinstance(dates, pd.DatetimeIndex):
        delta_days = (dates - pd.Timestamp(val_date)).days
    else:
        dates = pd.to_datetime(dates)
        delta_days = (dates - pd.Timestamp(val_date)).days
    dt = np.asarray(delta_days, dtype=float) / DAYS_PER_YEAR
    return dt

def map_spreads(ratings: np.ndarray, spread_map: Dict[str, float]) -> np.ndarray:
    """Maps rating labels to spread floats."""
    # Default to 0.0 if rating not found
    flat = ratings.ravel()
    codes, uniques = pd.factorize(flat, sort=False)
    mapped = np.fromiter((spread_map.get(u, 0.0) for u in uniques), dtype=float, count=len(uniques))
    spreads = mapped[codes].reshape(ratings.shape)
    return spreads

def compute_maturity_flags(self, cashflows: np.ndarray) -> np.ndarray:
    """Identify years in which a bond is active based on 2d cashflow table (n_years x n_bonds)"""
    
    # Shape
    n_bonds = cashflows.shape[1]
    n_years = cashflows.shape[0]
    
    # Find the index of last non-zero cashflow for each bond
    last_cf_idx = np.full(n_bonds, n_years, dtype=int)
    for bond_idx in range(n_bonds):
        non_zero = np.where(cashflows[:, bond_idx] != 0)[0]
        if len(non_zero) > 0:
            last_cf_idx[bond_idx] = non_zero[-1]

    # Create maturity flags: True at/before maturity, False after maturity
    # Shape: (n_years, n_bonds)
    year_indices = np.arange(n_years)[:, np.newaxis]
    return year_indices <= last_cf_idx[np.newaxis, :]


# --- Rates --- #

class Rates:

    def __init__(self, yields: np.ndarray, dates: List):

        if len(yields) != len(dates):
            raise ValueError("Yields and dates length mismatch.")

        self.yields = pd.Series(yields, index = dates)

    def interpolate(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Linear interpolation of yields to target dates."""
        combined_index = self.yields.index.union(dates).sort_values()
        interpolated = (
            self.yields
            .reindex(combined_index)
            .interpolate(method="time")
            .reindex(dates)
        )
        return interpolated
        
    def calc_fwds(self, val_date: datetime, dates: pd.DatetimeIndex = None) -> pd.Series:

        if dates is not None:
            y = self.interpolate(dates)
        else:
            y = self.yields.loc[self.yields.index >= val_date]

        if len(y) < 2:
            raise ValueError("Not enough data points after the valuation date to calculate forwards.")

        # Time to maturity in years
        t = (y.index - val_date).days / DAYS_PER_YEAR
        acc = np.power(1 + y.values, t)

        # Differences in time (years)
        dt = np.diff(t)

        # Compute forward rates for periods 1..n
        fwds_values = np.empty_like(y.values)
        fwds_values[0] = y.iloc[0]  # first forward equal to first yield
        fwds_values[1:] = (acc[1:] / acc[:-1]) ** (1 / dt) - 1

        return pd.Series(fwds_values, index=y.index, name="fwds")


class TransitionMatrix:
    r"""
    Class for holding a credit rating transition matrix.
    Matrix must be defined as a square matrix with labels.
    """

    def __init__(self, tmatrix: np.ndarray, labels: list[str]):

        self.tmatrix = tmatrix
        self.labels = np.array(labels)
        self.cum_tmatrix = np.cumsum(tmatrix, axis=1)  # cumulative probabilities for np.searchsorted

        # Labels
        self.label_to_idx = {l: i for i, l in enumerate(labels)}

        # Check transitions sum to 1
        bad_rows = np.where(~np.isclose(self.cum_tmatrix[:, -1], 1.0))[0]
        if bad_rows.size:
            raise ValueError(f"Transition probabilities for {self.labels[bad_rows]} do not sum to 1")


    def _transitions_vector(self, current_ratings: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Calculate the 1 step ratings migrations for a vector of length n_issuers: current_ratings."""
        cum_tmatrix_long = self.cum_tmatrix[current_ratings]
        next_ratings = np.sum(p > cum_tmatrix_long, axis = 1)
        return next_ratings

    def transitions(self, pX: np.ndarray, ratings_map: np.ndarray) -> np.ndarray:
        """
        Calculate the ratings migrations for n_issuers of n_years for n_sims.
        pX is an (n_sim x n_years x n_issuers) array of CDF(N(0,1)) variables.
        """

        # Get shape parameters
        n_sim, n_years, n_issuers = pX.shape
        assert n_issuers == len(ratings_map), "Mismatch in number of bond issuers."

        # Convert rating labels to indices
        initial_rating_indices = np.array([self.label_to_idx[r] for r in ratings_map])

        # Repeat Initial Ratings for all scenarios
        initial_ratings_array = np.broadcast_to(initial_rating_indices, (n_sim, n_issuers))                   # (n_sim x n_bonds)

        # Create holding results array for all sims. Add extra year to include starting ratings.
        transitions_idx = np.empty((n_sim, n_years + 1, n_issuers), dtype=int)                                # (n_sim x n_years x n_bonds)

        # Set t=0 ratings to initial ratings
        transitions_idx[:, 0] = initial_ratings_array

        # For each timestep and scenario, extract the vector of n_issuers ratings and apply _transitions_vector
        for t in range(n_years):
            for i in range(n_sim):
                current_ratings = transitions_idx[:, t][i]
                p = pX[i, t][:, np.newaxis]
                transitions_idx[i, t + 1, :] = self._transitions_vector(current_ratings, p)

        # Convert Indices back to labels
        transitions = self.labels[transitions_idx]

        return transitions

    def __str__(self):
        return str(pd.DataFrame(self.tmatrix, columns=self.labels, index=self.labels).to_markdown(floatfmt = ".1%"))


class SimulationResult:
    """
    Class for holding results of a CreditRiskModel run.
    """
    def __init__(
        self,
        E: Any,
        S: Any,
        I: Any,
        X: Any,
        pX: Any,
        transitions: Any,
        n_sim: int,
        n_years: int
     ):
        self.E = E
        self.S = S
        self.I = I
        self.X = X
        self.pX = pX
        self.transitions = transitions
        self.n_sim = n_sim
        self.n_years = n_years

class CreditRiskModel:
    """
    Monte Carlo factor-based Credit Risk Model based on the BRS Credit VaR Model.
    """

    def __init__(
        self,
        transition_matrix: TransitionMatrix,
        rho_e: float,
        rho_s: np.ndarray,
        issuer_ids: list,
        sector_map: pd.Series,
        ratings_map: pd.Series
    ):
        self.transition_matrix = transition_matrix
        self.rho_e = rho_e
        self.rho_s = rho_s
        self.issuer_ids = issuer_ids
        self.sector_map = sector_map
        self.ratings_map = ratings_map

    def run(self,  n_sim: int, n_years: int):

        n_issuers = len(self.ratings_map)
        n_issuer_sectors = self.sector_map.nunique()

        # Random draws
        E = np.random.normal(size=(n_sim, n_years))
        S = np.random.normal(size=(n_sim, n_years, n_issuer_sectors))
        I = np.random.normal(size=(n_sim, n_years, n_issuers))

        rho_s_i = self.rho_s[self.sector_map]
        X = np.sqrt(self.rho_e) * E[:, :, np.newaxis] + \
            np.sqrt(rho_s_i - self.rho_e)[np.newaxis, np.newaxis, :] * S[:, :, self.sector_map] +\
            np.sqrt(1 - rho_s_i)[np.newaxis, np.newaxis, :] * I

        pX = sp.stats.norm.cdf(X)

        # Calculate ratings transitions
        transitions = self.transition_matrix.transitions(pX, self.ratings_map)

        # format transitions into a dataframe to retain issuer id info.
        transitions = pd.DataFrame(
            transitions.reshape(n_sim * (n_years+1), n_issuers),
            index=pd.MultiIndex.from_product(
                [range(n_sim), range(n_years + 1)],
                names=["sim", "year"]
            ),
            columns=self.issuer_ids
        )

        return SimulationResult(E, S, I, X, pX, transitions, n_sim, n_years)
 
# ---- Assets --- #

class Issuers:
    """Data class for bond issuer details."""
    def __init__(
        self,
        ids: List[str],
        ratings: pd.Series,
        sectors: pd.Series,
        names: Optional[pd.Series] = None
    ):
        self.ids = ids
        self.ratings = ratings
        self.sectors = sectors
        self.names = names
        

 class BondSimulationResult:
    """Class to hold the results of a stochastic bond simulation and manipulate output."""

    def __init__(
        self,
        transitions,
        received_cashflows,
        recovery_payments,
        total_cashflows,
        pvs,
        dates,
        bond_ids
    ):
        self.transitions=transitions
        self.received_cashflows=received_cashflows
        self.recovery_payments=recovery_payments
        self.total_cashflows=total_cashflows
        self.pvs=pvs
        self.dates=dates
        self.bond_ids=bond_ids

    def to_dataframe(self, by_bond: bool = True):
        df = pd.DataFrame(
            data = {
                'rating': self.transitions.reshape(-1),
                'cashflow': self.total_cashflows.reshape(-1),
                'pv': self.pvs.reshape(-1)
            },
            index = pd.MultiIndex.from_product(
                iterables = [range(self.total_cashflows.shape[0]), self.dates, self.bond_ids],
                names = ["scenario", "date", "bond_id"]
            )
        ).reset_index()

        if not by_bond:
           return df.groupby(['scenario', 'date']).sum().drop(columns='bond_id').reset_index()
        else:
            return df


class Bonds:
    """Class for holding bond information and cashflows"""

    def __init__(
        self,
        ids: list,
        issuer_ids: pd.Series,
        recoveries: pd.Series,
        cashflows: pd.DataFrame,
        issuers: Issuers,
        descriptions: pd.Series | None = None
    ):
        self.ids = ids
        self.issuer_ids = issuer_ids
        self.recoveries = recoveries
        self.cashflows = cashflows[ids].sort_index()
        self.issuers = issuers
        self.descriptions = descriptions

    def year_end_cashflows(self, val_date: datetime | None = None) -> pd.DataFrame:
        """Returns year end cashflows."""
        cf = self.cashflows.copy()

        # Move cashflow dates to month-end (cashflow dates are usually beginning of month)
        cf = cf.resample("ME").sum()

        # Filter out cashflows before val_date
        if val_date is not None:
            cf = cf.loc[cf.index > val_date]

        # Return cashflows at year-end
        return cf.resample("YE").sum()

    def pv(self, val_date: pd.Timestamp, rates: Rates, spread_map: Dict[str, float]) -> pd.Series:
        """Calculates bond prices at val_date according to market data (rates, spreads)."""
        cfs = self.year_end_cashflows(val_date)
        dt = calc_year_frac(cfs.index, val_date)

        # Market Data
        base_yields = rates.interpolate(cfs.index).values
        current_ratings = self.issuer_ids.map(self.issuers.ratings).reindex(self.ids)
        spreads = self._get_spreads(current_ratings.values, spread_map) # (n_bonds,)

        # Discount Factors: (T x N)
        # (1 + r + s)^-t
        dfs = (1 + base_yields[:, None] + spreads[None, :]) ** (-dt[:, None])

        prices = (cfs.values * dfs).sum(axis=0)

        return pd.Series(prices, index=self.ids)

    def run_sim(self, val_date: datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):
        """Calculate cashflows and PV timeline for bonds in portfolio for a given set of economy."""

        # Get cashflows, dates
        cashflows_df = self.year_end_cashflows(val_date)
        cashflows = cashflows_df.values
        dates = cashflows_df.index

        # Shape parameters
        n_sim = sim_results.n_sim
        n_years_sim = sim_results.n_years
        n_years = cashflows.shape[0]
        n_bonds = cashflows.shape[1]

        # Recovery Rates
        recovery_rates = self.recoveries.values

        # Maturity flags
        not_matured = compute_maturity_flags(cashflows)

        # map issuer transitions to bonds
        bond_transitions = (
            # map from issuer ids to bond ids and ignore year 0 since this is initial rating.
            sim_results.transitions
            .loc[sim_results.transitions.index.get_level_values("year") != 0, self.issuer_ids]
            .set_axis(self.ids, axis=1)
            # convert to 3d array
            .to_numpy()
            .reshape(n_sim, n_years_sim, n_bonds)
        )

        # Cashflows received from non-defaulted bonds
        received_cashflows = cashflows[np.newaxis, :n_years_sim, :]  * (bond_transitions != DEFAULT_LABEL)

        # Identify first default
        first_default = (np.cumsum(~not_defaulted, axis=1) == 1)

        # Recovery Payments. Don't pay recovery if bond has already matured
        recovery_payments = first_default * recovery_rates * not_matured[np.newaxis, :n_years_sim, :]

        # Total Cashflows = Cashflows from non-defaulted + Recovery Payments
        total_cashflows =  received_cashflows +  recovery_payments

        # Map transitions to spreads using efficient vectorized lookup
        spreads = map_spreads(bond_transitions, spread_map)

        yields = rates.interpolate(dates).values
        dt = calc_dt(dates, val_date)

        pvs = np.zeros((n_sim, n_years_sim, n_bonds))

        for t in range(n_years_sim):                                                                # T* = (T-t-1)

            # Disocunt Factors
            dfs = np.power(
                1 + yields[np.newaxis, t+1:, np.newaxis]  + spreads[:, [t], :],
                -(dt[t+1:] - dt[t])[np.newaxis, :, np.newaxis]
            )                                                                                   # (n_sim x T* x n_bonds)

            # Discount all cashflows to t
            discounted_cfs = cashflows[np.newaxis, t+1: , :]  * dfs                             # (n_sim x T* x n_bonds)

            # Sum across years to get PV at t
            pvs[:, t, :] = discounted_cfs.sum(axis = 1)                                         # (n_sim x 1 x n_bonds)

        return BondSimulationResult(
            transitions=bond_transitions,
            received_cashflows=received_cashflows,
            recovery_payments=recovery_payments,
            total_cashflows=total_cashflows,
            pvs=pvs,
            dates=dates[:n_years_sim],
            bond_ids=self.ids
        )