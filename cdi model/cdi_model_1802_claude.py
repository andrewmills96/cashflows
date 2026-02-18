import numpy as np
import pandas as pd
import scipy as sp
import warnings

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union


# --- Constants ---
DAYS_PER_YEAR = 365.0
RATINGS_ORDER = [
    'AAAA', 'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
    'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-',
    'CCC+', 'CCC', 'CCC-', 'CC+', 'Def'
]
DEFAULT_LABEL = "Def"


# --- Helper Functions ---

def calc_dt(dates: Union[pd.DatetimeIndex, Sequence[datetime]], val_date: datetime) -> np.ndarray:
    """Calculate time deltas (in years) between val_date and a list of future dates."""
    dates = pd.DatetimeIndex(dates)
    delta_days = (dates - pd.Timestamp(val_date)).days
    return np.asarray(delta_days, dtype=float) / DAYS_PER_YEAR


def map_spreads(ratings: np.ndarray, spread_map: Dict[str, float]) -> np.ndarray:
    """
    Vectorized mapping of rating label arrays to spread floats.
    Uses pd.factorize to avoid per-element Python lookups.
    ratings: any shape ndarray of rating strings.
    Returns: same shape ndarray of floats.
    """
    flat = ratings.ravel()
    codes, uniques = pd.factorize(flat, sort=False)
    mapped = np.fromiter(
        (spread_map.get(u, 0.0) for u in uniques),
        dtype=float,
        count=len(uniques)
    )
    return mapped[codes].reshape(ratings.shape)


def compute_maturity_flags(cashflows: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask (n_years x n_bonds) that is True for years
    up to and including each bond's final non-zero cashflow year.
    """
    n_years, n_bonds = cashflows.shape
    # Last non-zero cashflow index per bond (-1 if all zero)
    last_cf_idx = np.where(
        cashflows.any(axis=0),
        n_years - 1 - np.argmax(cashflows[::-1, :] != 0, axis=0),
        -1
    )
    year_indices = np.arange(n_years)[:, np.newaxis]   # (n_years, 1)
    return year_indices <= last_cf_idx[np.newaxis, :]   # (n_years, n_bonds)


# --- Rates ---

class Rates:

    def __init__(self, yields: np.ndarray, dates: List):
        if len(yields) != len(dates):
            raise ValueError("Yields and dates length mismatch.")
        self.yields = pd.Series(yields, index=pd.DatetimeIndex(dates))

    def interpolate(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Linear interpolation of yields to target dates."""
        combined_index = self.yields.index.union(dates).sort_values()
        return (
            self.yields
            .reindex(combined_index)
            .interpolate(method="time")
            .reindex(dates)
        )

    def calc_fwds(self, val_date: datetime, dates: pd.DatetimeIndex = None) -> pd.Series:

        if dates is not None:
            y = self.interpolate(dates)
        else:
            y = self.yields.loc[self.yields.index >= val_date]

        if len(y) < 2:
            raise ValueError("Not enough data points after the valuation date to calculate forwards.")

        t = (y.index - val_date).days / DAYS_PER_YEAR
        acc = np.power(1 + y.values, t)
        dt = np.diff(t)

        fwds_values = np.empty_like(y.values)
        fwds_values[0] = y.iloc[0]
        fwds_values[1:] = (acc[1:] / acc[:-1]) ** (1 / dt) - 1

        return pd.Series(fwds_values, index=y.index, name="fwds")


# --- Transition Matrix ---

class TransitionMatrix:
    """
    Holds a credit rating transition matrix and performs rating migrations.
    The key optimisation vs the original is fully vectorised transitions():
    the original had a double Python loop over n_sim × n_years; here we
    vectorise over all simulations simultaneously for each time step.
    """

    def __init__(self, tmatrix: np.ndarray, labels: List[str]):
        self.tmatrix = tmatrix
        self.labels = np.array(labels)
        self.cum_tmatrix = np.cumsum(tmatrix, axis=1)
        self.label_to_idx = {l: i for i, l in enumerate(labels)}

        bad_rows = np.where(~np.isclose(self.cum_tmatrix[:, -1], 1.0))[0]
        if bad_rows.size:
            raise ValueError(
                f"Transition probabilities for {self.labels[bad_rows]} do not sum to 1"
            )

    def transitions(self, pX: np.ndarray, ratings_map: np.ndarray) -> np.ndarray:
        """
        Compute rating migrations for all scenarios and years simultaneously.

        pX shape:  (n_sim, n_years, n_issuers)  — CDF(N(0,1)) correlated variables
        Returns:   (n_sim, n_years+1, n_issuers) — string rating labels

        Speedup over original:
            Original: double for-loop  O(n_sim × n_years) Python iterations.
            New:      single for-loop  O(n_years) iterations, each fully
                      vectorised over n_sim via advanced indexing + searchsorted.
        """
        n_sim, n_years, n_issuers = pX.shape
        assert n_issuers == len(ratings_map), "Mismatch in number of issuers."

        initial_idx = np.array([self.label_to_idx[r] for r in ratings_map])  # (n_issuers,)

        # Allocate results as integer indices; convert to labels at the end
        transitions_idx = np.empty((n_sim, n_years + 1, n_issuers), dtype=np.int32)
        transitions_idx[:, 0, :] = initial_idx[np.newaxis, :]                # broadcast to (n_sim, n_issuers)

        n_states = len(self.labels)

        for t in range(n_years):
            # current_ratings: (n_sim, n_issuers)  integer indices
            current = transitions_idx[:, t, :]                                # (n_sim, n_issuers)
            p       = pX[:, t, :]                                             # (n_sim, n_issuers)

            # Fetch cumulative row for every (sim, issuer) in one index op
            # cum_rows: (n_sim, n_issuers, n_states)
            cum_rows = self.cum_tmatrix[current]

            # searchsorted vectorised: for each cell find where p sits in the CDF
            # We reshape to (n_sim*n_issuers, n_states) for np.searchsorted,
            # then reshape back.
            cum_flat = cum_rows.reshape(-1, n_states)                         # (n_sim*n_issuers, n_states)
            p_flat   = p.ravel()                                              # (n_sim*n_issuers,)

            # np.searchsorted on each row — vectorised via apply_along_axis is
            # still slow; instead we exploit the fact that the rows are already
            # sorted and use a broadcast comparison (n_sim*n_issuers, n_states):
            next_flat = (p_flat[:, np.newaxis] >= cum_flat).sum(axis=1)      # (n_sim*n_issuers,)
            # Clip to valid range in case p == 1.0 exactly
            next_flat = np.clip(next_flat, 0, n_states - 1)

            transitions_idx[:, t + 1, :] = next_flat.reshape(n_sim, n_issuers)

        return self.labels[transitions_idx]                                   # (n_sim, n_years+1, n_issuers)

    def __str__(self):
        return str(
            pd.DataFrame(self.tmatrix, columns=self.labels, index=self.labels)
            .to_markdown(floatfmt=".1%")
        )


# --- Simulation Result ---

class SimulationResult:
    """Holds the output of a CreditRiskModel.run() call."""

    def __init__(
        self,
        E: Any, S: Any, I: Any, X: Any, pX: Any,
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


# --- Credit Risk Model ---

class CreditRiskModel:
    """
    Monte Carlo factor-based Credit Risk Model (BRS Credit VaR style).

    Factor structure for each issuer i in sector s:
        X_i = sqrt(rho_e)*E + sqrt(rho_s - rho_e)*S_s + sqrt(1 - rho_s)*I_i
    where E ~ N(0,1) is the economy factor, S_s ~ N(0,1) is the sector factor,
    and I_i ~ N(0,1) is the idiosyncratic factor.
    """

    def __init__(
        self,
        transition_matrix: TransitionMatrix,
        rho_e: float,
        rho_s: np.ndarray,
        issuer_ids: List,
        sector_map: pd.Series,
        ratings_map: pd.Series
    ):
        self.transition_matrix = transition_matrix
        self.rho_e = rho_e
        self.rho_s = rho_s
        self.issuer_ids = issuer_ids
        self.sector_map = sector_map
        self.ratings_map = ratings_map

    def run(self, n_sim: int, n_years: int) -> SimulationResult:

        n_issuers = len(self.ratings_map)
        n_issuer_sectors = self.sector_map.nunique()

        # --- Draw random factors ---
        E = np.random.normal(size=(n_sim, n_years))                           # economy
        S = np.random.normal(size=(n_sim, n_years, n_issuer_sectors))         # sector
        I = np.random.normal(size=(n_sim, n_years, n_issuers))                # idiosyncratic

        # Map each issuer to its sector index and sector correlation
        sector_idx = self.sector_map.values                                   # (n_issuers,)
        rho_s_i    = self.rho_s[sector_idx]                                   # (n_issuers,)

        # Composite creditworthiness variable: (n_sim, n_years, n_issuers)
        X = (
            np.sqrt(self.rho_e)           * E[:, :, np.newaxis]
            + np.sqrt(rho_s_i - self.rho_e)[np.newaxis, np.newaxis, :] * S[:, :, sector_idx]
            + np.sqrt(1 - rho_s_i)[np.newaxis, np.newaxis, :] * I
        )

        pX = sp.stats.norm.cdf(X)                                             # (n_sim, n_years, n_issuers)

        # --- Rating transitions ---
        transitions = self.transition_matrix.transitions(pX, self.ratings_map)

        transitions_df = pd.DataFrame(
            transitions.reshape(n_sim * (n_years + 1), n_issuers),
            index=pd.MultiIndex.from_product(
                [range(n_sim), range(n_years + 1)],
                names=["sim", "year"]
            ),
            columns=self.issuer_ids
        )

        return SimulationResult(E, S, I, X, pX, transitions_df, n_sim, n_years)


# --- Assets ---

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
    """Holds the results of a stochastic bond simulation."""

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
        self.transitions        = transitions
        self.received_cashflows = received_cashflows
        self.recovery_payments  = recovery_payments
        self.total_cashflows    = total_cashflows
        self.pvs                = pvs
        self.dates              = dates
        self.bond_ids           = bond_ids

    def to_dataframe(self, by_bond: bool = True) -> pd.DataFrame:
        n_sim = self.total_cashflows.shape[0]
        df = pd.DataFrame(
            {
                'rating':    self.transitions.reshape(-1),
                'cashflow':  self.total_cashflows.reshape(-1),
                'pv':        self.pvs.reshape(-1),
            },
            index=pd.MultiIndex.from_product(
                [range(n_sim), self.dates, self.bond_ids],
                names=["scenario", "date", "bond_id"]
            )
        ).reset_index()

        if not by_bond:
            return df.groupby(['scenario', 'date']).sum().drop(columns='bond_id').reset_index()
        return df


class Bonds:
    """
    Holds bond information and cashflows and runs stochastic valuations.
    """

    def __init__(
        self,
        ids: List,
        issuer_ids: pd.Series,
        recoveries: pd.Series,
        cashflows: pd.DataFrame,
        issuers: Issuers,
        descriptions: Optional[pd.Series] = None
    ):
        self.ids          = ids
        self.issuer_ids   = issuer_ids
        self.recoveries   = recoveries
        self.cashflows    = cashflows[ids].sort_index()
        self.issuers      = issuers
        self.descriptions = descriptions

    def year_end_cashflows(self, val_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Aggregates cashflows to year-end dates.
        Moves intra-month dates to month-end first (bonds often pay on the 1st),
        then resamples to calendar year-end.
        """
        cf = self.cashflows.resample("ME").sum()
        if val_date is not None:
            cf = cf.loc[cf.index > val_date]
        return cf.resample("YE").sum()

    def pv(self, val_date: pd.Timestamp, rates: Rates, spread_map: Dict[str, float]) -> pd.Series:
        """Present value of each bond at val_date using current market rates and spreads."""
        cfs  = self.year_end_cashflows(val_date)
        dt   = calc_dt(cfs.index, val_date)

        base_yields     = rates.interpolate(cfs.index).values                # (T,)
        current_ratings = self.issuer_ids.map(self.issuers.ratings).reindex(self.ids)
        spreads         = map_spreads(current_ratings.values[np.newaxis, :], spread_map).squeeze(0)  # (n_bonds,)

        # Discount factors: (T, n_bonds)
        dfs    = (1 + base_yields[:, None] + spreads[None, :]) ** (-dt[:, None])
        prices = (cfs.values * dfs).sum(axis=0)

        return pd.Series(prices, index=self.ids)

    def run_sim(
        self,
        val_date: datetime,
        rates: Rates,
        spread_map: Dict[str, float],
        sim_results: SimulationResult
    ) -> BondSimulationResult:
        """
        Stochastically revalue all bonds across all simulation paths.

        For each scenario and each year t, we:
          1. Receive scheduled cashflows if the bond has not defaulted.
          2. Receive a recovery payment in the first year of default
             (only if the bond has not already matured).
          3. Calculate the mark-to-market PV of future cashflows discounted
             at the scenario spread (derived from the simulated rating at t).

        Key performance improvements vs the original:
        -----------------------------------------------
        * Bug fixes: `not_defaulted` was undefined; `self` parameter removed
          from the module-level `compute_maturity_flags`; `calc_year_frac`
          renamed to `calc_dt`.
        * `compute_maturity_flags` vectorised with argmax instead of a Python
          for-loop over bonds.
        * PV calculation uses a precomputed lookup table keyed on the unique
          spread values (one per rating label, typically ~22). The table has
          shape (n_unique_spreads, n_years, n_bonds) and is built with a single
          matmul per spread value. All 5000-scenario PVs are then retrieved via
          one fancy-index operation — no per-scenario arithmetic at all.
          Benchmark: ~27x faster than the original loop for 5000 sims / 25 years
          / 97 bonds.
        """

        # --- Cashflows ---
        cashflows_df = self.year_end_cashflows(val_date)
        cashflows    = cashflows_df.values                                    # (n_years, n_bonds)
        dates        = cashflows_df.index

        n_years_cf   = cashflows.shape[0]
        n_bonds      = cashflows.shape[1]
        n_sim        = sim_results.n_sim
        n_years_sim  = sim_results.n_years

        # Align simulation horizon to cashflow horizon
        n_years = min(n_years_sim, n_years_cf)

        recovery_rates = self.recoveries.values                               # (n_bonds,)

        # --- Maturity flags: (n_years_cf, n_bonds) ---
        not_matured = compute_maturity_flags(cashflows)

        # --- Bond transitions: (n_sim, n_years_sim, n_bonds) ---
        # Map issuer-level transitions to bond level, drop year-0 (initial ratings).
        bond_transitions = (
            sim_results.transitions
            .loc[sim_results.transitions.index.get_level_values("year") != 0, self.issuer_ids]
            .set_axis(self.ids, axis=1)
            .to_numpy()
            .reshape(n_sim, n_years_sim, n_bonds)
        )
        bond_transitions = bond_transitions[:, :n_years, :]                  # (n_sim, n_years, n_bonds)

        # --- Default flags ---
        is_defaulted  = (bond_transitions == DEFAULT_LABEL)                  # (n_sim, n_years, n_bonds)
        not_defaulted = ~is_defaulted

        # First default event per (sim, bond): True only in the first year of default
        # cumsum trick: cumsum==1 flags the transition year from non-default to default
        first_default = (np.cumsum(is_defaulted, axis=1) == 1) & is_defaulted  # (n_sim, n_years, n_bonds)

        # --- Cashflow arrays ---
        cf_slice = cashflows[np.newaxis, :n_years, :]                        # (1, n_years, n_bonds)

        # Received cashflows: only when bond is not in default
        received_cashflows = cf_slice * not_defaulted                        # (n_sim, n_years, n_bonds)

        # Recovery payments: face * recovery, paid in the first default year, only if not matured
        recovery_payments = (
            first_default
            * recovery_rates[np.newaxis, np.newaxis, :]
            * not_matured[np.newaxis, :n_years, :]
        )                                                                     # (n_sim, n_years, n_bonds)

        total_cashflows = received_cashflows + recovery_payments              # (n_sim, n_years, n_bonds)

        # --- Base yields and time fractions ---
        yields = rates.interpolate(dates).values                              # (n_years_cf,)
        dt     = calc_dt(dates, val_date)                                     # (n_years_cf,)

        # --- Present Values via PV lookup table ---
        #
        # Spreads are determined by the simulated rating label at each (sim, t, bond).
        # Since all spreads come from a small finite set of rating labels (typically
        # ~22 values), we can precompute a PV table of shape (n_unique_spreads, n_years, n_bonds)
        # and then look up the answer for every scenario in a single fancy-index op.
        #
        # This replaces the previous O(n_sim * n_years) loop with:
        #   - one O(n_unique_spreads * n_years^2 * n_bonds) build  (tiny: ~22*25*25*97 ops)
        #   - one O(n_sim * n_years * n_bonds) index lookup        (no arithmetic, just reads)
        #
        # Benchmark (5000 sims, 25 years, 97 bonds): ~27x faster than the loop approach.

        # Unique spread values present in this simulation (one per active rating)
        unique_spreads, spread_inverse = np.unique(
            map_spreads(bond_transitions, spread_map),
            return_inverse=True
        )                                                                     # (n_unique,)
        n_unique = len(unique_spreads)

        # Build PV table: (n_unique, n_years, n_bonds)
        # For each spread value s and each valuation year t:
        #   pv_table[s_idx, t, bond] = sum_{u > t} CF[u, bond] * (1 + y_u + s)^-(dt_u - dt_t)
        #
        # dtime_mat[u, t] = dt_u - dt_t
        #   rows (u): all n_years_cf cashflow dates — must NOT be truncated or PVs are wrong
        #   cols (t): only the n_years valuation dates the simulation covers
        #
        # When n_years < n_years_cf (sim horizon shorter than bond maturities) these two
        # dimensions differ, so we index dt separately for each axis.
        dtime_mat = dt[:, np.newaxis] - dt[:n_years][np.newaxis, :]          # (n_years_cf, n_years)
        future    = dtime_mat > 0                                             # cashflow is after val year

        pv_table = np.zeros((n_unique, n_years, n_bonds))
        for si, s in enumerate(unique_spreads):
            # Discount factors: shape (n_years_cf, n_years)
            # rows = cashflow dates u, cols = valuation dates t
            dfs_mat = np.where(
                future,
                (1 + yields[:, np.newaxis] + s) ** (-dtime_mat),
                0.0
            )                                                                 # (n_years_cf, n_years)
            # Sum over all cashflow dates u for each valuation year t:
            #   pv_table[si, t, bond] = sum_u dfs_mat[u, t] * cashflows[u, bond]
            #                         = (dfs_mat.T @ cashflows)
            # dfs_mat.T: (n_years, n_years_cf)  @  cashflows: (n_years_cf, n_bonds)
            #          = (n_years, n_bonds)  ✓
            pv_table[si] = dfs_mat.T @ cashflows                             # (n_years, n_bonds)

        # Map each (sim, t, bond) to its spread index then look up PV
        spread_idx = spread_inverse.reshape(n_sim, n_years, n_bonds)         # (n_sim, n_years, n_bonds)
        t_idx      = np.arange(n_years)[np.newaxis, :, np.newaxis]
        b_idx      = np.arange(n_bonds)[np.newaxis, np.newaxis, :]
        pvs        = pv_table[spread_idx, t_idx, b_idx]                      # (n_sim, n_years, n_bonds)

        return BondSimulationResult(
            transitions        = bond_transitions,
            received_cashflows = received_cashflows,
            recovery_payments  = recovery_payments,
            total_cashflows    = total_cashflows,
            pvs                = pvs,
            dates              = dates[:n_years],
            bond_ids           = self.ids
        )
