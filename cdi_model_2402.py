import numpy as np
import pandas as pd
import scipy as sp
from datetime import datetime
from typing import Any, List, Sequence, Union, Optional, Dict, Mapping, Tuple
from dataclasses import dataclass
from functools import reduce
import refinitiv.data as rd
import warnings

# --- Constants ---
DAYS_PER_YEAR = 365.0
RATINGS_ORDER = [
    'AAAA', 'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
    'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-',
    'CCC+', 'CCC', 'CCC-', 'CC+', 'Def'
]
DEFAULT_LABEL = "Def"

# --- Helper Functions --- #
def calc_year_frac(dates: Union[pd.DatetimeIndex, List[pd.Timestamp]], val_date: pd.Timestamp) -> np.ndarray:
    """Calculates year fractions between a list of dates and a valuation date (Act/365)."""
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.to_datetime(dates)

    delta_days = (dates - val_date).days
    return np.asarray(delta_days, dtype=float) / DAYS_PER_YEAR

def calc_dt(dates: Union[pd.DatetimeIndex, Sequence[datetime]], val_date: datetime) -> np.ndarray:
    """
    Calculate year fractions between a sequence of dates (or a pandas.DatetimeIndex)
    and a single valuation date.

    Parameters
    ----------
    dates : Union[pd.DatetimeIndex, Sequence[datetime]]
        A list, array, or pandas DatetimeIndex of date objects.
    val_date : datetime
        The valuation date to measure from.

    Returns
    -------
    np.ndarray
        vector of year fractions (shape: [t, ]).
    """

    # check dates are in the right format and calculate the number of days from val_date
    if isinstance(dates, pd.DatetimeIndex):
        delta_days = (dates - pd.Timestamp(val_date)).days
    else:
        dates = pd.to_datetime(dates)
        delta_days = (dates - pd.Timestamp(val_date)).days

    # standardise
    dt = np.asarray(delta_days, dtype=float) / DAYS_PER_YEAR

    return dt

def fit_to_shape(a, shape):
    """
    Function to convert the shape of an array `a` to `shape` by trimming and then zero-padding as needed.
    Works for 2D and 3D target shapes.
    """
    # If target is 3D but array is 2D, expand dimensions
    if len(shape) == 3 and a.ndim == 2:
        a = np.expand_dims(a, axis=-1)

    # Trim array to target shape
    trimmed = a[tuple(slice(0, min(a.shape[i], shape[i])) for i in range(len(shape)))]

    # Compute padding for each dimension
    pad_widths = []
    for i in range(len(shape)):
        pad_end = max(0, shape[i] - trimmed.shape[i])
        pad_widths.append((0, pad_end))

    # If the original array has fewer dimensions, pad those too
    while len(pad_widths) < trimmed.ndim:
        pad_widths.append((0, 0))

    # Apply padding
    padded = np.pad(trimmed, pad_widths, mode='constant')

    return padded

def get_refinitiv_bund_yields():
    """Get live bund yields from Refinitiv."""
    session = rd.open_session()
    germany_bonds = rd.discovery.Chain(name="0#DEBMK=")
    df = rd.get_data(universe=germany_bonds.constituents, fields=["MATUR_DATE", "CF_YIELD"])
    rd.close_session()

    # format
    df['CF_YIELD'] /= 100
    df["MATUR_DATE"] = pd.to_datetime(df["MATUR_DATE"])

    # add t
    today = pd.Timestamp.today().normalize()
    df["t"] = (df["MATUR_DATE"] - today).dt.days / 365.25

    # rename columns
    df = df.rename(columns={'MATUR_DATE':'date', 'CF_YIELD': 'yield'})

    return df[['date', 't', 'yield']]

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

    Uses np.flip() rather than [::-1] slicing to ensure a contiguous array
    is passed to np.argmax, avoiding stride-related issues on some numpy builds.
    """
    n_years, n_bonds = cashflows.shape
    # Flip along axis=0 (time axis) into a fresh contiguous array, then find
    # the first non-zero row — that corresponds to the last non-zero in the original.
    flipped      = np.flip(cashflows, axis=0)                                # contiguous copy
    has_cashflow = cashflows.any(axis=0)                                     # (n_bonds,)
    last_cf_idx  = np.where(
        has_cashflow,
        n_years - 1 - np.argmax(flipped != 0, axis=0),
        -1
    )                                                                        # (n_bonds,)
    year_indices = np.arange(n_years)[:, np.newaxis]                         # (n_years, 1)
    return year_indices <= last_cf_idx[np.newaxis, :]                        # (n_years, n_bonds)

def total_returns(pvs: np.ndarray, cashflows: np.ndarray, fwds: np.ndarray, a0: float):
    """Calculates the annual total return for a portfolio of bonds"""
    returns = np.empty((pvs.shape[0], pvs.shape[1]), dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        returns[:, 0] = (pvs[:, 0] + cashflows[:, 0]) / a0 - 1
        returns[:, 1:] = (pvs[:, 1:] + cashflows[:, 1:]) / pvs[:, :-1] - 1

    returns = np.where(np.isfinite(returns), returns, fwds)

    #checks
    assert returns.shape == (pvs.shape[0], pvs.shape[1])
    assert not np.isnan(returns).any()

    return returns

def allocate_bond_sim(bond_sim: BondSimulationResult, allocation: pd.Series, shape: tuple, target = 'cashflows'):

    # Ids and Notionals
    ids = allocation.index.to_list()
    allocations = allocation.values

    # map bond_id -> index in cashflows
    bond_idx = {bid: i for i, bid in enumerate(bond_sim.bond_ids)}

    # indices of the bonds you want (in the correct order)
    selected_idx = [bond_idx[id] for id in ids]

    # get cashflows or pvs
    if target == 'pvs':
        nominal = bond_sim.pvs[:, :, selected_idx]
    elif target == 'cashflows':
        nominal = bond_sim.total_cashflows[:, :, selected_idx]
    else:
        raise TypeError('Invalid Option Selected')

    net = nominal * allocations
    total = net.sum(axis=2)

    return fit_to_shape(total, shape)

# --- Rates --- #

class Rates:
    """
    Handles discount rate curves and interpolation.

    Inputs
    ----------
    data : pd.DataFrame
        Input DataFrame containing 'date' and 'yield' columns.
    """

    def __init__(self, yields: np.ndarray, dates: List):

        # Validation
        if len(yields) != len(dates):
            raise ValueError("Yields and dates length mismatch.")

        self.yields = pd.Series(yields, index = dates)

    def calc_fwds(self, val_date: datetime, dates: pd.DatetimeIndex = None) -> pd.Series:
        """
        Calculate implied forward rates starting from the valuation date.

        Parameters
        ----------
        val_date : datetime
            The valuation date.

        Returns
        -------
        pd.Series
            Forward rates indexed by date.
        """

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

        fwds = pd.Series(fwds_values, index=y.index, name="fwds")

        return fwds

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

# --- Risk Model --- #

class TransitionMatrix:
    r"""
    Class for holding a credit rating transition matrix.
    Matrix must be defined as a square matrix with labels.
    """

    def __init__(self, tmatrix: np.ndarray, labels: list[str]):

        # Check Shape of ratings and labels match
        if tmatrix.ndim != 2 or tmatrix.shape[0] != tmatrix.shape[1]:
            raise IndexError("Transition matrix must be square and 2-dimensional")
        if len(labels) != tmatrix.shape[0]:
            raise IndexError("Labels length must match matrix dimension")

        # Get Transition Matrix Details
        self.tmatrix = tmatrix
        self.cum_tmatrix = np.cumsum(tmatrix, axis=1)  # cumulative probabilities for np.searchsorted

        # Labels
        self.labels = np.array(labels)
        self.label_to_idx = {l: i for i, l in enumerate(labels)}

        # Check transitions sum to 1
        bad_rows = np.where(~np.isclose(self.cum_tmatrix[:, -1], 1.0))[0]
        if bad_rows.size:
            raise ValueError(f"Transition probabilities for {self.labels[bad_rows]} do not sum to 1")

    def indices_to_labels(self, indices: np.ndarray) -> np.ndarray:
        return self.labels[indices]

    def fundamental_matrix(self):
        """
        Calculate the fundamental matrix for a transition matrix, which shows the expected number of visits to each state before default.
        """
        # The transition matrix (P) can be broken down into a matrix of transient states (Q),
        # a matrix of default probabilities (R) and an absorbing state matrix (D):
        # P = [Q  R]
        #     [O  D]

        P = self.tmatrix

        # Separate non-absorbing states (Q)
        Q = P[:-1, :-1]

        # Fundamental Matrix (Expected time in transient states before default)
        # N = (I - Q)^-1
        N = np.linalg.inv((np.eye(len(Q)) - Q))

        labels = self.labels[:-1]

        return pd.DataFrame(N, index = labels, columns = labels)

    def time_to_default(self):
        """
        Calculate the expected time to default for each rating.
        """

        # Get fundamental matrix
        N = self.fundamental_matrix().values

        # Expected time to default: t = N @ 1
        t = N @ np.ones(len(N))

        return pd.Series(t, index=self.labels[:-1], name = 'Time to Default')

    def _transitions_vector(self, current_ratings: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Calculate the 1 step ratings migrations for a vector of length n_issuers: current_ratings.
        p is a (n_issuers x 1) vector of simulated normal CDF probabilities for each issuers.
        """

        # Get the cumulative transition matrix entries for each of the n_issuers based on current ratings.
        cum_tmatrix_long = self.cum_tmatrix[current_ratings]

        # Count how many cumulative probabilities are less than each p value
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
        transitions = self.indices_to_labels(transitions_idx)

        return transitions

    def __str__(self):
        return str(pd.DataFrame(self.tmatrix, columns=self.labels, index=self.labels).to_markdown(floatfmt = ".1%"))


class SimulationResult:
    """
    Class for holding results of a CreditRiskModel run.
    Contains methods to perform analytics on results.
    """

    def __init__(self,
        E: Any,
        S: Any,
        I: Any,
        X: Any,
        pX: Any,
        transitions: Any,
        n_sim: int,
        n_years: int):

        self.E = E
        self.S = S
        self.I = I
        self.X = X
        self.pX = pX
        self.transitions = transitions
        self.n_sim = n_sim
        self.n_years = n_years

    def get_realised_transition_matrix(self, ratings_order = RATINGS_ORDER):

        n_sim, n_years, n_bonds = self.transitions.shape

        # --- Construct DataFrame ---

        df = (
            pd.DataFrame({
                "rating": self.transitions.reshape(-1)
            }, index=pd.MultiIndex.from_product(
                [range(n_sim), range(n_years), range(n_bonds)],
                names=["scenario", "year", "bond"]
            ))
            .reset_index()
            .sort_values(["scenario", "bond", "year"], ignore_index=True)
        )

        # --- Compute transition probabilities ---
        df["prev_rating"] = df.groupby(["scenario", "bond"])["rating"].shift(1)
        df_trans = df.dropna(subset=["prev_rating"], how="any")

        transition_counts = (
            df_trans.groupby(["prev_rating", "rating"]).size().reset_index(name="count")
        )

        transition_counts["probability"] = (
            transition_counts.groupby("prev_rating")["count"]
            .apply(lambda x: x / x.sum())
            .values
        )

        # --- Pivot to matrix format ---
        transition_matrix = (
            transition_counts.pivot_table(
                index="prev_rating",
                columns="rating",
                values="probability",
                fill_value=0
            )
        )

        # --- Ensure all ratings in expected order (fill missing transitions with 0) ---
        missing_rows = set(ratings_order) - set(transition_matrix.index)
        missing_cols = set(ratings_order) - set(transition_matrix.columns)
        for r in missing_rows: transition_matrix.loc[r] = 0
        for c in missing_cols: transition_matrix[c] = 0

        transition_matrix = transition_matrix.loc[ratings_order, ratings_order]

        return pd.DataFrame(transition_matrix.values, index=ratings_order, columns = ratings_order)

    def rating_mix_over_time(self, ratings_order = RATINGS_ORDER) -> pd.DataFrame:
        """
        Parameters
        ----------
        transitions : ndarray (n_sim, n_years, n_issuers)
            Rating labels

        Returns
        -------
        DataFrame indexed by year, columns = ratings,
        values = average portfolio weight
        """

        n_sim, n_years, n_issuers = self.transitions.shape

        mixes = []

        for t in range(n_years):
            # Flatten sims × issuers
            ratings_t = self.transitions[:, t, :].ravel()

            # Rating proportions
            mix_t = pd.Series(ratings_t).value_counts(normalize=True).rename(t)
            mixes.append(mix_t)

        df = pd.DataFrame(mixes).fillna(0.0)
        df.index.name = "Year"

        return df[ratings_order]


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
        self.validate_inputs()

    def validate_inputs(self) -> None:
        """Validate consistency of inputs for the credit risk model."""

        # --- Type checks ---
        if not isinstance(self.sector_map, pd.Series):
            raise TypeError("sector_map must be a pandas Series.")

        if not isinstance(self.ratings_map, pd.Series):
            raise TypeError("ratings_map must be a pandas Series.")

        if not isinstance(self.rho_s, np.ndarray):
            raise TypeError("rho_s must be a numpy array.")

        if not np.isscalar(self.rho_e):
            raise TypeError("rho_e must be a scalar.")

        # --- Index consistency across Series ---
        sector_idx = self.sector_map.index
        ratings_idx = self.ratings_map.index

        if not sector_idx.equals(ratings_idx):
            raise ValueError(
                "sector_map and ratings_map must have identical indexes "
                "(same issuers, same order)."
            )

        if sector_idx.has_duplicates:
            raise ValueError("Issuer index contains duplicates.")

        # --- Sector correlation consistency ---
        n_issuer_sectors = self.sector_map.nunique()
        n_sectors = len(self.rho_s)

        if n_sectors < n_issuer_sectors:
            raise ValueError(
                f"{n_issuer_sectors} sectors exist in sector_map, "
                f"but only {n_sectors} sector correlations have been defined."
            )

        if n_sectors > n_issuer_sectors:
            warnings.warn(
                f"{n_sectors} sector correlations are defined but only "
                f"{n_issuer_sectors} sectors exist in sector_map. "
                f"Only the first {n_issuer_sectors} will be used.",
                UserWarning
            )

        # --- Correlation bounds ---
        if not (-1.0 <= self.rho_e <= 1.0):
            raise ValueError("rho_e must be between -1 and 1.")

        if np.any((self.rho_s < -1.0) | (self.rho_s > 1.0)):
            raise ValueError("All values in rho_s must be between -1 and 1.")

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

# --- Assets --- #

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
    """
    Class to hold the results of a stochastic bond simulation and manipulate output.
    """
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

    def realised_transition_matrix(self, ratings_order: list =  RATINGS_ORDER):
        """
        Calculate the observed transition matrix in the bond simulation.
        """

        # Shape parameters
        n_sim, n_years, n_bonds = self.transitions.shape

        # convert to df
        df = pd.DataFrame(
            data = {
                'rating': self.transitions.reshape(-1)
            },
            index = pd.MultiIndex.from_product(
                iterables = [range(n_sim), range(n_years), range(n_bonds)],
                names = ["scenario", "year", "bond"]
            )
        ).reset_index().sort_values(['scenario', 'bond', 'year'])

        # specify transitions
        df['prev_rating'] = df.groupby(['scenario', 'bond'])['rating'].shift(1)

        df_trans = df.dropna(subset=['prev_rating'])
        transition_counts = df_trans.groupby(['prev_rating', 'rating']).size().reset_index(name='count')
        transition_counts['probability'] = (
            transition_counts.groupby(['prev_rating'])['count']
            .apply(lambda x: x / x.sum()).values
        )

        # Matrix
        transition_matrix = (
            transition_counts.pivot_table(
                columns='rating', index='prev_rating', values='probability', fill_value=0
            )
        )

        # Order raws and columns
        transition_matrix = transition_matrix.loc[ratings_order, ratings_order]
        return transition_matrix


class Bonds:
    """
    Class for holding details and cashflows for a set of bonds.

    Attributes
    ----------
    details:
        - ids: bond ids. Usually the ISIN for a bond but can be any unique identifier as long as it is applied consistently.
        - issuer_id: indexed series of issuer ids for each bond
        - recoveries: indexed series of recovery rate for each bond in the event of default.
        - decriptions: optional indexed series of bond descriptions

    cashflows : pd.DataFrame
        Cashflow ladder with a DatetimeIndex and bond IDs as columns.
        Cashflows must be standardised to a single unit of Notional.
    issuers : Issuers
        Issuers object holding the details of the bond issurs. Contains data on ratings and sector for each issuer.
    """

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
        self.validate_inputs()

    def validate_inputs(self) -> None:
        """Validate internal consistency of bond inputs."""

        # --- ids ---
        if not isinstance(self.ids, (list, tuple)):
            raise TypeError("ids must be a list or tuple.")

        if len(self.ids) == 0:
            raise ValueError("ids cannot be empty.")

        if len(set(self.ids)) != len(self.ids):
            raise ValueError("ids must be unique.")

        # --- Series alignment ---
        for name, s in {
            "issuer_ids": self.issuer_ids,
            "recoveries": self.recoveries,
        }.items():
            if not isinstance(s, pd.Series):
                raise TypeError(f"{name} must be a pandas Series.")

            if not s.index.isin(self.ids).all():
                raise ValueError(f"{name} index must match bond ids.")

        # --- Recoveries ---
        if ((self.recoveries < 0) | (self.recoveries > 1)).any():
            raise ValueError("recoveries must be between 0 and 1.")

        # --- Cashflows ---
        cf = self.cashflows

        if not isinstance(cf, pd.DataFrame):
            raise TypeError("cashflows must be a pandas DataFrame.")

        # Ensure DatetimeIndex (or convertible date column)
        if not isinstance(cf.index, pd.DatetimeIndex):
            date_col = next((c for c in cf.columns if c.lower() == "date"), None)
            if date_col is None:
                raise ValueError("cashflows must have a DatetimeIndex or a 'date' column.")
            cf = cf.set_index(pd.to_datetime(cf[date_col], errors="raise")).drop(columns=date_col)

        # Ensure cashflow columns match ids
        if not set(cf.columns).issubset(self.ids):
            raise ValueError("cashflow columns must be a subset of bond ids.")

        # Ensure numeric cashflows
        if not cf.apply(pd.api.types.is_numeric_dtype).all():
            raise TypeError("all cashflow columns must be numeric.")

        # Sort index and match columns to bond id order.
        self.cashflows = cf[self.ids].sort_index()

        # --- Issuers ---
        # if not isinstance(self.issuers, Issuers):
        #     raise TypeError("issuers must be an Issuers object.")

        missing_issuers = set(self.issuer_ids.values) - set(self.issuers.ids)
        if missing_issuers:
            raise ValueError(f"issuer_ids missing in issuers object: {missing_issuers}")

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

        cfs = self.year_end_cashflows(val_date)
        dt = calc_dt(cfs.index, val_date)

        base_yields = rates.interpolate(cfs.index).values                # (T,)
        current_ratings = self.issuer_ids.map(self.issuers.ratings).reindex(self.ids)
        spreads = map_spreads(current_ratings.values[np.newaxis, :], spread_map).squeeze(0)  # (n_bonds,)

        # Discount factors: (T, n_bonds)
        dfs    = (1 + base_yields[:, np.newaxis] + spreads[np.newaxis, :]) ** (-dt[:, np.newaxis])
        prices = (cfs.values * dfs).sum(axis=0)

        return pd.Series(prices, index=self.ids)

    def _run_sim_pv(
            self,
            cashflows: np.ndarray,
            bond_transitions: np.ndarray,
            spread_map: dict,
            yields: np.ndarray,
            dt: np.ndarray
        ) -> np.ndarray:

        # Shape
        n_years_cf, n_bonds = cashflows.shape
        n_sim, n_years_sim, n_bonds = bond_transitions.shape

        # Take min n_years to return output for
        n_years = min(n_years_cf, n_years_sim)

        # Check that cashflows and transitions contain the same number of bonds
        assert cashflows.shape[1] == bond_transitions.shape[2], "n_bonds inconsistent"

        # Since all spreads come from a finite set of rating labels,
        # we can precompute a PV table of shape (n_unique_spreads, n_years, n_bonds)

        # Get unique spread values
        unique_spreads, spread_inverse = np.unique(map_spreads(bond_transitions, spread_map), return_inverse=True)
        n_unique = len(unique_spreads)

        # Create a matrix of dts for each year in the simulation
        dtime_mat = dt[:, np.newaxis] - dt[:n_years][np.newaxis, :]          # (n_years_cf, n_years)
        future = dtime_mat > 0                                               # cashflow is after val year

        # Create a matrix of (1 + yields)
        yield_mat = (1 + yields)[:, np.newaxis] * np.ones((1, n_years))   # (n_years_cf, n_years)

        # Build PV table: (n_unique, n_years, n_bonds)
        # For each spread value s and each valuation year t:
        # pv_table[s_idx, t, bond] = sum_{u > t} (1 + y_u + s)^-(dt_u - dt_t) * CF[u, bond]
        #                          = (dfs_mat.T @ cashflows)
        pv_table = np.zeros((n_unique, n_years, n_bonds))
        for si, s in enumerate(unique_spreads):
            # Discount factors: shape (n_years_cf, n_years)
            # rows = cashflow dates u
            # cols = valuation dates t
            dfs_mat = np.zeros_like(dtime_mat)
            dfs_mat[future] = (yield_mat[future] + s) ** (-dtime_mat[future])

            # Sum over all cashflow dates for each valuation year
            pv_table[si] = dfs_mat.T @ cashflows                             # (n_years, n_bonds)

        # Map each (sim, t, bond) to its spread index then look up PV
        spread_idx = spread_inverse.reshape(n_sim, n_years, n_bonds)         # (n_sim, n_years, n_bonds)
        t_idx = np.arange(n_years)[np.newaxis, :, np.newaxis]
        b_idx = np.arange(n_bonds)[np.newaxis, np.newaxis, :]
        pvs = pv_table[spread_idx, t_idx, b_idx]                      # (n_sim, n_years, n_bonds)

        return pvs

    def run_sim(self, val_date: datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):
        """
        Project cashflows and revalue bonds across all simulation paths.

        For each scenario and each year t, we:
          1. Receive scheduled cashflows if the bond has not defaulted.
          2. Receive a recovery payment in the first year of default
             (only if the bond has not already matured).
          3. Calculate the PV of future cashflows discounted
             at the scenario spread.

        Returns a BondSimulationResult containing:
        - transitions: simulated bond rating transitions
        - cashflows: realized cashflows (including defaults and recoveries)
        - pvs: present values across simulation dates
        - dates: valuation dates
        - bond_ids: bond identifiers
        """
        # --- Cashflows ---
        cashflows_df = self.year_end_cashflows(val_date)
        cashflows    = cashflows_df.values                                    # (n_years, n_bonds)
        dates        = cashflows_df.index

        n_years_cf   = cashflows.shape[0]
        n_bonds      = cashflows.shape[1]
        n_sim        = sim_results.n_sim
        n_years_sim  = sim_results.n_years

        # Simulation result will cover the minimum of years and the sim and cashflow years.
        n_years = min(n_years_sim, n_years_cf)

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

        # --- Maturity flags: (n_years_cf, n_bonds) ---
        not_matured = compute_maturity_flags(cashflows)

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
            * self.recoveries.values[np.newaxis, np.newaxis, :]
            * not_matured[np.newaxis, :n_years, :]
        )                                                                     # (n_sim, n_years, n_bonds)

        # Total Cashflows
        total_cashflows = received_cashflows + recovery_payments              # (n_sim, n_years, n_bonds)

        # Base yields and time fractions
        yields = rates.interpolate(dates).values                              # (n_years_cf,)
        dt = calc_dt(dates, val_date)                                         # (n_years_cf,)

        # PVs
        pvs = self._run_sim_pv(cashflows, bond_transitions, spread_map, yields, dt)

        return BondSimulationResult(
            transitions = bond_transitions,
            received_cashflows = received_cashflows,
            recovery_payments = recovery_payments,
            total_cashflows = total_cashflows,
            pvs = pvs,
            dates = dates[:n_years],
            bond_ids = self.ids
        )


# --- Liabilities --- #

class Liabilities:
    """Container for liability cashflows."""

    def __init__(self, cashflows: pd.Series, dates: List):
        # Check dates and cashflows are the same length
        if len(cashflows) != len(dates):
            raise ValueError("Cashflows and dates length mismatch.")

        # Store cashflows as series
        self.cashflows = pd.Series(cashflows, index = pd.to_datetime(dates))

    def pv(self, val_date: datetime, rates: Union[Rates, float], shift=0.0, timeline = False, n_years: float = None) -> float:
        """
        Calculate PV of liabilities for a specified date and discount rate.
        """

        # Get cashflows beyond val date
        cashflows = self.cashflows.loc[self.cashflows.index > val_date]
        dates = cashflows.index
        dt = calc_dt(dates, val_date)

        if np.isscalar(rates):
            net_yield = pd.Series(rates + shift, index=cashflows.index).values
        else:
            yields_int = rates.interpolate(dates)
            net_yield = (yields_int + shift).values

        df = (1 + net_yield) ** -dt
        pv = (cashflows * df).sum()

        if timeline:
            pv = np.array([
                    np.sum(cashflows.values[i + 1:, np.newaxis] * ((1 + net_yield[i+1:, np.newaxis])**-(dt[i + 1:, np.newaxis] - dt[i])))
                    for i in range(len(dates))
            ])
            pv = pv[:n_years] if n_years is not None else pv
        else:
            pv = (cashflows * (1 + net_yield) ** -dt).sum()

        return pv

    def __str__(self):
        return pd.DataFrame(self.cashflows).to_markdown(floatfmt='.0f')

# --- CDI --- #

class CDISimulationResult:
    """Class to hold the output of a CDI Simulation Result."""
    def __init__(
        self,
        cdi_sim: pd.DataFrame,
        cashflows: np.ndarray,
        pvs: np.ndarray
    ):
        self.cdi_sim = cdi_sim
        self.cashflows = cashflows
        self.pvs = pvs


class CDIMandate_Fox:
    """
    Class implementing the Fox CDI mandate.
    """

    def __init__(
        self,
        liabilities: Liabilities,
        cash: float,
        bonds: Bonds,
        cdi_allocation: pd.Series,
        cmbp_allocation: pd.Series,
        heubeck_liabilities: float,
        r_gaap: float,
        r_ifrs: float,
        cmbp_margin: float,
        asset_buffer: np.ndarray,
        mortality_buffer: float,
        fee: float,
        performance_cap: float

    ):
        self.liabilities=liabilities
        self.cash=cash
        self.bonds=bonds
        self.cdi_allocation=cdi_allocation

        self.cmbp_allocation=cmbp_allocation
        self.heubeck_liabilities=heubeck_liabilities
        self.r_gaap=r_gaap
        self.r_ifrs=r_ifrs
        self.cmbp_margin=cmbp_margin
        self.asset_buffer=asset_buffer
        self.mortality_buffer=mortality_buffer
        self.fee=fee
        self.performance_cap =performance_cap

    def run(self, val_date: datetime, rates : Rates, spread_map: dict, sim_results: SimulationResult):
        """Run Fox CDI simulation."""

        # 1. Calculate Required Liability Metrics
        max_years = sim_results.n_years # simulation shouldn't exceed number of years that the economy was simulated for.
        liab_cashflow_series = self.liabilities.cashflows.loc[val_date:][:max_years]
        liab_cashflows = liab_cashflow_series.values
        dates = pd.DatetimeIndex(liab_cashflow_series.index)
        dt = calc_dt(dates, val_date)
        T = len(dates)

        # PV Timeline
        liab_pv_gaap = self.liabilities.pv(val_date, rates = self.r_gaap, timeline=True, n_years=max_years)
        liab_pv_ifrs = self.liabilities.pv(val_date, rates = self.r_ifrs, timeline=True, n_years=max_years)

        # Meltdown liabilities
        cum_liab_cashflows = liab_cashflows.cumsum()
        meltdown_liabilities = np.maximum(
            (self.heubeck_liabilities  - cum_liab_cashflows) * self.mortality_buffer * ((1 + self.r_gaap) ** dt),
            0
        )

        # Next 2 liabilities
        full_cfs = self.liabilities.cashflows.loc[val_date:].values
        next_2_liabs = np.zeros_like(liab_cashflows)
        next_2_liabs = full_cfs[1:T+1] + full_cfs [2:T+2]

        # 2. Calculate 1-year forward rates for cash returns (and cmbp beyond expiry).
        fwds = rates.calc_fwds(val_date, dates).values

        # 3. Price Assets at val_date
        bond_prices = self.bonds.pv(val_date, rates, spread_map)
        cdi_t0 = (self.cdi_allocation * bond_prices).sum()
        cmbp_t0 = (self.cmbp_allocation * bond_prices).sum()
        total_assets = cdi_t0 + self.cash

        # 4. Calculate No Default CDI Cashflows
        cfs = self.bonds.year_end_cashflows(val_date)
        expected_cdi_cashflow = (cfs * self.cdi_allocation).sum(axis = 1).values
        if len(expected_cdi_cashflow) < T:
            expected_cdi_cashflow = np.pad(expected_cdi_cashflow, (0, T - len(expected_cdi_cashflow)))
        else:
            expected_cdi_cashflow = expected_cdi_cashflow[:T]

        # 5. Get Simulated CDI Asset Results

        # Simulate cashflows and pvs for full bond universe
        bond_sim = self.bonds.run_sim(val_date, rates, spread_map, sim_results)

        # Set target shape to force length of asset and liability cashflows to match.
        n_sim = sim_results.n_sim
        shape = (n_sim, T)

        # Calculate cashflows and pvs for CDI portfolio using the cdi allocation
        asset_cfs = allocate_bond_sim(bond_sim, self.cdi_allocation, shape, target ='cashflows')
        asset_pvs = allocate_bond_sim(bond_sim, self.cdi_allocation, shape, target ='pvs')

        # 6. Simulate Cashflows and PVs for Cashflow Matching Bund Portfolio.

        cmbp_cashflows = allocate_bond_sim(bond_sim, self.cmbp_allocation, shape, target ='cashflows')
        cmbp_pvs = allocate_bond_sim(bond_sim, self.cmbp_allocation, shape, target ='pvs')

        # Calculate annual total returns of CMBP for each scenario
        bt = total_returns(cmbp_pvs, cmbp_cashflows, fwds, cmbp_t0)

        # Calculate total returns of cdi portfolio to compare
        cdi_returns = total_returns(asset_pvs, asset_cfs, fwds, a0 = cdi_t0)

        # 7. Run Waterfall
        # t=0 values
        day_0_hgb_gap = self.heubeck_liabilities * self.mortality_buffer - (total_assets + self.asset_buffer)
        hgb_gap_t = np.full(n_sim, day_0_hgb_gap)
        cash_t = np.full(n_sim, self.cash)
        assets_t = np.full(n_sim, total_assets)
        bund_comparator = np.full(n_sim, total_assets)
        total_payment = np.zeros(n_sim)

        # Initialise results dictionary
        def replicate(x, n_sim): return np.tile(x, (n_sim, 1))
        results = {
            "dt": replicate(dt, n_sim),
            "liab_cashflow": replicate(liab_cashflows, n_sim),
            "liab_pv_gaap": replicate(liab_pv_gaap, n_sim),
            "liab_pv_ifrs": replicate(liab_pv_ifrs, n_sim),
            "meltdown_liabilities": replicate(meltdown_liabilities, n_sim),
            "next_2_liabs": replicate(next_2_liabs, n_sim),
            "bund_yield": replicate(rates.interpolate(dates).values, n_sim),
            "bund_fwds": replicate(fwds, n_sim),
            "expected_cdi_cashflow": replicate(expected_cdi_cashflow, n_sim),
            "asset_cashflow": asset_cfs,
            "remaining_asset_pv": asset_pvs,
            "cdi_return": cdi_returns,
            "net_cdi_return": np.zeros(shape),
            "cmbp_cashflow": cmbp_cashflows,
            "cmbp_pv": cmbp_pvs,
            "bt": bt,
            "fee": np.zeros(shape),
            "cash": np.zeros(shape),
            "assets": np.zeros(shape),
            "net_asset_return": np.zeros(shape),
            "bund_comparator": np.zeros(shape),
            "asset_buffer": np.zeros(shape),
            "meltdown_assets": np.zeros(shape),
            "hgb_gap": np.zeros(shape),
            "hgb_payment": np.zeros(shape),
            "performance_payment": np.zeros(shape),
            "additional_payment": np.zeros(shape),
            "total_hgb_payments": np.zeros(shape)
        }
        for t in range(T):

            # Calculate fee as % of AUM as at t
            fee_t = self.fee * (
                cash_t * (1 + fwds[t]) + asset_pvs[:, t]  + asset_cfs[:, t]
            )

            # Update cash balance, accounting for fees
            cash_t = (
                cash_t * ((1 + fwds[t]))
                + asset_cfs[:, t]
                - liab_cashflows[t]
                - fee_t
            )

            # Update assets
            prev_assets_t = assets_t
            assets_t = cash_t + asset_pvs[:, t]

            ## HGB Gap Guarante ##
            # Add asset buffer for meltdown calcs.
            asset_buffer = self.asset_buffer * ((1 + self.r_ifrs) **(t+1)) if t < 10 else 0.0
            meltdown_assets_t = assets_t + asset_buffer

            # HGB Gap. Gap can't increase from previous year's gap and can't be less than 0
            hgb_gap_t = np.clip(meltdown_liabilities[t] - meltdown_assets_t, 0, hgb_gap_t)

            # HGB Gap Payment. We will pay into the fund in the event that there is not enough money to cover next two years' liabilities
            # the maximum total payment (including from previous years) is the current hgb gap
            hgb_payment = np.minimum(
                np.maximum(next_2_liabs[t] - assets_t, 0),
                hgb_gap_t - total_payment
            )

            # add payment to cash and assets values
            cash_t += hgb_payment
            assets_t += hgb_payment
            total_payment += hgb_payment

            ## Additional Payment year 10.
            # In year 10 (t=9), the client will make an additonal pmt, up to 18.325m (asset buffer) if the scheme is underfunded.
            if t == 9:
                additional_payment = np.maximum(
                    np.minimum(
                        asset_buffer,
                        liab_pv_gaap[t] - assets_t,
                        1.1*liab_pv_ifrs[t] - assets_t
                    ),
                    0
                )
            else:
                additional_payment = 0.0

            cash_t += additional_payment
            assets_t += additional_payment

            ## Performance Guarantee.
            # Update Cashflow Matching Bund Portfolio (include additional payment)
            bund_comparator = bund_comparator * (1 + bt[:, t] + self.cmbp_margin) - liab_cashflows[t] + additional_payment

            # At year 25 (t = 24), if the schemes assets are below the cashflow matching bund portfolio,
            # we will need to pay in the difference, up to a maximum of performance_cap
            performance_payment = np.clip(bund_comparator - assets_t, 0, self.performance_cap) if t == 24 else 0.0

            # Calculate total net asset return for the year (excluding liability and additional payment)
            net_asset_return_t = (assets_t + liab_cashflows[t] - additional_payment)/prev_assets_t - 1

            for key, value in {
                "cash": cash_t,
                "assets": assets_t,
                "asset_buffer": asset_buffer,
                "net_asset_return": net_asset_return_t,
                "meltdown_assets": meltdown_assets_t,
                "hgb_gap": hgb_gap_t,
                "hgb_payment": hgb_payment,
                "bund_comparator": bund_comparator,
                "performance_payment" : performance_payment,
                "fee":fee_t,
                "additional_payment": additional_payment
            }.items():
                results[key][:, t] = value

        flat = {k: v.reshape(-1) for k, v in results.items()}
        index = pd.MultiIndex.from_product(
            [np.arange(n_sim), dates], names=["scenario", "date"]
        )
        df = pd.DataFrame(flat, index=index).reset_index()
        df['funding_level_gaap'] = df['assets']/df['liab_pv_gaap']
        df['funding_level_ifrs'] = df['assets']/df['liab_pv_ifrs']
        df['net_cdi_return'] = df['cdi_return'] * (1-self.fee) - self.fee
        df['net_bt_return'] = df['bt'] + self.cmbp_margin

        return df