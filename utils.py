import numpy as np
import pandas as pd
import scipy as sp
import datetime
from dataclasses import dataclass
from typing import List, Sequence, Mapping, Optional, Union, Any


DAYS_PER_YEAR = 365

def calc_dt(dates: List[datetime.datetime], val_date: datetime.datetime):
    return np.array(((dates - val_date).days / DAYS_PER_YEAR))[:, None]


class BaseDataClass:

    typeChecks = {
        'date': pd.api.types.is_datetime64_any_dtype,
        'numeric': pd.api.types.is_numeric_dtype,
        'string': pd.api.types.is_string_dtype
    }

    REQUIRED_COLUMNS: dict
    data: pd.DataFrame

    def validate_inputs(self):
        """confirm that the DataFrame contains exactly the correct columns"""

        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        # check that the dataFrame has exectly the right columns
        req = set(self.REQUIRED_COLUMNS)
        act = set(self.data.columns)

        missing_cols = req - act
        extra_cols = act - req

        if missing_cols:
            raise ValueError(f"Missing required column(s) {missing_cols}")
        if extra_cols:
            raise ValueError(f"Unexpected column(s) {extra_cols}")

        # check for correct types in each column
        bad_types= []
        for col, req in self.REQUIRED_COLUMNS.items():
            if req is not None and not self.typeChecks[req](self.data[col]):
                bad_types.append(f"Column {col}: {req} expected, {self.data[col].dtype} found")

        if bad_types:
            raise TypeError(f"Bad column type(s) {bad_types}")


class Rates(BaseDataClass):
    """
    Class to hold a discount rate curve.
    Input must be a dataframe of dates and yields
    Valuation date must be provided
    Columns must be: date, yield
    """

    REQUIRED_COLUMNS = {"date": 'date', "yield": 'numeric'}

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.validate_inputs()

        # yields and fwds as a date-indexed Series
        self.yields = data.set_index('date')['yield']

    # Calculate implied forward rates from the specified yield curve
    def calc_fwds(self, val_date: datetime.datetime) -> pd.Series:

        # ensure you are only using yields after val date
        y = self.yields.loc[val_date:]

        t = np.array((y.index - val_date).days / 365.0)
        acc = (1 + y) ** t

        # forwards calculated as the percentage change in total accumulation factors for each time index t
        fwds = (acc / acc.shift()) ** (1 / (t - np.roll(t, 1))) - 1

        # first forward set to first yield
        fwds.iloc[0] = y.iloc[0]
        fwds.name = 'fwds'
        return fwds

    # Linearly interpolates yields to the specified dates.
    def interpolate(self, dates: pd.DatetimeIndex) -> pd.Series:

        if not isinstance(dates, pd.DatetimeIndex):
            raise TypeError("dates must be a pd.DatetimeIndex.")

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
    Input must be a dataframe of cashflows and dates
    Columns must be: date, cashflow
    """

    REQUIRED_COLUMNS = {"date": 'date', "cashflow": 'numeric'}

    def __init__(self, data: pd.DataFrame, name='liabilities'):
        self.name = name
        self.data = data.copy()
        self.validate_inputs()
        self.data.set_index("date", inplace=True)
        self.cashflows = self.data["cashflow"]

    def to_dataframe(self):
        return self.cashflows.to_frame()

    def to_series(self):
        return self.cashflows

    def pv(self, rates: "Rates | float", val_date: datetime.datetime, shift=0.0) -> float:
        """
        Calculate PV of liabilities for a specified date and discount rate.
        """

        # Subset of Cashflows and dates beyond valuation date
        cashflows = self.cashflows.loc[val_date:]
        dates = pd.to_datetime(cashflows.index, dayfirst = True)
        t = (dates-val_date) / pd.Timedelta(days=365)

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


class TransitionMatrix:
    """
    Class for holding a credit rating transition matrix.
    Matrix must be defined as a square matrix with labels.
    """

    def __init__(self, tmatrix: np.ndarray, labels: list[str]):
        if tmatrix.ndim != 2:
            raise IndexError("2-d transition matrix expected")
        if tmatrix.shape[0] != tmatrix.shape[1]:
            raise IndexError("Square transition matrix expected")
        if len(labels) != tmatrix.shape[0]:
            raise IndexError("labels is not the same length as one side of the transition matrix")

        self.tmatrix = pd.DataFrame(tmatrix, index=labels, columns=labels)
        self.labels = labels
        self.tm = {label_: np.cumsum(tmatrix[index_, :]) for (index_, label_) in enumerate(labels)}

        bad_rows = [k for k,v in self.tm.items() if not np.isclose(v[-1], 1)]
        if bad_rows:
            raise ValueError(f"Transition probabilities for the following states do not sum to 1.0: {bad_rows}")

    def transition(self, label: str, prob: float) -> str:
        return self.labels[np.searchsorted(self.tm[label], prob)]

    def transitionv(self, label, prob):
        return [self.transition(label_, prob_) for (label_, prob_) in zip(label, prob)]

    def __str__(self):
        return str(self.tmatrix)


class Issuers(BaseDataClass):
    """
    Class for holding data on bond issuers.
    Columns must be: id, sector, rating
    """

    REQUIRED_COLUMNS = {"id": None, "sector": None, "rating": None}

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.validate_inputs()
        self.n_sectors = self.data['sector'].nunique()
        self.n_issuers = self.data['id'].nunique()
        self.ids = np.array([str(x) for x in self.data['id']])

    def validate_inputs(self):
        super().validate_inputs()

        # Also check that ids are unique
        duplicates = set(self.data.loc[self.data['id'].duplicated(), 'id'])
        if duplicates:
            raise KeyError(f"Duplicated IDs detected: {duplicates}")

    def __str__(self):
        return str(self.data)


@dataclass
class SimulationResult:
    """Container for results of a CreditRiskModel run."""
    E: Any
    S: Any
    I: Any
    X: Any
    pX: Any
    transitions: Any
    transitions_df: pd.DataFrame
    n_sim: int
    n_years: int


class CreditRiskModel:
    """
    Class to define a simulation of the Economy and sectors for each bond issuer.
    The output of this simulation defines the ratings migrations and point at which an issuer defaults.
    """

    def __init__(
            self,
            issuers: Issuers,
            rho_e: int,
            rho_s: np.ndarray,
            transition_matrix: TransitionMatrix
    ):

        self.rho_e = rho_e
        self.rho_s = rho_s
        self.transition_matrix = transition_matrix

        self.issuers = issuers
        self.n_sectors = issuers.n_sectors
        self.n_issuers = issuers.n_issuers
        self.sector_map = np.array(issuers.data['sector'])
        self.ratings_map = np.array(issuers.data['rating'])

        if len(rho_s) < issuers.n_sectors:
            raise ValueError("One or more sectors in the Issuers object do not have a sector correlation defined.")

    def run(self, n_sim: int, n_years: int) -> SimulationResult:

        E = np.random.normal(size=[n_sim, n_years])
        S = np.random.normal(size=[n_sim, n_years, self.n_sectors])
        I = np.random.normal(size=[n_sim, n_years, self.n_issuers])
        X = np.sqrt(self.rho_e) * E[:, :, np.newaxis] + \
            np.sqrt(self.rho_s[self.sector_map] - self.rho_e)[np.newaxis, np.newaxis, :] * S[:, :, self.sector_map] + \
            np.sqrt(1 - self.rho_s[self.sector_map])[np.newaxis, np.newaxis, :] * I

        pX = sp.stats.norm.cdf(X)

        transitions = []
        for s in range(n_sim):
            rating = self.ratings_map
            transitions.append(np.array(
                [rating := self.transition_matrix.transitionv(rating, pX[s, t, :]) for t in range(n_years)])
            )

        transitions_df = (
            pd.concat(
                pd.DataFrame(t)
                .assign(scenario=s)
                .reset_index(names='year')
                for s, t in enumerate(transitions)
            )
            .set_axis(['year'] + list(self.issuers.ids) + ['scenario'], axis=1)
            .melt(id_vars=['scenario', 'year'], var_name='issuer_id', value_name='rating')
        )

        return SimulationResult(E=E, S=S, I=I, X=X, pX=pX, transitions=transitions, transitions_df=transitions_df, n_sim=n_sim, n_years=n_years)


class Asset(BaseDataClass):
    """
    Base class for an asset object.
    """
    def __init__(self, data):
        self.data = data

    def pv(self):
        pass

    def pv_timeline(self):
        pass

    def cashflow_timeline(self):
        pass


class Bonds(Asset):
    """
    Class for holding data on bonds in a portfolio.
    Contains three elements:
    - data: a dataframe containing details for each bond. columns are: id, issuer_id, notional, recovery
    - cashflows: a cashflows table with dates as the index and bond ids as the columns
    - issuers: details of the bond issuers including sector and rating
    """

    REQUIRED_COLUMNS = {'id': 'string' ,'issuer_id' : 'string', 'notional': 'numeric', 'recovery': 'numeric'}

    def __init__(self, data: pd.DataFrame, cashflow_table: pd.DataFrame, issuers: Issuers):
        self.data = data
        self.cashflow_table = cashflow_table.set_index('date')
        self.issuers = issuers
        self.validate_inputs()

        # Gather data for the bonds
        self.ids = np.array([str(x) for x in self.data['id']])
        self.issuer_map = np.array(self.data['issuer_id'])
        self.notional_map = np.array(self.data['notional'])
        self.recovery_map = np.array(self.data['recovery'])
        self.n_bonds = len(self.data)

    def validate_inputs(self):
        super().validate_inputs()

        # check that the columns for the cashflow table are included in data map
        cf_table_ids = set(self.cashflow_table.columns)
        bond_data_ids = set(self.data['id'])

        missing = cf_table_ids - bond_data_ids
        extra = bond_data_ids - cf_table_ids
        if missing:
            raise KeyError(f"Bond mapping data missing for bond ids: {missing}")
        if extra:
            raise KeyError(f"Cashflows missing for bond ids: {extra}")

        # check that there is information for all issuers in the issuer object
        bond_issuers = set([str(x) for x in self.data['issuer_id']])
        issuer_ids = set(self.issuers.ids)

        missing_issuers = bond_issuers - issuer_ids

        if missing_issuers:
            raise KeyError(f"Issuer information missing for the follwing issuer ids: {missing_issuers}")

    def pv(self, val_date: datetime.datetime, rates: Rates, spread_map: dict):
        """
        Calculates the PV of the cashflows for a given set of rates and spread at a valuation date.
        This PV is based only on the cashflows and does not account for the allocation to this bond object.
        """
        # Only consider cashflows beyond val_date
        cfs_future = self.cashflow_table.loc[self.cashflow_table.index >= val_date]
        if cfs_future.empty:
            return 0.0

        dates = cfs_future.index
        cashflows = cfs_future.to_numpy()

        # Get risk free discount rates for target dates
        yields = rates.interpolate(dates).to_numpy()[:, None]

        # Get spreads based on issuer ratings
        merged = self.data.merge(self.issuers.data[['id', 'rating']], how = 'left', left_on = 'issuer_id', right_on = 'id')
        spreads = merged["rating"].map(spread_map).to_numpy()

        # Broadcast to same shape as cashflows
        yields_mat = np.broadcast_to(yields, cashflows.shape)
        spreads_mat = np.broadcast_to(spreads, cashflows.shape)

        # Compute discount factors
        # dt = np.array(((dates - val_date).days / 365))[:, None]
        dt = calc_dt(dates, val_date)
        dfs = (1 + yields_mat + spreads_mat) ** (-dt)

        # Calc PV
        pv = np.sum(cashflows * dfs)

        return pv

    def expected_cashflows(self, allocation: float, val_date: datetime.datetime, rates: Rates, spread_map: dict):
        # Adjust cashflows according to allocation
        raw_pv = self.pv(val_date, rates, spread_map)
        ratio = allocation / raw_pv
        cashflows_adj = self.cashflow_table.loc[val_date:] * ratio
        return cashflows_adj

    def sim_pv(self, allocation: float, val_date: datetime.datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):
        """
        Calculates the PV timeline for all bonds in the object for a given transitions simulation.
        """
        # Adjust cashflows according to allocation
        raw_pv = self.pv(val_date, rates, spread_map)
        ratio = allocation / raw_pv
        cashflows_adj = self.cashflow_table.loc[val_date:] * ratio

        # Convert to arrays
        cashflows = cashflows_adj.to_numpy()
        dates = cashflows_adj.index
        n_years = len(dates)

        # Get simulated transitions
        transitions_df = sim_results.transitions_df.copy()

        # Check there's enough years in sim for bond cashflows
        assert n_years <= sim_results.n_years, "Simulation doesn't contain enough years for all bonds."

        # filter and pivot transitions and map to issuers for selected bonds
        transitions_wide = (
            transitions_df.query("year < @n_years")
            .pivot(index=["scenario", "year"], columns="issuer_id", values="rating")
            .loc[:, self.issuer_map]
        )

        # Map ratings to spreads
        spreads = transitions_wide.map(lambda r: spread_map.get(r, 0.0))

        # get discount rates for dates and calculate time differences from val_date
        yields = rates.interpolate(dates).to_numpy()[:, None]
        dt = ((dates - val_date).days / DAYS_PER_YEAR).to_numpy()[:, None]

        # calculate pv of each bond at each timestep in the simulation as the discounted sum of expected future cashflows
        pvs = []
        for s in range(sim_results.n_sim):
            transition_s = transitions_wide.xs(s, level="scenario").to_numpy()
            spread_s = spreads.xs(s, level="scenario").to_numpy()

            # PV timeline
            pv_timeline = np.empty_like(cashflows)
            for t in range(len(dt)):
                df = (1 + yields[t+1:] + spread_s[t]) ** (-(dt[t+1:] - dt[t]))
                pv_timeline[t] = (cashflows[t+1:] * df).sum(axis=0)

            # Zero out defaulted bonds
            pv_timeline *= (transition_s != "Default")

            pvs.append(pv_timeline)

        results = (
            pd.DataFrame(
                {"pv": np.array(pvs).reshape(-1)},
                index=pd.MultiIndex.from_product(
                    [range(sim_results.n_sim), cashflows_adj.index, self.ids],
                    names=["scenario", "date", "bond_id"],
                ),
            )
            .reset_index()
        )

        return results

    def sim_cashflows(self, allocation: float, val_date: datetime.datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):
        """
        Calculates the Simulated cashflows for all bonds in the object for a given transitions simulation.
        """

        # Adjust cashflows according to allocation
        raw_pv = self.pv(val_date, rates, spread_map)
        ratio = allocation / raw_pv
        cashflows_adj = self.cashflow_table.loc[val_date:] * ratio

        # Convert to arrays
        cashflows = cashflows_adj.to_numpy()
        dates = cashflows_adj.index
        n_years = len(dates)

        # Get simulated transitions
        transitions_df = sim_results.transitions_df.copy()

        # Check there's enough years in sim for bond cashflows
        assert n_years <= sim_results.n_years, "Simulation doesn't contain enough years for all bonds."

        # filter and pivot transitions and map to issuers for selected bonds
        transitions_wide = (
            transitions_df.query("year < @n_years")
            .pivot(index=["scenario", "year"], columns="issuer_id", values="rating")
            .loc[:, self.issuer_map]
        )

        # identify years in which we expect bonds to still be active
        years = np.arange(n_years)[:, None]
        maturity_idx = n_years - np.argmax((cashflows != 0)[::-1, :], axis=0)
        active = (years < maturity_idx).astype(int)

        # Calculate the expected recovery flows that will be received in the event a bond defaults. will be 0 if bond has matured
        recovery_flows = self.recovery_map * self.notional_map * active * ratio

        # Identify the default year for each bond in each scenario and reshape into a (n_sim x n_year x n_bond) array.
        defaulted = (transitions_wide == "Default").astype(int)
        first_default = defaulted.groupby(level="scenario").cumsum().eq(1)
        first_default_arr = np.array([first_default.xs(i, level='scenario').values for i in range(sim_results.n_sim)])

        # Identify the years in which a bonds is still active (hasn't defaulted) and reshape into a (n_sim x n_year x n_bond) array.
        not_defaulted = (transitions_wide != "Default").astype(int)
        not_defaulted_arr = np.array([not_defaulted.xs(i, level='scenario').values for i in range(sim_results.n_sim)])

        # Calculate total cashflows as teh expected cashflows from non-defaulted bonds plus recovery flows from defaulted bonds
        total_cashflows = cashflows * not_defaulted_arr + recovery_flows * first_default_arr

        # Convert to dataframe to return
        results = pd.DataFrame(
            {
                "cashflow": total_cashflows.reshape(-1),
            },
            index=pd.MultiIndex.from_product(
                [range(sim_results.n_sim), dates, self.ids],
                names=["scenario", "date", "bond_id"]
            )
        ).reset_index()

        return results

    def run_sim(self,allocation: float, val_date: datetime.datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):

        # pvs
        pvs = self.sim_pv(
            allocation=allocation,
            val_date=val_date,
            rates=rates,
            spread_map=spread_map,
            sim_results=sim_results
        )

        # cashflows
        cashflows = self.sim_cashflows(
            allocation=allocation,
            val_date=val_date,
            rates=rates,
            spread_map=spread_map,
            sim_results=sim_results
        )

        # merge
        combined = pd.merge(cashflows, pvs, how = 'outer')

        # sum over all bond ids
        aggregated = combined.groupby(['scenario', 'date']).sum().reset_index().drop(columns='bond_id')

        return aggregated


class Portfolio:
    """
    Class to hold a portfolio of assets. Objects must inherit from Asset class.
    """
    def __init__(self, asset_list: List[Asset], allocations: List[float]):
        self.assets = asset_list
        self.allocations = allocations
        self.total_allocation = np.sum(self.allocations)

    def run_sim(self, val_date: datetime.datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):

        # Simulate cashflows and pvs for each asset
        sims = [
            asset.run_sim(allocation, val_date, rates, spread_map, sim_results)
            for asset, allocation in zip(self.assets, self.allocations)
        ]

        # Combine and sum
        combined = pd.concat(sims).groupby(['date', 'scenario']).sum().reset_index()

        return combined


@dataclass
class CDISimulationResult:
    """Container for CDI simulation outputs."""
    cdi_results: pd.DataFrame
    bond_results: pd.DataFrame
    expected_pv_payment: float


class CDIMandate:
    """
    Base class to hold information for CDI Mandates.
    """

    def __init__(self, liabilities: Liabilities, portfolio: Portfolio, cash: float):
        self.liabilities = liabilities
        self.assets = portfolio
        self.cash = cash

    def run(self, val_date: datetime.datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):
        pass


class CDIMandate_Fox(CDIMandate):
    """
    Class for the Fox CDI mandate.
    """

    def __init__(
        self, liabilities: Liabilities, portfolio: Portfolio, cash: float,
        asset_buffer: np.ndarray, gaap_int: float, mortality_buffer: float
    ):

        self.liabilities = liabilities
        self.portfolio = portfolio
        self.cash = cash

        # Fox specific parameters
        self.asset_buffer = asset_buffer
        self.gaap_int = gaap_int
        self.mortality_buffer = mortality_buffer

    def _payment_expected_value(self, cdi_results: pd.DataFrame, val_date: datetime.datetime, rates: Rates):

        # Take a copy of results
        df = cdi_results.copy()

        # Discount Rates
        df['t'] = calc_dt(pd.Index(df['date']), val_date)
        yields = rates.yields.to_frame(name='yield')
        merged = df.merge(yields, how='left', left_on='date', right_index=True).set_index('date')
        merged['payment_pv']=merged['payment']*(1+merged['yield'])**(-merged['t'])
        payments = merged.groupby(['scenario']).sum()['payment_pv']
        expected_value = payments.mean()
        return expected_value

    def run(self, val_date: datetime.datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):

        # Base Date Calcs. Liabilities are valued at the GAAP interest, not discount rate
        L0_gaap = self.liabilities.pv(rates=self.gaap_int, val_date=val_date)
        mortality_risk = self.mortality_buffer / L0_gaap
        day_0_hgb_gap = (L0_gaap + self.mortality_buffer) - (self.cash + self.portfolio.total_allocation + self.asset_buffer[0])

        # Liability Cashflows & Dates
        liability_cashflows = self.liabilities.cashflows.loc[val_date:]
        dates = liability_cashflows.index
        dt = (dates - val_date).days / 365
        T = len(liability_cashflows.index)

        # Liability PVs
        liability_pvs = np.array([
            (liability_cashflows.iloc[i+1:] * (1 + self.gaap_int) ** -(dt[i+1:] - dt[i])).sum()
            for i in range(T)
        ])

        # Meltdown Liabilities
        cumulative_cf = liability_cashflows.cumsum()
        meltdown_liabilities = np.maximum(
            (L0_gaap - cumulative_cf) * (1 + mortality_risk) * (1 + self.gaap_int) ** dt, 0
        )

        # Next 2 Years of Liabilities
        next_2_liabs = (liability_cashflows.shift(-1) + liability_cashflows.shift(-2)).fillna(0)

        # Asset Data
        starting_cash = self.cash
        bond_df = self.portfolio.run_sim(val_date, rates, spread_map, sim_results).set_index('date')

        # check that the dates match
        bond_dates = bond_df.index.unique()
        assert bond_dates.isin(dates).all(), "There is a mismatch between bond and liability dates. Review bond cashflows dates."
        Tb = len(bond_dates)

        # Forward rates
        fwds = rates.calc_fwds(val_date).to_numpy()

        # Convert liability Series to numpy arrays
        liability_cf_np = liability_cashflows.to_numpy()
        meltdown_liab_np = meltdown_liabilities.to_numpy()
        next_2_liabs_np = next_2_liabs.to_numpy()

        # Run Simulation
        results = []
        n_sim = len(bond_df['scenario'].unique())
        for scenario in range(n_sim):
            # Asset cashflows and PV per scenario
            df_scenario = bond_df[bond_df['scenario'] == scenario]
            cashflow = df_scenario['cashflow'].to_numpy()
            pv = df_scenario['pv'].to_numpy()

            # Ensure length of asset data matches liability
            cashflow = np.pad(cashflow, (0, T - Tb))
            pv = np.pad(pv, (0, T - Tb))

            # Initialise arrays
            cash = np.zeros(T)
            assets = np.zeros(T)
            meltdown_assets = np.zeros(T)
            hgb_gap = np.zeros(T)
            payment = np.zeros(T)

            # Simulation loop
            for t in range(T):
                prev_cash = starting_cash if t == 0 else cash[t - 1]
                prev_gap = day_0_hgb_gap if t == 0 else hgb_gap[t - 1]

                # Assets
                cash[t] = prev_cash * (1 + fwds[t]) + cashflow[t] - liability_cf_np[t]
                assets[t] = cash[t] + pv[t]
                buffer_t = self.asset_buffer[t+1] if t < (len(self.asset_buffer)-2) else 0

                meltdown_assets[t] = assets[t] + buffer_t
                hgb_gap[t] = np.clip(meltdown_liab_np[t] - meltdown_assets[t], 0, prev_gap)

                # Optional Year 11 additonal  payment if needed
                if t == 10 and assets[t] < liability_pvs[t]:
                    extra = min(liability_pvs[t] - assets[t], self.asset_buffer[t+1])
                    cash[t] += extra
                    assets[t] += extra

                # Payment triggered if assets fall below next 2 liabilities
                if assets[t] < next_2_liabs_np[t] and payment[:t].sum() == 0:
                    payment[t] = hgb_gap[t]

            # Store results
            results.append(pd.DataFrame({
                'date': dates,
                'scenario': scenario,
                'liability_cashflows': liability_cf_np,
                'liability_pvs': liability_pvs,
                'meltdown_liabilities': meltdown_liab_np,
                'next_2_liabs': next_2_liabs_np,
                'asset_cashflow': cashflow,
                'remaining_asset_pv': pv,
                'cash': cash,
                'assets': assets,
                'meltdown_assets': meltdown_assets,
                'hgb_gap': hgb_gap,
                'payment': payment
            }))

        # Final Output
        cdi_results = pd.concat(results, ignore_index=True)

        expected_pv_payment = self._payment_expected_value(cdi_results, val_date, rates)

        return CDISimulationResult(
            cdi_results = cdi_results,
            bond_results = bond_df,
            expected_pv_payment = expected_pv_payment
        )