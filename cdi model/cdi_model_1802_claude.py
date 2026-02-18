---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
Cell In[15], line 2
      1 val_date = pd.to_datetime('01-01-2026', dayfirst = True)
----> 2 bond_sim = bonds.run_sim(val_date, rates, spread_map, sim_results)

File c:\Users\millsan\git_repos\cashflow_modelling\scripts\bond_sim_testing.py:57, in run_sim(self, val_date, rates, spread_map, sim_results)
     51 n_years, n_bonds = cashflows.shape
     52 # Last non-zero cashflow index per bond (-1 if all zero)
     53 last_cf_idx = np.where(
     54     cashflows.any(axis=0),
     55     n_years - 1 - np.argmax(cashflows[::-1, :] != 0, axis=0),
     56     -1
---> 57 )
     58 year_indices = np.arange(n_years)[:, np.newaxis]   # (n_years, 1)
     59 return year_indices <= last_cf_idx[np.newaxis, :]

IndexError: boolean index did not match indexed array along dimension 1; dimension is 1 but corresponding boolean dimension is 25
