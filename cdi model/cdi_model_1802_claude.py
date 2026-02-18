---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
Cell In[19], line 56
     49 for si, s in enumerate(unique_spreads):
     50     # Discount factors: shape (n_years_cf, n_years)
     51     # rows = cashflow dates u, cols = valuation dates t
     52     # Avoid np.where eager evaluation of the power expression on non-future
     53     # cells — if any yield is NaN those cells would propagate into the result
     54     # even after masking. Instead compute only where future=True.
     55     dfs_mat = np.zeros_like(dtime_mat)
---> 56     dfs_mat[future] = (one_plus_y[:, np.newaxis] + s)[future] ** (-dtime_mat[future])
     58     # Sum over all cashflow dates u for each valuation year t:
     59     #   pv_table[si, t, bond] = sum_u dfs_mat[u, t] * cashflows[u, bond]
     60     #                         = (dfs_mat.T @ cashflows)
     61     # dfs_mat.T: (n_years, n_years_cf)  @  cashflows: (n_years_cf, n_bonds)
     62     #          = (n_years, n_bonds)  ✓
     63     pv_table[si] = dfs_mat.T @ cashflows                             # (n_years, n_bonds)

IndexError: boolean index did not match indexed array along dimension 1; dimension is 1 but corresponding boolean dimension is 25
