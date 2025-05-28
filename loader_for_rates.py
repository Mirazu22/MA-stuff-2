# -*- coding: utf-8 -*-
"""
Spyder-Editor

Dies ist eine temporäre Skriptdatei.
"""

import pandas as pd

# Paths to your CSV files
f_u = "results/csv/u_per_occ_numOMN…csv"  # steady‐state unemployment counts
f_e = f_u.replace("u_per_occ_num", "e_per_occ_num")  # employment counts
f_v = f_u.replace("u_per_occ_num", "v_per_occ_num")  # vacancy counts

# Load data
df_u = pd.read_csv(f_u)
df_e = pd.read_csv(f_e)
df_v = pd.read_csv(f_v)

# Define steady‐state window
t_cols = [f"t{i}" for i in range(25, 76)]

# Compute percentages
steady_u = df_u[t_cols].mean(axis=1) / (df_u[t_cols] + df_e[t_cols]).mean(axis=1) * 100
steady_v = df_v[t_cols].mean(axis=1) / (df_v[t_cols] + df_e[t_cols]).mean(axis=1) * 100

# Assemble output
df_ss = df_u[['id', 'label']].copy()
df_ss['u_ss_pct'] = steady_u
df_ss['v_ss_pct'] = steady_v

# Save
df_ss.to_csv("u_v_steady_state.csv", index=False)

df_ss.head()
