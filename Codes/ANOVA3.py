import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ---------------------------
# 1. DATA
# ---------------------------
data = {
    'Gravimetry': {
        0: {
            50: [42.52, 42.35, 54.38],   
            300: [289.0, 272.7, 281.5],
            500: [451.78, 613.14, 515.75],
            800: [721.9, 748.5, 739.7],
            1000: [1023.44, 1077.73, 1135.10]
        },
        35: {
            50: [30.59, 57.20, 48.90],
            300: [..., ..., ...],  
            500: [621.26, 495.42, 666.57],
            800: [..., ..., ...],  
            1000: [1211.90, 1192.27, 1209.46]
        },
        100: {
            50: [65.13, 106.20, 68.98],
            300: [316.1, 300.0, 290.9],
            500: [629.58, 589.18, 615.15],
            800: [765.7, 708.9, 780],
            1000: [1164.02, 1204.34, 1162.66]
        }
    },
    'Spec_350nm': {
        0: {
            50: [43.29, 25.67, 30.30],   
            300: [259.7, 251.1, 264.1],
            500: [402.60, 445.89, 454.55],
            800: [718.65, 787.9, 805.2],
            1000: [878.79, 900.43, 900.43]
        },
        35: {
            50: [82.25, 69.26, 34.63],
            300: [..., ..., ...],  
            500: [411.26, 454.55, 441.56],
            800: [..., ..., ...],  
            1000: [848.48, 835.50, 835.50]
        },
        100: {
            50: [34.63, 38.96, 34.63],
            300: [216.5, 242.4, 242.4],
            500: [428.57, 424.24, 419.91],
            800: [688.3, 779.2, 770.6],
            1000: [835.50, 861.47, 848.48]
        }
    },
    'Spec_560nm': {
        0: {
            50: [44.50, 37.47, 37.00],   
            300: [257.6, 245.9, 281.0],
            500: [442.62, 463.70, 461.36],
            800: [779.9, 751.8, 761.1],
            1000: [911.01, 927.40, 922.72]
        },
        35: {
            50: [39.81, 53.86, 46.84],
            300: [..., ..., ...],  
            500: [430.91, 449.65, 470.73],
            800: [..., ..., ...], 
            1000: [894.61, 859.48, 871.19]
        },
        100: {
            50: [35.12, 37.47, 46.84],
            300: [257.6, 304.4, 304.4],
            500: [440.28, 433.26, 428.57],
            800: [761.1, 730.7, 808.0],
            1000: [925.06, 899.30, 866.51]
        }
    }
}

# ---------------------------
# 2. SMART DATA PARSER 
# ---------------------------
rows = []

for tech, sal_dict in data.items():
    for sal, tog_dict in sal_dict.items():
        for tog, replicates in tog_dict.items():
            if isinstance(replicates, list) and len(replicates) > 0:
                for val in replicates:
                    if isinstance(val, (int, float)):
                        rows.append([tech, sal, tog, val])
                    else:
                        print(f"Warning: Skipping non-numeric value '{val}' for {tech}, Sal={sal}, TOG={tog}")

# Convert to DataFrame
df = pd.DataFrame(rows, columns=['Technique', 'Salinity', 'TOG_Nominal', 'Measurement'])

# Convert to categorical for ANOVA
df['Technique'] = df['Technique'].astype('category')
df['Salinity'] = df['Salinity'].astype('category')
df['TOG_Nominal'] = df['TOG_Nominal'].astype('category')

print(f"Total valid rows loaded: {len(df)}")  
print(df.head(10))

# ---------------------------
# 3. DESCRIPTIVE STATS 
# ---------------------------
print("\nGroup counts (checking for missing combinations):")
print(df.groupby(['Salinity', 'TOG_Nominal']).size())

# ---------------------------
# 4. 3-WAY ANOVA WITH TYPE II SUMS OF SQUARES
# ---------------------------

model = ols('Measurement ~ C(Technique) * C(Salinity) * C(TOG_Nominal)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("\n=== UNBALANCED 3-WAY ANOVA TABLE (Type II SS) ===")
print(anova_table)

# ---------------------------
# 5. CHECK ASSUMPTIONS (ON THE RESIDUALS OF THE MODEL)
# ---------------------------
residuals = model.resid

# Normality
_, p_norm = stats.shapiro(residuals)
print(f"\nShapiro-Wilk p-value (residuals): {p_norm:.4f}")
if p_norm > 0.05:
    print("Residuals appear normally distributed.")
else:
    print("Residuals deviate from normality (ANOVA is fairly robust if sample size is large enough).")

# Homogeneity of variance (Levene's test - groups with missing data are fine, just fewer groups)
df['Group'] = df['Technique'].astype(str) + '_' + df['Salinity'].astype(str) + '_' + df['TOG_Nominal'].astype(str)
groups = [df[df['Group'] == g]['Measurement'] for g in df['Group'].unique()]
_, p_var = stats.levene(*groups)
print(f"Levene's p-value: {p_var:.4f}")
if p_var > 0.05:
    print("Variances appear homogeneous across groups.")
else:
    print("Variances are not homogeneous. You might consider transforming the data (e.g., log) or using robust methods.")

# ---------------------------
# 6. POST-HOC TEST (Compare techniques where you have complete data)
# ---------------------------
# Since TOG=1000 & Salinity=100 is fully populated across all 3 techniques:
subset = df[(df['TOG_Nominal'] == 1000) & (df['Salinity'] == 100)]
print(f"\nNumber of rows in subset (TOG=1000, Sal=100): {len(subset)}")  # Should be 9

if len(subset) > 0:
    tukey_subset = pairwise_tukeyhsd(subset['Measurement'], subset['Technique'], alpha=0.05)
    print("\nTukey HSD (Techniques at TOG=1000, Salinity=100):")
    print(tukey_subset)
else:
    print("No complete data for this subset. Pick another combination.")



# ---------------------------
# ENHANCEMENT 1: EFFECT SIZES (Partial Eta Squared)
# ---------------------------
print("\n=== EFFECT SIZES (Partial Eta Squared) ===")
# Partial Eta^2 = SS_effect / (SS_effect + SS_residual)
ss_residual = anova_table.loc['Residual', 'sum_sq']
for index, row in anova_table.iterrows():
    if index != 'Residual':
        ss_effect = row['sum_sq']
        eta_sq = ss_effect / (ss_effect + ss_residual)
        print(f"{index}: eta^2 = {eta_sq:.4f}")  # <-- Fixed to avoid Windows encoding error

# ---------------------------
# ENHANCEMENT 2: SYSTEMATIC POST-HOC (Simple Effects)
# ---------------------------
print("\n=== SYSTEMATIC POST-HOC (Tukey HSD for each Salinity & TOG combination) ===")
print("Note: Applying Bonferroni correction (alpha = 0.05 / 15 comparisons = 0.0033)")
print("Only p-values < 0.0033 are considered significant here.\n")

# Get all unique combinations that actually have data
combinations = df.groupby(['Salinity', 'TOG_Nominal']).size()
combinations = combinations[combinations > 0].index.tolist()

significant_results = []

for sal, tog in combinations:
    subset = df[(df['Salinity'] == sal) & (df['TOG_Nominal'] == tog)]
    # Only run Tukey if we have all 3 techniques present
    if len(subset['Technique'].unique()) == 3:
        tukey = pairwise_tukeyhsd(subset['Measurement'], subset['Technique'], alpha=0.05)
        print(f"--- Salinity: {sal} g/L, TOG: {tog} ppm ---")
        print(tukey)
        print("-" * 50)
    else:
        print(f"Skipping Salinity: {sal}, TOG: {tog} (missing technique data)")

print("\nFor the paper: Report the significant pairwise comparisons (p < 0.05) from the tables above,")
print("but explicitly mention that due to the multiple comparisons, only p < 0.0033 is strictly 'significant' under Bonferroni.")