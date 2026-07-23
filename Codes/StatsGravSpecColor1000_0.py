import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# Example data - replace with your actual data
# Each sub-list represents triplicate measurements for one condition

# Conditions 1, 2, and 3 are GRAVIMETRY, SPECTROFOTOMETRY, and COLORIMETRY (1000 ppm + 0 g/L) 

data = {
    'Gravimetry': [1023.44, 1077.73, 1135.10],
    'Spec. (350 nm)': [878.79, 900.43, 900.43],
    'Spec. (560 nm)': [911.01, 927.40, 922.72]
}


1164.017991
1204.341534
1162.662808

835.4978355
861.4718615
848.4848485

925.058548
899.2974239
866.5105386


# Set your alpha level
alpha = 0.1

# -------------------
# Data Preparation
# -------------------
# Convert to long format for analysis
df = pd.DataFrame([(k, v) for k in data for v in data[k]], 
                  columns=['Technique', 'Oil Content [ppm]'])

# -------------------
# Descriptive Statistics
# -------------------
print("Descriptive Statistics:")
print(df.groupby('Technique').describe())

# -------------------
# Visualization
# -------------------
plt.figure(figsize=(8, 6))

# Boxplot with custom style:
# - linewidth = 2 for all lines (box edges, whiskers, caps, median)
# - linecolor = 'black'
# - box fill = light blue with alpha = 0.5 (semi‑transparent)
# - median line thickness also increased
sns.boxplot(
    x='Technique', y='Oil Content [ppm]', data=df,
    linewidth=2,
    boxprops=dict(facecolor='lightblue', alpha=0.5, edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=2),
    capprops=dict(color='black', linewidth=2),
    medianprops=dict(color='black', linewidth=2),
    flierprops=dict(marker='o', markersize=6, markerfacecolor='black', markeredgecolor='black')
)

# Swarmplot with larger black points (size = 8, alpha = 0.5)
sns.swarmplot(
    x='Technique', y='Oil Content [ppm]', data=df,
    color='black', size=8, alpha=0.5
)

# Axis labels and title
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Technique',fontsize=18)
plt.ylabel('Oil Content [ppm]',fontsize=18)


#plt.title('Measurement Distribution Across Conditions')
plt.show()

# -------------------
# Normality Check (Shapiro-Wilk test on residuals)
# -------------------
# First perform ANOVA to get residuals
model = stats.f_oneway(*data.values())
residuals = np.concatenate([np.array(data[c]) - np.mean(data[c]) for c in data])
_, p_norm = stats.shapiro(residuals)

print(f"\nNormality test (Shapiro-Wilk) p-value: {p_norm:.4f}")
if p_norm > alpha:
    print("Residuals appear normally distributed (fail to reject H0)")
else:
    print("Residuals do not appear normally distributed (reject H0)")

# -------------------
# Homogeneity of Variance Check (Levene's test)
# -------------------
_, p_var = stats.levene(*data.values())
print(f"\nHomogeneity of variance test (Levene's) p-value: {p_var:.4f}")
if p_var > alpha:
    print("Variances appear homogeneous (fail to reject H0)")
else:
    print("Variances do not appear homogeneous (reject H0)")

# -------------------
# Main Statistical Test
# -------------------
if p_norm > alpha and p_var > alpha:
    # Parametric ANOVA
    print("\nPerforming one-way ANOVA:")
    f_val, p_val = stats.f_oneway(*data.values())
    print(f"F-value: {f_val:.4f}, p-value: {p_val:.4f}")
    
    if p_val < alpha:
        print(f"Significant differences found (p < {alpha})")
        
        # Post-hoc Tukey's HSD if ANOVA is significant
        print("\nPost-hoc Tukey HSD test:")
        tukey = pairwise_tukeyhsd(df['Measurement'], df['Condition'], alpha=alpha)
        print(tukey)
        
        # Plot the Tukey results
        tukey.plot_simultaneous()
        plt.title('Tukey HSD Confidence Intervals')
        plt.show()
    else:
        print(f"No significant differences found (p >= {alpha})")
else:
    # Non-parametric Kruskal-Wallis
    print("\nPerforming Kruskal-Wallis test (non-parametric):")
    h_val, p_val = stats.kruskal(*data.values())
    print(f"H-value: {h_val:.4f}, p-value: {p_val:.4f}")
    
    if p_val < alpha:
        print(f"Significant differences found (p < {alpha})")
        
        # Post-hoc Dunn's test if Kruskal-Wallis is significant
        try:
            from scikit_posthocs import posthoc_dunn
            print("\nPost-hoc Dunn's test with Bonferroni correction:")
            dunn_results = posthoc_dunn(df, val_col='Measurement', group_col='Condition', p_adjust='bonferroni')
            print(dunn_results)
        except ImportError:
            print("\nFor Dunn's post-hoc test, please install scikit-posthocs: pip install scikit-posthocs")
    else:
        print(f"No significant differences found (p >= {alpha})")