import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Given data (concentration in ppm, absorbance)
x = np.array([20, 200, 400, 600, 800, 1200])
yVis = np.array([0.008, 0.129, 0.183, 0.247, 0.375, 0.484])
yUV = np.array([0.009, 0.070, 0.099, 0.131, 0.211, 0.257])

def calibration_analysis(x, y, label):
    """Perform linear regression and calculate analytical parameters."""
    # Linear regression using scipy
    slope, intercept, r_value, p_value, std_err_slope = stats.linregress(x, y)
    r_squared = r_value ** 2
    
    # Predicted values and residuals
    y_pred = slope * x + intercept
    residuals = y - y_pred
    n = len(x)
    # Residual standard deviation (RMSE)
    rmse = np.sqrt(np.sum(residuals**2) / (n - 2))
    
    # Standard error of the intercept
    mean_x = np.mean(x)
    ss_xx = np.sum((x - mean_x)**2)
    se_intercept = rmse * np.sqrt(1/n + mean_x**2 / ss_xx)
    
    # t-test for intercept (null hypothesis intercept = 0)
    t_intercept = intercept / se_intercept
    # two-tailed p-value
    p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), df=n-2))
    
    # LOD and LOQ
    lod = 3.3 * rmse / slope
    loq = 10 * rmse / slope
    
    # Durbin-Watson statistic (simple calculation)
    diff = np.diff(residuals)
    dw = np.sum(diff**2) / np.sum(residuals**2)
    
    # Print results
    print(f"\n===== Calibration curve: {label} =====")
    print(f"Equation: y = {slope:.6f} x + {intercept:.6f}")
    print(f"R² = {r_squared:.6f}")
    print(f"Slope standard error: {std_err_slope:.6f}")
    print(f"Intercept standard error: {se_intercept:.6f}")
    print(f"Intercept t-test: t = {t_intercept:.4f}, p = {p_intercept:.4f}")
    if p_intercept < 0.05:
        print("Intercept is significantly different from zero (p < 0.05).")
    else:
        print("Intercept is not significantly different from zero (p >= 0.05).")
    print(f"Residual standard deviation (RMSE): {rmse:.6f}")
    print(f"LOD (3.3 * RMSE / slope): {lod:.2f} ppm")
    print(f"LOQ (10 * RMSE / slope): {loq:.2f} ppm")
    print(f"Linear range tested: {min(x)} - {max(x)} ppm")
    print(f"Durbin-Watson statistic: {dw:.4f} (near 2 indicates no autocorrelation)")

    # Residual plot
    plt.figure(figsize=(7, 4))
    plt.scatter(x, residuals, color='black', marker='x', s=80, linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Concentration (ppm)', fontsize=14)
    plt.ylabel('Residuals (absorbance)', fontsize=14)
    plt.title(f'Residual plot for {label}', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(f'Residuals_{label}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'slope': slope, 'intercept': intercept, 'r_squared': r_squared,
        'slope_se': std_err_slope, 'intercept_se': se_intercept,
        'rmse': rmse, 'lod': lod, 'loq': loq,
        'p_intercept': p_intercept, 'dw': dw
    }

# Analyze both curves
res_vis = calibration_analysis(x, yVis, "Visible (560 nm)")
res_uv = calibration_analysis(x, yUV, "UV (350 nm)")

# Plot calibration curves with regression line and statistics on the plot
def plot_calibration(x, y, label, slope, intercept, r_squared):
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color='black', marker='x', s=100, linewidth=2)
    x_fit = np.linspace(0, 1200, 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='black', linewidth=2.5,
             label=f'Fit: y = {slope:.6f}x + {intercept:.4f}\nR² = {r_squared:.4f}')
    plt.xlabel('Concentration (ppm)', fontsize=18)
    plt.ylabel('Absorbance', fontsize=18)
    plt.xlim(0, 1200)
    plt.ylim(0, max(y)*1.1)
    plt.tick_params(axis='both', labelsize=16, size=8, width=2)
    plt.legend(fontsize=14, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'Curve_{label}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Plot curves with intercept
plot_calibration(x, yVis, "Vis", res_vis['slope'], res_vis['intercept'], res_vis['r_squared'])
plot_calibration(x, yUV, "UV", res_uv['slope'], res_uv['intercept'], res_uv['r_squared'])

# Print a summary table (copy-paste friendly)
print("\n===== Summary Table (copy for paper) =====")
print("Technique | Slope (ppm⁻¹) | Intercept | R² | RMSE | LOD (ppm) | LOQ (ppm)")
print(f"Visible (560 nm) | {res_vis['slope']:.6f} | {res_vis['intercept']:.6f} | {res_vis['r_squared']:.4f} | {res_vis['rmse']:.6f} | {res_vis['lod']:.2f} | {res_vis['loq']:.2f}")
print(f"UV (350 nm)     | {res_uv['slope']:.6f} | {res_uv['intercept']:.6f} | {res_uv['r_squared']:.4f} | {res_uv['rmse']:.6f} | {res_uv['lod']:.2f} | {res_uv['loq']:.2f}")