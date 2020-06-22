# Retrieved from https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
#[1] M. Duarte.  "Curve fitting," Jupyter Notebook.
#       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

# Import packages
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import curve_fit

# Define function to return the confidence bands
def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor="")

    return ax

# Directory
directory = 'C:/Users/Nikki/Desktop/Internship AM/Epsilon Project/Data'
os.chdir(directory)

# Retrieve data
df = pd.read_csv("accuracies.csv")
y = df['test']
x = df['samples']

# Define function to be fitted
def equation(s,a,b,c):
    return (1-a) - b * (s**c)                              # Return a power function

guess = [0, 1, -0.5] # Initial guess for the parameters
p, cov = curve_fit(equation, x, y, guess, maxfev=5000)     # Retrieve fitted parameters and their covariance
perr = np.sqrt(np.diag(cov))                               # Calculate the standard deviation of the parameters
print(p)
print(perr)

y_model = equation(x, p[0], p[1], p[2])                    # model using the fitted parameters

# T statistic
n = y.size                                                 # number of observations
m = p.size                                                 # number of parameters
dof = n - m                                                # degrees of freedom
t = stats.t.ppf(0.975, n - m)                              # t statistic used to determine confidence and prediction intervals

# Error estimation
resid = y - y_model
chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error

# Plotting the data
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, "o", color="#b9cfe7", markersize=8, markeredgewidth=1, markeredgecolor="b", markerfacecolor="None")
ax.plot(x, y_model, "-", color="0.1", linewidth=1.5, alpha=0.5, label="Fit")

x2 = [20, 40, 80, 160, 240, 320, 401, 481, 561, 641, 721, 802, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
#x2 = x
y2 = equation(x2, p[0], p[1], p[2])

# Plotting the confidence interval
plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)

# Plotting the prediction interval
pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
ax.plot(x2, y2 + pi, "--", color="0.5")

# Set borders of the figure
ax.spines["top"].set_color("0.5")
ax.spines["bottom"].set_color("0.5")
ax.spines["left"].set_color("0.5")
ax.spines["right"].set_color("0.5")
ax.get_xaxis().set_tick_params(direction="out")
ax.get_yaxis().set_tick_params(direction="out")
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# Set labels of the figure
#plt.title("", fontsize="14", fontweight="bold")
plt.xlabel("Sample size")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xlim(np.min(x) - 50, np.max(x) + 1000)

# Customize legend
handles, labels = ax.get_legend_handles_labels()
display = (0, 1)
anyArtist = plt.Line2D((0, 1), (0, 0), color="#b9cfe7")
legend = plt.legend(
    [handle for i, handle in enumerate(handles) if i in display] + [anyArtist],
    [label for i, label in enumerate(labels) if i in display] + ["95% Confidence Limits"],
    loc=9, bbox_to_anchor=(0, -0.21, 1., 0.102), ncol=3, mode="expand")
frame = legend.get_frame().set_edgecolor("0.5")

# Save and show figure
plt.tight_layout()
plt.savefig("filename.png", bbox_extra_artists=(legend,), bbox_inches="tight")
plt.show()