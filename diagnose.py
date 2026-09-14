import bucex as bx
import matplotlib.pyplot as plt

# change to your local path to the fit.bucex file
fit = bx.FitResult.load("results/serra/uccle_independent_20260914T100026_454684Z/TXx/fit.bucex")

print(bx.scientific_summary(fit, threshold=35))
fit.plot("trace", parameters=["xi", "sigma", "sd.level", "sd.slope", "sd.seasonal"])
fit.plot("acf", parameters=["xi", "sigma", "sd.level", "sd.slope", "sd.seasonal"])
plt.show()