import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")

def bootstrap_yield_curve(maturities, par_rates):
    """
    Bootstrap zero rates from par swap rates.
    maturities: list of years [1, 2, 3, 5, 7, 10]
    par_rates:  list of annualised par rates
    """
    zero_rates = {}
    discount_factors = {}
 
    for i, (T, par) in enumerate(zip(maturities, par_rates)):
        if i == 0:
            z = par  # first zero = par for 1yr
        else:
            # sum of discounted coupons from previous periods
            coupon_pv = sum(par * discount_factors[t] for t in maturities[:i])
            df = (1 - coupon_pv) / (1 + par)
            z  = -np.log(df) / T
        discount_factors[T] = np.exp(-z * T)
        zero_rates[T] = round(z * 100, 4)
    # add Nelson-Siegel
    # extract forward rate 
    # add interpolation
    return zero_rates
 
maturities = [1, 2, 3, 5]
par_rates  = [0.04, 0.042, 0.044, 0.047]
print("\n3. Yield Curve Bootstrap")
for T, z in bootstrap_yield_curve(maturities, par_rates).items():
    print(f"   {T}Y zero rate: {z}%")
