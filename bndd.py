import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")
def bond_analytics(face, coupon_rate, ytm, n_periods, freq=2):
    """
    Compute bond price, modified duration and convexity.
    freq: coupon payments per year (2 = semi-annual)
    """
    c  = face * coupon_rate / freq
    r  = ytm / freq
    t  = np.arange(1, n_periods * freq + 1)
    cf = np.full(len(t), c)
    cf[-1] += face
    pv       = cf / (1 + r)**t
    price    = pv.sum()
    duration = (t * pv).sum() / price / freq           # duration in years)
    mod_dur  = duration / (1 + ytm / freq)             # Modified duration
    convex   = ((t**2 + t) * pv).sum() / price / (freq**2 * (1 + r)**2)
    # add DV01 (dollar duration), yield curve sens
    return {"Price": round(price, 4),
            "MacaulayDur": round(duration, 4),
            "ModDur": round(mod_dur, 4),
            "Convexity": round(convex, 4)}
 
print("\n7. Bond Duration & Convexity")
for k, v in bond_analytics(1000, 0.05, 0.06, 10).items():
    print(f"   {k}: {v}")
