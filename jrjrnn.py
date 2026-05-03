import numpy as np
import pandas as pd
import warnings
#dcf

def dcf_valuation(fcf_list, wacc, terminal_growth):
    """
    Simple DCF: discount explicit FCF forecasts + terminal value.
    fcf_list: list of free cash flows [year1, year2, ..., yearN]
    wacc:     weighted average cost of capital
    terminal_growth: perpetuity growth rate
    """
    n         = len(fcf_list)
    pv_fcf    = sum(fcf / (1 + wacc)**t for t, fcf in enumerate(fcf_list, 1))
    terminal  = fcf_list[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_term   = terminal / (1 + wacc)**n
    return {"PV_FCF": round(pv_fcf, 2),
            "PV_Terminal": round(pv_term, 2),
            "Enterprise_Value": round(pv_fcf + pv_term, 2)}
 
fcf = [120, 135, 150, 168, 185]
print("\n5. DCF Valuation")
for k, v in dcf_valuation(fcf, wacc=0.09, terminal_growth=0.025).items():
    print(f"   {k}: ${v:,.0f}M")

