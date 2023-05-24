from sklearn.metrics import accuracy_score, recall_score
import numpy as np
import xxhash

def fairness_metrics(df_fm, protected_attribute, target):
    
    fair_met = {# Statistical Parity
                "SP_a_1": np.nan,
                "SP_a_0": np.nan,
                "DI": np.nan, # Disparate Impact
                "SPD": np.nan, # Statistical Parity Difference
                # Equal Opportunity
                "EO_a_1": np.nan,
                "EO_a_0": np.nan,
                "EOD": np.nan, # Equal Opportunity Difference
                # Overall Accuracy
                "OA_a_1": np.nan,
                "OA_a_0": np.nan,
                "OAD": np.nan, # Overall Accuracy Difference
                }
    
    # Filtering datasets for fairness metrics
    df_a_1 = df_fm.loc[df_fm[protected_attribute+"_1"]==1]
    df_a_0 = df_fm.loc[df_fm[protected_attribute+"_1"]==0]

    # Calculate Statistical Parity per group
    SP_a_1 = df_a_1.loc[df_a_1["y_pred"]==1].shape[0] / df_a_1.shape[0]
    SP_a_0 = df_a_0.loc[df_a_0["y_pred"]==1].shape[0] / df_a_0.shape[0]
    fair_met["SP_a_1"] = SP_a_1
    fair_met["SP_a_0"] = SP_a_0

    # Disparate Impact
    DI = SP_a_0 / SP_a_1
    fair_met["DI"] = DI
    
    # Statistical Parity Difference
    SPD = SP_a_1 - SP_a_0
    fair_met["SPD"] = SPD

    # Equal Opportunity
    EO_a_1 = recall_score(df_a_1[target], df_a_1['y_pred'])
    EO_a_0 = recall_score(df_a_0[target], df_a_0['y_pred'])
    fair_met["EO_a_1"] = EO_a_1
    fair_met["EO_a_0"] = EO_a_0

    # Equal Opportunity Difference
    EOD = EO_a_1 - EO_a_0
    fair_met["EOD"] = EOD

    # Overall Accuracy
    OA_a_1 = accuracy_score(df_a_1[target], df_a_1['y_pred'])
    OA_a_0 = accuracy_score(df_a_0[target], df_a_0['y_pred'])
    fair_met["OA_a_1"] = OA_a_1
    fair_met["OA_a_0"] = OA_a_0

    # Accuracy per Group Difference
    OAD = OA_a_1 - OA_a_0
    fair_met["OAD"] = OAD
    
    return fair_met    



def IVE_LH(val_seed, k, epsilon, optimal=True):
    """
    Indicator-Vector-Encoding (IVE) for Local Hashing (LH) Mechanisms.
    """
    
    g=2 # BLH parameter
    if optimal:
        g = int(np.round(np.exp(epsilon))) + 1 # OLH parameter
    
    ive_lh = np.zeros(k)

    for v in range(k):
        if val_seed[0] == (xxhash.xxh32(str(v), seed=val_seed[1]).intdigest() % g):
            ive_lh[v] = 1
    
    return ive_lh    
    
def IVE_SS(ss, k):
    """
    Indicator-Vector-Encoding (IVE) for Subset Selection (SS) Mechanism.
    """
    
    ive_ss = np.zeros(k)
    ive_ss[ss] = 1
    
    return ive_ss    
    
def IVE_THE(hist, k, thresh):
    """
    Indicator-Vector-Encoding (IVE) for Thresholding with Histogram Encoding (THE) Mechanism.
    """
    
    ss_the = np.where(hist > thresh)[0]
    
    ive_the = np.zeros(k)
    
    if len(ss_the) > 0:
        ive_the[ss_the] = 1
    
    return ive_the    
    
    
    