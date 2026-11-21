"""Segmentation helpers: RFM, k-means, KPI profiling, matching, stability."""
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

KM_FEATS = ["opd_n", "lab_n", "adm_n", "recency", "revenue", "growth", "monsoon_share", "age", "chronic_share", "labs_per_visit"]
LOG_FEATS = ["opd_n", "lab_n", "adm_n", "recency", "revenue", "labs_per_visit"]
TOTAL_BEDS = 140


def model_matrix(f):
    X = f[KM_FEATS].copy()
    X["revenue"] = X.revenue.clip(upper=X.revenue.quantile(0.995))       # winsorise the extreme tail
    for c in LOG_FEATS: X[c] = np.log1p(X[c])
    return X


def scaled(f):
    X = model_matrix(f)
    return StandardScaler().fit_transform(X), X


# ---------------------------------------------------------------- RFM
def _q5(s, ascending=True):
    r = s.rank(method="first", ascending=ascending)
    return pd.qcut(r, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def rfm(f):
    out = pd.DataFrame(index=f.index)
    out["R"] = _q5(f.recency, ascending=False)           # most recent -> 5
    out["F"] = _q5(f.enc_n + f.lab_n * 0.01)              # tiny lab tie-breaker
    out["M"] = _q5(f.revenue)
    def name(r):
        R, F, M = r.R, r.F, r.M
        if R >= 4 and F >= 4 and M >= 4: return "R1 Champions"
        if R >= 3 and M >= 4: return "R2 High spenders"
        if R >= 3 and F >= 3: return "R3 Steady regulars"
        if R <= 2 and (F >= 3 or M >= 4): return "R4 Slipping valuable"
        if R >= 4: return "R5 Recent light users"
        return "R6 Dormant light users"
    out["rfm_segment"] = out.apply(name, axis=1)
    out["rfm_score"] = out.R.astype(str) + out.F.astype(str) + out.M.astype(str)
    return out


# ---------------------------------------------------------------- k-means selection
def k_sweep(Xs, ks=range(2, 11), sample=8000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xs), min(sample, len(Xs)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=seed).fit(Xs)
        sizes = np.bincount(km.labels_, minlength=k) / len(Xs)
        rows.append(dict(k=k, inertia=km.inertia_, silhouette=silhouette_score(Xs[idx], km.labels_[idx]),
                         davies_bouldin=davies_bouldin_score(Xs, km.labels_), min_share=sizes.min()))
    return pd.DataFrame(rows)


def fit_km(Xs, k, seed=0):
    return KMeans(k, n_init=20, random_state=seed).fit(Xs)


# ---------------------------------------------------------------- outcomes (admission level)
def outcomes(D, seg, start, end):
    """Per-patient clinical/operational outcomes inside the window, plus index-admission table."""
    a = D["adm"].copy()
    a = a.sort_values(["patient_id", "admit_date"])
    a["next_admit"] = a.groupby("patient_id").admit_date.shift(-1)
    a["readm30"] = ((a.next_admit - a.discharge_date).dt.days.between(0, 30))
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    w = a[(a.admit_date >= start) & (a.admit_date <= end)].copy()
    # right-censoring: discharges in the last 30 days of the data cannot show a 30-day readmission
    w["eligible"] = w.discharge_date <= (pd.Timestamp("2025-12-31") - pd.Timedelta(days=30))
    # bed-days clipped to the window
    ov = (np.minimum(w.discharge_date, end + pd.Timedelta(days=1)) - np.maximum(w.admit_date, start)).dt.days.clip(lower=0)
    w["bed_days"] = ov
    w["segment"] = w.patient_id.map(seg)
    return w[w.segment.notna()]


def profile(f, seg_col, w_adm, start, end, labels_order=None):
    """Segment profile table: size, revenue/volume share, LOS, occupancy contribution, readmission."""
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    g = f.groupby(seg_col)
    t = pd.DataFrame({"patients": g.size()})
    t["share_patients_%"] = 100 * t.patients / t.patients.sum()
    t["share_revenue_%"] = 100 * g.revenue.sum() / f.revenue.sum()
    t["share_visits_%"] = 100 * g.enc_n.sum() / f.enc_n.sum()
    t["visits_per_patient"] = g.enc_n.mean()
    t["opd_per_patient"] = g.opd_n.mean()
    t["admissions_per_100"] = 100 * g.adm_n.mean()
    t["revenue_per_patient"] = g.revenue.mean()
    t["median_recency_days"] = g.recency.median()
    t["mean_age"] = g.age.mean()
    t["chronic_share"] = g.chronic_share.mean()
    t["growth"] = g.growth.mean()
    t["monsoon_share"] = g.monsoon_share.mean()
    ad = w_adm.groupby("segment")
    t["admissions"] = ad.size().reindex(t.index).fillna(0).astype(int)
    t["avg_LOS_days"] = ad.los.mean().reindex(t.index)
    t["bed_days"] = ad.bed_days.sum().reindex(t.index).fillna(0)
    t["share_bed_days_%"] = 100 * t.bed_days / t.bed_days.sum()
    t["occupancy_contrib_pp"] = 100 * t.bed_days / (TOTAL_BEDS * days)
    el = w_adm[w_adm.eligible].groupby("segment")
    t["readmit_30d_%"] = 100 * el.readm30.mean().reindex(t.index)
    t["readmit_denominator"] = el.size().reindex(t.index).fillna(0).astype(int)
    # too few stays to quote a rate (n < 30): leave blank rather than show a misleading 0% or 100%
    t.loc[t.admissions < 30, "avg_LOS_days"] = np.nan
    t.loc[t.readmit_denominator < 30, "readmit_30d_%"] = np.nan
    return t


# ---------------------------------------------------------------- KPI separation
def kpi_separation(f, seg_col, w_adm):
    """Eta-squared + Kruskal-Wallis p for KPIs. 'outcome' = not used as a clustering input."""
    rows = []
    def eta(vals, grp):
        d = pd.DataFrame({"v": vals, "g": grp}).dropna()
        if d.g.nunique() < 2: return np.nan, np.nan
        gm = d.v.mean(); ssb = d.groupby("g").v.apply(lambda x: len(x) * (x.mean() - gm) ** 2).sum()
        sst = ((d.v - gm) ** 2).sum()
        p = stats.kruskal(*[x.values for _, x in d.groupby("g").v]).pvalue
        return ssb / sst, p
    seg = f[seg_col]
    spec = []
    # clustering inputs (for reference)
    for k in ["revenue", "enc_n", "recency"]:
        spec.append((k, "input to segmentation", f[k], seg))
    # independent KPIs
    pat = w_adm.groupby("patient_id").agg(los=("los", "mean"), bed=("bed_days", "sum"))
    spec.append(("avg LOS (admitted patients)", "outcome (not an input)", pat.los, seg.reindex(pat.index)))
    spec.append(("bed-days per patient", "outcome (not an input)", f.index.map(pat.bed).fillna(0).to_series(index=f.index), seg))
    e = w_adm[w_adm.eligible]
    spec.append(("30-day readmission (index stays)", "outcome (not an input)", e.readm30.astype(float), e.segment))
    spec.append(("lab tests per patient", "outcome-ish", f.lab_n, seg))
    for name, kind, v, g in spec:
        v = pd.Series(np.asarray(v, dtype=float), index=getattr(v, "index", None)) if not isinstance(v, pd.Series) else v
        eta_, p = eta(v.values if hasattr(v, "values") else v, np.asarray(g))
        rows.append(dict(kpi=name, kind=kind, eta_squared=eta_, kruskal_p=p))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- matching + stability
def match_labels(profA, profB):
    """Hungarian matching of segments in B to segments in A by distance of standardised centroid profiles."""
    cols = ["opd_n", "adm_n", "lab_n", "recency", "revenue", "growth", "monsoon_share", "age", "chronic_share"]
    A, B = profA[cols], profB[cols]
    mu, sd = pd.concat([A, B]).mean(), pd.concat([A, B]).std()
    Az, Bz = (A - mu) / sd, (B - mu) / sd
    cost = np.linalg.norm(Az.values[:, None, :] - Bz.values[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return {B.index[j]: A.index[i] for i, j in zip(r, c)}, cost


def change_rate(a, b):
    """% of entities (present in both) whose segment differs."""
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    return 100 * (j.iloc[:, 0] != j.iloc[:, 1]).mean(), len(j)
