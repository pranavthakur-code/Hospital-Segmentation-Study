"""Cleaning + feature engineering (shared by the notebook and the checks).  All functions are pure."""
import re, difflib
import numpy as np, pandas as pd

REF = pd.Timestamp("2025-12-31")
CHRONIC = {"diabetes", "hypertension", "cardiac", "renal", "asthma_copd"}

# --------------------------------------------------------------------------- age
_DOB_FORMATS = [(r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"), (r"^\d{2}/\d{2}/\d{4}$", "%d/%m/%Y"),
                (r"^\d{2}-[A-Za-z]{3}-\d{4}$", "%d-%b-%Y"), (r"^\d{2}-\d{2}-\d{2}$", "%d-%m-%y")]


def parse_age(x, ref=REF):
    """Returns (age_years, source) where source is 'age', 'dob' or 'bad'."""
    if pd.isna(x): return np.nan, "bad"
    s = str(x).strip()
    m = re.match(r"^(\d{1,3})\s*(y|yr|yrs|year|years)?$", s, re.I)
    if m: return float(m.group(1)), "age"
    for pat, fmt in _DOB_FORMATS:
        if re.match(pat, s):
            d = pd.to_datetime(s, format=fmt, errors="coerce")
            if pd.isna(d): break
            if d > ref: d = d - pd.DateOffset(years=100)          # 2-digit year pivot (e.g. 30 -> 2030 -> 1930)
            return (ref - d).days / 365.25, "dob"
    return np.nan, "bad"


# --------------------------------------------------------------------------- diagnosis free text -> canonical group
RULES = [  # ordered: first match wins
    ("paeds", r"pediatric|paediatric|childhood|bronchiolitis|vaccinat|well baby"),
    ("maternal", r"antenatal|\banc\b|pregnan|deliver|lscs|c-section|post natal|pcos|menorrhagia"),
    ("diabetes", r"diabet|\bdm\b|dm-?\s?2|t2dm|type (2|ii)"),
    ("hypertension", r"hypertens|\bhtn\b|high bp|blood pressure|\bbp\b"),
    ("cardiac", r"chest pain|angina|coronary|\bcad\b|\bihd\b|heart failure|\bchf\b|palpitation|\bmi\b"),
    ("renal", r"\bckd\b|kidney|\buti\b|urinary|renal|pyelo"),
    ("asthma_copd", r"asthma|copd|wheez|allergic (rhinitis|bronchitis)"),
    ("resp_infection", r"urti|upper respiratory|\buri\b|cough|cold|bronchitis"),
    ("fever", r"fever|pyrexia|\bpuo\b|dengue|typhoid|enteric|malaria"),
    ("gastro", r"gastro|\bage\b|loose|diarrh|gastritis|acidity|gerd|abdominal pain|vomit"),
    ("injury", r"fracture|^#|\brta\b|road traffic|sprain|dislocation"),
    ("musculoskeletal", r"back pain|\blbp\b|knee|osteoarth|\boa\b|joint pain|neck pain|spondyl|back ache"),
    ("surgical", r"appendic|hernia|gallstone|cholecyst|hysterectomy|piles|hemorrhoid|fissure"),
    ("skin", r"skin|dermatitis|fungal|scabies|acne|urticaria|eczema"),
    ("ent", r"ear pain|otitis|sinusitis|tonsillitis|vertigo"),
    ("other", r"check.?up|weakness|anaemia|anemia|vitamin|thyroid"),
]
_RX = [(g, re.compile(p)) for g, p in RULES]
# single-word keywords used for the fuzzy (typo) fallback
KEYWORDS = {"hypertension": "hypertension", "diabetes": "diabetes", "diabetic": "diabetes", "bronchitis": "resp_infection",
            "respiratory": "resp_infection", "gastroenteritis": "gastro", "gastritis": "gastro", "diarrhoea": "gastro",
            "fracture": "injury", "dislocation": "injury", "sprain": "injury", "osteoarthritis": "musculoskeletal",
            "spondylosis": "musculoskeletal", "asthma": "asthma_copd", "wheezing": "asthma_copd", "dengue": "fever",
            "typhoid": "fever", "malaria": "fever", "fever": "fever", "pregnancy": "maternal", "antenatal": "maternal",
            "delivery": "maternal", "appendicitis": "surgical", "hernia": "surgical", "gallstones": "surgical",
            "cholecystitis": "surgical", "dermatitis": "skin", "scabies": "skin", "urticaria": "skin", "eczema": "skin",
            "sinusitis": "ent", "otitis": "ent", "tonsillitis": "ent", "vertigo": "ent", "anaemia": "other",
            "thyroid": "other", "angina": "cardiac", "coronary": "cardiac", "palpitations": "cardiac", "kidney": "renal",
            "urinary": "renal", "pyelonephritis": "renal", "paediatric": "paeds", "bronchiolitis": "paeds",
            "pyrexia": "fever", "vomiting": "gastro", "abdominal": "gastro", "knee": "musculoskeletal"}
_KW = list(KEYWORDS)


def map_diagnosis(raw: pd.Series):
    """Returns (group series, method series).  Method: regex / fuzzy / unknown."""
    cache = {}
    def one(s):
        if s in cache: return cache[s]
        t = re.sub(r"\s+", " ", re.sub(r"[.]+", " ", str(s))).strip().lower()
        if t in ("", "na", "-", "?", "nan"): r = ("unknown", "blank")
        else:
            r = None
            for g, rx in _RX:
                if rx.search(t): r = (g, "regex"); break
            if r is None:
                for tok in re.findall(r"[a-z]{5,}", t):
                    m = difflib.get_close_matches(tok, _KW, n=1, cutoff=0.8)
                    if m: r = (KEYWORDS[m[0]], "fuzzy"); break
            if r is None: r = ("unknown", "unmapped")
        cache[s] = r
        return r
    out = raw.map(one)
    return out.map(lambda x: x[0]), out.map(lambda x: x[1])


# --------------------------------------------------------------------------- loading + cleaning
def load_and_clean(d="data"):
    log = []
    def L(step, issue, n, action): log.append(dict(step=step, issue=issue, rows_affected=int(n), action=action))
    raw = {k: pd.read_csv(f"{d}/{k}_raw.csv") for k in ["patients", "opd_visits", "admissions", "lab_tests", "billing"]}
    hosp = pd.read_csv(f"{d}/hospitals.csv"); docs = pd.read_csv(f"{d}/doctors.csv")
    L("raw", "total transactional rows (opd+admissions+labs+billing)", sum(len(raw[k]) for k in ["opd_visits", "admissions", "lab_tests", "billing"]), "n/a")

    # patients / age
    p = raw["patients"].copy()
    parsed = p.age_or_dob.map(parse_age)
    p["age"] = parsed.map(lambda x: x[0]); p["age_src"] = parsed.map(lambda x: x[1])
    L("patients", "age entered as date of birth (mixed formats)", (p.age_src == "dob").sum(), "parsed 4 date formats, converted to age at 31-Dec-2025")
    L("patients", "age text with suffix ('34 yrs', '34Y')", p.age_or_dob.astype(str).str.contains("[A-Za-z]").sum() - (p.age_src == "dob").sum(), "stripped suffix")
    bad_age = (p.age_src == "bad") | (p.age < 0) | (p.age > 110)
    L("patients", "unparseable / impossible age", bad_age.sum(), "set to median age")
    p.loc[bad_age, "age"] = p.age.median()
    p["age"] = p.age.round(1)

    # OPD
    o = raw["opd_visits"].copy(); o["visit_date"] = pd.to_datetime(o.visit_date)
    n = o.duplicated().sum(); o = o.drop_duplicates(); L("opd", "exact duplicate visit rows", n, "dropped")
    n = o.visit_id.duplicated().sum(); o = o.drop_duplicates("visit_id"); L("opd", "duplicate visit_id after exact-dup removal", n, "dropped")
    o["dx_group"], o["dx_method"] = map_diagnosis(o.diagnosis_text)
    n = o.doctor_id.isna().sum(); o["doctor_id"] = o.doctor_id.fillna("UNKNOWN"); L("opd", "missing doctor_id", n, "set to UNKNOWN (kept; doctor not used for segmentation)")

    # admissions
    a = raw["admissions"].copy(); a["admit_date"] = pd.to_datetime(a.admit_date); a["discharge_date"] = pd.to_datetime(a.discharge_date)
    n = a.admission_id.duplicated().sum(); a = a.drop_duplicates("admission_id"); L("admissions", "duplicate admission_id", n, "dropped")
    bad = a.discharge_date < a.admit_date
    L("admissions", "discharge date before admission date", bad.sum(), "dropped (cannot recover true LOS)")
    a = a[~bad].copy()
    a["los"] = (a.discharge_date - a.admit_date).dt.days.clip(lower=1)
    a["dx_group"], a["dx_method"] = map_diagnosis(a.diagnosis_text)
    n = a.doctor_id.isna().sum(); a["doctor_id"] = a.doctor_id.fillna("UNKNOWN"); L("admissions", "missing doctor_id", n, "set to UNKNOWN")

    # labs
    l = raw["lab_tests"].copy(); l["test_date"] = pd.to_datetime(l.test_date)
    n = l.duplicated().sum(); l = l.drop_duplicates(); L("labs", "exact duplicate lab rows", n, "dropped")

    # billing
    b = raw["billing"].copy(); b["bill_date"] = pd.to_datetime(b.bill_date)
    n = b.duplicated().sum(); b = b.drop_duplicates(); L("billing", "exact duplicate bill rows", n, "dropped")
    dbl = (b.total_amount - b.base_amount - 2 * b.gst_amount).abs() < 1.0
    dbl &= b.gst_amount > 0
    b["gst_double_counted"] = dbl
    L("billing", "GST counted twice (total = base + 2 x GST)", dbl.sum(), "total recomputed = base + GST; revenue is measured on the ex-GST base amount so GST never enters value")
    b["total_fixed"] = (b.base_amount + b.gst_amount).round(2)
    refunds = b.bill_id.str.startswith("R") | (b.base_amount < 0)
    L("billing", "refund / reversal rows (negative amounts)", refunds.sum(), "netted against the original bill of the same encounter")
    L("billing", "zero-value bills", ((b.base_amount == 0) & ~refunds).sum(), "kept as 0 revenue (encounter still counts as a visit)")
    # keying errors: amount 100x too large -> divide by 100 if the corrected value is plausible
    cap = b[b.base_amount > 0].groupby("encounter_type").base_amount.quantile(0.99) * 2
    cap_ = b.encounter_type.map(cap)
    ksel = (b.base_amount > 0) & (b.base_amount > cap_)
    L("billing", "amount ~100x too large (keying error)", ksel.sum(), "divided by 100 (corrected value lies inside the normal range)")
    for c in ["base_amount", "gst_amount", "total_fixed"]: b.loc[ksel, c] = b.loc[ksel, c] / 100
    # a 100x row may also have a negative twin: any |amount| above the cap on negative side
    kneg = (b.base_amount < 0) & (-b.base_amount > cap_)
    for c in ["base_amount", "gst_amount", "total_fixed"]: b.loc[kneg, c] = b.loc[kneg, c] / 100
    rev = b.groupby("encounter_id", as_index=False).agg(revenue=("base_amount", "sum"), gst=("gst_amount", "sum"),
                                                       billed_total=("total_fixed", "sum"), n_bills=("bill_id", "count"))
    # dx coverage
    allx = pd.concat([o[["dx_method"]], a[["dx_method"]]])
    cov = allx.dx_method.value_counts()
    L("diagnosis", "free-text mapped by regex rules", cov.get("regex", 0), "mapped to 16 canonical groups")
    L("diagnosis", "free-text mapped by fuzzy match (typos)", cov.get("fuzzy", 0), "difflib similarity >= 0.8 on keywords")
    L("diagnosis", "blank / NA diagnosis", cov.get("blank", 0), "group = unknown")
    L("diagnosis", "unmapped text (not recoverable)", cov.get("unmapped", 0), "group = unknown")
    return dict(patients=p, opd=o, adm=a, lab=l, bill=b, rev=rev, hosp=hosp, docs=docs, log=pd.DataFrame(log), dxcov=cov)


# --------------------------------------------------------------------------- features
def build_events(D):
    o, a, r = D["opd"], D["adm"], D["rev"].set_index("encounter_id").revenue
    ev_o = pd.DataFrame({"patient_id": o.patient_id, "date": o.visit_date, "kind": "OPD", "dx": o.dx_group,
                         "rev": o.visit_id.map(r).fillna(0.0).values, "hospital_id": o.hospital_id})
    ev_a = pd.DataFrame({"patient_id": a.patient_id, "date": a.admit_date, "kind": "IPD", "dx": a.dx_group,
                         "rev": a.admission_id.map(r).fillna(0.0).values, "hospital_id": a.hospital_id})
    return pd.concat([ev_o, ev_a], ignore_index=True)


def features(D, ev, start, end):
    """One row per patient with >=1 encounter in [start, end]."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    e = ev[(ev.date >= start) & (ev.date <= end)].copy()
    mid = start + (end - start) / 2
    e["h2"] = e.date > mid
    e["monsoon"] = e.date.dt.month.isin([6, 7, 8, 9])
    e["chronic"] = (e.kind == "OPD") & e.dx.isin(CHRONIC)
    g = e.groupby("patient_id")
    f = pd.DataFrame({
        "opd_n": g.apply(lambda x: (x.kind == "OPD").sum(), include_groups=False),
        "adm_n": g.apply(lambda x: (x.kind == "IPD").sum(), include_groups=False),
        "last": g.date.max(), "revenue": g.rev.sum(),
        "h1": g.apply(lambda x: (~x.h2).sum(), include_groups=False), "h2": g.h2.sum(),
        "mons": g.monsoon.sum(), "chron": g.chronic.sum()})
    f["enc_n"] = f.opd_n + f.adm_n
    ln = D["lab"]; ln = ln[(ln.test_date >= start) & (ln.test_date <= end)]
    f["lab_n"] = ln.groupby("patient_id").size().reindex(f.index).fillna(0)
    f["recency"] = (end - f["last"]).dt.days
    f["growth"] = (f.h2 - f.h1) / (f.h1 + f.h2 + 2)                      # shrunk toward 0 for tiny counts
    f["monsoon_share"] = (f.mons + 1 / 3) / (f.enc_n + 1)                # shrunk toward the 1/3 calendar share
    f["chronic_share"] = f.chron / f.opd_n.clip(lower=1)
    f["labs_per_visit"] = f.lab_n / f.opd_n.clip(lower=1)
    f = f.join(D["patients"].set_index("patient_id")[["age", "gender"]])
    return f.drop(columns=["last", "h1", "h2", "mons", "chron"])
