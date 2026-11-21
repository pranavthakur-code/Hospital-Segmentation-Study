"""
Synthetic data generator for the Hyderabad Community Clinics segmentation study.

Everything here is SIMULATED. It reproduces the data problems described in the brief:
  * free-text diagnosis (spelling variants, abbreviations, typos, blanks)
  * age stored as an integer string in some rows and as a date of birth in others
  * billing totals where GST is counted twice (all of hospital H4, plus random rows elsewhere)
plus a few realistic extras (duplicates, refunds, 100x keying errors, bad dates, missing doctors).

Run:  python generate_data.py  [output_dir]
Seeded, so the output is reproducible.
"""
import sys, os
import numpy as np
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "data"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2025)

N_PAT = 24000
P1 = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
P2 = (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
END = P2[1]

# ----------------------------------------------------------------------------- reference data
hospitals = pd.DataFrame({
    "hospital_id": ["H1", "H2", "H3", "H4"],
    "hospital_name": ["Kukatpally Community Clinic", "Secunderabad Community Clinic",
                      "Gachibowli Community Clinic", "Charminar Community Clinic"],
    "beds": [40, 30, 45, 25]})
hosp_w = np.array([0.30, 0.22, 0.33, 0.15])

DEPTS = ["General Medicine", "Cardiology", "Pulmonology", "Nephrology", "Obstetrics & Gynaecology",
         "Orthopaedics", "General Surgery", "Dermatology", "Paediatrics", "ENT"]
doc_rows = []
for h in hospitals.hospital_id:
    for d in DEPTS:
        n = 3 if d == "General Medicine" else (2 if d in ("Orthopaedics", "Paediatrics", "Obstetrics & Gynaecology") else 1)
        for _ in range(n):
            doc_rows.append((h, d))
doctors = pd.DataFrame(doc_rows, columns=["hospital_id", "department"])
doctors.insert(0, "doctor_id", [f"DR{i:03d}" for i in range(1, len(doctors) + 1)])
doc_lookup = {(h, d): g.doctor_id.values for (h, d), g in doctors.groupby(["hospital_id", "department"])}

# diagnosis groups: department + free-text variants
GROUPS = {
    "diabetes": ("General Medicine", ["Type 2 diabetes mellitus", "T2DM", "DM type 2", "diabetes mellitus - II", "Diabetes",
                                      "sugar (diabetes) follow up", "DM-2 uncontrolled", "type ii dm"]),
    "hypertension": ("General Medicine", ["Hypertension", "HTN", "High BP", "essential hypertension", "htn on treatment",
                                          "raised blood pressure", "BP follow-up"]),
    "resp_infection": ("General Medicine", ["URTI", "upper respiratory tract infection", "URI", "cough and cold", "common cold",
                                            "acute bronchitis", "viral bronchitis", "sore throat / cold"]),
    "fever": ("General Medicine", ["viral fever", "fever", "Fever with chills", "pyrexia of unknown origin", "PUO", "dengue fever",
                                   "dengue", "typhoid fever", "enteric fever", "malaria", "fever/body pains"]),
    "gastro": ("General Medicine", ["gastroenteritis", "AGE", "acute gastro enteritis", "loose motions", "diarrhoea", "gastritis",
                                    "acidity / GERD", "abdominal pain", "vomiting & loose stools"]),
    "musculoskeletal": ("Orthopaedics", ["low back pain", "LBP", "knee pain", "osteoarthritis knee", "OA knee", "joint pain",
                                         "neck pain", "cervical spondylosis", "back ache"]),
    "cardiac": ("Cardiology", ["chest pain", "angina", "coronary artery disease", "CAD", "IHD", "heart failure", "CHF",
                               "palpitations", "MI"]),
    "asthma_copd": ("Pulmonology", ["asthma", "bronchial asthma", "COPD", "wheezing", "allergic rhinitis", "allergic bronchitis"]),
    "renal": ("Nephrology", ["CKD", "chronic kidney disease", "UTI", "urinary tract infection", "renal calculi", "kidney stone",
                             "pyelonephritis"]),
    "maternal": ("Obstetrics & Gynaecology", ["antenatal check up", "ANC visit", "pregnancy 28 wks", "normal delivery", "LSCS",
                                              "C-section", "post natal check", "PCOS", "menorrhagia"]),
    "injury": ("Orthopaedics", ["fracture", "# radius", "fracture tibia", "road traffic accident", "RTA - injuries", "sprain ankle",
                                "dislocation shoulder", "hip fracture"]),
    "surgical": ("General Surgery", ["appendicitis", "hernia", "inguinal hernia", "gallstones", "cholecystitis", "cholecystectomy",
                                     "hysterectomy", "piles / hemorrhoids", "fissure in ano"]),
    "skin": ("Dermatology", ["skin allergy", "dermatitis", "fungal infection", "scabies", "acne", "urticaria", "eczema"]),
    "paeds": ("Paediatrics", ["pediatric fever", "childhood diarrhea", "bronchiolitis", "tonsillitis", "vaccination", "well baby check"]),
    "ent": ("ENT", ["ear pain", "otitis media", "sinusitis", "tonsillitis", "vertigo"]),
    "other": ("General Medicine", ["general check up", "health check-up", "weakness", "anaemia", "vitamin D deficiency", "thyroid",
                                   "hypothyroidism"]),
}

LABS = {  # test: (variants, cost)
    "hba1c": (["HbA1c", "HBA1C", "glycosylated hb"], 450), "sugar": (["FBS", "PPBS", "blood sugar random", "RBS"], 120),
    "lipid": (["Lipid profile", "lipid panel"], 600), "cbc": (["CBC", "complete blood count", "cbc/esr"], 350),
    "lft": (["LFT", "liver function test"], 700), "kft": (["KFT", "renal function test", "creatinine/urea"], 650),
    "tsh": (["TSH", "thyroid profile"], 500), "dengue": (["Dengue NS1", "dengue IgM"], 900),
    "malaria": (["Malaria antigen", "MP smear"], 400), "urine": (["Urine routine", "urine R/E"], 150),
    "ecg": (["ECG"], 300), "xray": (["X-ray chest", "chest xray"], 500),
}

# archetype parameters  (opd mean/yr, adm rate/yr, LOS mean, 30d readmit prob, lab per visit, seasonal share, p_active, spend scale)
ARCH = {
    "A_healthy":   dict(opd=2.0, adm=0.02, los=2.5, readm=0.03, lab=0.25, seas=0.15, act=0.75, spend=1.0),
    "B_chronic":   dict(opd=10., adm=0.15, los=3.5, readm=0.08, lab=1.10, seas=0.10, act=0.95, spend=1.25),
    "C_episodic":  dict(opd=4.0, adm=0.35, los=2.8, readm=0.06, lab=0.50, seas=0.15, act=0.80, spend=1.0),
    "D_seasonal":  dict(opd=3.5, adm=0.20, los=3.0, readm=0.05, lab=0.80, seas=0.75, act=0.85, spend=0.9),
    "E_surgical":  dict(opd=4.0, adm=1.20, los=6.5, readm=0.09, lab=0.80, seas=0.10, act=0.80, spend=1.6),
    "F_frail":     dict(opd=8.0, adm=1.80, los=8.0, readm=0.19, lab=1.20, seas=0.15, act=0.95, spend=1.4),
    "G_maternal":  dict(opd=6.0, adm=0.60, los=3.0, readm=0.04, lab=0.90, seas=0.10, act=0.70, spend=1.2),
    "H_lapsing":   dict(opd=3.0, adm=0.10, los=3.0, readm=0.06, lab=0.30, seas=0.15, act=1.00, spend=0.9),
}
ARCH_P = {"A_healthy": .34, "B_chronic": .20, "C_episodic": .12, "D_seasonal": .10, "E_surgical": .07,
          "F_frail": .06, "G_maternal": .05, "H_lapsing": .06}
TRANS = {  # archetype in period 1 -> period 2
    "A_healthy": {"A_healthy": .88, "C_episodic": .05, "B_chronic": .04, "D_seasonal": .03},
    "B_chronic": {"B_chronic": .90, "F_frail": .04, "A_healthy": .03, "E_surgical": .03},
    "C_episodic": {"C_episodic": .85, "A_healthy": .10, "E_surgical": .05},
    "D_seasonal": {"D_seasonal": .88, "A_healthy": .12},
    "E_surgical": {"E_surgical": .75, "A_healthy": .15, "F_frail": .10},
    "F_frail": {"F_frail": .90, "E_surgical": .10},
    "G_maternal": {"G_maternal": .50, "A_healthy": .50},
    "H_lapsing": {"H_lapsing": 1.0},
}
# outpatient / inpatient diagnosis-group mixes per archetype
OPD_MIX = {
    "A_healthy": {"resp_infection": .22, "fever": .18, "gastro": .15, "skin": .10, "musculoskeletal": .10, "other": .15, "ent": .06, "paeds": .04},
    "B_chronic": {"diabetes": .38, "hypertension": .34, "cardiac": .08, "renal": .05, "asthma_copd": .05, "other": .10},
    "C_episodic": {"musculoskeletal": .20, "gastro": .18, "injury": .15, "resp_infection": .15, "skin": .10, "ent": .10, "fever": .12},
    "D_seasonal": {"fever": .35, "resp_infection": .25, "asthma_copd": .20, "gastro": .12, "skin": .08},
    "E_surgical": {"surgical": .30, "injury": .20, "musculoskeletal": .20, "cardiac": .10, "other": .20},
    "F_frail": {"hypertension": .22, "diabetes": .20, "cardiac": .20, "renal": .14, "asthma_copd": .10, "musculoskeletal": .08, "other": .06},
    "G_maternal": {"maternal": .80, "other": .15, "skin": .05},
    "H_lapsing": {"resp_infection": .25, "fever": .20, "gastro": .15, "musculoskeletal": .15, "other": .15, "skin": .10},
}
IPD_MIX = {
    "A_healthy": {"gastro": .30, "fever": .30, "injury": .15, "surgical": .15, "other": .10},
    "B_chronic": {"diabetes": .25, "cardiac": .25, "hypertension": .15, "renal": .20, "asthma_copd": .15},
    "C_episodic": {"injury": .30, "surgical": .25, "gastro": .20, "fever": .15, "other": .10},
    "D_seasonal": {"fever": .50, "asthma_copd": .25, "gastro": .15, "resp_infection": .10},
    "E_surgical": {"surgical": .50, "injury": .30, "cardiac": .10, "renal": .10},
    "F_frail": {"cardiac": .30, "renal": .20, "asthma_copd": .18, "diabetes": .12, "injury": .12, "hypertension": .08},
    "G_maternal": {"maternal": .95, "other": .05},
    "H_lapsing": {"gastro": .30, "fever": .30, "surgical": .20, "injury": .20},
}
WARD = ["General", "Semi-private", "Private", "ICU"]
WARD_RATE = dict(zip(WARD, [1500, 2800, 4500, 9000]))
WARD_P = {k: [.62, .25, .11, .02] for k in ARCH}
WARD_P["F_frail"] = [.50, .22, .12, .16]
WARD_P["E_surgical"] = [.40, .30, .24, .06]
WARD_P["B_chronic"] = [.60, .25, .10, .05]
CONSULT = {"General Medicine": 350, "Cardiology": 700, "Pulmonology": 600, "Nephrology": 650, "Obstetrics & Gynaecology": 550,
           "Orthopaedics": 550, "General Surgery": 600, "Dermatology": 500, "Paediatrics": 400, "ENT": 450}


def pick(d):
    k = list(d); p = np.array(list(d.values()), dtype=float); return k[rng.choice(len(k), p=p / p.sum())]


def sample_dates(n, start, seas, g):
    """n dates in the year: share `seas` in monsoon (Jun-Sep), rest by a beta curve tilted by growth g."""
    days = (END if start == P2[0] else P1[1]) - start
    span = days.days + 1
    is_seas = rng.random(n) < seas
    frac = rng.beta(np.exp(g), np.exp(-g), n)
    base = (frac * (span - 1)).astype(int)
    mon_start = (pd.Timestamp(start.year, 6, 1) - start).days
    mon = mon_start + rng.integers(0, 122, n)
    off = np.where(is_seas, mon, base)
    return start + pd.to_timedelta(np.clip(off, 0, span - 1), unit="D")


# ----------------------------------------------------------------------------- patients
arch_names = list(ARCH_P)
arch1 = rng.choice(arch_names, N_PAT, p=list(ARCH_P.values()))
arch2 = np.array([rng.choice(list(TRANS[a]), p=list(TRANS[a].values())) for a in arch1])
is_new = rng.random(N_PAT) < 0.10                       # registered during 2025
prim_h = rng.choice(4, N_PAT, p=hosp_w)
female = np.where(np.isin(arch1, ["G_maternal"]) | np.isin(arch2, ["G_maternal"]), True, rng.random(N_PAT) < 0.5)
age_mu = {"A_healthy": (35, 14), "B_chronic": (58, 10), "C_episodic": (34, 14), "D_seasonal": (30, 18), "E_surgical": (52, 12),
          "F_frail": (74, 7), "G_maternal": (28, 4), "H_lapsing": (38, 15)}
age = np.array([rng.normal(*age_mu[a]) for a in arch1]).clip(1, 95).round().astype(int)
patients = pd.DataFrame({"patient_id": [f"P{i:06d}" for i in range(1, N_PAT + 1)],
                         "gender": np.where(female, "F", "M"), "age_years": age})
patients["_arch1"], patients["_arch2"], patients["_new"], patients["_prim"] = arch1, arch2, is_new, prim_h


def age_text(a):
    """Age as integer text for ~80% of rows, a date of birth in mixed formats for the rest."""
    r = rng.random()
    if r < 0.80:
        return rng.choice([str(a), f"{a} yrs", f"{a}Y", f"{a} years"], p=[.7, .12, .1, .08])
    dob = pd.Timestamp("2024-07-01") - pd.Timedelta(days=int(a * 365.25 + rng.integers(0, 365)))
    fmt = rng.choice(["%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d-%m-%y"], p=[.45, .25, .2, .1])
    return dob.strftime(fmt)


patients["age_or_dob"] = [age_text(a) for a in age]

# ----------------------------------------------------------------------------- events
opd, adm, lab = [], [], []
oid = aid = lid = 0


def make_period(i, pid, arch, start, end, hosp_i, g, pat_new):
    global oid, aid, lid
    P = ARCH[arch]
    act = P["act"] if not (arch == "H_lapsing" and start == P2[0]) else 0.15
    if pat_new and start == P1[0]:
        return
    if (not pat_new) and rng.random() > act:
        return
    mean = P["opd"] * (0.35 if (arch == "H_lapsing" and start == P2[0]) else 1.0)
    n_opd = rng.negative_binomial(3, 3 / (3 + mean))
    n_adm = rng.poisson(P["adm"] * 0.68)
    if arch == "H_lapsing" and start == P1[0]:       # lapsing patients are only seen early in P1
        pass
    if n_opd + n_adm == 0:
        n_opd = 1
    seas = P["seas"]
    if arch == "H_lapsing" and start == P1[0]:
        d_opd = sample_dates(n_opd, start, seas, -1.0)       # visits cluster early in the year
        d_adm = sample_dates(n_adm, start, seas, -1.0)
    else:
        d_opd = sample_dates(n_opd, start, seas, g)
        d_adm = sample_dates(n_adm, start, seas, g)
    if pat_new:                                              # new patients start mid-year
        shift = pd.to_timedelta(rng.integers(0, 200), unit="D")
        d_opd, d_adm = d_opd + shift, d_adm + shift
    for d in d_opd:
        if d > END: continue
        grp = pick(OPD_MIX[arch]); dept = GROUPS[grp][0]
        h = hosp_i if rng.random() < 0.85 else rng.integers(0, 4)
        hid = hospitals.hospital_id[h]
        doc = rng.choice(doc_lookup[(hid, dept)])
        oid += 1
        txt = rng.choice(GROUPS[grp][1])
        opd.append((f"V{oid:07d}", pid, hid, d, dept, doc, txt, grp))
        k = rng.poisson(P["lab"] * 0.55)
        if k:
            names = ["hba1c", "sugar", "lipid", "kft"] if grp in ("diabetes", "hypertension", "renal") else \
                    ["cbc", "dengue", "malaria", "urine"] if grp == "fever" else ["cbc", "urine", "xray", "tsh", "lft", "ecg"]
            for _ in range(k):
                t = rng.choice(names); lid += 1
                lab.append((f"L{lid:07d}", pid, f"V{oid:07d}", hid, d + pd.Timedelta(days=int(rng.integers(0, 3))), rng.choice(LABS[t][0]), LABS[t][1], t))
    queue = list(d_adm)
    while queue:
        d = queue.pop(0)
        if d > END: continue
        grp = pick(IPD_MIX[arch]); dept = GROUPS[grp][0]
        h = hosp_i if rng.random() < 0.85 else rng.integers(0, 4)
        hid = hospitals.hospital_id[h]
        los = max(1, int(round(rng.gamma(2.2, P["los"] / 2.2))))
        ward = rng.choice(WARD, p=WARD_P[arch])
        aid += 1
        dis = d + pd.Timedelta(days=los)
        doc = rng.choice(doc_lookup[(hid, dept)])
        adm.append((f"A{aid:07d}", pid, hid, d, dis, dept, doc, rng.choice(GROUPS[grp][1]), ward, grp, arch))
        if rng.random() < P["readm"]:
            queue.append(dis + pd.Timedelta(days=int(rng.integers(3, 31))))


for i in range(N_PAT):
    pid = patients.patient_id.iat[i]
    g = rng.normal(0, 0.5)
    make_period(i, pid, arch1[i], P1[0], P1[1], prim_h[i], g, False)
    make_period(i, pid, arch2[i], P2[0], P2[1], prim_h[i], g if rng.random() < .5 else rng.normal(0, .5), bool(is_new[i]))

opd = pd.DataFrame(opd, columns=["visit_id", "patient_id", "hospital_id", "visit_date", "department", "doctor_id", "diagnosis_text", "_grp"])
adm = pd.DataFrame(adm, columns=["admission_id", "patient_id", "hospital_id", "admit_date", "discharge_date", "department", "doctor_id",
                                 "diagnosis_text", "ward_type", "_grp", "_arch"])
lab = pd.DataFrame(lab, columns=["lab_id", "patient_id", "visit_id", "hospital_id", "test_date", "test_name", "_cost", "_t"])

# ----------------------------------------------------------------------------- billing
arch_of = lambda pid_series, period: None
pat_arch = patients.set_index("patient_id")[["_arch1", "_arch2"]]
opd["_arch"] = np.where(opd.visit_date <= P1[1], pat_arch._arch1.reindex(opd.patient_id).values, pat_arch._arch2.reindex(opd.patient_id).values)
lab_cost = lab.groupby("visit_id")._cost.sum()
spend = lambda a: np.array([ARCH[x]["spend"] for x in a])
n = len(opd)
pharm = rng.lognormal(np.log(380), 0.7, n) * spend(opd._arch)
opd_base = np.array([CONSULT[d] for d in opd.department]) * rng.uniform(0.9, 1.15, n) + pharm + opd.visit_id.map(lab_cost).fillna(0).values
bill_opd = pd.DataFrame({"patient_id": opd.patient_id, "hospital_id": opd.hospital_id, "encounter_type": "OPD", "encounter_id": opd.visit_id,
                         "bill_date": opd.visit_date, "base_amount": opd_base.round(2), "gst_pct": 5})
na = len(adm)
days = (adm.discharge_date - adm.admit_date).dt.days.clip(lower=1).values
rate = adm.ward_type.map(WARD_RATE).values
surg_flag = adm._grp.isin(["surgical", "maternal", "injury"]).values | (rng.random(na) < 0.06)
proc = np.where(surg_flag, rng.lognormal(np.log(32000), 0.75, na), rng.lognormal(np.log(4500), 0.6, na))
adm_base = (days * rate + proc + days * rng.lognormal(np.log(900), 0.5, na)) * spend(adm._arch)
bill_ipd = pd.DataFrame({"patient_id": adm.patient_id, "hospital_id": adm.hospital_id, "encounter_type": "IPD", "encounter_id": adm.admission_id,
                         "bill_date": adm.discharge_date, "base_amount": adm_base.round(2), "gst_pct": 12})
bill = pd.concat([bill_opd, bill_ipd], ignore_index=True)
bill = bill[bill.bill_date <= END].reset_index(drop=True)
bill["gst_amount"] = (bill.base_amount * bill.gst_pct / 100).round(2)
bill["total_amount"] = (bill.base_amount + bill.gst_amount).round(2)
bill["payment_mode"] = rng.choice(["Cash", "UPI", "Card", "Insurance"], len(bill), p=[.35, .38, .12, .15])
bill.insert(0, "bill_id", [f"B{i:07d}" for i in range(1, len(bill) + 1)])
# --- inject billing problems
dbl = (bill.hospital_id == "H4") | (rng.random(len(bill)) < 0.02)               # GST counted twice
bill.loc[dbl, "total_amount"] = (bill.base_amount + 2 * bill.gst_amount)[dbl].round(2)
ix = rng.choice(len(bill), int(0.0005 * len(bill)), replace=False)               # 100x keying errors
bill.loc[ix, ["base_amount", "gst_amount", "total_amount"]] *= 100
ix = rng.choice(len(bill), int(0.003 * len(bill)), replace=False)                # zero-value bills
bill.loc[ix, ["base_amount", "gst_amount", "total_amount"]] = 0
refund = bill.sample(int(0.003 * len(bill)), random_state=3).copy()              # reversal rows
refund[["base_amount", "gst_amount", "total_amount"]] *= -1
refund["bill_id"] = [f"R{i:06d}" for i in range(len(refund))]
bill = pd.concat([bill, refund], ignore_index=True)

# ----------------------------------------------------------------------------- inject extras
def corrupt(s):
    r = rng.random()
    if r < 0.06 and len(s) > 4:                         # typo: drop or swap a character
        i = rng.integers(1, len(s) - 2)
        return s[:i] + s[i + 1:] if rng.random() < .5 else s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if r < 0.14: return s.upper()
    if r < 0.22: return s.lower()
    if r < 0.26: return "  " + s + " .."
    if r < 0.29: return rng.choice(["", "NA", "-", "?"])
    return s


for df in (opd, adm):
    df["diagnosis_text"] = [corrupt(s) for s in df.diagnosis_text]
opd.loc[rng.random(len(opd)) < 0.01, "doctor_id"] = None
adm.loc[rng.random(len(adm)) < 0.01, "doctor_id"] = None
bad = rng.random(len(adm)) < 0.001                       # discharge before admission (keying error)
adm.loc[bad, "discharge_date"] = adm.loc[bad, "admit_date"] - pd.Timedelta(days=2)
for name, df, key in (("opd", opd, "visit_id"), ("bill", bill, "bill_id"), ("lab", lab, "lab_id")):
    dup = df.sample(int(0.004 * len(df)), random_state=5)
    if name == "opd": opd = pd.concat([opd, dup], ignore_index=True)
    if name == "bill": bill = pd.concat([bill, dup], ignore_index=True)
    if name == "lab": lab = pd.concat([lab, dup], ignore_index=True)

# ----------------------------------------------------------------------------- write
truth = patients[["patient_id", "_arch1", "_arch2"]].rename(columns={"_arch1": "archetype_2024", "_arch2": "archetype_2025"})
patients_out = patients[["patient_id", "gender", "age_or_dob"]].copy()
reg = pd.Series(pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 365 * 5, N_PAT), unit="D"))
first_enc = pd.concat([opd[["patient_id", "visit_date"]].rename(columns={"visit_date": "d"}),
                       adm[["patient_id", "admit_date"]].rename(columns={"admit_date": "d"})]).groupby("patient_id").d.min()
newreg = patients.patient_id.map(first_enc) - pd.to_timedelta(rng.integers(0, 20, N_PAT), unit="D")
patients_out["registration_date"] = np.where(is_new, newreg, reg)
patients_out["registration_date"] = pd.to_datetime(patients_out["registration_date"]).dt.date

opd.drop(columns=["_grp", "_arch"]).to_csv(f"{OUT}/opd_visits_raw.csv", index=False)
adm.drop(columns=["_grp", "_arch"]).to_csv(f"{OUT}/admissions_raw.csv", index=False)
lab.drop(columns=["_cost", "_t"]).to_csv(f"{OUT}/lab_tests_raw.csv", index=False)
bill.to_csv(f"{OUT}/billing_raw.csv", index=False)
patients_out.to_csv(f"{OUT}/patients_raw.csv", index=False)
hospitals.to_csv(f"{OUT}/hospitals.csv", index=False)
doctors.to_csv(f"{OUT}/doctors.csv", index=False)
truth.to_csv(f"{OUT}/_simulation_truth_do_not_use_for_modelling.csv", index=False)
tot = len(opd) + len(adm) + len(lab) + len(bill)
print(f"opd {len(opd):,} | admissions {len(adm):,} | labs {len(lab):,} | billing {len(bill):,} | total transactional rows {tot:,}")
