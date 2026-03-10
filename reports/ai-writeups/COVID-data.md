### 3.1 National COVID‑19 Registry (case notifications + tests + hospitalisations)

**What it is**

A Ministry of Health registry that:

- Captures **all confirmed SARS‑CoV‑2 infections** (PCR and RAT) across hospitals, primary care, and Public Health Preparedness Clinics.
- Includes all **COVID‑19‑related hospitalisations and severe cases**, plus dates of test and hospital admission.

**Used in**

- **Frailty & boosting:** _Real‑World Effectiveness of Boosting Against Omicron Hospitalization in Older Adults, Stratified by Frailty_ (Vaccines 2025; 10.3390/vaccines13060565). They explicitly say they used national records of SARS‑CoV‑2 infections, hospitalisations, and vaccinations, with outcomes ascertained via the national COVID‑19 registry and mandatory notifications.
- **Cardiac outcomes vs RSV/flu:** _Cardiac Events in Adults Hospitalized for RSV vs COVID‑19 or Influenza_ (JAMA Netw Open 2025; 10.1001/jamanetworkopen.2025.11764) – COVID hospitalisations and all COVID test results are taken from this registry.
- **Long‑term sequelae after Omicron hospitalisation** and **RSV vs Omicron vs influenza PASC** (10.1016/j.ijid.2025.107946; 10.1016/j.cmi.2025.04.022) both rely on national COVID hospitalisation data as the index events.
- **Bivalent booster & PASC:** _Bivalent Boosters and Risk of Postacute Sequelae Following Vaccine‑Breakthrough SARS‑CoV‑2 Omicron Infection_ (Clin Infect Dis 2025; 10.1093/cid/ciae598) builds a population‑based cohort of adult Omicron infections from national databases.

**Key variables you can expect**

- Person identifier, age, sex, nationality
- Date and type of positive test (PCR/RAT)
- Classification of hospitalisation as COVID‑related
- Indicator for reinfection (spaced from prior episodes)

**Limitations**

- Misses infections that are never tested or never reported (especially late Omicron era).
- Limited clinical detail (no vitals, troponins, etc.).
- Variant often assigned by **time window using genomic surveillance**, not per‑case WGS.

**Value for CADENCE**

- Backbone for defining **infection timelines and variant eras** at the patient level.
- Key exposure for Q2/Q3 (post‑COVID MI risk and risk score) and time‑varying covariate for Q1 (era‑specific acute outcomes).

---

### 3.2 National Immunisation Registry (COVID + influenza)

**What it is**

National registry with:

- All COVID‑19 vaccine doses (product, dose number, date, bivalent vs monovalent).
- Seasonal influenza vaccines (and possibly others).

**Used in**

- **Cardiac events RSV vs COVID vs flu:** vaccination status for both COVID‑19 and influenza was obtained by linking to the National Immunisation Registry.
- **Frailty & boosting:** they use national records of COVID vaccinations, including booster roll‑out and bivalent introduction, to define vaccination status and time since last dose.
- **Bivalent booster & PASC:** exposure definition is “last booster ancestral vs bivalent mRNA” – clearly using product‑level data from this registry.

**Limitations**

- Tiny fraction vaccinated overseas may have incomplete or late‑entered records.
- No immunogenicity measures (those come from separate serology cohorts).

**Value for CADENCE**

- Exactly what you need to model **time‑since‑vaccine, product type, and hybrid immunity** in your MI/HF outcomes and risk score models.

---

### 3.3 Mediclaims – National Healthcare Claims Database

**What it is**

National database of all reimbursed encounters (public + private), with:

- ICD‑10 diagnoses, procedures, admission/discharge dates.
- Used to derive comorbidity indices, frailty, and multi‑system sequelae.

**Used in**

- **Cardiac events RSV vs COVID vs flu:** RSV and influenza hospitalisations are _identified from RSV- and influenza-specific ICD‑10 codes recorded in Mediclaims_; cardiovascular events during the index stay are also ICD‑10‑based in Mediclaims, and Mediclaims provides comorbidity and immunocompromised status.
- **Frailty & boosting:** Hospital Frailty Risk Score (HFRS) is computed using ICD‑10 codes from the national healthcare claims database over the previous 4 years; Mediclaims also supplies comorbidities and utilisation history.
- **PASC-style studies** (RSV vs COVID vs flu long-term sequelae; Omicron vs influenza sequelae; Remdesivir & sequelae; Bivalent booster & PASC) all use Mediclaims to identify incident diagnoses across organ systems 31–365 days post‑infection or hospitalisation.

**Limitations**

- Classical claims limitations: diagnosis misclassification, especially for mild conditions / symptoms.
- Doesn’t hold granular clinical data (lab values, imaging reports).

**Value for CADENCE**

- Fills in **comorbidities, non‑cardiac outcomes, and frailty** for all CADENCE patients, including those treated outside your cardiac centres.
- Gives you a ready‑made, nationally consistent way to measure **long-term multi‑organ outcomes** alongside CADENCE’s gold‑standard MI/HF data.

---

### 3.4 National Diabetes Database (NDD)

**What it is**

A disease registry capturing diagnosed diabetes cases, used to identify **incident T2D**.

**Used in**

- _Risk of New‑Onset Type 2 Diabetes Among Vaccinated Adults After Omicron or Delta Variant SARS‑CoV‑2 Infection_ (JAMA Netw Open 2025; 10.1001/jamanetworkopen.2025.2959) — they build a national cohort of test‑positive and test‑negative adults, with SARS‑CoV‑2 status from national COVID databases and new‑onset T2D from a national diabetes registry.

**Limitations**

- Relies on engagement with the health‑care system and registry enrolment; early or diet‑controlled T2D may be under‑captured.

**Value for CADENCE**

- Lets you treat **post‑COVID T2D** as a mediator or competing risk for MI/HF in your longitudinal models.
- Also a proof‑of‑concept that similar disease registries might be linkable (CKD, cancer, etc.).

---

### 3.5 Multi‑organ long‑COVID / PASC outcome definitions

**What it is**

A curated library of ICD‑10 code sets for multi‑system post‑acute sequelae, plus analysis pipelines for:

- New‑incident diagnoses 31–365 days after infection or discharge.
- Organ system groupings: cardiovascular, neurological, respiratory, autoimmune, psychiatric, etc.

**Used in**

- **Bivalent booster & PASC** (CID 10.1093/cid/ciae598).
- **Remdesivir & long-term sequelae after COVID hospitalisation** (CMI 10.1016/j.cmi.2025.06.016).
- **Long-term multisystemic sequelae: Omicron hospitalisation vs influenza** (IJID 10.1016/j.ijid.2025.107946) and **RSV vs Omicron vs influenza** (CMI 10.1016/j.cmi.2025.04.022).

**Value for CADENCE**

- You inherit a **tested set of outcome definitions** for cardiovascular and non‑cardiovascular PASC.
- You can compare:

  - **Claims‑based CV outcomes** vs
  - **Registry‑verified MI/HF outcomes** from CADENCE, which is methodologically powerful.

---

### 3.6 HF / IHD cohorts and heart‑failure outcomes

**What it is**

Heart‑failure and ischaemic heart disease are defined from national claims and hospital data, and linked to COVID/Vaccination data.

**Used in**

- _Omicron SARS‑CoV‑2 outcomes in vaccinated individuals with heart failure and ischaemic heart disease_ (Ann Acad Med Singap 2025; 10.47102/annals-acadmedsg.202535) compares COVID hospitalisation and severe disease risk in HF/IHD vs matched non‑HF/IHD during Omicron, using registries for infection and vaccination plus claims/registries to define HF/IHD.
- _Risk of new‑onset heart failure and heart failure exacerbations following COVID‑19, influenza or respiratory-syncytial-virus hospitalisation_ (Eur J Prev Cardiol 2025; 10.1093/eurjpc/zwaf714) uses national RVI hospitalisations and then follows patients for 180 days to detect new HF diagnoses and HF readmissions via Mediclaims.

**Limitations**

- HF defined from ICD‑10 + readmission patterns; not as granular as echo‑based clinical HF.

**Value for CADENCE**

- You can _validate and refine_ these HF concepts with your richer cardiology data.
- You can run **very nuanced post‑COVID HF and MI analyses** restricted to your MI population, while benchmarking against PREPARE’s broader national HF findings.

---

### 3.7 Pregnancy and obstetric cohort

**What it is**

Linked data on pregnant women with mild Omicron infections, including:

- Infection status (from national COVID registry)
- Vaccination status
- Pregnancy/obstetric data
- Long‑COVID‑like multisystem outcomes.

**Used in**

- _Long COVID‑19 in pregnancy: increased risk but modest incidence following mild Omicron infection in a boosted obstetric cohort during endemicity_ (AJOG 2025; 10.1016/j.ajog.2025.03.004).

**Value for CADENCE**

- More niche, but it shows PREPARE can work with **specialised registries (obstetric/perinatal)** and apply the same PASC framework — reassuring for doing analogous work in HF/MI.

---

### 3.8 Paediatric cohort

**What it is**

Child‑level dataset with linked infection (dengue and SARS‑CoV‑2), hospitalisations and post‑acute diagnoses.

**Used in**

- _Long‑term Sequelae Following Dengue Infection vs SARS‑CoV‑2 Infection in a Pediatric Population_ (Open Forum Infect Dis 2025; 10.1093/ofid/ofaf134).
- Paediatric arm of the RSV vs Omicron vs influenza long‑term sequelae study (CMI 10.1016/j.cmi.2025.04.022).

**Value for CADENCE**

- Less central (CADENCE is adult MI‑focused), but it confirms that the same linkage framework works across **all ages**.

---

### 3.9 Frailty datasets (HFRS + CFS)

**What it is**

- **Hospital Frailty Risk Score (HFRS)** from 4 years of Mediclaims ICD‑10 history.
- **Clinical Frailty Scale (CFS)** from a national 2019 community survey of adults ≥60.

**Used in**

- Vaccines 2025 frailty/boosting paper stratifies older adults into low, intermediate and high frailty and estimates booster effectiveness within each stratum.

**Value for CADENCE**

- Lets you explore whether **post‑COVID MI risk and in‑hospital mortality** differ across frailty levels, which is _exactly_ the kind of nuance Q3 can exploit.

---

### 3.10 Digital contact‑tracing (DCT) dataset

**What it is**

National logs of Bluetooth/proximity‑based contacts between index cases and exposed individuals.

**Used in**

- _Utilization of population-wide digital contact tracing to estimate real-world vaccine effectiveness in a pandemic setting_ (CMI 2025; 10.1016/j.cmi.2025.06.014). They assemble a huge cohort of older adults and a nested dataset of case–contact pairs from DCT logs to estimate vaccine effectiveness “given exposure” against Delta and Omicron.

**Value for CADENCE**

- Not essential to first‑pass MI questions, but potentially useful for:

  - Understanding **exposure intensity** among CADENCE patients.
  - Reducing bias in any vaccine‑effectiveness sub‑analyses you might eventually do with MI/HF outcomes.

---

### 3.11 Serology and antibody‑kinetics cohort

**What it is**

A smaller prospective cohort of booster recipients with:

- Serial anti‑S IgG/IgA levels against wild‑type and Omicron BA.1 at multiple timepoints (0, 28, 180, 360 days).

**Used in**

- _Modeling Antibody Kinetics Post‑mRNA Booster Vaccination and Protection Durations Against SARS-CoV-2 Infection_ (J Med Virol 2025; 10.1002/jmv.70521).

**Value for CADENCE**

- Mostly conceptual: helps you justify **time‑since‑booster functional forms** in your statistical models (e.g., assuming hazard of infection and sequelae increases as antibody titres wane).

---

### 3.12 Within‑host viral dynamics cohort

**What it is**

A high‑resolution dataset of serial RT‑PCR cycle‑threshold (Ct) values and clinical metadata from Delta cases.

**Used in**

- _Defining the Critical Requisites for Accurate Simulation of SARS‑CoV‑2 Viral Dynamics: Patient Characteristics and Data Collection Protocol_ (J Med Virol 2025; 10.1002/jmv.70174), which describes how such data are collected and used in simulations.
- _Age‑ and vaccination status‑dependent isolation guidelines based on simulation of SARS‑CoV‑2 Delta cases in Singapore_ (Communications Medicine 2025; 10.1038/s43856-025-00797-8) calibrates isolation recommendations to these viral‑load trajectories.

**Value for CADENCE**

- Not directly linkable at scale, but conceptually helps you reason about **acute thrombotic risk windows** (e.g., peak risk days relative to symptom onset / highest viral loads).

---

### 3.13 Human mobility, NPIs and hospital infection-control data

**What they are**

- Aggregated **mobility matrices** between regions and over time; used to improve Rt estimation and epidemic response.
- **Masking and hospital infection‑control policy timelines**, coupled with surveillance of health‑care–associated non‑SARS-CoV‑2 respiratory viruses.
- **Vaccination‑differentiated safe‑management measures (VDS)** and their timing, used when analysing vaccine uptake among hesitant individuals.

**Value for CADENCE**

- Feed into **era‑level or time‑varying covariates** in your Q1 analyses:

  - Was the MI occurring during a period of extreme system strain?
  - Was hospital masking universal or relaxed?
  - Were unvaccinated people subject to access restrictions?

---

## 4. How this lines up with your project

Putting it succinctly:

- PREPARE already has **exactly the core national datasets** you need:

  - COVID case & hospitalisation registry
  - Vaccination registry
  - National claims (Mediclaims)
  - Disease registries (e.g. diabetes)
  - Long‑COVID / PASC outcome definitions.

- On top of that, they have **specialty cohorts** (HF/IHD, pregnancy, paediatrics), **frailty data**, digital contact tracing, and modelling‑oriented viral/serology datasets that can enrich more advanced analyses.

- Your CADENCE slides already assume this structure: PREPARE supplying infection/variant/vaccination/severity data via TRUST, and CADENCE supplying granular MI/HF clinical data, with the three question pillars (acute outcomes, post‑COVID incidence, and risk scoring) built on top.

If you want, next step I can help you turn this into:

- A 1–2 page **formal data‑sharing specification** for the PREPARE side, or
- A **table** mapping each CADENCE question (Q1–Q3) to precise variables from each PREPARE dataset (COVID registry, NIR, Mediclaims, etc.), which you can drop into your protocol.

## 1. Core person-level COVID data

### Infection & testing

| Wishlist item                                             | Status                               | Notes / source                                                                                                                                                                                           |
| --------------------------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Positive test date(s) for SARS-CoV-2                      | ✅ **Have (PREPARE)**                | From national COVID registry used across PREPARE papers (case notifications + PCR/RAT results).                                                                                                          |
| Test type (PCR vs RAT)                                    | ✅ **Have (PREPARE)**                | Explicitly recorded in case-notification systems; used when distinguishing PCR-confirmed vs other infections.                                                                                            |
| First infection vs reinfection flag                       | ✅ **Have (PREPARE)**                | Reinfections usually defined as positive test ≥90 days after prior one in national datasets.                                                                                                             |
| Testing _setting_ (hospital vs primary care vs community) | ⚠️ **Partial / unclear**             | Some papers stratify by hospitalisation but not by testing venue; registry likely has facility codes, but this isn’t clearly described — treat as “available but needs confirmation.”                    |
| Symptom onset date                                        | ❓ **Mostly don’t have / uncertain** | Not consistently reported in registry-based national studies; usually they anchor on test date or admission date. You should assume this is missing for most people and only available in small cohorts. |

### Variant / lineage

| Wishlist item                                                                | Status                 | Notes                                                                                                                                                                   |
| ---------------------------------------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Variant **era** tag per infection (pre-Delta, Delta, Omicron subwaves, etc.) | ✅ **Have (PREPARE)**  | All the VE/PASC papers classify cases by wave/era using national genomic surveillance + calendar time.                                                                  |
| Individual-level variant (WGS / SGTF)                                        | ⚠️ **Partial at best** | They acknowledge limited sequencing and even highlight “strain misclassification” in your slides under sensitivity analysis. So: only a subset will have exact lineage. |

### Vaccination history

| Wishlist item                                     | Status                | Notes                                                                                            |
| ------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------ |
| All COVID vaccine doses, dates, and product       | ✅ **Have (PREPARE)** | From National Immunisation Registry, widely used in frailty-boosting, T2D, HF, and PASC studies. |
| Flag for bivalent vs monovalent formulation       | ✅ **Have (PREPARE)** | Required for the bivalent booster vs ancestral booster PASC paper.                               |
| Time since last dose at infection / MI / HF event | ✅ **Derivable**      | Once you link NIR to CADENCE via TRUST, this is trivial to compute.                              |

### Acute COVID severity & treatment

| Wishlist item                                                           | Status                                      | Notes                                                                                                                                                                                        |
| ----------------------------------------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Indicator of COVID-related hospitalisation                              | ✅ **Have (PREPARE)**                       | National COVID registry + Mediclaims used to define COVID hospitalisations across multiple papers.                                                                                           |
| ICU / HDU admission, length of stay                                     | ✅ **Likely have (PREPARE)**                | Used in several COVID-severity and outcomes papers; at minimum, ICU status is available.                                                                                                     |
| Respiratory support level (O₂, NIV, IMV)                                | ⚠️ **Probably partial**                     | Clearly available in hospital EHR for COVID cohorts; not always described in national-scale registry papers. Assume this exists but may be more work to standardise across hospitals.        |
| In-hospital COVID therapies (remdesivir, steroids, IL-6/JAK inhibitors) | ✅/⚠️ **Remdesivir: have; others: unclear** | The remdesivir-long-term-sequelae paper uses detailed treatment data; it’s almost certainly from hospital medication records. Availability of _all_ drugs at population level is less clear. |

---

## 2. Long-term outcomes, comorbidities & frailty

### Long-term post-COVID sequelae

| Wishlist item                                                                 | Status                          | Notes                                                                                                                                                              |
| ----------------------------------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Multi-organ new-incident diagnoses 31–365 days post infection/hospitalisation | ✅ **Have (PREPARE)**           | PREPARE has a full PASC outcome library (CV, neuro, respiratory, autoimmune, psych) based on ICD-10 in Mediclaims, used in several Omicron/RSV/dengue PASC papers. |
| Cardiovascular sequelae (MI, HF, arrhythmias, stroke, VTE, myocarditis)       | ✅ **Have (PREPARE + CADENCE)** | Claims-based definitions already exist; CADENCE adds richer clinical confirmation for MI/HF and procedures.                                                        |
| New-onset type 2 diabetes                                                     | ✅ **Have (PREPARE)**           | From National Diabetes Database + claims in the T2D paper.                                                                                                         |

### Comorbidities & frailty

| Wishlist item                                                                    | Status                      | Notes                                                                                                                                     |
| -------------------------------------------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline comorbidities (Charlson, specific CV risk factors)                      | ✅ **Have (PREPARE)**       | From Mediclaims ICD-10 history in essentially all large cohort papers.                                                                    |
| Hospital Frailty Risk Score (HFRS)                                               | ✅ **Have (PREPARE)**       | Explicitly calculated from 4-year ICD-10 data in the frailty/boosting paper.                                                              |
| Clinical Frailty Scale (CFS)                                                     | ⚠️ **Subset only**          | Comes from a 2019 community survey of ≥60-year-olds; only available for those surveyed.                                                   |
| Detailed symptom burden (fatigue, cognitive issues, etc.) beyond coded diagnoses | ❓ **Generally don’t have** | National claims & registries only capture diagnoses, not patient-reported outcomes; may exist in small research cohorts but not at scale. |

---

## 3. Exposure context & system-level covariates

### Digital contact tracing & mobility

| Wishlist item                                                             | Status                            | Notes                                                                                                                      |
| ------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Digital contact-tracing case–contact pairs                                | ✅ **Have (PREPARE)**             | Used to estimate vaccine effectiveness given exposure in the DCT CMI paper; data are large but require special governance. |
| Individual “exposure intensity” metrics (number of logged contacts, etc.) | ⚠️ **Derivable but not standard** | Raw DCT logs allow this; PREPARE hasn’t published that specific metric yet, so it would be new work.                       |
| Human mobility matrices (area-to-area flows)                              | ✅ **Have (PREPARE)**             | Used in the Rt/mobility modelling framework paper.                                                                         |
| Masking and hospital infection-control policy timelines                   | ✅ **Have (PREPARE)**             | Used in the AJIC time-series analysis of non-SARS respiratory viruses.                                                     |

---

## 4. Sub-cohort flags / specialised registries

| Wishlist item                                  | Status                          | Notes                                                                                                                 |
| ---------------------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Pregnancy & obstetric status                   | ✅ **Have (PREPARE)**           | Linked obstetric + infection data used in the pregnancy Long-COVID paper.                                             |
| Paediatric cohort indicator and paeds outcomes | ✅ **Have (PREPARE)**           | Used in the paediatric dengue vs COVID sequelae paper and RSV/Omicron/flu paeds analyses.                             |
| Known HF / IHD at baseline                     | ✅ **Have (PREPARE + CADENCE)** | For national work, HF/IHD is built from Mediclaims; for CADENCE patients you also have cardiology-grade confirmation. |

---

## 5. What CADENCE / SingCLOUD has vs lacks (non-COVID side)

From your slides:

- SingCLOUD **already has**: ICD-10 event diagnosis (MI), PCI/CABG, meds, cardiac labs (troponin, lipids, etc.), echo/angiography findings, chest X-rays, vitals, etc.
- For the Q3 risk-score slide, you explicitly mark **symptomatology (e.g. chest pain) and NYHA class as MISSING**; troponin/CRP, creatinine/urea, and vitals are available.

On the **COVID side**, CADENCE/SingCLOUD by itself:

- ❌ does **not** hold infection dates, variant, vaccination, or COVID-severity data — these are entirely on PREPARE/MOH and must come in via TRUST linkage.

---

## 6. TL;DR

- **Already have (somewhere in the ecosystem):**

  - Infection dates, reinfection status, variant _eras_, vaccination and booster history, COVID hospitalisations, basic severity, remdesivir exposure, comorbidities, frailty (HFRS), multi-organ long-COVID outcomes, T2D incidence, HF/IHD flags, pregnancy/paeds flags, DCT exposures (raw), mobility, NPI timelines.

- **Clearly missing or only partial:**

  - Symptom onset date, full individual-level variant sequencing, detailed symptom-level long-COVID data, complete oxygen/ventilation details across _all_ hospitals, full CFS coverage, and any COVID data inside SingCLOUD prior to TRUST linkage.

If you like, next step I can turn this into a 1-page “data inventory” sheet you can literally paste into a protocol or send to PREPARE as “here is what we think exists and what we still need clarified.”

---

Notes:

- Consider using SMIR/NRDO Dataset first to collate Cardio outcomes

## SYMPTOMS DATA - FUTILE/BIAS EFFORT

- KB found that it's impossible for them to tease apart pure COVID presentations versus COVID+other presentations. (For example, SOB might be due to asthma, but also due to COVID.) This was during a smaller study looking at resources used on COVID versus other conditions.

- We don't have access to the information that would enable us to understand which patients are pure COVID presentations and which are COVID+other presentations. Don't try to tease them apart.

- Hence there will always be some bias when it comes to outcome classification, when we are looking at some of the symptoms that may overlapping with other conditions.

## Outcomes

- Table 1s should be updated by February 2026 of all the papers should be ready
- KB wants us to prepare each of the different projects like a manuscript, meaning all of the things we are intending to present, should be done in a manuscript format.
- They want our findings in a table format, especially in for formal meetings, manuscript-level

- Need to showcase selection flowchart, who gets selected, who doesnt get selected, and who gets excluded, that way they can see if there might be something wrong with the numbers when they see the different segregated cohorts.

- Start doing some of these standard models looking at the different outcomes

- KB's stance is, as analysts, there can be a lot of back and forth with the clinicians, but at some point we should be putting our foot down and saying, "this is the data we have, this is the analysis we have, start writing." Alvin agrees with this approach if the leads are junior, but if the leads are senior, then they might have a different kind of style that they want to follow.

- ! Show flowchart, show Table 1, and then show the types of outcomes we are projecting to see if there are any issues with the numbers.

- KB wants to bring in other analysts to the January meeting, but February is when we should have almost all of the findings ready for publication.

- Ad-hoc bias analysis - for example, during Omicron where surveillance wasn't that strict, we might have to do some ad-hoc bias analysis to show that the findings are still valid.
- Ad-hoc: In later years, when we are not sure if someone has COVID or not, using some pre-determined parameters, we can do a bias analysis to see if we can flush out
- Ask Rizwan to see if there was a clinical procedure change
- Do a 2x2 table for bias analysis
- When we are writing the manuscript, we will have to state that we think that the unexposed cohort will be greater impacted in risk by X amount, during this period of time when surveillance was weaker.
- "Right now, what we've reported is that we see an increased rate of this variant versus the other variant at this point. But what is the degree of assumed misclassification must we have before we are losing the association significance between the two factors?

## Next steps:

- Find codes for NSTEMI in mediclaims, and then start working on getting some outcomes done for Table 1 in Question 1.
- Define Variant Eras in COVID dataset from individual level.
-

- Look at VHD dataset again (COVID) and see if there are low hanging fruits for papers

- GRACE score may be difficult, we missing symptoms, functionality score, etc.

- Do question 1, salvage question 2, pretend like question 3 is dead until new data comes

[text](<../PhD Ideas/segmented_infographic/section1.html>)
