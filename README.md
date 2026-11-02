# 🎓 LPU Placement Intelligence & Student Readiness Prediction Platform

**Version:** 2.2.0  
**Domain:** Supervised Machine Learning, Explainable AI (XAI), Skill Gap Diagnosis & Mentor Analytics  
**Primary Discipline Scope:** B.Tech & MCA across Computer Science & Engineering (CSE), Information Technology (IT), AI & Data Engineering, Electronics & Communication (ECE), and related computing disciplines.

---

> [!IMPORTANT]
> ### ⚠️ Mandatory Transparency Notice: Academic Prototype & Synthetic Data
> The dataset (`data/lpu_placement_dataset.csv`, 2,000 student records) is synthesized algorithmically using correlated latent distributions, domain-informed engineering rubrics, and placement rules strictly for demonstrating end-to-end supervised machine learning pipelines, transparent readiness scoring, and mentor feedback workflows.
> - **Evaluation on this dataset measures agreement with synthetic rubric labels.**
> - **These results do NOT establish accuracy on real LPU students or guarantee actual corporate placement outcomes.**
> - **This platform is an independent student academic project; it has no official affiliation with or endorsement from Lovely Professional University (LPU) administration or its Division of Career Services / Training and Placement (TPO).**

---

## 📑 Table of Contents
1. [Platform Overview & Purpose](#-1-platform-overview--purpose)
2. [Core Conceptual Framework: Model vs. Readiness vs. Completeness](#-2-core-conceptual-framework)
3. [What You Can Do in This Project (By User Role)](#-3-what-you-can-do-in-this-project)
   - [For Students](#-for-students)
   - [For Faculty Mentors & Career Coaches](#-for-faculty-mentors--career-coaches)
   - [For TPO Placement Officers & Institutional Admins](#-for-tpo-placement-officers--institutional-admins)
   - [For Machine Learning Engineers & Researchers](#-for-machine-learning-engineers--researchers)
4. [Tour of the 7 Interactive Web Views](#-4-tour-of-the-7-interactive-web-views)
5. [Deterministic 5-Pillar Readiness Model](#-5-deterministic-5-pillar-readiness-model)
6. [Skill-Gap Analysis & Benchmark Tolerance Logic](#-6-skill-gap-analysis--benchmark-tolerance-logic)
7. [Active Backlogs Feature Importance Audit](#-7-active-backlogs-feature-importance-audit)
8. [System Architecture & Technology Stack](#-8-system-architecture--technology-stack)
9. [Project Directory & File Structure](#-9-project-directory--file-structure)
10. [Setup, Installation & Running Instructions](#-10-setup-installation--running-instructions)
11. [Pre-Configured Demo Accounts (RBAC)](#-11-pre-configured-demo-accounts-rbac)
12. [Automated Test Suites & Quality Assurance](#-12-automated-test-suites--quality-assurance)
13. [Complete REST API Reference](#-13-complete-rest-api-reference)
14. [Real-World University Ingestion Roadmap](#-14-real-world-university-ingestion-roadmap)

---

## 🎯 1. Platform Overview & Purpose

The **LPU Placement Intelligence & Student Readiness Prediction Platform** is a full-stack, machine learning-driven web application designed to bridge the gap between academic evaluation and industry recruitment expectations. 

Traditional placement portals typically offer nothing more than binary pass/fail filters or opaque black-box likelihood percentages. This platform replaces opaque guessing with a **dual-engine architecture**:
1. **Empirical Machine Learning Engine:** Evaluates candidate profiles against historical outcome patterns using calibrated classification models and decision tree feature attribution.
2. **Pedagogical Diagnostic Engine:** Delivers a fully transparent, rubric-based readiness score across five institutional pillars, pinpointing specific competency gaps against target industry job profiles.

---

## 🧠 2. Core Conceptual Framework

The system strictly decouples three fundamental metrics that are often conflated in campus analytics:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             STUDENT ASSESSMENT                              │
└──────────────────────┬───────────────────────────────┬──────────────────────┘
                       │                               │
                       ▼                               ▼
       ┌───────────────────────────────┐ ┌───────────────────────────┐
       │     Supervised Model          │ │   Transparent Readiness   │
       │   (Demo Model Estimate)       │ │ (Deterministic 5-Pillars) │
       ├───────────────────────────────┤ ├───────────────────────────┤
       │ • Demo Model Estimate (%)     │ │ • Technical Pillar  (30%) │
       │ • Calibrated P(Y=1) on synth  │ │ • Aptitude Pillar   (25%) │
       │ • Brier Score: 0.0516         │ │ • Comm. Pillar      (20%) │
       │ • ROC-AUC: 0.9827             │ │ • Projects Pillar   (15%) │
       │ • Whole integer percentage    │ │ • Internship Pillar (10%) │
       │ • Non-validated disclaimer    │ │ • Canonical 3-Tier Bands  │
       └───────────────┬───────────────┘ └─────────────┬─────────────┘
                       │                               │
                       └───────────────┬───────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │    Prescriptive Diagnostics   │
                       │   & Role Benchmark Tolerance  │
                       ├───────────────────────────────┤
                       │ • Gap = Score - Benchmark     │
                       │ • ±3.0 Pt Tolerance Logic     │
                       │ • Non-Guaranteed Action Plan  │
                       │ • Input Completeness Tooltip  │
                       └───────────────────────────────┘
```

### Metric 1: Demo Model Estimate (Supervised ML)
- **What it is:** The estimated probability of clearing campus placement recruitment, produced by a supervised machine learning model (Logistic Regression Champion with Platt-scaling calibration) trained on synthetic historical cohorts.
- **Display standard:** Consistently formatted as a whole percentage integer (e.g., `87%`) across all UI cards, tables, reports, and exports.
- **Mandatory disclaimer:** Every display is paired with the transparency note:  
  *“Generated by a model trained on synthetic profiles. Not validated for real student placement outcomes.”*
- **Integrity commitment:** It is never referred to as "real-world placement chance," "certainty," or "model confidence."

### Metric 2: Transparent Student Readiness Score (Pedagogical Assessment)
- **What it is:** A deterministic, explainable institutional score from $0$ to $100$ computed via transparent weighted rubrics across five preparation pillars (DSA, Aptitude, Communication, Projects, Internships).
- **Canonical 3-Tier Classification:**
  - 🔴 **Needs attention:** Score $< 40$
  - 🟡 **Developing:** $40 \le \text{Score} < 70$
  - 🟢 **Strong preparation:** $70 \le \text{Score} \le 100$
- **Backward compatibility:** Legacy 4-tier model outputs (`Foundation`, `Developing`, `Placement Ready`, `Advanced`) map deterministically into this 3-tier framework.

### Metric 3: Input Completeness Tracker
- **What it is:** An operational audit metric ($0 - 100\%$) indicating how many core and optional assessment fields the student has completed.
- **Clarification:** It does **not** reflect how strong the student is; it merely reflects data submission completeness. An explicit tooltip clarifies this distinction.

### Metric 4: Simulated CTC Regressor (Secondary / Experimental)
- **What it is:** An auxiliary Gradient Boosting regression baseline providing an illustrative annual compensation range (in LPA).
- **Positioning:** Kept in a dedicated, demoted experimental section with prominent simulation badges so students do not misinterpret salary estimates as job offers.

---

## 🚀 3. What You Can Do in This Project

### 🎓 For Students
1. **Self-Assessment Evaluation:**
   - Complete a guided 5-step wizard covering **Academics & Eligibility**, **Coding & DSA**, **Cognitive Aptitude**, **Communication & Projects**, and **Career Aspirations**.
   - Receive instantaneous feedback with real-time field validation (e.g. CGPA $\le 10.0$, attendance $\le 100\%$).
2. **Instant 1-Click Archetype Presets:**
   - Test the system instantly using pre-populated student archetypes:
     - *Preset 0: Advanced DSA & Full-Stack Lead* (High readiness, 0 backlogs)
     - *Preset 1: Balanced Full-Stack Developer* (Mid-high readiness, 0 backlogs)
     - *Preset 2: High CGPA Academic Specialist* (High academics, moderate DSA)
     - *Preset 3: At-Risk Candidate with Active Backlogs* (Active backlogs, needs urgent remediation)
3. **Institutional Eligibility Verification:**
   - Immediate clearance feedback against campus recruitment criteria:
     - Maximum active backlogs ($\le 0$ or $\le 1$ depending on drive tier).
     - Minimum CGPA thresholds ($\ge 6.0$ or $\ge 7.0$).
     - Minimum mandatory attendance ($\ge 75\%$).
4. **Interactive Skill-Gap Diagnosis:**
   - Compare competencies against 7 industry role benchmarks:
     - *Software Development Engineer (SDE 1)*
     - *Frontend Engineer*
     - *Backend Engineer*
     - *Data Engineer*
     - *DevOps & Cloud Engineer*
     - *Machine Learning Engineer*
     - *QA & SDET Automation Engineer*
   - View an interactive multi-axis **Chart.js Radar Chart** showing your profile overlaid against target benchmarks.
   - Inspect the **Skill-Gap Table** with signed differences ($\text{Gap} = \text{Score} - \text{Benchmark}$) and $\pm 3.0$ pt tolerance tags.
5. **Actionable Improvement Roadmap:**
   - Receive a non-guaranteed, prioritized milestone plan structured into **Immediate (30-Day)**, **Medium-Term (60-Day)**, and **Final (90-Day)** preparation sprints.
6. **Longitudinal History & Official Report:**
   - Track progress over time across multiple assessment attempts.
   - View visual line charts showing score progression.
   - Generate and download a formatted, comprehensive student evaluation report.
7. **Read Mentor Coaching Notes:**
   - Access personalized feedback and tactical advice left by assigned faculty mentors.

---

### 👥 For Faculty Mentors & Career Coaches
1. **Dedicated Mentor Hub:**
   - Sign in using mentor credentials (`mentor1` / `Mentor@123`) to unlock the mentor portal.
2. **Assigned Student Roster:**
   - View a cohort table of assigned advisees displaying Registration Number, Student Name, Program, Readiness Score, Canonical Category, Demo Model Estimate (whole %), and Assessment Count.
3. **Deep-Dive Student Inspection:**
   - Click on any student record to load their full diagnostic report, historical assessment logs, and pillar breakdown.
4. **Submit Direct Coaching Feedback:**
   - Write structured mentor notes, action items, and follow-up schedules. Feedback is immediately persisted in SQLite and delivered to the student's dashboard.

---

### 🏛️ For TPO Placement Officers & Institutional Admins
1. **Central Institutional Monitoring:**
   - Sign in as administrator (`admin1` / `Admin@123`) to access full university cohort analytics.
2. **Cohort Risk Stratification:**
   - Track cohort-wide distribution across the canonical readiness categories (*Needs attention*, *Developing*, *Strong preparation*).
   - Identify candidates at risk due to active backlogs, low attendance, or insufficient coding practice.
3. **Batch Prediction via CSV:**
   - Download the official institutional template (`GET /api/sample-csv`).
   - Upload bulk batches of up to hundreds of student records simultaneously (`POST /api/batch-predict`).
   - Instantly receive batch analytics: overall clearance rate, mean readiness score, mean simulated compensation, and individual results.

---

### 🔬 For Machine Learning Engineers & Researchers
1. **Explainable AI (XAI) Tree Attribution:**
   - Inspect tree-based feature attribution calculated via Saabas path decomposition.
   - Understand positive and negative contributors pushing individual predictions up or down.
2. **Strictly Sorted Feature Importance Chart:**
   - Explore global ensemble feature importances sorted in strictly descending order with explicit axis units (`Relative Importance (%)`).
   - Access an accessible semantic HTML table displaying exact numerical percentages.
3. **Audited Active Backlogs Behavior:**
   - Review transparent documentation detailing why `active_backlogs` ranks as the top feature (~12.2%) based on synthetic label rules and tree split mechanics.
4. **Reproducible Model Pipeline:**
   - Inspect scikit-learn preprocessing pipelines (`StandardScaler`, `OneHotEncoder`) fitted strictly on training data without leakage.
   - Retrain models with a single command (`python src/train_models.py`) and inspect saved evaluation metrics in `models/model_metrics.json`.

---

## 🧭 4. Tour of the 7 Interactive Web Views

The user interface uses a **modern 250px compact sidebar layout** with an adaptive top bar, persistent breadcrumbs, and a mobile drawer.

```
┌──────────────┬───────────────────────────────────────────────────────────────┐
│  SIDEBAR     │ TOP BAR: Breadcrumbs | Academic Prototype Badge | User Session │
│              ├───────────────────────────────────────────────────────────────┤
│ 🏠 Overview  │                                                               │
│ 📝 Assessment│                       ACTIVE VIEW AREA                        │
│ 🎯 Plan      │                                                               │
│ 📜 History   │  (Dynamically rendered without full-page browser reloads)      │
│ 👥 Mentor    │                                                               │
│ 📊 Cohorts   │                                                               │
│ 🔬 Method    │                                                               │
└──────────────┴───────────────────────────────────────────────────────────────┘
```

1. 🏠 **Overview:**
   - Top KPI summary cards: **Demo Model Estimate** (whole %), **Readiness Assessment** (0-100 & canonical tier), and **Input Completeness** (with explanatory tooltip).
   - 1-click preset selector to test different student archetypes instantly.
   - Institutional eligibility status banner (Backlogs, CGPA, Attendance).
2. 📝 **Assessment:**
   - Interactive 5-step wizard form collecting academic, technical, cognitive, and communicative metrics.
   - Live completeness bar updating in real time as fields are completed.
   - Client-side error validation before dispatching to the REST API.
3. 🎯 **Improvement Plan:**
   - Target career role selector dynamically adjusting benchmark curves.
   - Multi-axis Chart.js Radar Chart comparing student score against target benchmark.
   - Comprehensive **Skill-Gap Table** with tolerance status tags.
   - Prioritized preparation milestones for 30, 60, and 90 days.
   - Secondary experimental section for simulated CTC estimation.
4. 📜 **History:**
   - Longitudinal progress timeline displaying all past submissions for the active student.
   - Score trajectory charts highlighting improvements across assessment cycles.
   - Complete student evaluation report export.
5. 👥 **Mentor Hub:**
   - Roster table of assigned students with live readiness indicators.
   - Student record drawer to inspect specific strengths and gap areas.
   - Feedback submission module allowing mentors to publish guidance notes directly to students.
6. 📊 **Cohorts (TPO Batch Hub):**
   - Drag-and-drop CSV batch upload component.
   - One-click template CSV download button.
   - Institutional aggregation charts: tier distribution, placement likelihood spread, and compensation histogram.
7. 🔬 **Methodology & XAI:**
   - Model evaluation benchmarks (Accuracy, Precision, Recall, F1, ROC-AUC, Brier score).
   - Horizontal bar chart of feature importances sorted strictly descending with accessible table fallback.
   - Formal audit findings regarding the `active_backlogs` contribution.
   - Data provenance and synthetic generation disclosures.

---

## 📐 5. Deterministic 5-Pillar Readiness Model

The Student Readiness Score ($0 - 100$) is computed via an auditable, transparent pedagogical rubric independent of any black-box machine learning weights:

$$\text{Readiness Score} = \sum_{i=1}^{5} w_i \times P_i$$

### Pillar Weights & Compositions:

| Pillar ($P_i$) | Weight ($w_i$) | Underlying Sub-Components | Anti-Gaming Safeguards |
| :--- | :---: | :--- | :--- |
| **1. Technical Competency** | **30%** | DSA Assessment score (12%), Coding Contest Rating (10%), Web/Cloud/System Design (8%) | Normalized to 0-100 scale; contest ratings scaled against 2000 Elo cap |
| **2. Aptitude & Problem Solving** | **25%** | Quantitative Aptitude (10%), Logical Reasoning (10%), Verbal Ability (5%) | Standardized percentile conversion |
| **3. Communication & Interpersonal** | **20%** | Mock Interview rating (10%), Group Discussion rating (6%), Presentation rating (4%) | Proctored 0-100 rubric scale |
| **4. Projects & Engineering Practice** | **15%** | Verified Full-Stack/ML Projects (10%), GitHub Code Quality & Commits (5%) | Project count capped at $\max = 4$ to prevent quantity-over-quality gaming |
| **5. Internships & Real-World Experience** | **10%** | Industry Internship Count (6%), Total Internship Duration in Months (4%) | Internship count capped at $\max = 3$; duration capped at $\max = 6\text{ months}$ |

### Canonical 3-Tier Categorization:

```
  0                     40                     70                   100
  ┌─────────────────────┬──────────────────────┬──────────────────────┐
  │   Needs attention   │      Developing      │  Strong preparation  │
  └─────────────────────┴──────────────────────┴──────────────────────┘
```
- **Needs attention ($< 40$):** Significant fundamental gaps in coding, aptitude, or core eligibility. Priority focus on remedial coursework.
- **Developing ($40 \le \text{Score} < 70$):** Good baseline competencies; requires focused practice in data structures, mock interviews, and project refinement.
- **Strong preparation ($70 \le \text{Score} \le 100$):** High readiness across all five pillars; primed for top-tier competitive campus placement drives.

---

## 📊 6. Skill-Gap Analysis & Benchmark Tolerance Logic

The platform calculates competency gaps across 6 core technical and cognitive dimensions:
1. **DSA & Algorithmic Problem Solving**
2. **System Design & Web Architecture**
3. **Quantitative Aptitude & Mathematics**
4. **Logical & Analytical Reasoning**
5. **Technical Communication & Articulation**
6. **Project Implementation & Production Code**

### Mathematical Formulation
$$\text{Gap} = \text{Student Competency Score} - \text{Role Benchmark Score}$$

A positive gap ($+$) indicates that the student exceeds the benchmark, while a negative gap ($-$) indicates that the student is below the benchmark.

### $\pm 3.0$ Pt Benchmark Tolerance Classification:
To avoid penalizing students for minor statistical variances, the platform implements a defensible $\pm 3.0$ pt tolerance band:

| Gap Range ($\Delta$) | Status Label | Badge Styling | Meaning |
| :--- | :--- | :--- | :--- |
| $\Delta \ge +10.0$ | **Above benchmark** | Green badge (`exceeds`) | Candidate strongly outperforms the role expectations |
| $+3.0 < \Delta < +10.0$ | **Meets benchmark** | Blue badge (`on-target`) | Candidate comfortably satisfies the role requirements |
| $-3.0 \le \Delta \le +3.0$ | **Within benchmark tolerance** | Teal badge (`tolerance`) | Candidate is essentially on target within reasonable variation |
| $-10.0 \le \Delta < -3.0$ | **Below benchmark** | Yellow badge (`below`) | Candidate requires targeted revision before interviews |
| $\Delta < -10.0$ | **Critical gap** | Red badge (`critical`) | High-priority preparation required to prevent interview elimination |

> [!NOTE]
> Under this rule, negative gaps are **never** labeled as "On Target".

---

## 🔬 7. Active Backlogs Feature Importance Audit

During feature importance extraction of the Random Forest ensemble, `active_backlogs` was observed to account for **12.19%** of total tree split attribution, ranking as the single highest feature. 

The audit verified that this behavior is mathematically and logically sound based on three verifiable factors:

1. **Synthetic Label Generation Rule (`src/data_generator.py`):**
   ```python
   backlog_penalty = active_backlogs * 0.15 + history_backlogs * 0.03
   # Strict probability cap:
   if active_backlogs >= 2:
       placement_prob = min(placement_prob, 45.0)
   ```
   Profiles with 2 or more active backlogs are capped below $45\%$ placement probability.
2. **Supervised Binary Classification Target (`src/train_models.py`):**
   ```python
   df["placed_binary"] = ((df["placement_probability"] >= 50.0) & (df["active_backlogs"] <= 1)).astype(int)
   ```
   Candidates with $> 1$ active backlog cannot receive a positive placement label, modeling the strict eligibility criteria enforced by tier-1 and tier-2 campus recruiters.
3. **Information Gain & Gini Impurity Reduction:**
   Decision trees select splits that maximize Gini impurity reduction. Because splitting on `active_backlogs <= 1` cleanly isolates unplaced candidates with high purity, tree algorithms repeatedly select this feature at or near the root nodes.
4. **Methodological Transparency:**
   Impurity-based feature importance naturally reflects features that create sharp binary partitions. The platform explains this behavior in the Methodology section rather than artificially suppressing the feature's importance.

---

## 🏗️ 8. System Architecture & Technology Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       BROWSER CLIENT (HTML5 / ES6)                         │
│  • Compact 250px Desktop Sidebar Layout + Mobile Responsive Drawer         │
│  • Chart.js Radar & Bar Visualizations                                      │
│  • Single-Page View Routing (7 Purpose-Built Views)                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP / REST (Bearer Auth)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FLASK APPLICATION BACKEND                            │
│  • app.py: REST Controllers, CORS, Static Serving, Error Handlers           │
│  • src/auth_service.py: PBKDF2 Password Hashing & HMAC-SHA256 Tokens        │
│  • src/validator.py: Server-Side Strict Schema & Range Validation           │
│  • src/readiness_scorer.py: Deterministic 5-Pillar Pedagogical Scorer       │
│  • src/ml_engine.py: ML Inference, Tree Attribution, Roadmaps, Gaps         │
└──────────────────┬───────────────────────────────────────┬──────────────────┘
                   │                                       │
                   ▼                                       ▼
┌──────────────────────────────────────┐ ┌────────────────────────────────────┐
│      PERSISTENCE (SQLite3)           │ │     TRAINED ML MODELS (PKL)        │
│  • data/assessment_history.db        │ │  • placement_classifier.pkl (LogReg│
│  • assessments, users, feedback      │ │  • readiness_classifier.pkl (RF)   │
│  • Dynamic DATABASE_PATH redirection │ │  • preprocessor.pkl (Scalers/Enc)  │
└──────────────────────────────────────┘ └────────────────────────────────────┘
```

- **Frontend:** Vanilla modern HTML5, glassmorphic CSS variables, responsive flexbox/grid layout, Chart.js for data visualization, zero runtime npm framework bloat.
- **Backend:** Python 3.10+ Flask microframework with secure session authentication, JSON request validation, and clean RESTful design.
- **Machine Learning:** scikit-learn classification and regression pipelines, StandardScaler, OneHotEncoder, pure NumPy baseline algorithms.
- **Database:** SQLite 3 with automatic schema migration, parameterized queries to prevent SQL injection, and test database isolation.

---

## 📁 9. Project Directory & File Structure

```
lpu-placement-intelligence/
├── data/
│   ├── lpu_placement_dataset.csv           # 2,000 synthetic student records (v2.2 schema)
│   ├── real_student_placement_schema.json  # Reference schema for future university data
│   └── assessment_history.db               # Production SQLite database
├── docs/
│   ├── architecture.md                     # Detailed system topology and security design
│   ├── dataset.md                          # Data dictionary and synthetic generation rules
│   ├── model_card.md                       # ML governance, evaluation metrics, and audit notes
│   └── implementation_status.md            # Verified audit and development checklist
├── models/
│   ├── preprocessor.pkl                    # Feature scalers fitted strictly on training data
│   ├── placement_classifier.pkl            # Champion Calibrated Logistic Regression model
│   ├── readiness_classifier.pkl            # Champion Random Forest Readiness model
│   ├── ctc_regressor.pkl                   # Gradient Boosting Simulated Salary Regressor
│   ├── prob_regressor.pkl                  # Auxiliary Likelihood Regressor
│   ├── role_classifier.pkl                 # Multi-output Career Role Matcher
│   └── model_metrics.json                  # Held-out test evaluation benchmarks
├── scripts/
│   └── cleanup_test_records.py             # Database sanitation utility for test artifacts
├── src/
│   ├── __init__.py
│   ├── auth_service.py                     # PBKDF2 hashing, HMAC session tokens & RBAC
│   ├── data_generator.py                   # Multivariate synthetic student profile generator
│   ├── history_store.py                    # SQLite storage layer with DATABASE_PATH isolation
│   ├── ml_algorithms.py                    # Pure NumPy baselines (Dummy classifier, AUC, Brier)
│   ├── ml_engine.py                        # Central inference engine, tree attribution, roadmaps
│   ├── readiness_scorer.py                 # Deterministic 5-pillar scorer & 3-tier categorization
│   ├── train_models.py                     # Leakage-free training and cross-validation pipeline
│   └── validator.py                        # Strict validation rules for inputs and batch CSVs
├── static/
│   ├── css/
│   │   └── style.css                       # Modern responsive glassmorphic design system
│   ├── js/
│   │   └── app.js                          # Frontend client controller & Chart.js renderer
│   └── index.html                          # Single-page application containing all 7 views
├── .env.example                            # Environment variable configuration template
├── app.py                                  # Main Flask application entry point
├── requirements.txt                        # Minimal pinned Python dependencies
├── test_pipeline.py                        # Isolated unit and integration test suite
├── test_live_server.py                     # Live REST API verification suite (79 checks)
└── README.md                               # Master documentation & user guide
```

---

## 💻 10. Setup, Installation & Running Instructions

### Prerequisites
- **Python 3.10+** (tested on Python 3.10, 3.11, 3.12, 3.14 on Windows, Linux, and macOS)
- Any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox, Brave, Safari)

### Step 1: Clone or Navigate to the Workspace
```powershell
cd lpu-placement-intelligence
```

### Step 2: Set Up Python Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1

# (Or on Linux / macOS):
# source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Step 3: Run the Web Application
```powershell
python app.py
```
*(Alternatively, execute directly without activating: `.venv\Scripts\python.exe app.py`)*

### Step 4: Open in Your Web Browser
Navigate to either of the following local URLs:
- **[http://localhost:5000](http://localhost:5000)**
- **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🔐 11. Pre-Configured Demo Accounts (RBAC)

The platform comes pre-seeded with test accounts in SQLite:

| Role | Username | Password | Purpose & Privileges |
| :--- | :--- | :--- | :--- |
| **Student** | `student1` | `Student@123` | Self-assessment wizard, longitudinal assessment history, read mentor feedback |
| **Mentor** | `mentor1` | `Mentor@123` | Access Mentor Hub, view assigned student profiles, submit coaching feedback notes |
| **Admin** | `admin1` | `Admin@123` | Central institutional analytics, batch CSV predictions, system monitoring |

*Note: Unauthenticated visitors browse in Guest Mode, where they can test presets and explore public demo records (`is_demo = 1`). Private student assessments are secured behind session tokens.*

---

## 🧪 12. Automated Test Suites & Quality Assurance

The codebase includes two independent test suites:

### A. Isolated Unit & Integration Suite (`test_pipeline.py`)
This suite tests readiness calculations, canonical tier classifications, model loading, input validation, and database operations. It uses `DATABASE_PATH` redirection to `data/test_assessment_history.db` with automated setup and teardown to ensure zero database pollution:
```powershell
python test_pipeline.py
```
*Expected Output:*
```
..............
----------------------------------------------------------------------
Ran 14 tests in 1.843s

OK
```

### B. Live Server HTTP Verification Suite (`test_live_server.py`)
This suite performs 79 end-to-end HTTP contract checks against the live running server:
```powershell
python test_live_server.py http://127.0.0.1:5000
```
*Expected Output:*
```
========================================================================
 VERIFICATION RESULTS: 79 Passed | 0 Failed | Duration: 1.16s
========================================================================
[SUCCESS] ALL LIVE SERVER REST ENDPOINTS VERIFIED AND RESPONDING FLAWLESSLY!
```

### C. Retraining the Machine Learning Models (Optional)
To regenerate synthetic training data and fit models from scratch:
```powershell
python src/train_models.py
```

### D. Purging Test Artifacts from Database (Optional)
```powershell
python scripts/cleanup_test_records.py --apply
```

---

## 📡 13. Complete REST API Reference

All requests and responses use JSON. Authenticated endpoints require the header `Authorization: Bearer <session_token>`.

### Public & Diagnostic Endpoints

#### 1. System Health Check
- **Endpoint:** `GET /api/health`
- **Response:**
  ```json
  {
    "status": "healthy",
    "version": "2.2.0",
    "platform": "LPU Placement Intelligence & Student Readiness Prediction Platform",
    "supervised_model_loaded": true,
    "readiness_classifier_loaded": true,
    "demo_mode": true,
    "disclaimer": "Academic research prototype evaluated on synthetic rubric benchmarks; not affiliated with official LPU recruitment administration."
  }
  ```

#### 2. Get Presets & Career Roles
- **Endpoint:** `GET /api/presets`
- **Response:** Returns list of preset student profiles and 7 supported career roles with their benchmark curves.

#### 3. Model Performance Benchmarks
- **Endpoint:** `GET /api/model-performance`
- **Response:** Returns evaluation metrics (ROC-AUC, Brier score, accuracy, F1) for all candidate classifiers and sorted feature importances.

#### 4. Download Sample CSV
- **Endpoint:** `GET /api/sample-csv`
- **Response:** Returns a downloadable CSV template with pre-filled student headers for batch evaluation.

---

### Prediction & Diagnostic Endpoints

#### 5. Evaluate Student Assessment
- **Endpoint:** `POST /api/predict`
- **Request Body:** JSON object containing student academic, technical, cognitive, and communicative scores.
- **Key Response Fields:**
  - `demo_model_estimate_pct`: Whole integer percentage (e.g. `87%`).
  - `demo_model_estimate_display`: Formatted string (`"87%"`).
  - `demo_model_disclaimer`: Mandatory synthetic data notice.
  - `readiness_score`: Float between $0.0$ and $100.0$.
  - `readiness_category`: Canonical tier (`"Needs attention"`, `"Developing"`, `"Strong preparation"`).
  - `completeness_pct`: Input completeness ($0-100\%$).
  - `completeness_note`: Explanation clarifying data completeness vs performance.
  - `eligibility`: Institutional clearance status and criteria checks.
  - `skill_gaps`: Dictionary of competencies with signed gaps and tolerance statuses.
  - `roadmap`: Prioritized preparation milestones for 30, 60, and 90 days.
  - `experimental_simulated_ctc`: Secondary simulated compensation estimate in LPA.

#### 6. Batch Cohort Evaluation
- **Endpoint:** `POST /api/batch-predict`
- **Request Body:** JSON array of student assessment records or multipart CSV file.
- **Response:** Summary statistics (mean readiness, clearance rate, mean simulated CTC) and individual student results.

---

### History & Report Endpoints

#### 7. Retrieve Assessment History
- **Endpoint:** `GET /api/history/<student_id>`
- **Response:** List of longitudinal historical assessments for the student, sorted newest first.

#### 8. Retrieve Full Assessment Report
- **Endpoint:** `GET /api/report/<assessment_id>`
- **Response:** Full structured diagnostic report for printing or PDF generation.

---

### Authentication & Mentor Endpoints

#### 9. User Authentication (Login)
- **Endpoint:** `POST /api/auth/login`
- **Request Body:** `{"username": "mentor1", "password": "Mentor@123"}`
- **Response:** `{"status": "success", "token": "...", "user": {"username": "mentor1", "role": "mentor"}}`

#### 10. Verify Current Session
- **Endpoint:** `GET /api/auth/me`
- **Headers:** `Authorization: Bearer <token>`
- **Response:** User identity and role verification.

#### 11. Mentor Assigned Students Roster
- **Endpoint:** `GET /api/mentor/students`
- **Headers:** `Authorization: Bearer <mentor_token>`
- **Response:** List of assigned students with current readiness status and demo estimates.

#### 12. Submit Coaching Feedback
- **Endpoint:** `POST /api/mentor/feedback`
- **Headers:** `Authorization: Bearer <mentor_token>`
- **Request Body:**
  ```json
  {
    "student_id": "LPU_DEMO_ADV01",
    "feedback_text": "Strong DSA performance. Focus on system design scalability for upcoming drives.",
    "action_items": "Complete 3 mock interviews and revise distributed caching.",
    "recommended_role": "Backend Engineer"
  }
  ```
- **Response:** `{"status": "success", "message": "Feedback saved successfully."}`

#### 13. Student Feedback Retrieval
- **Endpoint:** `GET /api/student/feedback/<student_id>`
- **Response:** List of coaching notes submitted by mentors for the student.

---

## 🔮 14. Real-World University Ingestion Roadmap

To transition this academic prototype to live university deployment with actual student data, the following requirements must be addressed:

1. **Schema Compliance:**
   - Ingest data in accordance with [**`data/real_student_placement_schema.json`**](file:///c:/Users/Lenovo/.gemini/antigravity-ide/scratch/lpu-placement-intelligence/data/real_student_placement_schema.json).
2. **Recruitment Participation Tracking:**
   - Explicitly record `recruitment_participation_status`. Students pursuing higher education (GATE/GRE), civil services, or family businesses must be excluded from unplaced outcome counts to avoid penalizing placement rates.
3. **Temporal Cross-Validation:**
   - Discontinue randomized train/test splits. Use chronological cohort splits (e.g., train on 2023-2024 and 2024-2025 cohorts; evaluate on 2025-2026 cohort) to prevent future data leakage.
4. **Empirical Probability Calibration:**
   - Calibrate placement probabilities against real corporate hiring outcomes using Platt Scaling or Isotonic Regression, monitoring Brier score and reliability curves.
5. **Verified Assessment Sourcing:**
   - Replace self-reported student inputs with verified institutional feeds: proctored coding test platforms (e.g. HackerRank/CodeChef institutional APIs), LMS attendance records, and faculty mentor interview evaluations.
6. **Data Privacy & Governance:**
   - Enforce student anonymization and ensure full compliance with applicable student privacy regulations before loading production academic records.

---

## 📄 License & Attribution
- **Project:** LPU Placement Intelligence & Student Readiness Prediction Platform (v2.2.0)
- **Purpose:** Academic Research & Student Placement Diagnostic Prototype
- **Notice:** Not affiliated with or endorsed by Lovely Professional University (LPU). All university trademarks remain the property of their respective owners.
