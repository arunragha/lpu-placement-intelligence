# Dataset Documentation: LPU Placement Intelligence Platform

## 1. Overview & Provenance

- **File Path**: `data/lpu_placement_dataset.csv`
- **Total Records**: 2,000 unique student profiles
- **Total Columns**: 30 features + target variables
- **Dataset Nature**: **Synthetic Algorithmic Demonstration Dataset**
- **License / Usage**: Academic research and pedagogical evaluation prototype only.

> [!WARNING]
> **Academic Prototype & Synthetic Provenance Notice**:
> This dataset was generated algorithmically based on engineering college campus placement rubrics and domain rules. It does **not** contain real Lovely Professional University (LPU) student records, nor is it endorsed or provided by the official LPU placement cell or administration. It is created solely for demonstrating end-to-end supervised machine learning pipelines, transparent readiness scoring, and mentor feedback workflows.

---

## 2. Ethical Considerations & Excluded Demographics

To uphold fair and unbiased machine learning standards in higher education:
- **Demographic Exclusions**: Gender, caste, religion, parental income, home state, and socio-economic indicators are **strictly excluded** from model training and prediction features.
- **Personally Identifiable Information (PII)**: Student names, registration numbers, email addresses, phone numbers, and physical addresses are strictly excluded from all training pipelines and model inputs.
- **Post-Placement Leakage**: Company offer status, final CTC, signing bonuses, and post-placement evaluation metrics are never used as prediction features. Only pre-placement attributes available prior to recruitment season are considered.

---

## 3. Schema & Feature Definitions

The dataset contains pre-placement academic, technical, aptitude, and co-curricular observations:

### A. Academic & Eligibility Metrics
| Feature Name | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `student_id` | String | Identifier | Unique synthetic student identifier (excluded from training) |
| `branch` | Categorical | CSE, IT, AI & Data, ECE, Mech | Academic engineering department |
| `cgpa` | Float | 5.00 – 10.00 | Cumulative Grade Point Average |
| `tenth_percentage` | Float | 50.0 – 99.0 | High School / 10th Board examination percentage |
| `twelfth_percentage` | Float | 50.0 – 99.0 | Senior Secondary / 12th Board examination percentage |
| `attendance_percentage` | Float | 60.0 – 100.0 | Overall academic course attendance percentage |
| `active_backlogs` | Integer | 0 – 3 | Current uncleared backlogs at the time of recruitment |
| `history_backlogs` | Integer | 0 – 6 | Total past backlogs cleared prior to placement season |

### B. Technical Coding & System Competencies
| Feature Name | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `dsa_problems_solved` | Integer | 0 – 750 | Verified coding platform problems solved (LeetCode, GFG) |
| `dsa_score` | Float | 20.0 – 100.0 | Standardized internal Data Structures & Algorithms test |
| `coding_rating` | Float | 900 – 2350 | Competitive programming rating (CodeChef / Codeforces scale) |
| `web_cloud_score` | Float | 20.0 – 100.0 | Full-stack web, cloud, and API practical assessment |
| `system_design_score` | Float | 15.0 – 100.0 | Architecture and scalable system design evaluation |

### C. Cognitive & Aptitude Assessments
| Feature Name | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `quantitative_aptitude` | Float | 30.0 – 100.0 | Numerical problem solving and mathematics assessment |
| `logical_reasoning` | Float | 32.0 – 100.0 | Analytical reasoning and pattern recognition |
| `verbal_ability` | Float | 30.0 – 100.0 | English comprehension and grammar evaluation |

### D. Soft Skills & Communication
| Feature Name | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `mock_interview_score` | Float | 25.0 – 100.0 | Faculty / industry panel mock interview score |
| `gd_score` | Float | 25.0 – 100.0 | Group Discussion evaluation score |
| `communication_rating` | Float | 1.0 – 5.0 | Professional spoken English and presentation rubric rating |

### E. Practical Experience & Co-Curriculars
| Feature Name | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `projects_count` | Integer | 0 – 8 | Significant portfolio projects completed |
| `github_quality` | Float | 15.0 – 100.0 | GitHub code commit consistency and documentation score |
| `internships_count` | Integer | 0 – 3 | Number of corporate / research internships completed |
| `internship_duration_months` | Integer | 0 – 14 | Total cumulative internship experience in months |
| `certifications_count` | Integer | 0 – 7 | Industry certifications earned (AWS, Google Cloud, etc.) |
| `hackathons_participated` | Integer | 0 – 8 | National or collegiate hackathons participated |
| `hackathons_won` | Integer | 0 – 5 | Hackathons awarded top prize or podium finish |

---

## 4. Target Variables & Labels

1. **`placement_status` (Binary Target)**:
   - `1`: Placed through campus recruitment drive during the observation window.
   - `0`: Not placed during the campus recruitment cycle.
   - *Distribution*: 1,065 Placed (53.25%), 935 Not Placed (46.75%).

2. **`readiness_tier` (Multi-Class Target)**:
   - `Foundation` (< 50 readiness index; foundational learning needed).
   - `Developing` (50–69 readiness index; core competencies established, gaps in advanced topics).
   - `Placement Ready` (70–84 readiness index; ready for tier-1 & tier-2 placement drives).
   - `Advanced` (>= 85 readiness index; high-trajectory dream/super-dream candidate).

3. **`expected_ctc_lpa` (Continuous Regression Target)**:
   - Salary package in Lakhs Per Annum (LPA) for placed candidates (range: 3.5 LPA – 38.0 LPA).

4. **`target_role` (Multi-Class Career Target)**:
   - SDE, Full Stack Developer, AI/ML Engineer, Cloud & DevOps, Data Analyst, QA Engineer, Associate Consultant.

---

## 5. Statistical Distribution Summary

| Metric | Minimum | Maximum | Mean | Median |
| :--- | :--- | :--- | :--- | :--- |
| **CGPA** | 5.00 | 9.95 | 7.54 | 7.51 |
| **Active Backlogs** | 0 | 3 | 0.26 | 0 |
| **DSA Problems Solved** | 0 | 750 | 149.96 | 126.0 |
| **Coding Rating** | 900 | 2350 | 1379.59 | 1341.0 |
| **Quantitative Aptitude** | 30.0 | 100.0 | 59.95 | 60.1 |
| **Mock Interview Score** | 25.0 | 100.0 | 57.64 | 57.0 |
| **Internship Duration (Mos)** | 0 | 14 | 4.21 | 3.0 |

---

## 6. Real-World Institutional Cohort Schema Requirements

To transition this platform from an academic prototype to a real university production deployment, the university placement division would need to integrate:

1. **ERP / SIS Integration**:
   - Bi-directional sync with university student management system (e.g. UMS) for verified CGPA, course credits, and real-time backlog status.
2. **Coding Lab Telemetry**:
   - API webhooks from institutional coding platforms (HackerRank, LeetCode, GitHub Classroom) to capture authentic commit history and verified problem solutions.
3. **Assessment Centre Rubrics**:
   - Digitized scoring rubrics from official Career Development Center (CDC) mock interview and group discussion sessions.
4. **Company Drive Eligibility Logs**:
   - Historical job descriptions, eligibility criteria (minimum 10th/12th/CGPA, maximum backlogs), and student application/shortlisting/selection logs over at least 3 consecutive graduating batches.
5. **Data Protection & Consent**:
   - Institutional Review Board (IRB) approval, student data privacy consent in accordance with the Digital Personal Data Protection (DPDP) Act, and role-based cryptographic access control.

