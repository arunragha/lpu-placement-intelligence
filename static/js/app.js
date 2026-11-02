/**
 * LPU Placement Intelligence & Student Readiness Prediction Platform
 * Version: 2.2.0
 * Frontend Application Controller, Visualization, & API Client
 */

// Global Application State
const state = {
  theme: localStorage.getItem('lpu_theme') || 'dark',
  currentTab: 'overview',
  presets: [],
  currentPrediction: null,
  modelMetadata: null,
  radarChart: null,
  tpoTierChart: null,
  tpoBranchChart: null,
  featImpChart: null,
  batchStudents: [],
  token: localStorage.getItem('lpu_token') || null,
  currentUser: null,
  assignedStudents: []
};

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  initAuth();
  setupNavListeners();
  setupHeroListeners();
  setupFormListeners();
  setupExperimentalToggle();
  setupTPOBatchProcessor();
  setupHistorySearch();
  setupAuthModalListeners();
  setupMentorHubListeners();
  
  await loadPresets();
  await loadModelPerformance();
  await loadRecentHistory();
  await loadMentorStudents();
});

/* ==========================================================================
   1. Theme & Navigation
   ========================================================================== */

function initTheme() {
  document.documentElement.setAttribute('data-theme', state.theme);
  const themeBtn = document.getElementById('themeToggleBtn');
  if (themeBtn) {
    themeBtn.innerHTML = state.theme === 'dark' ? '☀️' : '🌙';
    themeBtn.addEventListener('click', () => {
      state.theme = state.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', state.theme);
      localStorage.setItem('lpu_theme', state.theme);
      themeBtn.innerHTML = state.theme === 'dark' ? '☀️' : '🌙';
      refreshAllCharts();
    });
  }
}

function setupNavListeners() {
  const tabs = document.querySelectorAll('.nav-item-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const targetView = tab.getAttribute('data-tab');
      switchView(targetView);
      closeMobileDrawer();
    });
  });

  // Mobile drawer toggle
  const mobileBtn = document.getElementById('mobileMenuBtn');
  const overlay = document.getElementById('sidebarOverlay');
  if (mobileBtn) {
    mobileBtn.addEventListener('click', toggleMobileDrawer);
  }
  if (overlay) {
    overlay.addEventListener('click', closeMobileDrawer);
  }

  // Plan tab buttons
  const btnEmptyAssess = document.getElementById('btnEmptyGoAssessment');
  const btnEmptyDemo = document.getElementById('btnEmptyLoadArchetype');
  const btnPlanReassess = document.getElementById('btnPlanReassess');

  if (btnEmptyAssess) btnEmptyAssess.addEventListener('click', () => switchView('assessment'));
  if (btnEmptyDemo) btnEmptyDemo.addEventListener('click', () => applyPresetAndEvaluate('LPU_DEMO_ADV01'));
  if (btnPlanReassess) btnPlanReassess.addEventListener('click', () => switchView('assessment'));
}

function toggleMobileDrawer() {
  const sidebar = document.getElementById('appSidebar');
  const overlay = document.getElementById('sidebarOverlay');
  if (sidebar && overlay) {
    const isOpen = sidebar.classList.contains('open');
    if (isOpen) {
      closeMobileDrawer();
    } else {
      sidebar.classList.add('open');
      overlay.classList.add('active');
    }
  }
}

function closeMobileDrawer() {
  const sidebar = document.getElementById('appSidebar');
  const overlay = document.getElementById('sidebarOverlay');
  if (sidebar) sidebar.classList.remove('open');
  if (overlay) overlay.classList.remove('active');
}

function switchView(viewId) {
  // Normalize aliases
  let normalized = viewId;
  if (viewId === 'tpo') normalized = 'cohorts';
  if (viewId === 'governance') normalized = 'methodology';

  state.currentTab = normalized;
  document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));
  const target = document.getElementById(`view-${normalized}`);
  if (target) {
    target.classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // Update sidebar active buttons
  document.querySelectorAll('.nav-item-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-tab') === normalized);
  });

  // Update topbar title
  const topbarTitle = document.getElementById('topbarCurrentView');
  if (topbarTitle) {
    const titles = {
      'overview': 'Platform Overview',
      'assessment': 'Student Assessment',
      'plan': 'Improvement Plan & Diagnostics',
      'history': 'Assessment History',
      'mentor': 'Faculty Mentor Advisory Hub',
      'cohorts': 'Cohort Analytics & Batch Processor',
      'methodology': 'ML Benchmarks & Governance'
    };
    topbarTitle.innerText = titles[normalized] || 'Platform Overview';
  }
}

function setupHeroListeners() {
  const btnStart = document.getElementById('btnHeroStart');
  const btnDemo = document.getElementById('btnHeroDemo');
  const btnGov = document.getElementById('btnHeroGovernance');

  if (btnStart) {
    btnStart.addEventListener('click', () => {
      switchView('assessment');
    });
  }

  if (btnDemo) {
    btnDemo.addEventListener('click', () => {
      applyPresetAndEvaluate('LPU_DEMO_ADV01');
    });
  }

  if (btnGov) {
    btnGov.addEventListener('click', () => {
      switchView('methodology');
    });
  }

  const btnOverviewLoad = document.getElementById('btnOverviewLoadPreset');
  const overviewSelect = document.getElementById('overviewPresetDropdown');
  if (btnOverviewLoad && overviewSelect) {
    btnOverviewLoad.addEventListener('click', () => {
      const pId = overviewSelect.value;
      if (pId) applyPresetAndEvaluate(pId);
    });
  }
}

/* ==========================================================================
   1B. User Authentication & Session
   ========================================================================== */

function initAuth() {
  state.token = localStorage.getItem('lpu_token');
  checkCurrentUser();
}

async function checkCurrentUser() {
  if (!state.token) {
    updateUserSessionUI(null);
    return;
  }
  try {
    const res = await fetch('/api/auth/me', {
      headers: { 'Authorization': `Bearer ${state.token}` }
    });
    const data = await res.json();
    if (data.status === 'success' && data.authenticated) {
      state.currentUser = data.user;
      updateUserSessionUI(data.user);
    } else {
      localStorage.removeItem('lpu_token');
      state.token = null;
      state.currentUser = null;
      updateUserSessionUI(null);
    }
  } catch (e) {
    console.error('Auth verification error:', e);
    updateUserSessionUI(null);
  }
}

function updateUserSessionUI(user) {
  const chip = document.getElementById('userSessionLabel');
  const authBtn = document.getElementById('btnOpenAuthModal');
  const mentorTab = document.getElementById('navTabMentor');
  
  if (user) {
    if (chip) chip.innerText = `${user.role === 'mentor' ? '👨‍🏫' : '🎓'} ${user.full_name || user.username}`;
    if (authBtn) {
      authBtn.innerText = 'Log Out';
      authBtn.onclick = handleLogout;
    }
    if (mentorTab) {
      mentorTab.style.display = (user.role === 'mentor' || user.role === 'admin') ? 'flex' : 'none';
    }
  } else {
    if (chip) chip.innerText = 'Guest (Demo Mode)';
    if (authBtn) {
      authBtn.innerText = 'Sign In';
      authBtn.onclick = openAuthModal;
    }
    if (mentorTab) mentorTab.style.display = 'flex';
  }
}

function openAuthModal() {
  const modal = document.getElementById('authModal');
  if (modal) modal.style.display = 'flex';
}

function closeAuthModal() {
  const modal = document.getElementById('authModal');
  if (modal) modal.style.display = 'none';
}

function handleLogout() {
  localStorage.removeItem('lpu_token');
  state.token = null;
  state.currentUser = null;
  updateUserSessionUI(null);
  loadRecentHistory();
}

function setupAuthModalListeners() {
  const closeBtn = document.getElementById('btnCloseAuthModal');
  const tabSignIn = document.getElementById('authTabSignIn');
  const tabRegister = document.getElementById('authTabRegister');
  const regFields = document.getElementById('authRegisterFields');
  const submitText = document.getElementById('btnSubmitAuthText');
  const form = document.getElementById('authForm');
  const quickStudent = document.getElementById('quickLoginStudent');
  const quickMentor = document.getElementById('quickLoginMentor');
  const errAlert = document.getElementById('authErrorAlert');

  let isRegisterMode = false;

  if (closeBtn) closeBtn.onclick = closeAuthModal;

  if (tabSignIn && tabRegister) {
    tabSignIn.onclick = () => {
      isRegisterMode = false;
      tabSignIn.classList.add('active');
      tabRegister.classList.remove('active');
      if (regFields) regFields.style.display = 'none';
      if (submitText) submitText.innerText = 'Sign In to Platform';
      if (errAlert) errAlert.style.display = 'none';
    };
    tabRegister.onclick = () => {
      isRegisterMode = true;
      tabRegister.classList.add('active');
      tabSignIn.classList.remove('active');
      if (regFields) regFields.style.display = 'block';
      if (submitText) submitText.innerText = 'Create Account';
      if (errAlert) errAlert.style.display = 'none';
    };
  }

  // Quick 1-click Demo logins
  if (quickStudent) {
    quickStudent.onclick = async () => {
      await performLogin('student1', 'Student@123');
    };
  }
  if (quickMentor) {
    quickMentor.onclick = async () => {
      await performLogin('mentor1', 'Mentor@123');
    };
  }

  if (form) {
    form.onsubmit = async (e) => {
      e.preventDefault();
      const uname = document.getElementById('authUsername').value.trim();
      const pwd = document.getElementById('authPassword').value;

      if (isRegisterMode) {
        const fname = document.getElementById('authFullName').value.trim() || uname;
        const role = document.getElementById('authRole').value;
        const email = document.getElementById('authEmail').value.trim();
        await performRegister(uname, pwd, role, fname, email);
      } else {
        await performLogin(uname, pwd);
      }
    };
  }
}

async function performLogin(username, password) {
  const errAlert = document.getElementById('authErrorAlert');
  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    const data = await res.json();
    if (res.status === 200 && data.status === 'success') {
      state.token = data.data.token;
      state.currentUser = data.data.user;
      localStorage.setItem('lpu_token', state.token);
      updateUserSessionUI(state.currentUser);
      closeAuthModal();
      loadRecentHistory();
      if (state.currentUser.role === 'mentor' || state.currentUser.role === 'admin') {
        loadMentorStudents();
      }
    } else {
      if (errAlert) {
        errAlert.style.display = 'block';
        errAlert.innerText = data.message || 'Login failed.';
      }
    }
  } catch (err) {
    if (errAlert) {
      errAlert.style.display = 'block';
      errAlert.innerText = 'Network connection failed.';
    }
  }
}

async function performRegister(username, password, role, fullName, email) {
  const errAlert = document.getElementById('authErrorAlert');
  try {
    const res = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username, password, role,
        full_name: fullName, email,
        student_reference_id: username.toUpperCase()
      })
    });
    const data = await res.json();
    if (res.status === 201 && data.status === 'success') {
      state.token = data.data.token;
      state.currentUser = data.data.user;
      localStorage.setItem('lpu_token', state.token);
      updateUserSessionUI(state.currentUser);
      closeAuthModal();
      loadRecentHistory();
    } else {
      if (errAlert) {
        errAlert.style.display = 'block';
        errAlert.innerText = data.message || 'Registration failed.';
      }
    }
  } catch (err) {
    if (errAlert) {
      errAlert.style.display = 'block';
      errAlert.innerText = 'Network error during registration.';
    }
  }
}

function setupExperimentalToggle() {
  const btn = document.getElementById('toggleExperimentalBtn');
  const content = document.getElementById('experimentalContent');
  if (btn && content) {
    btn.addEventListener('click', () => {
      content.style.display = content.style.display === 'none' ? 'block' : 'none';
    });
  }
}

/* ==========================================================================
   2. Demo Presets & Archetypes
   ========================================================================== */

async function loadPresets() {
  try {
    const res = await fetch('/api/presets');
    const data = await res.json();
    if (data.status === 'success') {
      state.presets = data.presets;
      populatePresetDropdown();
    }
  } catch (err) {
    console.error('Error fetching presets:', err);
  }
}

function populatePresetDropdown() {
  const dropdown = document.getElementById('presetDropdown');
  const overviewDropdown = document.getElementById('overviewPresetDropdown');
  const loadBtn = document.getElementById('btnLoadPreset');

  [dropdown, overviewDropdown].forEach(dd => {
    if (!dd) return;
    dd.innerHTML = '';
    state.presets.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.preset_id;
      opt.textContent = p.preset_name;
      dd.appendChild(opt);
    });
  });

  if (loadBtn && dropdown) {
    loadBtn.addEventListener('click', () => {
      applyPresetDataToForm(dropdown.value);
      submitAssessmentForm();
    });
  }
}

function applyPresetDataToForm(presetId) {
  const preset = state.presets.find(p => p.preset_id === presetId);
  if (!preset) return;

  const d = preset.data;
  for (const [key, val] of Object.entries(d)) {
    const input = document.getElementById(`field_${key}`);
    const badge = document.getElementById(`badge_${key}`);
    if (input) {
      input.value = val;
      if (badge) {
        badge.innerText = (key.includes('percentage') || key.includes('attendance')) ? `${val}%` : val;
      }
    }
  }
  hideValidationAlert();
}

async function applyPresetAndEvaluate(presetId) {
  applyPresetDataToForm(presetId);
  switchView('plan');
  await submitAssessmentForm();
}

/* ==========================================================================
   3. Form Wizard & Validation Handlers
   ========================================================================== */

function setupFormListeners() {
  // Sync sliders with numeric display badges
  const rangeInputs = document.querySelectorAll('input[type="range"]');
  rangeInputs.forEach(range => {
    const key = range.id.replace('field_', '');
    const badge = document.getElementById(`badge_${key}`);
    range.addEventListener('input', (e) => {
      if (badge) {
        const val = e.target.value;
        badge.innerText = (key.includes('percentage') || key.includes('attendance')) ? `${val}%` : val;
      }
    });
  });

  // Wizard tab switching
  const formTabs = document.querySelectorAll('.form-tab-btn');
  formTabs.forEach(btn => {
    btn.addEventListener('click', () => {
      formTabs.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const targetCat = btn.getAttribute('data-cat');
      document.querySelectorAll('.form-cat-group').forEach(group => {
        group.style.display = group.getAttribute('data-cat') === targetCat ? 'grid' : 'none';
      });
    });
  });

  // Form submit
  const form = document.getElementById('studentAssessmentForm');
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      submitAssessmentForm();
    });
  }
}

function collectFormData() {
  const fields = [
    'student_id', 'program', 'branch', 'semester', 'cgpa', 'active_backlogs',
    'attendance_percentage', 'tenth_percentage', 'twelfth_percentage', 'history_backlogs',
    'target_role', 'weekly_prep_hours', 'score_source',
    'dsa_problems_solved', 'dsa_score', 'coding_rating', 'web_cloud_score', 'system_design_score',
    'quantitative_aptitude', 'logical_reasoning', 'verbal_ability', 'mock_interview_score',
    'gd_score', 'communication_rating',
    'projects_count', 'github_quality', 'internships_count', 'internship_duration_months',
    'certifications_count', 'hackathons_participated', 'hackathons_won'
  ];

  const payload = {};
  fields.forEach(f => {
    const el = document.getElementById(`field_${f}`);
    if (el) {
      if (el.type === 'number' || el.type === 'range') {
        payload[f] = parseFloat(el.value);
      } else {
        payload[f] = el.value;
      }
    }
  });
  return payload;
}

function showValidationAlert(errors) {
  const alertBox = document.getElementById('assessmentValidationAlert');
  if (!alertBox) return;
  alertBox.style.display = 'block';
  alertBox.innerHTML = `
    <strong>⚠️ Input Validation Errors Detected:</strong>
    <ul style="margin-top: 0.4rem; margin-left: 1.25rem;">
      ${errors.map(err => `<li>${err}</li>`).join('')}
    </ul>
  `;
  alertBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hideValidationAlert() {
  const alertBox = document.getElementById('assessmentValidationAlert');
  if (alertBox) {
    alertBox.style.display = 'none';
    alertBox.innerHTML = '';
  }
}

/* ==========================================================================
   4. Prediction API & Results Visualization
   ========================================================================== */

async function submitAssessmentForm() {
  const payload = collectFormData();
  const submitBtn = document.getElementById('btnSubmitAssessment');
  if (submitBtn) {
    submitBtn.innerHTML = '<span>⚡ Analyzing Profile via Supervised ML Engine...</span>';
    submitBtn.disabled = true;
  }

  hideValidationAlert();

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (res.status === 200 && data.status === 'success') {
      state.currentPrediction = data.result;
      renderPredictionResults(data.result);
      switchView('plan');
      loadRecentHistory();
    } else {
      const errors = data.errors || [data.message || 'Error occurred during assessment evaluation.'];
      showValidationAlert(errors);
      switchView('assessment');
    }
  } catch (err) {
    console.error('Prediction request failed:', err);
    showValidationAlert(['Server connection error. Please verify the Flask backend is running.']);
    switchView('assessment');
  } finally {
    if (submitBtn) {
      submitBtn.innerHTML = '<span>⚡ Run Supervised ML Readiness Evaluation</span>';
      submitBtn.disabled = false;
    }
  }
}

function renderPredictionResults(result) {
  const pred = result.prediction;
  const comp = result.completeness;
  const elig = result.eligibility;
  const explain = result.explainability;
  const gaps = result.gap_analysis;
  const radar = result.radar_scores;
  const roadmap = result.roadmap;
  const readiness = result.readiness_assessment;

  // Header meta bar
  const headStu = document.getElementById('planHeaderStudentId');
  const headMeta = document.getElementById('planHeaderMeta');
  if (headStu) headStu.innerText = `Student: ${pred.student_id || 'LPU2026_CANDIDATE'}`;
  if (headMeta) headMeta.innerText = `${pred.branch || 'Computer Science & Engineering'} • Semester ${pred.semester || 6} • Target: ${pred.target_role || 'SDE'}`;

  // Hide empty state, show results
  const emptyState = document.getElementById('planEmptyState');
  const resultsSec = document.getElementById('assessmentResultsSection');
  if (emptyState) emptyState.style.display = 'none';
  if (resultsSec) resultsSec.style.display = 'block';

  // 1. Demo Model Estimate KPI (Whole percentage integer)
  const supOutcome = pred.supervised_placement_outcome || (pred.simulated_placement_prob >= 50.0 ? 'Likely Placed' : 'Placement Support Needed');
  const demoPct = pred.demo_model_estimate_pct !== undefined ? pred.demo_model_estimate_pct : Math.round(pred.supervised_placement_prob !== undefined && pred.supervised_placement_prob !== null ? pred.supervised_placement_prob : pred.simulated_placement_prob);

  const outcomeText = document.getElementById('kpiPlacementOutcomeText');
  const outcomeBadge = document.getElementById('kpiPlacementOutcomeBadge');
  const supProbEl = document.getElementById('kpiSupervisedPlacementProb');
  const modelStatusEl = document.getElementById('kpiPlacementModelStatus');

  if (outcomeText) outcomeText.innerText = supOutcome;
  if (outcomeBadge) {
    outcomeBadge.style.color = supOutcome === 'Likely Placed' ? '#10b981' : '#ef4444';
    outcomeBadge.style.borderColor = supOutcome === 'Likely Placed' ? '#10b981' : '#ef4444';
    outcomeBadge.style.background = supOutcome === 'Likely Placed' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)';
  }
  if (supProbEl) supProbEl.innerText = `${demoPct}%`;
  if (modelStatusEl) {
    modelStatusEl.innerText = pred.demo_model_estimate_notice || 'Generated by a model trained on synthetic profiles. Not validated for real student placement outcomes.';
  }

  // 2. Canonical 3-Tier Readiness Assessment KPI
  const scoreValEl = document.getElementById('kpiReadinessScoreValue');
  const bandTextEl = document.getElementById('kpiReadinessBandText');
  const bandBadgeEl = document.getElementById('kpiReadinessBandBadge');

  const canonicalCategory = pred.readiness_category || (readiness && readiness.readiness_category) || (readiness && readiness.readiness_band) || 'Developing';
  const totalScore = readiness ? readiness.total_score : 65.0;

  if (scoreValEl) scoreValEl.innerText = `${totalScore} / 100`;
  if (bandTextEl) bandTextEl.innerText = canonicalCategory;
  if (bandBadgeEl) {
    const bClass = canonicalCategory === 'Strong preparation' ? 'tier-advanced' : canonicalCategory === 'Developing' ? 'tier-developing' : 'tier-foundation';
    bandBadgeEl.className = 'tier-badge ' + bClass;
  }
  if (readiness) {
    renderReadinessBars(readiness);
  }

  // 3. Input Completeness KPI
  const compEl = document.getElementById('kpiCompleteness');
  if (compEl) {
    compEl.innerText = `${comp.completeness_pct}%`;
  }
  const compSubtext = document.getElementById('kpiCompletenessSubtext');
  if (compSubtext) {
    compSubtext.innerText = comp.explanation || 'Percentage of required assessment fields supplied. This is not model confidence or placement likelihood.';
  }

  // 4. Institutional Eligibility Check KPI
  const eligEl = document.getElementById('kpiEligibility');
  if (eligEl) {
    eligEl.innerText = elig.status;
    eligEl.style.color = elig.status === 'Fully Eligible' ? '#10b981' : elig.status === 'Partially Eligible' ? '#f59e0b' : '#ef4444';
  }
  const eligSubtext = document.getElementById('kpiEligibilitySubtext');
  if (eligSubtext) {
    eligSubtext.innerText = `Clearance: ~${elig.estimated_drive_clearance_pct}% of campus recruitment drives`;
  }

  // 5. Target Role Alignment KPI
  const roleEl = document.getElementById('kpiTargetRoleDisplay');
  if (roleEl) {
    roleEl.innerText = pred.target_role;
  }
  const roleFitEl = document.getElementById('kpiRoleFitScore');
  if (roleFitEl) {
    const targetFit = pred.top_roles.find(r => r.is_student_target) || pred.top_roles[0];
    roleFitEl.innerText = `Model Fit Score: ${targetFit ? targetFit.fit_score : '--'}% (Recommended: ${pred.recommended_role})`;
  }

  // 6. Plain Language Explanation
  const descEl = document.getElementById('tierDescriptionText');
  if (descEl) {
    descEl.innerText = pred.tier_description;
  }

  const eligWarning = document.getElementById('eligibilityWarningBox');
  if (eligWarning) {
    if (elig.unmet_conditions && elig.unmet_conditions.length > 0) {
      eligWarning.style.display = 'block';
      eligWarning.innerHTML = `⚠️ <strong>Recruitment Eligibility Notice:</strong> ${elig.unmet_conditions.join(' ')}`;
    } else {
      eligWarning.style.display = 'none';
    }
  }

  // 7. Mentor Feedback Section (if present)
  renderMentorFeedbackNotes(result.mentor_feedback);

  // 8. Tree Path Feature Attribution List
  renderTreePathAttribution(explain);

  // 9. 7-Pillar Competency Radar Chart
  renderRadarChart(radar);

  // 10. Skill Gap Table
  renderGapTable(gaps, radar);

  // 11. Actionable 30/60/90-Day Roadmap
  renderRoadmapTimeline(roadmap);

  // 12. Secondary Experimental Simulation Values
  const expSim = pred.experimental_simulated_ctc || {};
  const simProb = document.getElementById('kpiPlacementProb');
  if (simProb) simProb.innerText = `${pred.simulated_placement_prob}%`;
  const simCtc = document.getElementById('kpiExpectedCTC');
  if (simCtc) simCtc.innerText = `₹${expSim.expected_ctc_lpa || pred.simulated_expected_ctc_lpa} LPA`;
  const simRange = document.getElementById('kpiCTCRange');
  if (simRange) simRange.innerText = `Simulated Bracket: ${expSim.ctc_bracket || pred.simulated_ctc_bracket}`;
}

function renderReadinessBars(readiness) {
  const container = document.getElementById('readinessBarsContainer');
  const missingNotice = document.getElementById('readinessMissingNotice');
  if (!container || !readiness || !readiness.breakdown) return;

  container.innerHTML = '';
  const colors = {
    technical: '#6366f1',
    aptitude: '#3b82f6',
    communication: '#8b5cf6',
    projects: '#10b981',
    internships: '#f59e0b'
  };

  for (const [key, b] of Object.entries(readiness.breakdown)) {
    const barItem = document.createElement('div');
    barItem.className = 'readiness-bar-item';
    const color = colors[key] || '#6366f1';
    barItem.innerHTML = `
      <div class="readiness-bar-header">
        <span>${b.name} (${b.weight_pct}% weight)</span>
        <span>Subscore: <strong>${b.subscore} / 100</strong> → Contribution: <strong>${b.contribution} pts</strong></span>
      </div>
      <div class="readiness-bar-track">
        <div class="readiness-bar-fill" style="width: ${b.subscore}%; background: ${color};"></div>
      </div>
    `;
    container.appendChild(barItem);
  }

  if (missingNotice) {
    if (readiness.missing_fields && readiness.missing_fields.length > 0) {
      missingNotice.style.display = 'block';
      missingNotice.innerHTML = `ℹ️ <em>Self-reported notes: Unassessed or missing fields evaluated on baseline neutral values (${readiness.missing_fields.join(', ')}).</em>`;
    } else {
      missingNotice.style.display = 'none';
    }
  }
}

function renderMentorFeedbackNotes(feedbackList) {
  const section = document.getElementById('mentorFeedbackSection');
  const listContainer = document.getElementById('mentorFeedbackListContainer');
  if (!section || !listContainer) return;

  if (feedbackList && feedbackList.length > 0) {
    section.style.display = 'block';
    listContainer.innerHTML = '';
    feedbackList.forEach(fb => {
      const item = document.createElement('div');
      item.style.padding = '0.75rem';
      item.style.background = 'var(--bg-glass)';
      item.style.borderRadius = 'var(--radius-sm)';
      item.style.border = '1px solid var(--border-glass)';
      item.innerHTML = `
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #a5b4fc; font-weight: 700; margin-bottom: 0.25rem;">
          <span>👤 ${fb.mentor_username}</span>
          <span>${fb.created_at || 'Recent'}</span>
        </div>
        <div style="font-size: 0.88rem; color: var(--text-primary); margin-bottom: 0.25rem;">
          ${fb.feedback_notes}
        </div>
        ${fb.recommended_focus ? `<div style="font-size: 0.78rem; color: var(--lpu-orange);">🎯 Recommended Focus: <strong>${fb.recommended_focus}</strong></div>` : ''}
      `;
      listContainer.appendChild(item);
    });
  } else {
    section.style.display = 'none';
  }
}

function setupMentorHubListeners() {
  const btnRefresh = document.getElementById('btnRefreshMentorRoster');
  const bandFilter = document.getElementById('mentorBandFilter');
  const feedbackForm = document.getElementById('mentorFeedbackForm');

  if (btnRefresh) {
    btnRefresh.onclick = loadMentorStudents;
  }

  if (bandFilter) {
    bandFilter.onchange = () => {
      filterMentorStudents(bandFilter.value);
    };
  }

  if (feedbackForm) {
    feedbackForm.onsubmit = async (e) => {
      e.preventDefault();
      await submitMentorFeedback();
    };
  }
}

async function loadMentorStudents() {
  const tbody = document.getElementById('mentorStudentsTableBody');
  const selectStu = document.getElementById('feedbackStudentId');
  if (!tbody) return;

  tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">Loading assigned student roster...</td></tr>';

  try {
    const headers = state.token ? { 'Authorization': `Bearer ${state.token}` } : {};
    const res = await fetch('/api/mentor/students', { headers });
    const data = await res.json();

    if (data.status === 'success' && data.students) {
      state.assignedStudents = data.students;
      renderMentorStudentsTable(data.students);

      if (selectStu) {
        selectStu.innerHTML = '';
        data.students.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s.student_id;
          opt.textContent = `${s.student_id} (${s.target_role || s.branch})`;
          selectStu.appendChild(opt);
        });
      }
    } else {
      tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No assigned students found on roster.</td></tr>';
    }
  } catch (err) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #ef4444; padding: 1.5rem;">Failed to load roster from backend.</td></tr>';
  }
}

function filterMentorStudents(band) {
  if (band === 'ALL') {
    renderMentorStudentsTable(state.assignedStudents);
  } else {
    const filtered = state.assignedStudents.filter(s => (s.readiness_category || s.readiness_band) === band);
    renderMentorStudentsTable(filtered);
  }
}

function renderMentorStudentsTable(students) {
  const tbody = document.getElementById('mentorStudentsTableBody');
  if (!tbody) return;

  if (students.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No students match the selected filter.</td></tr>';
    return;
  }

  tbody.innerHTML = '';
  students.forEach(s => {
    const tr = document.createElement('tr');
    const category = s.readiness_category || s.readiness_band || 'Developing';
    const bandBadge = category === 'Strong preparation' ? 'tier-advanced' : category === 'Developing' ? 'tier-developing' : 'tier-foundation';
    const demoEstimate = s.demo_model_estimate !== undefined ? s.demo_model_estimate : Math.round(s.placement_prob || 0);

    tr.innerHTML = `
      <td><strong>${s.student_id}</strong></td>
      <td>${s.branch}</td>
      <td>${s.target_role || '--'}</td>
      <td style="font-weight: 700;">${s.cgpa ? s.cgpa.toFixed(2) : '--'}</td>
      <td><span class="tier-badge ${bandBadge}" style="font-size: 0.72rem; padding: 0.15rem 0.5rem;">${category}</span></td>
      <td style="font-weight: 800; color: #6366f1;">${s.readiness_score ? s.readiness_score.toFixed(1) : '--'} / 100</td>
      <td style="color: #10b981; font-weight: 700;">${demoEstimate}%</td>
      <td>
        <button type="button" class="btn-secondary" style="font-size: 0.75rem; padding: 0.25rem 0.55rem;" onclick="loadStudentAssessmentIntoView('${s.student_id}', ${s.latest_assessment_id || 'null'})">
          View Report
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadStudentAssessmentIntoView(studentId, assessmentId) {
  if (assessmentId) {
    try {
      const headers = state.token ? { 'Authorization': `Bearer ${state.token}` } : {};
      const res = await fetch(`/api/report/${assessmentId}`, { headers });
      const data = await res.json();
      if (data.status === 'success' && data.report) {
        state.currentPrediction = data.report.prediction_result;
        renderPredictionResults(data.report.prediction_result);
        switchView('plan');
        return;
      }
    } catch (e) {}
  }
  
  const preset = state.presets.find(p => p.data.student_id === studentId);
  if (preset) {
    applyPresetAndEvaluate(preset.preset_id);
  } else {
    switchView('history');
    const input = document.getElementById('historySearchInput');
    if (input) {
      input.value = studentId;
      document.getElementById('btnSearchHistory').click();
    }
  }
}

async function submitMentorFeedback() {
  const stuId = document.getElementById('feedbackStudentId').value;
  const focus = document.getElementById('feedbackRecommendedFocus').value;
  const notes = document.getElementById('feedbackNotesText').value.trim();
  const alertBox = document.getElementById('feedbackStatusAlert');

  if (!notes) {
    if (alertBox) {
      alertBox.style.color = '#ef4444';
      alertBox.innerText = 'Please enter counseling feedback notes.';
    }
    return;
  }

  try {
    const headers = { 'Content-Type': 'application/json' };
    if (state.token) headers['Authorization'] = `Bearer ${state.token}`;

    const res = await fetch('/api/mentor/feedback', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        assessment_id: state.currentPrediction ? (state.currentPrediction.assessment_history_id || 0) : 0,
        student_id: stuId,
        feedback_notes: notes,
        recommended_focus: focus
      })
    });

    const data = await res.json();
    if (data.status === 'success') {
      if (alertBox) {
        alertBox.style.color = '#10b981';
        alertBox.innerText = '✅ Feedback logged successfully for ' + stuId;
      }
      document.getElementById('feedbackNotesText').value = '';
    } else {
      if (alertBox) {
        alertBox.style.color = '#ef4444';
        alertBox.innerText = data.message || 'Failed to record feedback.';
      }
    }
  } catch (err) {
    if (alertBox) {
      alertBox.style.color = '#ef4444';
      alertBox.innerText = 'Network error while saving feedback.';
    }
  }
}

function renderTreePathAttribution(explain) {
  const driversHeader = document.getElementById('attributionDriversHeader');
  const mitigatorsHeader = document.getElementById('attributionMitigatorsHeader');
  if (driversHeader) driversHeader.innerText = explain.driver_label || '▲ Primary Factors:';
  if (mitigatorsHeader) mitigatorsHeader.innerText = explain.mitigator_label || '▼ Counter-Factors:';

  const posContainer = document.getElementById('shapPositiveDrivers');
  const negContainer = document.getElementById('shapNegativeBottlenecks');
  if (!posContainer || !negContainer) return;

  posContainer.innerHTML = '';
  negContainer.innerHTML = '';

  const isAtRisk = explain.is_at_risk_tier;

  explain.primary_drivers.forEach(item => {
    posContainer.appendChild(createAttributionElement(item, isAtRisk ? 'negative' : 'positive'));
  });

  explain.secondary_mitigators.forEach(item => {
    negContainer.appendChild(createAttributionElement(item, isAtRisk ? 'positive' : 'negative'));
  });
}

function createAttributionElement(item, sentiment) {
  const el = document.createElement('div');
  el.className = `attribution-item ${sentiment}`;

  const barWidth = Math.min(100, Math.max(15, Math.abs(item.path_contribution * 300)));

  el.innerHTML = `
    <div class="attr-info">
      <div class="attr-name">${item.display_name}</div>
      <div class="attr-val">Input Value: <strong>${item.raw_value}</strong></div>
    </div>
    <div class="attr-bar-container">
      <div class="attr-bar ${sentiment}" style="width: ${barWidth}px;"></div>
      <div class="attr-score" style="color: ${sentiment === 'positive' ? '#10b981' : '#ef4444'};">
        ${item.path_contribution > 0 ? '+' : ''}${(item.path_contribution * 100).toFixed(1)}%
      </div>
    </div>
  `;
  return el;
}

function renderRadarChart(radar) {
  const canvas = document.getElementById('skillRadarCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (state.radarChart) {
    state.radarChart.destroy();
  }

  const isDark = state.theme === 'dark';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
  const labelColor = isDark ? '#9ca3af' : '#475569';

  state.radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: radar.labels,
      datasets: [
        {
          label: 'Student Assessed Score',
          data: radar.student_scores,
          backgroundColor: 'rgba(99, 102, 241, 0.25)',
          borderColor: '#6366f1',
          borderWidth: 2.5,
          pointBackgroundColor: '#818cf8',
          pointBorderColor: '#ffffff',
          pointHoverRadius: 6
        },
        {
          label: `Role Benchmark (${radar.target_role_name})`,
          data: radar.target_role_scores,
          backgroundColor: 'rgba(16, 185, 129, 0.12)',
          borderColor: '#10b981',
          borderWidth: 2,
          borderDash: [4, 4],
          pointBackgroundColor: '#10b981',
          pointHoverRadius: 5
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: {
            stepSize: 20,
            color: labelColor,
            backdropColor: 'transparent',
            font: { family: 'Outfit', size: 10 }
          },
          grid: { color: gridColor },
          angleLines: { color: gridColor },
          pointLabels: {
            color: isDark ? '#f3f4f6' : '#1f2937',
            font: { family: 'Outfit', size: 11, weight: '600' }
          }
        }
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: isDark ? '#f3f4f6' : '#1f2937',
            font: { family: 'Outfit', size: 11 }
          }
        }
      }
    }
  });

  const sub = document.getElementById('radarTargetSubtitle');
  if (sub) {
    sub.innerText = `Benchmarked against role expectations for ${radar.target_role_name}. Basis: ${radar.target_role_description}`;
  }
}

function renderGapTable(gaps, radar) {
  const tbody = document.getElementById('gapTableBody');
  const badgeEl = document.getElementById('targetTierBenchmarkBadge');
  if (!tbody) return;

  if (badgeEl) {
    badgeEl.innerText = `Benchmarked Role: ${gaps.target_role}`;
  }

  tbody.innerHTML = '';
  gaps.pillars_breakdown.forEach(row => {
    const tr = document.createElement('tr');
    const deltaVal = parseFloat(row.delta);
    const deltaColor = deltaVal >= 0 ? '#10b981' : deltaVal >= -3.0 ? '#60a5fa' : deltaVal >= -15.0 ? '#f59e0b' : '#ef4444';
    const deltaFormatted = `${deltaVal > 0 ? '+' : ''}${deltaVal.toFixed(1)} pts`;

    tr.innerHTML = `
      <td><strong>${row.pillar}</strong></td>
      <td>${row.student_score} pts</td>
      <td>${row.target_score} pts</td>
      <td style="font-family: var(--font-mono); font-weight: 700; color: ${deltaColor};">
        ${deltaFormatted}
      </td>
      <td>
        <span class="status-tag ${row.badge}">${row.status}</span>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function renderRoadmapTimeline(roadmap) {
  const container = document.getElementById('roadmapTimelineContainer');
  if (!container) return;

  container.innerHTML = '';
  roadmap.forEach(phase => {
    const card = document.createElement('div');
    card.className = 'roadmap-phase-card';
    card.innerHTML = `
      <div class="phase-header">
        <span class="phase-tag">${phase.phase}</span>
        <span style="font-size: 0.8rem; color: var(--text-muted);">${phase.focus}</span>
      </div>
      <div class="phase-title">${phase.title}</div>
      <ul class="phase-tasks">
        ${phase.milestones.map(m => `<li>${m}</li>`).join('')}
      </ul>
      <div class="phase-milestone">
        🎯 <strong>Target Deliverable:</strong> ${phase.target_deliverable}
      </div>
    `;
    container.appendChild(card);
  });
}

/* ==========================================================================
   5. TPO Cohort Analytics & Batch Processor
   ========================================================================== */

function setupTPOBatchProcessor() {
  const dropzone = document.getElementById('batchCsvDropzone');
  const fileInput = document.getElementById('batchCsvFileInput');
  if (!dropzone || !fileInput) return;

  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleBatchFileUpload(e.target.files[0]);
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '#6366f1';
  });
  dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = '';
  });
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '';
    if (e.dataTransfer.files.length > 0) {
      handleBatchFileUpload(e.dataTransfer.files[0]);
    }
  });
}

async function handleBatchFileUpload(file) {
  const statusEl = document.getElementById('tpoUploadStatus');
  const errorAlert = document.getElementById('tpoBatchErrorAlert');
  if (statusEl) statusEl.innerHTML = `<span style="color: var(--lpu-orange);">⏳ Uploading &amp; Processing ${file.name}...</span>`;
  if (errorAlert) errorAlert.style.display = 'none';

  const formData = new FormData();
  formData.append('batch_file', file);

  try {
    const res = await fetch('/api/batch-evaluate', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    if (res.status === 200 && data.status === 'success') {
      if (statusEl) statusEl.innerHTML = `<span style="color: #10b981;">✅ Processed ${data.analytics.total_records_evaluated} records successfully!</span>`;
      renderTPOAnalytics(data.analytics, data.students);
    } else {
      if (statusEl) statusEl.innerHTML = '';
      if (errorAlert) {
        errorAlert.style.display = 'block';
        errorAlert.innerHTML = `<strong>⚠️ Processing Error:</strong> ${data.message || 'Failed to process CSV.'}`;
      }
    }
  } catch (err) {
    if (statusEl) statusEl.innerHTML = '';
    if (errorAlert) {
      errorAlert.style.display = 'block';
      errorAlert.innerHTML = 'Network connection failed during CSV upload.';
    }
  }
}

function renderTPOAnalytics(analytics, students) {
  const dashboard = document.getElementById('tpoAnalyticsDashboard');
  if (!dashboard) return;
  dashboard.style.display = 'block';

  // KPIs
  document.getElementById('tpoTotalStudents').innerText = analytics.total_records_evaluated;
  document.getElementById('tpoInvalidRowCount').innerText = `${analytics.invalid_rows_count || 0} invalid rows excluded`;
  document.getElementById('tpoAvgCompleteness').innerText = `${analytics.average_completeness_pct}%`;
  document.getElementById('tpoAvgCTC').innerText = `₹${analytics.average_expected_ctc_lpa} LPA`;
  document.getElementById('tpoMedianCTC').innerText = `Median CTC: ₹${analytics.median_expected_ctc_lpa} LPA`;
  document.getElementById('tpoTopBottleneck').innerText = analytics.top_cohort_bottleneck;

  renderTPOTierChart(analytics.tier_distribution);
  renderTPOBranchChart(analytics.department_breakdown);
  renderTPOPriorityTable(students);
  renderTPOFullTable(students);
}

function renderTPOTierChart(dist) {
  const canvas = document.getElementById('tpoTierChartCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (state.tpoTierChart) state.tpoTierChart.destroy();

  const labels = Object.keys(dist);
  const data = Object.values(dist);

  const colors = {
    'Strong preparation': '#10b981',
    'Developing': '#3b82f6',
    'Needs attention': '#ef4444',
    'Advanced': '#a855f7',
    'Placement Ready': '#10b981',
    'Foundation': '#ef4444'
  };

  state.tpoTierChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: data,
        backgroundColor: labels.map(l => colors[l] || '#6366f1'),
        borderWidth: 2,
        borderColor: 'var(--bg-primary)'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { color: state.theme === 'dark' ? '#f3f4f6' : '#1f2937' }
        }
      }
    }
  });
}

function renderTPOBranchChart(deptBreakdown) {
  const canvas = document.getElementById('tpoBranchChartCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (state.tpoBranchChart) state.tpoBranchChart.destroy();

  const labels = Object.keys(deptBreakdown);
  const data = Object.values(deptBreakdown).map(d => d.count || d);

  state.tpoBranchChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels.map(l => l.replace('Engineering', 'Eng').replace('Computer Science &', 'CSE')),
      datasets: [{
        label: 'Students Enrolled',
        data: data,
        backgroundColor: 'rgba(99, 102, 241, 0.75)',
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: state.theme === 'dark' ? '#9ca3af' : '#475569', font: { size: 10 } } },
        y: { ticks: { color: state.theme === 'dark' ? '#9ca3af' : '#475569' } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

function renderTPOPriorityTable(students) {
  const tbody = document.getElementById('tpoPriorityTableBody');
  if (!tbody) return;

  const atRisk = students.filter(s => {
    const cat = (s.readiness_category || s.readiness_tier || '').toLowerCase();
    return cat.includes('needs attention') || cat.includes('foundation') || s.active_backlogs > 0;
  });

  tbody.innerHTML = '';
  if (atRisk.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 1rem;">No high-risk students flagged in this cohort batch.</td></tr>';
    return;
  }

  atRisk.forEach(s => {
    const tr = document.createElement('tr');
    const category = s.readiness_category || s.readiness_tier;
    const catBadge = category === 'Needs attention' || category === 'Foundation' ? 'tier-foundation' : 'tier-developing';
    tr.innerHTML = `
      <td><strong>${s.student_id}</strong></td>
      <td>${s.branch}</td>
      <td><span class="tier-badge ${catBadge}">${category}</span></td>
      <td style="color: ${s.active_backlogs > 0 ? '#ef4444' : '#10b981'}; font-weight: 700;">${s.active_backlogs}</td>
      <td>${s.cgpa.toFixed(2)}</td>
      <td style="color: #ef4444; font-weight: 600;">${s.primary_deficit || 'Core Programming Deficit'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderTPOFullTable(students) {
  const tbody = document.getElementById('tpoStudentTableBody');
  if (!tbody) return;

  tbody.innerHTML = '';
  students.forEach(s => {
    const tr = document.createElement('tr');
    const category = s.readiness_category || s.readiness_tier;
    const catBadge = category === 'Strong preparation' || category === 'Advanced' ? 'tier-advanced' : category === 'Developing' ? 'tier-developing' : 'tier-foundation';
    tr.innerHTML = `
      <td><strong>${s.student_id}</strong></td>
      <td>${s.branch}</td>
      <td>${s.target_role || '--'}</td>
      <td>${s.cgpa ? s.cgpa.toFixed(2) : '--'}</td>
      <td>${s.dsa_problems_solved || 0}</td>
      <td><span class="tier-badge ${catBadge}">${category}</span></td>
      <td>${s.completeness_pct ? Math.round(s.completeness_pct) : 100}%</td>
      <td style="color: ${s.eligibility_status === 'Fully Eligible' ? '#10b981' : '#f59e0b'}; font-weight: 600;">${s.eligibility_status || 'Eligible'}</td>
    `;
    tbody.appendChild(tr);
  });
}

/* ==========================================================================
   6. Assessment History & Progress Tracking
   ========================================================================== */

function setupHistorySearch() {
  const searchBtn = document.getElementById('btnSearchHistory');
  const input = document.getElementById('historySearchInput');
  const catFilter = document.getElementById('historyCategoryFilter');

  if (searchBtn && input) {
    searchBtn.addEventListener('click', () => {
      const q = input.value.trim();
      const cat = catFilter ? catFilter.value : 'ALL';
      if (q) queryStudentHistory(q);
      else loadRecentHistory(cat);
    });
    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        const q = input.value.trim();
        const cat = catFilter ? catFilter.value : 'ALL';
        if (q) queryStudentHistory(q);
        else loadRecentHistory(cat);
      }
    });
  }

  if (catFilter) {
    catFilter.addEventListener('change', () => {
      loadRecentHistory(catFilter.value);
    });
  }
}

async function queryStudentHistory(studentId) {
  try {
    const res = await fetch(`/api/history/${encodeURIComponent(studentId)}`);
    const data = await res.json();
    if (data.status === 'success') {
      renderHistoryTable(data.history, `Chronological History for Student ID: ${studentId}`);
    }
  } catch (err) {
    console.error('History query failed:', err);
  }
}

async function loadRecentHistory(category = 'ALL') {
  try {
    const url = category && category !== 'ALL' ? `/api/history?category=${encodeURIComponent(category)}` : '/api/history';
    const res = await fetch(url);
    const data = await res.json();
    if (data.status === 'success') {
      renderHistoryTable(data.recent_assessments, 'Recent Evaluations Saved to SQLite Store');
    }
  } catch (err) {
    console.error('Recent history query failed:', err);
  }
}

function renderHistoryTable(records, title) {
  const tbody = document.getElementById('historyTableBody');
  if (!tbody) return;

  tbody.innerHTML = '';
  if (!records || records.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No historical assessment records found matching criteria.</td></tr>';
    return;
  }

  records.forEach((r, idx) => {
    const tr = document.createElement('tr');
    const category = r.readiness_category || r.readiness_tier || 'Developing';
    const catBadge = category === 'Strong preparation' ? 'tier-advanced' : category === 'Developing' ? 'tier-developing' : 'tier-foundation';

    tr.innerHTML = `
      <td>#${r.id || (idx + 1)}</td>
      <td style="font-family: var(--font-mono); font-size: 0.8rem;">${r.timestamp}</td>
      <td><strong>${r.student_id}</strong></td>
      <td>${r.branch}</td>
      <td>${r.target_role || '--'}</td>
      <td>${r.cgpa ? r.cgpa.toFixed(2) : '--'}</td>
      <td><span class="tier-badge ${catBadge}">${category}</span></td>
      <td>${r.completeness_pct ? Math.round(r.completeness_pct) : 100}%</td>
      <td>
        <button type="button" class="btn-secondary" style="font-size: 0.75rem; padding: 0.25rem 0.55rem;" onclick="loadStudentAssessmentIntoView('${r.student_id}', ${r.id})">
          View Plan
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

/* ==========================================================================
   7. Model Performance & Governance
   ========================================================================== */

async function loadModelPerformance() {
  try {
    const res = await fetch('/api/model-performance');
    const data = await res.json();
    if (data.status === 'success') {
      state.modelMetadata = data.metadata;
      renderGovernanceBenchmarks(data.metadata);
    }
  } catch (err) {
    console.error('Failed to load model benchmarks:', err);
  }
}

function renderGovernanceBenchmarks(meta) {
  const clfTbody = document.getElementById('clfBenchmarkTableBody');
  const regTbody = document.getElementById('regBenchmarkTableBody');

  if (clfTbody && meta.classification_models_benchmark) {
    clfTbody.innerHTML = '';
    for (const [name, m] of Object.entries(meta.classification_models_benchmark)) {
      const isBest = name === meta.best_classifier_name;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>
          <strong>${name}</strong>
          ${isBest ? '<span class="badge-best">CHAMPION</span>' : ''}
        </td>
        <td style="font-weight: 700; color: #10b981;">${(m.accuracy * 100).toFixed(2)}%</td>
        <td>${m.macro_f1.toFixed(4)}</td>
        <td>${m.weighted_f1 ? m.weighted_f1.toFixed(4) : '--'}</td>
        <td>${(m.cv_accuracy_mean * 100).toFixed(2)}% (±${(m.cv_accuracy_std * 100).toFixed(2)}%)</td>
        <td>${m.cv_macro_f1_mean ? m.cv_macro_f1_mean.toFixed(4) : '--'}</td>
      `;
      clfTbody.appendChild(tr);
    }
  }

  if (regTbody && meta.regression_models_benchmark) {
    regTbody.innerHTML = '';
    for (const [name, m] of Object.entries(meta.regression_models_benchmark)) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${name}</strong></td>
        <td style="color: #6366f1; font-weight: 700;">${m.r2_score.toFixed(4)}</td>
        <td>${m.rmse.toFixed(2)} LPA</td>
        <td>${m.mae.toFixed(2)} LPA</td>
      `;
      regTbody.appendChild(tr);
    }
  }

  // Feature Importance Chart (Sorted strictly descending)
  if (meta.sorted_feature_importances) {
    renderFeatureImportanceChart(meta.sorted_feature_importances);
  } else if (meta.feature_importances) {
    renderFeatureImportanceChart(meta.feature_importances);
  }
}

function renderFeatureImportanceChart(featData) {
  const canvas = document.getElementById('featureImportanceCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (state.featImpChart) state.featImpChart.destroy();

  // Normalize into strictly descending sorted list
  let sortedList = [];
  if (Array.isArray(featData)) {
    sortedList = [...featData].sort((a, b) => b.relative_pct - a.relative_pct).slice(0, 12);
  } else {
    sortedList = Object.entries(featData)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12)
      .map(([k, v]) => ({
        feature: k,
        clean_name: k.replace(/_/g, ' '),
        importance: v,
        relative_pct: +(v * 100).toFixed(2)
      }));
  }

  const friendlyNames = {
    'active_backlogs': 'Active Backlogs Count',
    'dsa_score': 'DSA Assessment Score',
    'coding_rating': 'Competitive Coding Rating',
    'cgpa': 'University CGPA',
    'dsa_problems_solved': 'DSA Problems Solved',
    'system_design_score': 'System Design Proficiency',
    'mock_interview_score': 'Mock Interview Rating',
    'quantitative_aptitude': 'Quantitative Aptitude',
    'logical_reasoning': 'Logical Reasoning',
    'communication_rating': 'Spoken Communication',
    'web_cloud_score': 'Web & Cloud Frameworks',
    'github_quality': 'Project Quality Rubric',
    'projects_count': 'Capstone Projects Count',
    'internships_count': 'Completed Internships',
    'verbal_ability': 'Verbal & Language Ability'
  };

  const domainMap = {
    'active_backlogs': 'Academic Eligibility',
    'cgpa': 'Academic Record',
    'dsa_score': 'Technical Preparation',
    'coding_rating': 'Technical Preparation',
    'dsa_problems_solved': 'Technical Preparation',
    'system_design_score': 'Technical Preparation',
    'web_cloud_score': 'Technical Preparation',
    'mock_interview_score': 'Interview Readiness',
    'quantitative_aptitude': 'Aptitude & Problem Solving',
    'logical_reasoning': 'Aptitude & Problem Solving',
    'verbal_ability': 'Aptitude & Problem Solving',
    'communication_rating': 'Soft Skills & Communication',
    'github_quality': 'Projects & Experience',
    'projects_count': 'Projects & Experience',
    'internships_count': 'Projects & Experience'
  };

  const labels = sortedList.map(item => friendlyNames[item.feature] || item.clean_name || item.feature.replace(/_/g, ' '));
  const values = sortedList.map(item => item.relative_pct !== undefined ? item.relative_pct : +(item.importance * 100).toFixed(2));

  const isDark = state.theme === 'dark';

  state.featImpChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Relative Importance (%)',
        data: values,
        backgroundColor: sortedList.map((p, idx) => idx === 0 ? 'rgba(249, 115, 22, 0.85)' : 'rgba(99, 102, 241, 0.75)'),
        borderColor: sortedList.map((p, idx) => idx === 0 ? '#f97316' : '#6366f1'),
        borderWidth: 1,
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: 'Relative Importance (%)',
            color: isDark ? '#9ca3af' : '#475569',
            font: { family: 'Outfit', size: 12, weight: '600' }
          },
          ticks: {
            color: isDark ? '#9ca3af' : '#475569',
            callback: (v) => `${v}%`
          },
          grid: { color: isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.08)' }
        },
        y: {
          ticks: {
            color: isDark ? '#f3f4f6' : '#1f2937',
            font: { family: 'Outfit', size: 11, weight: '600' }
          },
          grid: { display: false }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => ` Relative Importance: ${c.raw}%`
          }
        }
      }
    }
  });

  // Populate accessible table view
  const tableBody = document.getElementById('accessibleFeatTableBody');
  if (tableBody) {
    tableBody.innerHTML = '';
    sortedList.forEach((item, idx) => {
      const featKey = item.feature;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight: 700; color: ${idx === 0 ? 'var(--lpu-orange)' : 'var(--text-primary)'};">#${idx + 1}</td>
        <td><strong>${friendlyNames[featKey] || featKey.replace(/_/g, ' ')}</strong> <code>(${featKey})</code></td>
        <td style="font-family: var(--font-mono); font-weight: 700; color: #10b981;">${item.relative_pct}%</td>
        <td><span class="spec-pill">${domainMap[featKey] || 'General'}</span></td>
      `;
      tableBody.appendChild(tr);
    });
  }
}

function refreshAllCharts() {
  if (state.currentPrediction && state.currentPrediction.radar_scores) {
    renderRadarChart(state.currentPrediction.radar_scores);
  }
  if (state.modelMetadata && state.modelMetadata.sorted_feature_importances) {
    renderFeatureImportanceChart(state.modelMetadata.sorted_feature_importances);
  } else if (state.modelMetadata && state.modelMetadata.feature_importances) {
    renderFeatureImportanceChart(state.modelMetadata.feature_importances);
  }
}
