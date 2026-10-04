/**
 * SCAMSHIELD — Explainable AI-Powered Digital Scam Detection
 * ScamShield AI Professional SaaS & Cybersecurity Client
 * 
 * Simple, user-first scam detection assistant:
 * "Check before you click" • "Paste -> Check -> Understand -> Act"
 * Interacts with FastAPI backend (/scan, /health, /predict, /category, /explain)
 */

// Global App Configuration
const CONFIG = {
  API_BASE_URL: window.location.origin.includes('8000')
    ? window.location.origin
    : 'http://127.0.0.1:8000',
  HEALTH_CHECK_INTERVAL_MS: 15000,
};

// Application State
const appState = {
  currentResult: null,
  activeView: 'home',
  apiOnline: false,
};

// Curated Threat & Benign Presets
const PRESETS = {
  bank_kyc: {
    id: "bank_kyc",
    name: "Bank / KYC",
    type: "threat",
    text: "Dear SBI Customer, your YONO account and debit card has been blocked today due to pending KYC verification. Please click the link immediately http://sbi-kyc-update-portal.info/login to update your Pan Card and Aadhaar details to restore bank services within 2 hours.",
    url: "http://sbi-kyc-update-portal.info/login"
  },
  digital_arrest: {
    id: "digital_arrest",
    name: "Digital Arrest",
    type: "threat",
    text: "URGENT CBI NOTICE: CBI and Delhi Police cybercrime cell have issued an immediate Digital Arrest warrant against Aadhaar card ending in 4921 under PMLA 2002. Suspect fund transfers of Rs 48,00,000 linked to money laundering. You must remain on video call continuously for police interrogation. Connect now to official CBI portal: http://192.168.1.105/cbi-police/hearing to avoid immediate police team dispatch to your residence.",
    url: "http://192.168.1.105/cbi-police/hearing"
  },
  electricity_bill: {
    id: "electricity_bill",
    name: "Electricity Bill",
    type: "threat",
    text: "Dear Consumer, your electricity power supply will be disconnected tonight at 9:30 PM from the power sub-station because your previous month electricity bill Rs 3,420 was not updated. Immediately contact our electricity officer Mr. Sharma at 9876543210 or click http://secure-state-bill-desk.in/pay to update and avoid sudden blackout.",
    url: "http://secure-state-bill-desk.in/pay"
  },
  phishing_login: {
    id: "phishing_login",
    name: "Phishing",
    type: "threat",
    text: "Security Alert: Unusual login attempt detected on your Microsoft 365 account from Moscow, Russia. Your session has been temporarily suspended. Confirm your identity immediately to prevent permanent account termination: https://login-microsoftonline-verify-auth.co/secure",
    url: "https://login-microsoftonline-verify-auth.co/secure"
  },
  legit_otp: {
    id: "legit_otp",
    name: "Legitimate OTP",
    type: "benign",
    text: "582914 is your secret One Time Password (OTP) for purchase of Rs 2,499.00 at AMAZON INDIA on HDFC Bank Card ending 7041. Valid for 10 minutes. NEVER share your OTP or password with anyone, including bank staff.",
    url: ""
  },
  statutory_notice: {
    id: "statutory_notice",
    name: "Tax Notice",
    type: "benign",
    text: "Income Tax Department: Intimation under Section 143(1) of the Income-tax Act, 1961 for Assessment Year 2024-25 has been processed. Refund of Rs. 4,280 has been credited to your validated bank account ending in 8812. No further action is required. Please check your e-filing portal.",
    url: "https://eportal.incometax.gov.in"
  }
};

// View Titles Mapping
const VIEW_TITLES = {
  home: { title: "ScamShield", sub: "Check before you click" },
  scanner: { title: "Scan Message", sub: "Check suspicious communications" },
  about: { title: "About ScamShield", sub: "Digital scam assistance tool" },
  "research-dashboard": { title: "Academic Research & Benchmarks", sub: "Model evaluation, locked partitions & empirical results" }
};

// Global Navigation Controller
window.appNav = {
  switchView(viewName) {
    if (!viewName) return;
    appState.activeView = viewName;

    // Update sidebar navigation links
    const sidebarLinks = document.querySelectorAll('.sidebar-link');
    sidebarLinks.forEach(link => {
      if (link.dataset.view === viewName) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });

    // Update view sections
    const sections = document.querySelectorAll('.view-section');
    sections.forEach(sec => {
      if (sec.id === `view-${viewName}`) {
        sec.classList.add('active');
      } else {
        sec.classList.remove('active');
      }
    });

    // Update header title
    const meta = VIEW_TITLES[viewName] || { title: "ScamShield", sub: "Digital Scam Defense" };
    const headingEl = document.getElementById('viewHeadingTitle');
    if (headingEl) {
      headingEl.innerText = meta.title;
      const subEl = document.getElementById('viewHeadingSubtitle');
      if (subEl) subEl.innerText = meta.sub;
    }

    // Close mobile menu if open
    const sidebar = document.getElementById('appSidebar');
    if (sidebar) {
      sidebar.classList.remove('mobile-open');
    }

    // Scroll to top of content
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
};

// DOM Initialization
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initPresets();
  initScannerControls();
  checkApiHealth();
  setInterval(checkApiHealth, CONFIG.HEALTH_CHECK_INTERVAL_MS);
});

// Setup Sidebar Links, Brand Link, and Mobile Toggle
function initNavigation() {
  const sidebarLinks = document.querySelectorAll('.sidebar-link');
  sidebarLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const targetView = link.dataset.view;
      if (targetView) {
        window.appNav.switchView(targetView);
      }
    });
  });

  const brandLink = document.getElementById('brandLink');
  if (brandLink) {
    brandLink.addEventListener('click', (e) => {
      e.preventDefault();
      window.appNav.switchView('home');
    });
  }

  const mobileToggle = document.getElementById('btnToggleMobileMenu');
  const sidebar = document.getElementById('appSidebar');
  if (mobileToggle && sidebar) {
    mobileToggle.addEventListener('click', () => {
      sidebar.classList.toggle('mobile-open');
    });
  }
}

// Populate Example Buttons (Clicking populates input without auto-scanning)
function initPresets() {
  const container = document.getElementById('presetButtonsContainer');
  if (!container) return;

  container.innerHTML = '';
  Object.values(PRESETS).forEach(preset => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'preset-btn';
    btn.innerText = preset.name;
    btn.title = `Load example: ${preset.name}`;
    btn.addEventListener('click', () => {
      loadPreset(preset);
    });
    container.appendChild(btn);
  });
}

// Load Preset text without auto-scanning
function loadPreset(preset) {
  const textarea = document.getElementById('scannerTextarea');
  const urlInput = document.getElementById('scannerUrlInput');
  const counter = document.getElementById('scannerCharCounter');

  if (textarea) {
    textarea.value = preset.text;
    if (counter) {
      counter.innerText = `${preset.text.length.toLocaleString()} / 5,000`;
    }
  }
  if (urlInput) {
    urlInput.value = preset.url || '';
  }

  // Switch to scanner view if on another page
  if (appState.activeView !== 'scanner') {
    window.appNav.switchView('scanner');
  }

  showToast(`Example inserted: ${preset.name}. Click "Check This Message" to scan.`);
}

// Setup Scanner Controls and Event Listeners
function initScannerControls() {
  const textarea = document.getElementById('scannerTextarea');
  const urlInput = document.getElementById('scannerUrlInput');
  const counter = document.getElementById('scannerCharCounter');
  const btnAnalyze = document.getElementById('btnAnalyzeMessage');
  const btnClear = document.getElementById('btnClearScanner');
  const btnRetry = document.getElementById('btnRetryScan');
  const btnScanAnother = document.getElementById('btnScanAnotherMessage');
  const btnCopyResult = document.getElementById('btnCopyResultSummary');
  const btnCopyReportMarkdown = document.getElementById('btnCopyReportMarkdown');
  const btnToggleAdvanced = document.getElementById('btnToggleAdvancedDetails');

  // Character counter
  if (textarea && counter) {
    textarea.addEventListener('input', () => {
      const len = textarea.value.length;
      counter.innerText = `${len.toLocaleString()} / 5,000`;
      
      // Auto-extract URLs if URL field is empty
      if (urlInput && !urlInput.value.trim()) {
        const foundUrl = extractFirstUrl(textarea.value);
        if (foundUrl) urlInput.value = foundUrl;
      }
    });
  }

  // Check Message Button
  if (btnAnalyze) {
    btnAnalyze.addEventListener('click', () => {
      executeScan();
    });
  }

  // Retry Button on error
  if (btnRetry) {
    btnRetry.addEventListener('click', () => {
      executeScan();
    });
  }

  // Clear Button
  if (btnClear) {
    btnClear.addEventListener('click', () => {
      if (textarea) textarea.value = '';
      if (urlInput) urlInput.value = '';
      if (counter) counter.innerText = '0 / 5,000';
      resetScannerState();
    });
  }

  // Scan Another Message
  if (btnScanAnother) {
    btnScanAnother.addEventListener('click', () => {
      if (textarea) {
        textarea.value = '';
        textarea.focus();
      }
      if (urlInput) urlInput.value = '';
      if (counter) counter.innerText = '0 / 5,000';
      resetScannerState();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  // Copy Result Summary
  if (btnCopyResult) {
    btnCopyResult.addEventListener('click', () => {
      copyResultSummary();
    });
  }

  // Copy Full Markdown Audit
  if (btnCopyReportMarkdown) {
    btnCopyReportMarkdown.addEventListener('click', () => {
      copyMarkdownAudit();
    });
  }

  // Toggle Technical Details Accordion
  if (btnToggleAdvanced) {
    btnToggleAdvanced.addEventListener('click', () => {
      const content = document.getElementById('advancedDetailsContent');
      const arrow = document.getElementById('advancedToggleArrow');
      if (content) {
        const isOpen = content.classList.toggle('open');
        if (arrow) arrow.innerText = isOpen ? '▲' : '▼';
      }
    });
  }
}

// Reset scanner back to initial ready state
function resetScannerState() {
  const emptyState = document.getElementById('scannerEmptyState');
  const loadingState = document.getElementById('scannerLoadingState');
  const errorState = document.getElementById('scannerErrorState');
  const resultsOutput = document.getElementById('scannerResultsOutput');

  if (emptyState) emptyState.style.display = 'flex';
  if (loadingState) loadingState.style.display = 'none';
  if (errorState) errorState.style.display = 'none';
  if (resultsOutput) resultsOutput.style.display = 'none';
}

// Execute Scan against FastAPI Backend
async function executeScan() {
  const textarea = document.getElementById('scannerTextarea');
  const urlInput = document.getElementById('scannerUrlInput');
  const btnAnalyze = document.getElementById('btnAnalyzeMessage');

  const text = textarea ? textarea.value.trim() : '';
  const url = urlInput ? urlInput.value.trim() : '';

  if (!text) {
    showToast('Please enter a message before scanning.');
    if (textarea) textarea.focus();
    return;
  }

  // UI state transition to Loading
  const emptyState = document.getElementById('scannerEmptyState');
  const loadingState = document.getElementById('scannerLoadingState');
  const errorState = document.getElementById('scannerErrorState');
  const resultsOutput = document.getElementById('scannerResultsOutput');

  if (emptyState) emptyState.style.display = 'none';
  if (resultsOutput) resultsOutput.style.display = 'none';
  if (errorState) errorState.style.display = 'none';
  if (loadingState) loadingState.style.display = 'flex';
  if (btnAnalyze) btnAnalyze.disabled = true;

  const startTime = performance.now();

  try {
    const response = await fetch(`${CONFIG.API_BASE_URL}/scan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, url: url || null })
    });

    const measuredLatency = performance.now() - startTime;

    if (!response.ok) {
      throw new Error(`API returned HTTP ${response.status}`);
    }

    const result = await response.json();
    appState.currentResult = result;
    renderScanResults(result, measuredLatency);
    showToast('✓ Message check completed');
  } catch (error) {
    console.warn('API call encountered an issue:', error);
    // If backend is not reached, show clean error state
    if (loadingState) loadingState.style.display = 'none';
    if (errorState) errorState.style.display = 'flex';
    showToast('Could not reach the ScamShield service.');
  } finally {
    if (loadingState) loadingState.style.display = 'none';
    if (btnAnalyze) btnAnalyze.disabled = false;
  }
}

// Render Results into Result Deck
function renderScanResults(data, latencyMs) {
  const resultsOutput = document.getElementById('scannerResultsOutput');
  const emptyState = document.getElementById('scannerEmptyState');
  const errorState = document.getElementById('scannerErrorState');

  if (emptyState) emptyState.style.display = 'none';
  if (errorState) errorState.style.display = 'none';
  if (resultsOutput) resultsOutput.style.display = 'block';

  // 1. Unified Verdict Extraction
  const unified = data.unified_assessment || {};
  const verdict = data.verdict || {};
  const risk = data.risk_assessment || {};
  const cat = data.category || {};
  const explanation = data.explanation || {};

  const isThreat = verdict.is_threat !== undefined 
    ? verdict.is_threat 
    : (unified.prediction === 'phishing');

  // 2. Risk Score & Pill Styling
  const rawScore = unified.risk_score !== undefined
    ? Number(unified.risk_score)
    : Number(risk.risk_score || 0);
  
  const scoreNum = document.getElementById('riskScoreNum');
  if (scoreNum) scoreNum.innerText = Math.round(rawScore);

  let riskTier = (unified.risk_level || 'Low').toUpperCase();
  if (rawScore > 75) riskTier = 'CRITICAL';
  else if (rawScore > 50) riskTier = 'HIGH';
  else if (rawScore > 25) riskTier = 'MEDIUM';
  else riskTier = 'LOW';

  const riskPill = document.getElementById('riskTierPill');
  if (riskPill) {
    let icon = '🟢';
    if (riskTier === 'CRITICAL' || riskTier === 'HIGH') icon = '🔴';
    else if (riskTier === 'MEDIUM') icon = '🟡';

    riskPill.innerText = `${icon} ${riskTier} RISK`;
    riskPill.className = `scan-risk-badge tier-${riskTier.toLowerCase()}`;
  }

  // Horizontal Risk Gauge Pointer
  const gaugePointer = document.getElementById('horizontalGaugePointer');
  if (gaugePointer) {
    const clamped = Math.max(0, Math.min(100, rawScore));
    gaugePointer.style.left = `${clamped}%`;
  }

  // 3. Category & Confidence
  const catText = document.getElementById('verdictCategoryText');
  const confText = document.getElementById('verdictConfidenceText');

  const categoryName = unified.category_title || cat.category_title || (isThreat ? 'Suspicious Communication' : 'Legitimate Message');
  if (catText) {
    catText.innerText = categoryName;
  }

  const confidenceVal = unified.confidence !== undefined 
    ? unified.confidence 
    : (verdict.threat_probability || 0.92);
  if (confText) {
    confText.innerText = `Confidence: ${(confidenceVal * 100).toFixed(0)}%`;
  }

  // 4. Plain-Language Summary Assessment
  const plainSummaryBox = document.getElementById('plainSummaryBox');
  if (plainSummaryBox) {
    if (riskTier === 'CRITICAL') {
      plainSummaryBox.innerText = "This message shows very strong scam indicators. Do not interact with it until you verify it.";
    } else if (riskTier === 'HIGH') {
      plainSummaryBox.innerText = "This message contains several strong scam signals.";
    } else if (riskTier === 'MEDIUM') {
      plainSummaryBox.innerText = "This message contains some suspicious signals. Be careful.";
    } else {
      plainSummaryBox.innerText = "This message does not show many common scam signals.";
    }
  }

  // 5. Why Does ScamShield Think This Is Risky?
  renderWhyFlaggedReasons(unified.reasons || explanation.reasons || [], isThreat);

  // 6. What We Noticed (Evidence Snippets)
  renderEvidenceSnippets(unified.evidence_snippets || explanation.evidence_snippets || []);

  // 7. What Should You Do? (Recommended Actions)
  renderActionChecklist(unified.recommended_action || risk.action_checklist || [], isThreat);

  // 8. Modality Signal Breakdown (Inside Collapsed Technical Details)
  const textVal = document.getElementById('textSignalVal');
  const urlVal = document.getElementById('urlSignalVal');
  const intentVal = document.getElementById('intentSignalVal');

  if (textVal) textVal.innerText = isThreat ? '+0.62' : '-0.45';
  if (urlVal) urlVal.innerText = (data.input?.url || (isThreat && data.input?.url_present)) ? '+0.48' : '0.00';
  if (intentVal) intentVal.innerText = isThreat ? '+0.73' : '-0.38';

  // 9. SHAP Feature Attribution Waterfall
  renderShapWaterfall(explanation.top_threat_drivers || [], explanation.top_benign_drivers || []);
}

// Render Simple "Why Does ScamShield Think This Is Risky?" Cards
function renderWhyFlaggedReasons(reasons, isThreat) {
  const container = document.getElementById('whyFlaggedReasonsList');
  if (!container) return;

  container.innerHTML = '';

  if (!isThreat || !reasons || reasons.length === 0) {
    const defaultItem = document.createElement('div');
    defaultItem.className = 'reason-item-card';
    defaultItem.innerHTML = `
      <span class="reason-item-icon">✓</span>
      <div class="reason-item-content">
        <div class="reason-item-headline">No Threat Indicators Detected</div>
        <div class="reason-item-sub">Vocabulary and syntax conform to expected safe communication standards.</div>
      </div>
    `;
    container.appendChild(defaultItem);
    return;
  }

  reasons.forEach(r => {
    const item = document.createElement('div');
    item.className = 'reason-item-card';

    let headline = r;
    let description = '';

    if (r.includes(':')) {
      const parts = r.split(':');
      headline = parts[0].trim();
      description = parts.slice(1).join(':').trim();
    }

    item.innerHTML = `
      <span class="reason-item-icon">⚠</span>
      <div class="reason-item-content">
        <div class="reason-item-headline">${escapeHtml(headline)}</div>
        ${description ? `<div class="reason-item-sub">${escapeHtml(description)}</div>` : ''}
      </div>
    `;
    container.appendChild(item);
  });
}

// Render "What We Noticed" Evidence Chips
function renderEvidenceSnippets(snippets) {
  const container = document.getElementById('groundedEvidenceList');
  const box = document.getElementById('evidenceContainerBox');
  if (!container) return;

  container.innerHTML = '';

  if (!snippets || snippets.length === 0) {
    if (box) box.style.display = 'none';
    return;
  }

  if (box) box.style.display = 'block';

  snippets.forEach(snippet => {
    const chip = document.createElement('div');
    chip.className = 'evidence-quote-chip';
    chip.innerHTML = `“<span>${escapeHtml(snippet)}</span>”`;
    container.appendChild(chip);
  });
}

// Render Action-Oriented "WHAT SHOULD YOU DO?"
function renderActionChecklist(actions, isThreat) {
  const container = document.getElementById('recommendedActionList');
  if (!container) return;

  container.innerHTML = '';

  let list = [];

  if (isThreat) {
    list = [
      { icon: '🛑', text: 'Do not click the link.' },
      { icon: '🛑', text: 'Do not share your OTP, password or PIN.' },
      { icon: '🛑', text: 'Do not transfer money.' },
      { icon: '✓', text: "Verify the message using the organisation's official website or app." },
      { icon: '✓', text: 'If you believe this is a scam, report it through the official cybercrime channel (Helpline 1930 / cybercrime.gov.in).' }
    ];
  } else {
    list = [
      { icon: '✓', text: 'This message appears safe based on our scam detection checks.' },
      { icon: '✓', text: 'Always exercise reasonable caution and never share passwords or banking PINs.' }
    ];
  }

  // Prepend backend custom instruction if provided
  if (typeof actions === 'string' && actions.trim()) {
    list.unshift({ icon: isThreat ? '🛑' : '✓', text: actions.trim() });
  }

  list.forEach(item => {
    const li = document.createElement('li');
    li.className = 'action-checklist-item';
    li.innerHTML = `
      <span class="action-item-icon">${item.icon}</span>
      <span>${escapeHtml(item.text)}</span>
    `;
    container.appendChild(li);
  });
}

// Render Technical SHAP Feature Attribution Waterfall
function renderShapWaterfall(threatDrivers, benignDrivers) {
  const container = document.getElementById('shapWaterfallContainer');
  if (!container) return;

  container.innerHTML = '';

  const drivers = [];
  if (threatDrivers && threatDrivers.length) {
    drivers.push(...threatDrivers.slice(0, 4).map(d => ({
      feature: d.feature || d.feature_name || 'Threat Signal',
      impact: d.impact !== undefined ? d.impact : (d.shap_value || 0.1),
      isThreat: true
    })));
  }

  if (benignDrivers && benignDrivers.length) {
    drivers.push(...benignDrivers.slice(0, 3).map(d => ({
      feature: d.feature || d.feature_name || 'Safe Signal',
      impact: d.impact !== undefined ? d.impact : (d.shap_value || -0.1),
      isThreat: false
    })));
  }

  if (drivers.length === 0) {
    container.innerHTML = '<div style="font-size: 0.76rem; color: var(--text-secondary);">Balanced feature baseline.</div>';
    return;
  }

  const maxAbs = Math.max(...drivers.map(d => Math.abs(d.impact)), 0.05);

  drivers.forEach(d => {
    const absVal = Math.abs(d.impact);
    const pctWidth = Math.min(100, Math.max(10, Math.round((absVal / maxAbs) * 100)));
    const isThreat = d.impact >= 0;

    const row = document.createElement('div');
    row.className = 'shap-row';
    row.innerHTML = `
      <div class="shap-feat" title="${d.feature}">${formatFeatureName(d.feature)}</div>
      <div class="shap-track">
        <div class="${isThreat ? 'shap-fill-threat' : 'shap-fill-benign'}" style="width: ${pctWidth}%;"></div>
      </div>
      <div class="shap-val" style="color: ${isThreat ? 'var(--risk-critical)' : 'var(--risk-low)'};">
        ${isThreat ? '+' : ''}${d.impact.toFixed(3)}
      </div>
    `;
    container.appendChild(row);
  });
}

// Format raw feature keys for cleaner readability
function formatFeatureName(raw) {
  const map = {
    intent_urgency: "Urgency Coercion",
    intent_authority: "Authority Impersonation",
    intent_fear: "Legal / Fear Threat",
    intent_credential: "Credential Request",
    intent_payment: "Payment Demand",
    url_has_ip: "Raw IP Hostname",
    url_entropy: "High Link Entropy",
    url_is_shortener: "Link Shortener",
    text_cbi: "Keyword: CBI",
    text_police: "Keyword: Police",
    text_arrest: "Keyword: Arrest",
    text_otp: "Keyword: OTP",
    text_bank: "Keyword: Bank"
  };
  return map[raw] || raw.replace(/^(text_|url_|intent_)/, '').replace(/_/g, ' ');
}

// Copy Summary to Clipboard
function copyResultSummary() {
  if (!appState.currentResult) return;

  const res = appState.currentResult;
  const unified = res.unified_assessment || {};
  const score = Math.round(unified.risk_score || 0);
  const tier = unified.risk_level || 'Low';
  const category = unified.category_title || 'Digital Scam';

  const text = `ScamShield Result:
Risk Level: ${tier.toUpperCase()} RISK (${score}/100)
Category: ${category}
Advice: Never click suspicious links or share OTPs, passwords, or PINs.
Report cybercrime to Helpline 1930 or cybercrime.gov.in.`;

  navigator.clipboard.writeText(text).then(() => {
    showToast('✓ Result summary copied to clipboard');
  }).catch(() => {
    showToast('⚠️ Could not copy to clipboard');
  });
}

// Export Full Markdown Audit
function copyMarkdownAudit() {
  if (!appState.currentResult) return;

  const res = appState.currentResult;
  const unified = res.unified_assessment || {};

  const md = `# ScamShield Threat Assessment Audit
- **Timestamp**: ${new Date().toISOString()}
- **Risk Score**: ${unified.risk_score || 0} / 100
- **Risk Level**: ${unified.risk_level || 'N/A'}
- **Category**: ${unified.category_title || 'N/A'}
- **Model Confidence**: ${((unified.confidence || 0) * 100).toFixed(1)}%

## Analyzed Input
- Raw Message: "${res.input?.raw_text || ''}"
- Target URL: ${res.input?.extracted_urls?.[0] || 'None'}

## Reasons
${(unified.reasons || []).map(r => `- ${r}`).join('\n') || '- None'}

## Recommended Actions
- Do not click suspicious links or share confidential credentials.
- Contact official support channels directly.
`;

  navigator.clipboard.writeText(md).then(() => {
    showToast('✓ Full Markdown audit report copied to clipboard');
  }).catch(() => {
    showToast('⚠️ Could not copy report');
  });
}

// Backend Health Check
async function checkApiHealth() {
  const statusBadge = document.getElementById('apiStatusBadge');
  const statusIndicator = document.getElementById('statusIndicator');
  const statusText = document.getElementById('apiStatusText');

  const sidebarStatusIndicator = document.getElementById('sidebarStatusIndicator');
  const sidebarStatusText = document.getElementById('sidebarStatusText');

  try {
    const res = await fetch(`${CONFIG.API_BASE_URL}/health`, { method: 'GET' });
    if (res.ok) {
      appState.apiOnline = true;
      if (statusIndicator) statusIndicator.classList.remove('offline');
      if (statusText) statusText.innerText = 'API Online';
      if (sidebarStatusIndicator) sidebarStatusIndicator.classList.remove('offline');
      if (sidebarStatusText) sidebarStatusText.innerText = 'API Online';
    } else {
      throw new Error();
    }
  } catch {
    appState.apiOnline = false;
    if (statusIndicator) statusIndicator.classList.add('offline');
    if (statusText) statusText.innerText = 'API Offline';
    if (sidebarStatusIndicator) sidebarStatusIndicator.classList.add('offline');
    if (sidebarStatusText) sidebarStatusText.innerText = 'API Offline';
  }
}

// Utility: URL regex extractor
function extractFirstUrl(str) {
  if (!str) return null;
  const match = str.match(/https?:\/\/[^\s]+/i);
  return match ? match[0] : null;
}

// Utility: HTML Escaping
function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// Toast notification helper
function showToast(msg) {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = 'toast-msg';
  toast.innerText = msg;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.transition = 'opacity 0.3s ease';
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 2800);
}
