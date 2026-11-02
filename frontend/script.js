/**
 * EstateIQ – Real-Estate Machine Learning Intelligence Dashboard
 * Pure Vanilla JavaScript Client
 */

// Presets data for California regions
const PRESETS = {
  "sf-bay": {
    MedInc: 8.3252,
    HouseAge: 41,
    AveRooms: 6.9841,
    AveBedrms: 1.0238,
    Population: 322,
    AveOccup: 2.5555,
    Latitude: 37.88,
    Longitude: -122.23,
  },
  "la-coastal": {
    MedInc: 5.6431,
    HouseAge: 32,
    AveRooms: 5.82,
    AveBedrms: 1.01,
    Population: 1100,
    AveOccup: 2.75,
    Latitude: 33.95,
    Longitude: -118.42,
  },
  "central-valley": {
    MedInc: 2.45,
    HouseAge: 25,
    AveRooms: 4.85,
    AveBedrms: 1.05,
    Population: 2150,
    AveOccup: 3.2,
    Latitude: 36.75,
    Longitude: -119.78,
  },
  "san-diego": {
    MedInc: 4.5,
    HouseAge: 18,
    AveRooms: 5.4,
    AveBedrms: 1.03,
    Population: 1850,
    AveOccup: 2.9,
    Latitude: 32.85,
    Longitude: -117.15,
  },
};

const FIELD_RULES = {
  MedInc: { min: 0.1, max: 25.0, label: "Median Income" },
  HouseAge: { min: 1, max: 100, label: "House Age" },
  AveRooms: { min: 0.5, max: 50.0, label: "Average Rooms" },
  AveBedrms: { min: 0.2, max: 20.0, label: "Average Bedrooms" },
  Population: { min: 1, max: 50000, label: "Population" },
  AveOccup: { min: 0.5, max: 50.0, label: "Average Occupancy" },
  Latitude: { min: 32.0, max: 43.0, label: "Latitude" },
  Longitude: { min: -125.0, max: -114.0, label: "Longitude" },
};

document.addEventListener("DOMContentLoaded", () => {
  initHealthCheck();
  loadModelAnalytics();
  setupEventListeners();
  // Populate first preset by default
  applyPreset("sf-bay");
});

/**
 * Check backend health status on startup
 */
async function initHealthCheck() {
  const indicator = document.getElementById("system-status-indicator");
  const statusText = document.getElementById("system-status-text");
  const activeBadge = document.getElementById("active-model-badge");

  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error(`Health status returned HTTP ${res.status}`);
    const data = await res.json();

    if (data.status === "healthy" && data.model_loaded) {
      indicator.className = "status-indicator online";
      statusText.textContent = "API Operational";
      activeBadge.textContent = data.best_model_name;
    } else {
      indicator.className = "status-indicator error";
      statusText.textContent = "Model Missing";
      activeBadge.textContent = "Action Required";
    }
  } catch (err) {
    console.error("Health check failed:", err);
    indicator.className = "status-indicator error";
    statusText.textContent = "Disconnected";
    activeBadge.textContent = "Offline";
  }
}

/**
 * Fetch and render model comparison metrics and feature importances
 */
async function loadModelAnalytics() {
  const tableBody = document.getElementById("models-table-body");
  const benchmarkStatus = document.getElementById("benchmark-status");
  const rmseBarsContainer = document.getElementById("rmse-bars-container");
  const importanceList = document.getElementById("importance-list");
  const importanceLabel = document.getElementById("importance-model-label");

  try {
    const res = await fetch("/models");
    if (!res.ok) throw new Error(`Failed to fetch models: HTTP ${res.status}`);
    const data = await res.json();

    benchmarkStatus.textContent = `${data.experiment_name || "EstateIQ Experiment"}`;
    importanceLabel.textContent = data.best_model;

    // 1. Render Model Comparison Table
    tableBody.innerHTML = "";
    const models = data.models || {};
    const bestModelKey = data.best_model;

    let maxRmse = 0.001;

    for (const [modelName, modelInfo] of Object.entries(models)) {
      const isBest = modelName === bestModelKey;
      const metrics = modelInfo.metrics;
      if (metrics.rmse > maxRmse) maxRmse = metrics.rmse;

      const row = document.createElement("tr");
      if (isBest) row.classList.add("selected-row");

      row.innerHTML = `
        <td><strong>${formatModelName(modelName)}</strong></td>
        <td>${metrics.mae.toFixed(4)} <small class="text-subtle">($${metrics.mae_usd.toLocaleString()})</small></td>
        <td>${metrics.rmse.toFixed(4)} <small class="text-subtle">($${metrics.rmse_usd.toLocaleString()})</small></td>
        <td>${metrics.r2_score.toFixed(4)}</td>
        <td>
          ${isBest
            ? '<span class="status-badge-best">★ Best Performer</span>'
            : '<span class="status-badge-alt">Evaluated</span>'}
        </td>
      `;
      tableBody.appendChild(row);
    }

    // 2. Render RMSE Comparison Bars
    rmseBarsContainer.innerHTML = "";
    for (const [modelName, modelInfo] of Object.entries(models)) {
      const isBest = modelName === bestModelKey;
      const rmse = modelInfo.metrics.rmse;
      const rmseUsd = modelInfo.metrics.rmse_usd;
      const pct = Math.min(100, Math.round((rmse / (maxRmse * 1.15)) * 100));

      const barRow = document.createElement("div");
      barRow.className = "bar-row";
      barRow.innerHTML = `
        <div class="bar-header">
          <span class="bar-label">${formatModelName(modelName)}</span>
          <span class="bar-value">RMSE: ${rmse.toFixed(4)} ($${rmseUsd.toLocaleString()})</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill ${isBest ? 'accent-emerald' : ''}" style="width: ${pct}%"></div>
        </div>
      `;
      rmseBarsContainer.appendChild(barRow);
    }

    // 3. Render Feature Importances
    const importances = data.best_model_feature_importances || {};
    importanceList.innerHTML = "";

    const sortedFeatures = Object.entries(importances).sort((a, b) => b[1] - a[1]);

    for (const [feat, val] of sortedFeatures) {
      const pctVal = (val * 100).toFixed(1);
      const item = document.createElement("div");
      item.className = "importance-item";
      item.innerHTML = `
        <div class="importance-meta">
          <span class="importance-name">${cleanFeatureLabel(feat)}</span>
          <span class="importance-pct">${pctVal}%</span>
        </div>
        <div class="importance-track">
          <div class="importance-bar" style="width: ${Math.max(2, parseFloat(pctVal))}%"></div>
        </div>
      `;
      importanceList.appendChild(item);
    }

  } catch (err) {
    console.error("Error loading model analytics:", err);
    tableBody.innerHTML = `<tr><td colspan="5" class="error-msg">Failed to load model metrics from backend.</td></tr>`;
    importanceList.innerHTML = `<div class="error-msg">Could not retrieve feature importance data.</div>`;
  }
}

/**
 * Configure UI event listeners
 */
function setupEventListeners() {
  // Preset buttons
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const presetKey = btn.getAttribute("data-preset");
      applyPreset(presetKey);
    });
  });

  // Clear button
  document.getElementById("clear-btn").addEventListener("click", () => {
    document.getElementById("property-form").reset();
    clearAllErrors();
    document.getElementById("result-placeholder").classList.remove("hidden");
    document.getElementById("result-content").classList.add("hidden");
    document.getElementById("result-model-badge").textContent = "Waiting for input";
  });

  // Form submission
  document.getElementById("property-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    await handleFormSubmit();
  });
}

/**
 * Populate form inputs from preset profile
 */
function applyPreset(presetKey) {
  const data = PRESETS[presetKey];
  if (!data) return;

  clearAllErrors();
  for (const [key, value] of Object.entries(data)) {
    const input = document.getElementById(key);
    if (input) input.value = value;
  }
}

/**
 * Validate input values before sending request
 */
function validateForm() {
  clearAllErrors();
  let isValid = true;
  const payload = {};

  for (const [field, rules] of Object.entries(FIELD_RULES)) {
    const input = document.getElementById(field);
    const errSpan = document.getElementById(`err-${field}`);
    const rawVal = input.value.trim();

    if (rawVal === "") {
      errSpan.textContent = `${rules.label} is required.`;
      isValid = false;
      continue;
    }

    const num = parseFloat(rawVal);
    if (isNaN(num)) {
      errSpan.textContent = `${rules.label} must be a number.`;
      isValid = false;
      continue;
    }

    if (num < rules.min || num > rules.max) {
      errSpan.textContent = `${rules.label} must be between ${rules.min} and ${rules.max}.`;
      isValid = false;
      continue;
    }

    payload[field] = num;
  }

  return { isValid, payload };
}

function clearAllErrors() {
  document.querySelectorAll(".error-msg").forEach((el) => {
    el.textContent = "";
  });
}

/**
 * Handle form submit and call POST /predict
 */
async function handleFormSubmit() {
  const { isValid, payload } = validateForm();
  if (!isValid) return;

  const submitBtn = document.getElementById("submit-btn");
  const spinner = document.getElementById("btn-spinner");

  try {
    submitBtn.disabled = true;
    spinner.classList.add("active");

    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      const msg = errData.detail || `Server returned error ${response.status}`;
      throw new Error(msg);
    }

    const data = await response.json();
    renderPredictionResult(data);

  } catch (err) {
    alert(`Prediction Error: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
    spinner.classList.remove("active");
  }
}

/**
 * Render prediction results in the result card
 */
function renderPredictionResult(data) {
  document.getElementById("result-placeholder").classList.add("hidden");
  const resultContent = document.getElementById("result-content");
  resultContent.classList.remove("hidden");

  // Format currency
  const priceVal = data.prediction_usd;
  document.getElementById("val-price").textContent = data.prediction_usd_formatted || `$${priceVal.toLocaleString()}`;

  // Model badge
  document.getElementById("result-model-badge").textContent = data.model_name;

  // Uncertainty interval
  const unc = data.uncertainty;
  if (unc) {
    document.getElementById("val-range-text").textContent =
      `${unc.lower_bound_formatted} – ${unc.upper_bound_formatted}`;
    document.getElementById("val-uncertainty-desc").textContent = unc.description;
  }

  // Engineered features
  const eng = data.engineered_features || {};
  document.getElementById("feat-bed-ratio").textContent =
    eng.bedrooms_per_room !== undefined ? eng.bedrooms_per_room.toFixed(3) : "-";
  document.getElementById("feat-rooms-ratio").textContent =
    eng.rooms_per_household !== undefined ? eng.rooms_per_household.toFixed(2) : "-";
  document.getElementById("feat-occup-ratio").textContent =
    eng.population_per_household !== undefined ? eng.population_per_household.toFixed(2) : "-";

  // Disclaimer
  if (data.disclaimer) {
    document.getElementById("val-disclaimer").textContent = data.disclaimer;
  }

  // Smooth scroll to result on mobile
  if (window.innerWidth < 960) {
    document.getElementById("result-card").scrollIntoView({ behavior: "smooth" });
  }
}

function formatModelName(name) {
  if (name === "HistGradientBoostingRegressor") return "HistGradientBoosting Regressor";
  if (name === "RandomForestRegressor") return "Random Forest Regressor";
  return name;
}

function cleanFeatureLabel(name) {
  const map = {
    MedInc: "Median Income (MedInc)",
    HouseAge: "House Age (HouseAge)",
    AveRooms: "Average Rooms (AveRooms)",
    AveBedrms: "Average Bedrooms (AveBedrms)",
    Population: "Population Count",
    AveOccup: "Average Occupancy",
    Latitude: "Geographical Latitude",
    Longitude: "Geographical Longitude",
    rooms_per_household: "Engineered: Rooms / Household",
    bedrooms_per_room: "Engineered: Bedrooms / Room",
    population_per_household: "Engineered: Pop / Household",
  };
  return map[name] || name;
}

