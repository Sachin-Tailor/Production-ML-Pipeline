/**
 * RetentionAI Client-Side Controller
 * High-performance UI interactions, live slider bindings, and API inference.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const form = document.getElementById("churn-form");
  const btnPredict = document.getElementById("btn-predict");
  const btnSpinner = document.getElementById("btn-spinner");
  const btnText = document.getElementById("btn-text");

  // Inputs & Labels
  const inputCreditScore = document.getElementById("input-credit-score");
  const valCreditScore = document.getElementById("val-credit-score");

  const inputBalance = document.getElementById("input-balance");
  const inputSalary = document.getElementById("input-salary");

  const inputAge = document.getElementById("input-age");
  const valAge = document.getElementById("val-age");

  const inputTenure = document.getElementById("input-tenure");
  const valTenure = document.getElementById("val-tenure");

  const inputProducts = document.getElementById("input-products");
  const valProducts = document.getElementById("val-products");

  const inputCreditCard = document.getElementById("input-credit-card");
  const inputActiveMember = document.getElementById("input-active-member");

  const inputCountry = document.getElementById("input-country");
  const inputGender = document.getElementById("input-gender");

  // Results elements
  const gaugeBar = document.getElementById("gauge-bar");
  const resPercentage = document.getElementById("res-percentage");
  const riskBanner = document.getElementById("risk-banner");
  const resRiskTier = document.getElementById("res-risk-tier");
  const resTargetLabel = document.getElementById("res-target-label");
  const resActionText = document.getElementById("res-action-text");
  const resLatency = document.getElementById("res-latency");

  // SVG Gauge circumference (r = 90)
  const CIRCUMFERENCE = 2 * Math.PI * 90; // ~565.48
  gaugeBar.style.strokeDasharray = `${CIRCUMFERENCE} ${CIRCUMFERENCE}`;
  gaugeBar.style.strokeDashoffset = CIRCUMFERENCE;

  // Live Slider Label Synchronization
  inputCreditScore.addEventListener("input", (e) => {
    valCreditScore.textContent = e.target.value;
  });
  inputAge.addEventListener("input", (e) => {
    valAge.textContent = e.target.value;
  });
  inputTenure.addEventListener("input", (e) => {
    valTenure.textContent = e.target.value;
  });
  inputProducts.addEventListener("input", (e) => {
    valProducts.textContent = e.target.value;
  });

  // Country Segmented Pills
  const countryPills = document.querySelectorAll("#country-pill-group .pill-btn");
  countryPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      countryPills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      inputCountry.value = pill.dataset.value;
      triggerPrediction();
    });
  });

  // Gender Segmented Pills
  const genderPills = document.querySelectorAll("#gender-pill-group .pill-btn");
  genderPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      genderPills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      inputGender.value = pill.dataset.value;
      triggerPrediction();
    });
  });

  // Checkbox toggles trigger prediction
  inputCreditCard.addEventListener("change", triggerPrediction);
  inputActiveMember.addEventListener("change", triggerPrediction);

  // Preset Configurations
  const presets = {
    high: {
      credit_score: 590,
      country: "Germany",
      gender: "Female",
      age: 54,
      tenure: 2,
      balance: 135000,
      salary: 110000,
      products: 3,
      has_credit_card: true,
      is_active_member: false,
    },
    low: {
      credit_score: 760,
      country: "France",
      gender: "Male",
      age: 27,
      tenure: 6,
      balance: 15000,
      salary: 65000,
      products: 1,
      has_credit_card: true,
      is_active_member: true,
    },
    moderate: {
      credit_score: 640,
      country: "Spain",
      gender: "Female",
      age: 41,
      tenure: 3,
      balance: 82000,
      salary: 95000,
      products: 2,
      has_credit_card: false,
      is_active_member: false,
    },
  };

  const presetButtons = document.querySelectorAll(".preset-chip");
  presetButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      presetButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const profile = presets[btn.dataset.preset];
      if (profile) {
        applyProfile(profile);
        triggerPrediction();
      }
    });
  });

  function applyProfile(p) {
    inputCreditScore.value = p.credit_score;
    valCreditScore.textContent = p.credit_score;

    inputBalance.value = p.balance;
    inputSalary.value = p.salary;

    inputAge.value = p.age;
    valAge.textContent = p.age;

    inputTenure.value = p.tenure;
    valTenure.textContent = p.tenure;

    inputProducts.value = p.products;
    valProducts.textContent = p.products;

    inputCreditCard.checked = p.has_credit_card;
    inputActiveMember.checked = p.is_active_member;

    inputCountry.value = p.country;
    countryPills.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.value === p.country);
    });

    inputGender.value = p.gender;
    genderPills.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.value === p.gender);
    });
  }

  // Recommendation generator based on business rules
  function generateRecommendation(riskLevel, probability, data) {
    if (riskLevel === "High") {
      return `🚨 <strong>Urgent Intervention Required (${(probability * 100).toFixed(1)}% churn risk)</strong>: Customer exhibits high-flight risk signals. Immediate action: Trigger an automated 1.5% APY deposit bonus and schedule a proactive consultation with a dedicated branch relationship manager.`;
    } else if (riskLevel === "Medium") {
      return `⚠️ <strong>Moderate Risk Detected (${(probability * 100).toFixed(1)}% churn risk)</strong>: Customer engagement has slowed. Recommended strategy: Deploy personalized email campaign highlighting loyalty rewards, fee-free transactions, and bundled digital banking perks.`;
    } else {
      return `🛡️ <strong>High Retention Health (${(probability * 100).toFixed(1)}% churn risk)</strong>: Customer profile is stable and loyal. Action: Eligible for premium credit card upgrades, cross-sell investment accounts, and loyalty milestone rewards.`;
    }
  }

  // Async API Inference Call
  async function triggerPrediction() {
    const payload = {
      credit_score: parseInt(inputCreditScore.value, 10),
      country: inputCountry.value,
      gender: inputGender.value,
      age: parseInt(inputAge.value, 10),
      tenure: parseInt(inputTenure.value, 10),
      balance: parseFloat(inputBalance.value) || 0.0,
      num_of_products: parseInt(inputProducts.value, 10),
      has_credit_card: inputCreditCard.checked ? 1 : 0,
      is_active_member: inputActiveMember.checked ? 1 : 0,
      estimated_salary: parseFloat(inputSalary.value) || 0.0,
    };

    btnSpinner.style.display = "inline-block";
    btnText.textContent = "Analyzing Patterns...";
    btnPredict.disabled = true;

    const startTime = performance.now();

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const elapsed = Math.round(performance.now() - startTime);
      resLatency.textContent = `${elapsed}ms`;

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const pred = data.prediction;

      // Update Gauge SVG
      const prob = pred.churn_probability;
      const percentage = Math.round(prob * 100);
      resPercentage.textContent = `${percentage}%`;

      const offset = CIRCUMFERENCE - prob * CIRCUMFERENCE;
      gaugeBar.style.strokeDashoffset = offset;

      // Adjust colors based on tier
      riskBanner.className = `risk-banner ${pred.risk_level.toLowerCase()}`;
      resRiskTier.textContent = `${pred.risk_level} Risk Tier`;
      resTargetLabel.textContent = pred.churn_label === "Churned" ? "Predicted to Churn" : "Retained Customer";

      if (pred.risk_level === "High") {
        gaugeBar.style.stroke = "var(--risk-high)";
      } else if (pred.risk_level === "Medium") {
        gaugeBar.style.stroke = "var(--risk-medium)";
      } else {
        gaugeBar.style.stroke = "var(--risk-low)";
      }

      // Update strategic action recommendation
      resActionText.innerHTML = generateRecommendation(pred.risk_level, prob, payload);
    } catch (err) {
      console.error("Prediction failed:", err);
      resActionText.textContent = "Inference service unreachable. Please ensure the server is running.";
    } finally {
      btnSpinner.style.display = "none";
      btnText.textContent = "Run Churn Risk Inference";
      btnPredict.disabled = false;
    }
  }

  // Form submission
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    triggerPrediction();
  });

  // Initial trigger on load with default preset
  triggerPrediction();
});
