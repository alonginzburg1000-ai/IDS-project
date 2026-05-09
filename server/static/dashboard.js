const POLL_MS = 1000;
const attackTypeLabels = ["dos", "probe", "r2l", "u2r"];
let attackChart = null;

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupSnifferControls();
  setupChart();
  refreshDashboard();
  window.setInterval(refreshDashboard, POLL_MS);
});

function setupTabs() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
      const tabId = button.dataset.tab;
      document.querySelectorAll(".tab-button").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(tabId).classList.add("active");
    });
  });
}

function setupSnifferControls() {
  document.getElementById("start-sniffer").addEventListener("click", () => {
    postJson("/api/sniffer/start").then(refreshDashboard).catch(handleControlError);
  });
  document.getElementById("stop-sniffer").addEventListener("click", () => {
    postJson("/api/sniffer/stop").then(refreshDashboard).catch(handleControlError);
  });
}

function setupChart() {
  const ctx = document.getElementById("attack-type-chart");
  attackChart = new Chart(ctx, {
    type: "pie",
    data: {
      labels: attackTypeLabels,
      datasets: [
        {
          data: [0, 0, 0, 0],
          backgroundColor: ["#ef4444", "#f97316", "#eab308", "#14b8a6"],
          borderColor: "#162033",
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: "#e5edf8",
          },
        },
      },
    },
  });
}

async function refreshDashboard() {
  try {
    const [health, traffic, suspicious, attackTypes, sniffer] = await Promise.all([
      fetchJson("/health"),
      fetchJson("/api/traffic"),
      fetchJson("/api/suspicious"),
      fetchJson("/api/stats/attack-types"),
      fetchJson("/api/sniffer/status"),
    ]);

    setOnlineState(true);
    updateMetrics(health);
    updateSnifferStatus(sniffer);
    renderTrafficTable(traffic.records || [], "traffic-table-body", "traffic-empty", false);
    renderTrafficTable(suspicious.records || [], "suspicious-table-body", "suspicious-empty", true);
    updateAttackChart(attackTypes.attack_types || {});
  } catch (error) {
    setOnlineState(false);
    console.error("dashboard refresh failed", error);
  }
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.json();
}

async function postJson(url) {
  const response = await fetch(url, {
    method: "POST",
    cache: "no-store",
  });
  if (!response.ok && response.status !== 202) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.json();
}

function setOnlineState(isOnline) {
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  dot.classList.toggle("online", isOnline);
  text.textContent = isOnline ? "מחובר" : "מנותק";
}

function updateMetrics(health) {
  document.getElementById("traffic-count").textContent = health.traffic_count ?? 0;
  document.getElementById("attack-count").textContent = health.attack_count ?? 0;
  document.getElementById("binary-threshold").textContent = health.binary_threshold ?? "0.55";
  document.getElementById("agent-state").textContent = health.agent_running ? "רץ" : "לא פעיל";
}

function updateSnifferStatus(sniffer) {
  const isRunning = Boolean(sniffer.sniffing);
  const isEnabled = sniffer.enabled !== false;
  const statusText = document.getElementById("sniffer-status-text");
  const startButton = document.getElementById("start-sniffer");
  const stopButton = document.getElementById("stop-sniffer");

  statusText.textContent = sniffer.display_status || (isRunning ? "הסנפה פעילה" : "הסנפה עצורה");
  startButton.disabled = !isEnabled || isRunning;
  stopButton.disabled = !isEnabled || !isRunning;
}

function handleControlError(error) {
  console.error("sniffer control failed", error);
  refreshDashboard();
}

function renderTrafficTable(records, tableBodyId, emptyId, suspiciousOnly) {
  const tbody = document.getElementById(tableBodyId);
  const empty = document.getElementById(emptyId);
  tbody.innerHTML = "";
  empty.style.display = records.length === 0 ? "block" : "none";

  for (const record of records) {
    const row = document.createElement("tr");
    if (record.binary_prediction === "attack") {
      row.classList.add("attack-row");
    }

    const columns = suspiciousOnly
      ? [
          formatTimestamp(record.timestamp),
          record.src_ip,
          record.dst_ip,
          displayValue(record.src_port),
          displayValue(record.dst_port),
          record.protocol,
          displayValue(record.attack_type),
          formatConfidence(record.binary_confidence),
        ]
      : [
          formatTimestamp(record.timestamp),
          record.src_ip,
          record.dst_ip,
          displayValue(record.src_port),
          displayValue(record.dst_port),
          record.protocol,
          renderPrediction(record.binary_prediction),
          formatConfidence(record.binary_confidence),
        ];

    for (const column of columns) {
      const cell = document.createElement("td");
      if (typeof column === "object" && column !== null) {
        cell.appendChild(column);
      } else {
        cell.textContent = column;
      }
      row.appendChild(cell);
    }
    tbody.appendChild(row);
  }
}

function renderPrediction(prediction) {
  const span = document.createElement("span");
  span.textContent = displayValue(prediction);
  span.className = prediction === "attack" ? "prediction-attack" : "prediction-normal";
  return span;
}

function updateAttackChart(counts) {
  const values = attackTypeLabels.map((label) => Number(counts[label] || 0));
  const total = values.reduce((sum, value) => sum + value, 0);
  document.getElementById("chart-empty").style.display = total === 0 ? "block" : "none";
  attackChart.data.datasets[0].data = values;
  attackChart.update();
}

function formatTimestamp(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return displayValue(value);
  }
  const date = new Date(numeric * 1000);
  return date.toLocaleString("he-IL");
}

function formatConfidence(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }
  return numeric.toFixed(4);
}

function displayValue(value) {
  return value === null || value === undefined || value === "" ? "-" : String(value);
}
