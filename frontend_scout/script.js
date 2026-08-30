const API_BASE = "https://ai-agent-scout.onrender.com";

const el = (id) => document.getElementById(id);

// ---------- health check ----------
async function checkHealth() {
  const blip = el("statusBlip");
  const text = el("statusText");
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error();
    blip.classList.add("online");
    text.textContent = "LINK ONLINE";
  } catch {
    blip.classList.add("offline");
    text.textContent = "LINK DOWN — RENDER MAY BE WAKING UP";
  }
}

// ---------- risk gauge ----------
function paintGauge(riskProbability) {
  const gauge = el("gauge");
  const pct = Math.round(riskProbability * 100);
  const deg = Math.round(riskProbability * 360);
  const color = riskProbability >= 0.6 ? "var(--amber)" : "var(--teal)";
  gauge.style.background = `conic-gradient(${color} ${deg}deg, var(--line) ${deg}deg)`;
  el("riskPct").textContent = `${pct}%`;
  el("riskLabel").textContent =
    riskProbability >= 0.6 ? "ELEVATED RISK" : riskProbability >= 0.35 ? "MODERATE RISK" : "LOW RISK";
}

// ---------- scan a shipment ----------
async function scanShipment(shipmentId) {
  const btn = el("scanBtn");
  const hint = el("formHint");
  const resultZone = el("resultZone");

  btn.disabled = true;
  btn.querySelector("span").textContent = "SCANNING…";
  hint.textContent = "";

  try {
    const res = await fetch(`${API_BASE}/predict/${shipmentId}`, { method: "POST" });
    if (res.status === 404) {
      hint.textContent = `No shipment found with ID ${shipmentId}.`;
      resultZone.hidden = true;
      return;
    }
    if (!res.ok) throw new Error(`Server returned ${res.status}`);

    const data = await res.json();
    renderTicket(data);
    resultZone.hidden = false;
  } catch (err) {
    hint.textContent = "Couldn't reach SCOUT. The API may be waking up — try again in a few seconds.";
  } finally {
    btn.disabled = false;
    btn.querySelector("span").textContent = "SCAN SHIPMENT";
  }
}

function renderTicket(data) {
  el("rShipmentId").textContent = `#${data.shipment_id}`;
  el("rOrigin").textContent = data.origin || "—";
  el("rDestination").textContent = data.destination || "—";
  el("rSupplier").textContent = data.supplier_name || "—";
  el("rMode").textContent = data.transport_mode || "—";

  el("rDelay").textContent =
    data.predicted_delay_days > 0
      ? `+${data.predicted_delay_days.toFixed(1)} days`
      : `${data.predicted_delay_days.toFixed(1)} days`;

  paintGauge(data.risk_probability);

  const factorsList = el("rFactors");
  factorsList.innerHTML = "";
  (data.top_risk_factors || []).forEach((f) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${f.feature}</span><span>${(f.importance * 100).toFixed(1)}%</span>`;
    factorsList.appendChild(li);
  });

  el("recoText").textContent = data.recommendation || "No recommendation generated.";
}

// ---------- flagged shipments board ----------
async function loadBoard() {
  const body = el("boardBody");
  body.innerHTML = `<tr><td colspan="7" class="board-empty">Loading manifest…</td></tr>`;

  try {
    const res = await fetch(`${API_BASE}/predictions?limit=25`);
    if (!res.ok) throw new Error();
    const rows = await res.json();

    if (!rows.length) {
      body.innerHTML = `<tr><td colspan="7" class="board-empty">No shipments scored yet. Scan one above.</td></tr>`;
      return;
    }

    body.innerHTML = "";
    rows
      .sort((a, b) => b.risk_probability - a.risk_probability)
      .forEach((r) => {
        const tr = document.createElement("tr");
        const riskClass = r.risk_probability >= 0.6 ? "high" : "low";
        const delay =
          r.predicted_delay_days > 0
            ? `+${Number(r.predicted_delay_days).toFixed(1)}d`
            : `${Number(r.predicted_delay_days).toFixed(1)}d`;
        const scored = r.prediction_date ? new Date(r.prediction_date).toLocaleString() : "—";

        tr.innerHTML = `
          <td class="mono">#${r.shipment_id}</td>
          <td>${r.supplier_name ?? "—"}</td>
          <td>${r.origin ?? "—"} → ${r.destination ?? "—"}</td>
          <td>${r.transport_mode ?? "—"}</td>
          <td><span class="risk-pill ${riskClass}">${Math.round(r.risk_probability * 100)}%</span></td>
          <td>${delay}</td>
          <td>${scored}</td>
        `;
        body.appendChild(tr);
      });
  } catch {
    body.innerHTML = `<tr><td colspan="7" class="board-empty">Couldn't load the board. Try refreshing.</td></tr>`;
  }
}

// ---------- wire up ----------
el("scanForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const id = el("shipmentId").value;
  if (id) scanShipment(id);
});

el("refreshBoard").addEventListener("click", loadBoard);

checkHealth();
loadBoard();
