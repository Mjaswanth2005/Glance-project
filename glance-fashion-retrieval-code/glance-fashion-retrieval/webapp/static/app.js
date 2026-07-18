const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const status = document.querySelector("#status");
const interpretation = document.querySelector("#interpretation");
const chips = document.querySelector("#chips");
const resultsHeading = document.querySelector("#results-title");
const resultsCaption = document.querySelector("#results-caption");
const results = document.querySelector("#results");
const message = document.querySelector("#message");
const detailPanel = document.querySelector("#detail-panel");
const detailContent = document.querySelector("#detail-content");
const resultTemplate = document.querySelector("#result-template");

function scoreLabel(score) {
  if (score >= 0.75) return "Strong result";
  if (score >= 0.45) return "Related result";
  return "Closest result";
}

function formatPercent(score) {
  return `${Math.round(score * 100)}%`;
}

function setMessage(text, visible = true) {
  message.textContent = text;
  message.hidden = !visible;
}

function renderInterpretation(parsedQuery) {
  const items = [
    ...parsedQuery.pairs.map((pair) => ({ label: [pair.color, pair.garment].filter(Boolean).join(" "), kind: "pair" })),
    ...parsedQuery.environmentHits.map((label) => ({ label, kind: "context" })),
    ...parsedQuery.styleHits.map((label) => ({ label, kind: "context" })),
  ];
  chips.replaceChildren();
  items.forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.dataset.kind = item.kind;
    chip.textContent = item.label;
    chips.append(chip);
  });
  interpretation.hidden = items.length === 0;
}

function displayTags(result) {
  const garments = result.garments.slice(0, 2).map((garment) => `${garment.color} ${garment.type}`);
  return [...garments, result.style || result.environment].filter(Boolean).join(" · ") || "Visual similarity";
}

function openDetail(result) {
  const garmentItems = result.garments.length
    ? result.garments.map((garment) => `<li>${garment.color} ${garment.type}</li>`).join("")
    : "<li>No garment regions detected</li>";
  const evidence = [
    ...result.explanation.matchedPairs.map((item) => `Matched ${item}`),
    ...result.explanation.partialPairs.map((item) => `Related garment, different colour: ${item}`),
    ...result.explanation.contextMatches.map((item) => `Context: ${item}`),
  ];
  detailContent.replaceChildren();
  const container = document.createElement("div");
  container.className = "detail-content";
  const imageMarkup = result.imageUrl ? `<img class="detail-image" src="${result.imageUrl}" alt="Retrieved fashion result" />` : "";
  container.innerHTML = `
    <p class="eyebrow">${scoreLabel(result.score)}</p>
    <h2 id="detail-title">${result.image_id}</h2>
    <p>${result.environment || "Unspecified setting"} · ${result.style || "Unspecified style"}</p>
    ${imageMarkup}
    <div class="score-breakdown">
      <div><span>Overall ranking signal</span><strong>${formatPercent(result.score)}</strong></div>
      <div><span>Garment and colour signal</span><strong>${formatPercent(result.structured_score)}</strong></div>
    </div>
    <div class="detail-list"><h3>Why this result</h3><ul>${(evidence.length ? evidence : ["Visual and stylistic similarity"]).map((item) => `<li>${item}</li>`).join("")}</ul></div>
    <div class="detail-list"><h3>Detected details</h3><ul>${garmentItems}</ul></div>
  `;
  detailContent.append(container);
  detailPanel.hidden = false;
  document.querySelector("#close-detail").focus();
}

function renderResults(searchResults) {
  results.replaceChildren();
  searchResults.forEach((result, index) => {
    const node = resultTemplate.content.cloneNode(true);
    const openButton = node.querySelector(".result-open");
    const image = node.querySelector("img");
    image.alt = `Result ${index + 1}: ${result.image_id}`;
    if (result.imageUrl) image.src = result.imageUrl;
    node.querySelector(".rank").textContent = `Result ${String(index + 1).padStart(2, "0")}`;
    node.querySelector(".score").textContent = scoreLabel(result.score);
    node.querySelector("h3").textContent = result.image_id;
    node.querySelector(".result-tags").textContent = displayTags(result);
    openButton.addEventListener("click", () => openDetail(result));
    results.append(node);
  });
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    status.className = `status ${data.ready ? "ready" : "unavailable"}`;
    status.lastElementChild.textContent = data.ready ? "Search ready" : "Setup required";
  } catch {
    status.className = "status unavailable";
    status.lastElementChild.textContent = "Server unavailable";
  }
}

async function previewInterpretation() {
  const query = queryInput.value.trim();
  if (!query) return;
  try {
    const response = await fetch(`/api/parse?q=${encodeURIComponent(query)}`);
    if (!response.ok) return;
    const data = await response.json();
    renderInterpretation(data.parsedQuery);
  } catch {
    interpretation.hidden = true;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  results.replaceChildren();
  setMessage("Finding garment matches and visual context…");
  resultsHeading.textContent = "Searching the collection.";
  resultsCaption.textContent = "Glance is interpreting your request and ranking the closest images.";
  await previewInterpretation();
  try {
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Search could not be completed.");
    renderInterpretation(data.parsedQuery);
    resultsHeading.textContent = data.results.length ? "Closest looks for your request." : "No results found.";
    resultsCaption.textContent = data.results.length ? `${data.results.length} ranked results. Select a look to see why it appeared.` : "Try a broader description or rebuild the index.";
    setMessage("", false);
    renderResults(data.results);
  } catch (error) {
    resultsHeading.textContent = "Search needs local setup.";
    resultsCaption.textContent = "The interface is ready; the retrieval model needs its local index and dependencies.";
    setMessage(error.message);
  }
});

document.querySelectorAll(".example").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.textContent;
    queryInput.focus();
    previewInterpretation();
  });
});

document.querySelector("#close-detail").addEventListener("click", () => {
  detailPanel.hidden = true;
});

checkHealth();
