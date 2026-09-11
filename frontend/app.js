const API_URL = "http://127.0.0.1:8000/api/query";
const queryInput = document.getElementById("queryInput");
const runQueryButton = document.getElementById("runQuery");
const statusText = document.getElementById("statusText");
const answerBlock = document.getElementById("answerBlock");
const retrievedList = document.getElementById("retrievedList");

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.style.color = isError ? "var(--error)" : "var(--accent)";
}

function renderAnswer(text) {
  answerBlock.classList.remove("empty");
  answerBlock.innerHTML = `
    <div class="result-card-item">
      <p>${text.replace(/\n/g, "<br />")}</p>
    </div>`;
}

function renderRetrieved(sourceUsed, contextData = "") {
  if (!sourceUsed) {
    retrievedList.classList.add("empty");
    retrievedList.innerHTML = `<p>No routing execution steps observed.</p>`;
    return;
  }

  retrievedList.classList.remove("empty");
  
  // Safe validation check to cleanly handle string split errors if backend values drop
  const descriptiveContext = contextData 
    ? contextData.replace(/\n/g, "<br />") 
    : "System extracted matching document fragments to synthesize optimal response vectors.";

  retrievedList.innerHTML = `
    <div class="result-card-item">
      <strong>Active Pipeline Asset</strong>
      <p>${sourceUsed}</p>
    </div>
    <div class="result-card-item">
      <strong>Execution Context</strong>
      <p style="font-size: 0.875rem; color: var(--muted); max-height: 200px; overflow-y: auto;">
        ${descriptiveContext}
      </p>
    </div>`;
}

async function runQuery() {
  const query = queryInput.value.trim();
  if (!query) {
    setStatus("Please enter a question before submitting.", true);
    return;
  }

  // Visual state mutation for processing initiation
  setStatus("Engaging RAG pipeline execution graph...");
  answerBlock.classList.add("empty");
  answerBlock.innerHTML = `<p>Awaiting model synthesis inference...</p>`;
  retrievedList.classList.add("empty");
  retrievedList.innerHTML = `<p>Evaluating semantic vector spaces...</p>`;
  
  runQueryButton.disabled = true;
  runQueryButton.textContent = "Processing Execution...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ user_query: query }),
    });

    if (!response.ok) {
      throw new Error(`Execution Pipeline Fault: ${response.status}`);
    }

    const data = await response.json();
    
    // Exact mapping alignment with your active FastAPI structured return values
    renderAnswer(data.ai_answer || "Null response payload returned.");
    renderRetrieved(data.source_used, data.active_context || "");
    
    setStatus("Inference cycle stabilized. Data rendered below.");
  } catch (error) {
    renderAnswer("Orchestration layer failed to return text results.");
    renderRetrieved("System Fallback State Exception");
    setStatus(`Error: ${error.message}`, true);
  } finally {
    runQueryButton.disabled = false;
    runQueryButton.textContent = "Run Query";
  }
}

runQueryButton.addEventListener("click", runQuery);
queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    runQuery();
  }
});