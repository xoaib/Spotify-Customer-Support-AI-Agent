document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('agent-form');
  const textarea = document.getElementById('customer-message');
  const charCount = document.getElementById('char-count');
  const submitBtn = document.getElementById('submit-btn');
  const btnSpinner = document.getElementById('btn-spinner');
  const btnText = submitBtn.querySelector('.btn-text');
  const resultSection = document.getElementById('result-section');
  const presetChips = document.getElementById('preset-chips');

  // Outputs
  const intentDisplay = document.getElementById('intent-display');
  const confidenceValue = document.getElementById('confidence-value');
  const confidenceFill = document.getElementById('confidence-fill');
  const actionBadge = document.getElementById('action-badge');
  const actionReason = document.getElementById('action-reason');
  const replyDisplay = document.getElementById('reply-display');
  const copyBtn = document.getElementById('copy-btn');
  const caseId = document.getElementById('case-id');
  const caseSimilarity = document.getElementById('case-similarity');
  const historicalQuestion = document.getElementById('historical-question');
  const historicalAnswer = document.getElementById('historical-answer');

  // Live character counter
  textarea.addEventListener('input', () => {
    const len = textarea.value.length;
    charCount.textContent = `${len} character${len === 1 ? '' : 's'}`;
  });

  // Preset chip clicks
  presetChips.addEventListener('click', (e) => {
    if (e.target.classList.contains('chip')) {
      const query = e.target.getAttribute('data-query');
      if (query) {
        textarea.value = query;
        textarea.dispatchEvent(new Event('input'));
        textarea.focus();
      }
    }
  });

  // Copy reply to clipboard
  copyBtn.addEventListener('click', async () => {
    const text = replyDisplay.textContent;
    if (text && text !== '-') {
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.textContent = 'Copied!';
        setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1800);
      } catch (err) {
        console.error('Failed to copy text: ', err);
      }
    }
  });

  // Form submission
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = textarea.value.trim();
    if (!query) return;

    // Loading state
    submitBtn.disabled = true;
    btnSpinner.style.display = 'inline-block';
    btnText.textContent = 'Analyzing...';

    try {
      const response = await fetch('/api/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: query })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      renderResults(data);
    } catch (error) {
      console.error('Error processing query:', error);
      alert('Failed to process message. Please ensure the backend server is running.');
    } finally {
      submitBtn.disabled = false;
      btnSpinner.style.display = 'none';
      btnText.textContent = 'Run Agent Pipeline';
    }
  });

  function renderResults(data) {
    resultSection.style.display = 'flex';

    // 1. Intent & Confidence
    intentDisplay.textContent = data.intent || 'unknown';
    const confPct = Math.round((data.confidence || 0) * 100);
    confidenceValue.textContent = `${confPct}%`;
    confidenceFill.style.width = `${confPct}%`;

    // 2. Action & Reason
    const isAutoHandle = data.action === 'AUTO_HANDLE';
    actionBadge.textContent = data.action;
    actionBadge.className = 'action-pill ' + (isAutoHandle ? 'auto-handle' : 'escalate');
    actionReason.textContent = data.reason || 'No explanation provided.';

    // 3. Drafted Reply
    replyDisplay.textContent = data.draft_reply || 'No reply generated.';

    // 4. Grounding (Top Retrieved Case)
    if (data.retrieved_cases && data.retrieved_cases.length > 0) {
      const topCase = data.retrieved_cases[0];
      caseId.textContent = topCase.case_id || 'Historical Match';
      const sim = (topCase.similarity !== undefined) ? topCase.similarity.toFixed(4) : 'N/A';
      caseSimilarity.textContent = `Similarity: ${sim}`;
      historicalQuestion.textContent = topCase.customer_text || 'No question recorded.';
      historicalAnswer.textContent = topCase.brand_reply || 'No resolution recorded.';
    } else {
      caseId.textContent = 'None';
      caseSimilarity.textContent = 'Similarity: 0.00';
      historicalQuestion.textContent = 'No historical match above threshold.';
      historicalAnswer.textContent = 'Routed to safe fallback resolution.';
    }

    // Smooth scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
});
