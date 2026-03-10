(() => {
  const chatWindow = document.getElementById("chat-window");
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const voiceBtn = document.getElementById("voice-btn");
  const typingEl = document.getElementById("typing-indicator");
  const langSelect = document.getElementById("language-select");
  const campusSelect = document.getElementById("campus-select");
  const suggestionsEl = document.getElementById("suggestions-dropdown");

  let suggestDebounceTimer = null;
  let isListening = false;

  function appendMessage(text, sender = "bot", meta = {}) {
    const msg = document.createElement("div");
    msg.className = `message ${sender}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = sender === "bot" ? "🤖" : "🧑";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = `<p>${text}</p>`;

    if (sender === "bot" && meta.messageId && window.CHAT_CONFIG.feedbackApiUrl) {
      const feedbackRow = document.createElement("div");
      feedbackRow.className = "feedback-row";
      feedbackRow.innerHTML = `
        <span class="feedback-label">Was this helpful?</span>
        <button type="button" class="feedback-btn up" data-value="up">Yes</button>
        <button type="button" class="feedback-btn down" data-value="down">No</button>
      `;

      const onClick = async (event) => {
        const value = event.currentTarget.getAttribute("data-value");
        try {
          await fetch(window.CHAT_CONFIG.feedbackApiUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message_id: meta.messageId, value }),
          });
          feedbackRow.innerHTML = `<span class="feedback-thanks">Thanks for your feedback.</span>`;
        } catch {
          feedbackRow.innerHTML = `<span class="feedback-thanks">Could not send feedback.</span>`;
        }
      };

      feedbackRow.querySelectorAll(".feedback-btn").forEach((btn) => btn.addEventListener("click", onClick));
      bubble.appendChild(feedbackRow);
    }

    msg.appendChild(avatar);
    msg.appendChild(bubble);
    chatWindow.appendChild(msg);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text) return;

    appendMessage(text, "user");
    input.value = "";
    suggestionsEl.classList.add("hidden");
    typingEl.classList.remove("hidden");
    sendBtn.disabled = true;

    try {
      const res = await fetch(window.CHAT_CONFIG.chatApiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          language: langSelect.value,
          campus: campusSelect.value,
        }),
      });

      const data = await res.json();
      if (!res.ok || data.error) {
        appendMessage(data.error || "Something went wrong. Please try again.", "bot");
      } else {
        appendMessage(data.answer, "bot", { messageId: data.message_id });
      }
    } catch (err) {
      appendMessage("Network error: " + err.message, "bot");
    } finally {
      typingEl.classList.add("hidden");
      sendBtn.disabled = false;
      input.focus();
    }
  }

  function hideSuggestions() {
    suggestionsEl.classList.add("hidden");
  }

  async function fetchSuggestions(prefix) {
    if (!window.CHAT_CONFIG.suggestApiUrl) return [];
    const params = new URLSearchParams({
      q: prefix,
      lang: langSelect.value,
      campus: campusSelect.value,
      limit: 8,
    });
    const res = await fetch(window.CHAT_CONFIG.suggestApiUrl + "?" + params);
    const data = await res.json();
    return data.suggestions || [];
  }

  function showSuggestions(suggestions) {
    if (!suggestions || suggestions.length === 0) {
      suggestionsEl.classList.add("hidden");
      return;
    }
    suggestionsEl.innerHTML = suggestions
      .map(
        (s) =>
          `<button type="button" class="suggestion-item" data-question="${(s.question || "").replace(/"/g, "&quot;")}">
            <span class="suggestion-q">${(s.question || "").slice(0, 60)}${(s.question || "").length > 60 ? "…" : ""}</span>
            ${s.category ? `<span class="suggestion-cat">${s.category}</span>` : ""}
          </button>`
      )
      .join("");
    suggestionsEl.classList.remove("hidden");

    suggestionsEl.querySelectorAll(".suggestion-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        const q = btn.getAttribute("data-question");
        if (q) {
          input.value = q;
          hideSuggestions();
          input.focus();
        }
      });
    });
  }

  function onInputChange() {
    clearTimeout(suggestDebounceTimer);
    const val = input.value.trim();
    suggestDebounceTimer = setTimeout(async () => {
      const suggestions = await fetchSuggestions(val);
      showSuggestions(suggestions);
    }, 200);
  }

  input.addEventListener("focus", () => onInputChange());
  input.addEventListener("input", () => onInputChange());
  input.addEventListener("blur", () => setTimeout(hideSuggestions, 150));

  sendBtn.addEventListener("click", sendMessage);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const topic = chip.getAttribute("data-topic");
      input.value = `Tell me about ${topic} at the campus.`;
      input.focus();
      onInputChange();
    });
  });

  function startVoiceInput() {
    if (!("webkitSpeechRecognition" in window) && !("SpeechRecognition" in window)) {
      appendMessage("Voice input is not supported in your browser. Try Chrome or Edge.", "bot");
      return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const rec = new SpeechRecognition();
    rec.lang = langSelect.value === "hi" ? "hi-IN" : langSelect.value === "mr" ? "mr-IN" : "en-IN";
    rec.continuous = false;
    rec.interimResults = false;

    rec.onstart = () => {
      isListening = true;
      voiceBtn.classList.add("listening");
      voiceBtn.title = "Listening…";
    };
    rec.onend = () => {
      isListening = false;
      voiceBtn.classList.remove("listening");
      voiceBtn.title = "Voice input";
    };
    rec.onresult = (e) => {
      const t = e.results[0][0].transcript;
      if (t) input.value = (input.value + " " + t).trim();
    };
    rec.onerror = () => {
      voiceBtn.classList.remove("listening");
      voiceBtn.title = "Voice input";
    };
    rec.start();
  }

  if (voiceBtn) {
    voiceBtn.addEventListener("click", () => {
      if (isListening) return;
      startVoiceInput();
    });
  }
})();
