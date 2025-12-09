/* =============================
   CHATBOT CLIENT SCRIPT (WORKS WITH HOME.HTML)
   ============================= */

console.log("Chatbot script loaded.");

const BASE_URL = "https://pomegranate-disease-updated-5.onrender.com";  // your backend URL

// UI Elements
const btnToggle = document.getElementById("chatbot-btn");
const chatBox = document.getElementById("chatbot-container");
const btnClose = document.getElementById("chatbot-close");
const msgBox = document.getElementById("chatbot-messages");
const inputBox = document.getElementById("chatbot-input");
const btnSend = document.getElementById("chatbot-send");
const btnMic = document.getElementById("chatbot-mic");
const btnStop = document.getElementById("chatbot-stop");
const recordingIndicator = document.getElementById("chatbot-recording-indicator");
const langSelect = document.getElementById("chatbot-language");

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// -------------------------------
// Chat UI Functions
// -------------------------------

function appendMessage(who, text) {
    const div = document.createElement("div");
    div.className = who === "bot" ? "chat-msg bot" : "chat-msg user";
    div.innerText = text;
    msgBox.appendChild(div);
    msgBox.scrollTop = msgBox.scrollHeight;
}

// -------------------------------
// Toggle Chat Window
// -------------------------------
btnToggle.addEventListener("click", () => {
    chatBox.classList.toggle("open");
});

btnClose.addEventListener("click", () => {
    chatBox.classList.remove("open");
});

// -------------------------------
// Send Text Message
// -------------------------------
btnSend.addEventListener("click", () => {
    sendTextMessage();
});

inputBox.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendTextMessage();
});

async function sendTextMessage() {
    const text = inputBox.value.trim();
    if (!text) return;

    appendMessage("user", text);
    inputBox.value = "";

    appendMessage("bot", "⏳ Thinking...");

    try {
        const res = await fetch(`${BASE_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user_message: text,
                language_code: langSelect.value,
                from_voice: false
            })
        });

        const data = await res.json();

        msgBox.lastChild.remove(); // remove "Thinking..."
        appendMessage("bot", data.bot_response);
    } catch (err) {
        msgBox.lastChild.remove();
        appendMessage("bot", "⚠️ Error contacting server.");
    }
}

// -------------------------------
// Voice Recording
// -------------------------------
btnMic.addEventListener("click", async () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
        mediaRecorder.onstop = handleRecordedAudio;

        isRecording = true;
        recordingIndicator.style.display = "inline";
        mediaRecorder.start();
    } catch (err) {
        appendMessage("bot", "🎤 Cannot access microphone.");
    }
}

function stopRecording() {
    if (!mediaRecorder) return;
    isRecording = false;
    recordingIndicator.style.display = "none";
    mediaRecorder.stop();
}

btnStop.addEventListener("click", stopRecording);

// -------------------------------
// Voice Upload to Backend
// -------------------------------
async function handleRecordedAudio() {
    const blob = new Blob(audioChunks, { type: "audio/webm" });
    const formData = new FormData();

    formData.append("file", blob, "speech.webm");
    formData.append("language_code", langSelect.value);

    appendMessage("user", "🎤 Processing your voice...");

    try {
        const res = await fetch(`${BASE_URL}/speech`, {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        if (data.transcribed_text?.startsWith("ERROR")) {
            appendMessage("bot", "⚠️ Could not recognize speech.");
            return;
        }

        appendMessage("user", data.transcribed_text);

        // Send text to chatbot
        sendTextMessageFromVoice(data.transcribed_text);

    } catch (err) {
        appendMessage("bot", "⚠️ Voice upload error.");
    }
}

// -------------------------------
// Send Message from Voice
// -------------------------------
async function sendTextMessageFromVoice(text) {
    appendMessage("bot", "⏳ Thinking...");

    try {
        const res = await fetch(`${BASE_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user_message: text,
                language_code: langSelect.value,
                from_voice: true
            })
        });

        const data = await res.json();

        msgBox.lastChild.remove();
        appendMessage("bot", data.bot_response);
    } catch {
        msgBox.lastChild.remove();
        appendMessage("bot", "⚠️ Error processing speech text.");
    }
}

// -------------------------------
// Initial welcome message
// -------------------------------
appendMessage("bot", "👋 Hello! Ask me anything about pomegranate crop diseases.");
