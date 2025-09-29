const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

const BACKEND_URL = "YOUR_CLOUD_RUN_SERVICE_URL"; // 👈 IMPORTANT: REPLACE THIS URL

// 💡 New variable to store the conversation session ID
let currentSessionId = null; 

function addMessage(text, sender) {
  const msg = document.createElement("div");
  msg.classList.add("message", sender);
  
  if (sender === "bot") {
    // 1. Replace raw newlines (\n) with HTML line breaks (<br>)
    let formattedText = text.replace(/\n/g, '<br>');
    
    // 2. Identify and wrap the suggested list items in a list structure
    // This targets lines starting with (1), (2), etc., which is what we asked the model to output.
    // It creates a basic structure for the list, though the line breaks handle most of the visual formatting.
    formattedText = formattedText.replace(/(\(\d+\) [^<]+)/g, '<p style="margin: 4px 0;">$1</p>');
    
    msg.innerHTML = formattedText;
  } else {
    // User messages are plain text
    msg.textContent = text;
  }
  
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  addMessage(message, "user");
  userInput.value = "";

  // 💡 Prepare the request body with the current message and session ID
  const requestBody = {
    message: message,
    // Add the session_id to the payload. It will be null on the first call.
    session_id: currentSessionId 
  };

  try {
    const res = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody), // Send the updated body
    });

    if (!res.ok) throw new Error("Network error or server returned error status");
    
    const data = await res.json();
    
    // 💡 Crucial update: Store the session_id returned by the server
    // This handles both the initial creation and subsequent updates
    if (data.session_id) {
        currentSessionId = data.session_id;
        console.log("Session ID updated:", currentSessionId);
    }
    
    addMessage(data.response || "No response", "bot");

  } catch (err) {
    console.error("Fetch Error:", err);
    addMessage("Error: Could not connect to server or process response. Check the console for details.", "bot");
  }
}

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});
