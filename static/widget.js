// Chat Widget v1.0 - Knowledge Brain Chatbot
(function() {
    // Configuration - Users can override these
    const CONFIG = {
        apiUrl: window.CHATBOT_API_URL || 'https://advanced-8rg07o3h9-gat6.vercel.app',
        botName: window.CHATBOT_NAME || 'Knowledge Bot',
        botAvatar: window.CHATBOT_AVATAR || '🧠',
        primaryColor: window.CHATBOT_COLOR || '#533483',
        greeting: window.CHATBOT_GREETING || 'Hello! Ask me anything about our documents.',
    };

    let sessionId = localStorage.getItem('chatbot_session') || generateSessionId();
    let isOpen = false;
    let isLoading = false;

    function generateSessionId() {
        const id = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('chatbot_session', id);
        return id;
    }

    // Create the widget HTML
    function createWidget() {
        const widget = document.createElement('div');
        widget.id = 'chatbot-widget';
        widget.innerHTML = `
            <style>
                #chatbot-widget * {
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                }
                
                .chatbot-button {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    background: ${CONFIG.primaryColor};
                    color: white;
                    border: none;
                    cursor: pointer;
                    font-size: 24px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                    z-index: 9999;
                    transition: transform 0.3s;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .chatbot-button:hover {
                    transform: scale(1.1);
                }
                
                .chatbot-button.hidden {
                    display: none;
                }
                
                .chatbot-window {
                    position: fixed;
                    bottom: 90px;
                    right: 20px;
                    width: 380px;
                    height: 500px;
                    background: #16213e;
                    border-radius: 16px;
                    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
                    z-index: 9999;
                    display: none;
                    flex-direction: column;
                    overflow: hidden;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }
                
                .chatbot-window.open {
                    display: flex;
                }
                
                .chatbot-header {
                    background: linear-gradient(135deg, #0f3460, ${CONFIG.primaryColor});
                    color: white;
                    padding: 16px 20px;
                    font-weight: bold;
                    font-size: 16px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                
                .chatbot-header-buttons {
                    margin-left: auto;
                    display: flex;
                    gap: 8px;
                }
                
                .chatbot-header-btn {
                    background: rgba(255,255,255,0.2);
                    border: none;
                    color: white;
                    width: 28px;
                    height: 28px;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 14px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .chatbot-header-btn:hover {
                    background: rgba(255,255,255,0.3);
                }
                
                .chatbot-messages {
                    flex: 1;
                    overflow-y: auto;
                    padding: 16px;
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }
                
                .chatbot-message {
                    display: flex;
                    gap: 8px;
                    max-width: 85%;
                    animation: chatbotFadeIn 0.3s;
                }
                
                @keyframes chatbotFadeIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                
                .chatbot-message.user {
                    align-self: flex-end;
                    flex-direction: row-reverse;
                }
                
                .chatbot-avatar {
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 14px;
                    flex-shrink: 0;
                }
                
                .chatbot-message.bot .chatbot-avatar {
                    background: ${CONFIG.primaryColor};
                }
                
                .chatbot-message.user .chatbot-avatar {
                    background: #0f3460;
                }
                
                .chatbot-message-content {
                    padding: 10px 14px;
                    border-radius: 12px;
                    font-size: 14px;
                    line-height: 1.5;
                    color: white;
                    word-wrap: break-word;
                }
                
                .chatbot-message.bot .chatbot-message-content {
                    background: #1a1a3e;
                }
                
                .chatbot-message.user .chatbot-message-content {
                    background: ${CONFIG.primaryColor};
                }
                
                .chatbot-sources {
                    margin-top: 4px;
                    font-size: 10px;
                    color: #a0aec0;
                    font-style: italic;
                }
                
                .chatbot-input-area {
                    display: flex;
                    padding: 12px;
                    border-top: 1px solid #1a1a3e;
                    gap: 8px;
                }
                
                .chatbot-input {
                    flex: 1;
                    padding: 10px 14px;
                    border: 1px solid #2d2d5e;
                    border-radius: 20px;
                    background: #1a1a3e;
                    color: white;
                    font-size: 14px;
                    outline: none;
                    font-family: inherit;
                }
                
                .chatbot-input:focus {
                    border-color: ${CONFIG.primaryColor};
                }
                
                .chatbot-send-btn {
                    padding: 10px 18px;
                    background: ${CONFIG.primaryColor};
                    color: white;
                    border: none;
                    border-radius: 20px;
                    cursor: pointer;
                    font-size: 14px;
                    font-family: inherit;
                }
                
                .chatbot-send-btn:hover {
                    opacity: 0.9;
                }
                
                .chatbot-typing {
                    display: flex;
                    gap: 4px;
                    padding: 10px 14px;
                }
                
                .chatbot-typing span {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: ${CONFIG.primaryColor};
                    animation: chatbotBounce 1.4s infinite;
                }
                
                .chatbot-typing span:nth-child(2) { animation-delay: 0.2s; }
                .chatbot-typing span:nth-child(3) { animation-delay: 0.4s; }
                
                @keyframes chatbotBounce {
                    0%, 60%, 100% { transform: translateY(0); }
                    30% { transform: translateY(-6px); }
                }
                
                @media (max-width: 480px) {
                    .chatbot-window {
                        width: 100%;
                        height: 100%;
                        bottom: 0;
                        right: 0;
                        border-radius: 0;
                    }
                }
            </style>
            
            <button class="chatbot-button" id="chatbot-toggle">
                ${CONFIG.botAvatar}
            </button>
            
            <div class="chatbot-window" id="chatbot-window">
                <div class="chatbot-header">
                    <span>${CONFIG.botAvatar}</span>
                    ${CONFIG.botName}
                    <div class="chatbot-header-buttons">
                        <button class="chatbot-header-btn" id="chatbot-clear">🔄</button>
                        <button class="chatbot-header-btn" id="chatbot-close">✕</button>
                    </div>
                </div>
                <div class="chatbot-messages" id="chatbot-messages">
                    <div class="chatbot-message bot">
                        <div class="chatbot-avatar">${CONFIG.botAvatar}</div>
                        <div class="chatbot-message-content">${CONFIG.greeting}</div>
                    </div>
                </div>
                <div class="chatbot-input-area">
                    <input type="text" class="chatbot-input" id="chatbot-input" placeholder="Ask a question..." autofocus>
                    <button class="chatbot-send-btn" id="chatbot-send">Send</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(widget);
        
        // Event listeners
        document.getElementById('chatbot-toggle').addEventListener('click', toggleChat);
        document.getElementById('chatbot-close').addEventListener('click', closeChat);
        document.getElementById('chatbot-clear').addEventListener('click', clearChat);
        document.getElementById('chatbot-send').addEventListener('click', sendMessage);
        document.getElementById('chatbot-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    }

    function toggleChat() {
        isOpen = !isOpen;
        const window = document.getElementById('chatbot-window');
        const button = document.getElementById('chatbot-toggle');
        
        if (isOpen) {
            window.classList.add('open');
            button.classList.add('hidden');
            document.getElementById('chatbot-input').focus();
        } else {
            window.classList.remove('open');
            button.classList.remove('hidden');
        }
    }

    function closeChat() {
        isOpen = false;
        document.getElementById('chatbot-window').classList.remove('open');
        document.getElementById('chatbot-toggle').classList.remove('hidden');
    }

    async function clearChat() {
        try {
            await fetch(`${CONFIG.apiUrl}/api/conversation/${sessionId}`, { method: 'DELETE' });
            sessionId = generateSessionId();
        } catch(e) {}
        
        document.getElementById('chatbot-messages').innerHTML = `
            <div class="chatbot-message bot">
                <div class="chatbot-avatar">${CONFIG.botAvatar}</div>
                <div class="chatbot-message-content">Chat cleared. Ask me anything!</div>
            </div>
        `;
    }

    function addMessage(text, role, sources = []) {
        const messagesDiv = document.getElementById('chatbot-messages');
        const div = document.createElement('div');
        div.className = `chatbot-message ${role}`;
        
        let html = `<div class="chatbot-avatar">${role === 'user' ? '👤' : CONFIG.botAvatar}</div>`;
        html += `<div class="chatbot-message-content">${text.replace(/\n/g, '<br>')}`;
        
        if (sources.length > 0) {
            html += '<div class="chatbot-sources">';
            const seen = new Set();
            sources.forEach(s => {
                if (!seen.has(s.document)) {
                    seen.add(s.document);
                    html += `📄 ${s.document} `;
                }
            });
            html += '</div>';
        }
        
        html += '</div>';
        div.innerHTML = html;
        messagesDiv.appendChild(div);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function showTyping() {
        const messagesDiv = document.getElementById('chatbot-messages');
        const div = document.createElement('div');
        div.className = 'chatbot-message bot';
        div.id = 'chatbot-typing';
        div.innerHTML = `
            <div class="chatbot-avatar">${CONFIG.botAvatar}</div>
            <div class="chatbot-message-content">
                <div class="chatbot-typing"><span></span><span></span><span></span></div>
            </div>
        `;
        messagesDiv.appendChild(div);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function hideTyping() {
        const el = document.getElementById('chatbot-typing');
        if (el) el.remove();
    }

    async function sendMessage() {
        const input = document.getElementById('chatbot-input');
        const question = input.value.trim();
        if (!question || isLoading) return;
        
        isLoading = true;
        addMessage(question, 'user');
        input.value = '';
        showTyping();
        
        try {
            const res = await fetch(`${CONFIG.apiUrl}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, session_id: sessionId })
            });
            
            const data = await res.json();
            hideTyping();
            
            if (data.answer) {
                addMessage(data.answer, 'bot', data.sources || []);
            } else {
                addMessage('Sorry, I could not process that question.', 'bot');
            }
        } catch (e) {
            hideTyping();
            addMessage('Sorry, an error occurred. Please try again.', 'bot');
        }
        
        isLoading = false;
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createWidget);
    } else {
        createWidget();
    }
})();