// widget.js – Full chat widget with session switcher, API key auth, and history
(function() {
    'use strict';

    const CONFIG = {
        apiUrl: window.CHATBOT_API_URL || window.location.origin,
        botName: window.CHATBOT_NAME || 'Knowledge Bot',
        botAvatar: window.CHATBOT_AVATAR || '🧠',
        primaryColor: window.CHATBOT_COLOR || '#6C63FF',
        secondaryColor: '#3F3D56',
        greeting: window.CHATBOT_GREETING || 'Hello! Ask me anything about our documents.',
        apiKey: window.CHATBOT_API_KEY || null,
    };

    console.log('🧠 Chat widget loaded');

    let sessionId = localStorage.getItem('chatbot_session') || 'session_' + Date.now();
    localStorage.setItem('chatbot_session', sessionId);

    let isOpen = false;
    let isLoading = false;
    let allSessions = [];

    function hasApiKey() {
        return CONFIG.apiKey && CONFIG.apiKey.length > 0;
    }

    // ── Build widget DOM ──
    function createWidget() {
        const widget = document.createElement('div');
        widget.id = 'chatbot-widget';
        widget.innerHTML = `
            <style>
                #chatbot-widget * { box-sizing: border-box; margin:0; padding:0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
                #chatbot-widget .chatbot-button {
                    position: fixed; bottom: 24px; right: 24px;
                    width: 60px; height: 60px; border-radius: 50%;
                    background: ${CONFIG.primaryColor}; color: #fff;
                    border: none; box-shadow: 0 6px 24px rgba(108, 99, 255, 0.4);
                    cursor: pointer; font-size: 28px; z-index: 99999;
                    display: flex; align-items: center; justify-content: center;
                    transition: transform 0.2s, box-shadow 0.2s;
                }
                #chatbot-widget .chatbot-button:hover { transform: scale(1.08); box-shadow: 0 8px 32px rgba(108, 99, 255, 0.5); }
                #chatbot-widget .chatbot-button.hidden { display: none; }
                #chatbot-widget .chatbot-window {
                    position: fixed; bottom: 100px; right: 24px;
                    width: 400px; max-width: calc(100vw - 48px);
                    height: 560px; max-height: calc(100vh - 140px);
                    background: #ffffff; border-radius: 20px;
                    box-shadow: 0 16px 60px rgba(0,0,0,0.25);
                    z-index: 99998; display: none;
                    flex-direction: column; overflow: hidden;
                    animation: slideUp 0.3s ease-out;
                }
                @keyframes slideUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
                #chatbot-widget .chatbot-window.open { display: flex; }
                #chatbot-widget .chatbot-header {
                    background: ${CONFIG.primaryColor}; color: #fff;
                    padding: 10px 16px;
                    display: flex; align-items: center; gap: 8px;
                    flex-shrink: 0;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                    flex-wrap: wrap;
                }
                #chatbot-widget .chatbot-header .bot-icon { font-size: 20px; }
                #chatbot-widget .chatbot-header .bot-name { font-size: 14px; font-weight: 600; flex: 1; min-width: 60px; }
                #chatbot-widget .chatbot-header .header-actions { display: flex; gap: 4px; flex-shrink: 0; }
                #chatbot-widget .chatbot-header .header-btn {
                    background: rgba(255,255,255,0.15); border: none; color: #fff;
                    width: 30px; height: 30px; border-radius: 6px; cursor: pointer; font-size: 14px;
                    display: flex; align-items: center; justify-content: center;
                    transition: background 0.2s;
                }
                #chatbot-widget .chatbot-header .header-btn:hover { background: rgba(255,255,255,0.3); }
                #chatbot-widget .chatbot-header .session-selector {
                    position: relative;
                    display: inline-block;
                }
                #chatbot-widget .chatbot-header .session-selector select {
                    background: rgba(255,255,255,0.2);
                    color: #fff;
                    border: none;
                    padding: 4px 8px;
                    border-radius: 12px;
                    font-size: 12px;
                    cursor: pointer;
                    outline: none;
                    max-width: 120px;
                    appearance: auto;
                }
                #chatbot-widget .chatbot-header .session-selector select option {
                    background: #2d2d5e;
                    color: #fff;
                }
                #chatbot-widget .chatbot-messages {
                    flex: 1; overflow-y: auto; padding: 12px 16px 8px 16px;
                    display: flex; flex-direction: column; gap: 10px;
                    background: #f8f9fc;
                }
                #chatbot-widget .chatbot-messages::-webkit-scrollbar { width: 4px; }
                #chatbot-widget .chatbot-messages::-webkit-scrollbar-thumb { background: #d0d5e0; border-radius: 4px; }
                #chatbot-widget .chatbot-message {
                    display: flex; gap: 10px; max-width: 85%;
                    animation: fadeIn 0.25s ease;
                }
                @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
                #chatbot-widget .chatbot-message.user { align-self: flex-end; flex-direction: row-reverse; }
                #chatbot-widget .chatbot-message .avatar {
                    width: 30px; height: 30px; border-radius: 50%;
                    flex-shrink: 0;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 14px;
                    background: ${CONFIG.primaryColor}; color: #fff;
                }
                #chatbot-widget .chatbot-message.user .avatar { background: ${CONFIG.secondaryColor}; }
                #chatbot-widget .chatbot-message .bubble {
                    padding: 10px 14px; border-radius: 14px;
                    font-size: 13px; line-height: 1.5;
                    background: #fff; color: #1e1e2f; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
                    word-break: break-word;
                }
                #chatbot-widget .chatbot-message.bot .bubble { border-bottom-left-radius: 4px; }
                #chatbot-widget .chatbot-message.user .bubble { background: ${CONFIG.primaryColor}; color: #fff; border-bottom-right-radius: 4px; }
                #chatbot-widget .chatbot-message .sources { margin-top: 4px; font-size: 10px; color: #8e95a9; display: flex; flex-wrap: wrap; gap: 4px 6px; }
                #chatbot-widget .chatbot-message .sources span { background: #f0f2f5; padding: 1px 8px; border-radius: 10px; }
                #chatbot-widget .chatbot-typing { display: flex; gap: 4px; padding: 6px 0; }
                #chatbot-widget .chatbot-typing span { width: 8px; height: 8px; border-radius: 50%; background: ${CONFIG.primaryColor}; animation: bounce 1.4s infinite; }
                #chatbot-widget .chatbot-typing span:nth-child(2) { animation-delay: 0.2s; }
                #chatbot-widget .chatbot-typing span:nth-child(3) { animation-delay: 0.4s; }
                @keyframes bounce { 0%,60%,100% { transform:translateY(0); } 30% { transform:translateY(-8px); } }
                #chatbot-widget .chatbot-input-area {
                    display: flex; gap: 8px; padding: 10px 14px;
                    background: #fff; border-top: 1px solid #eef0f4; flex-shrink: 0;
                }
                #chatbot-widget .chatbot-input-area input {
                    flex: 1; padding: 8px 14px; border: 1px solid #e2e6ed; border-radius: 20px;
                    font-size: 13px; outline: none; transition: border 0.2s; background: #f8f9fc;
                }
                #chatbot-widget .chatbot-input-area input:focus { border-color: ${CONFIG.primaryColor}; background: #fff; }
                #chatbot-widget .chatbot-input-area button {
                    padding: 8px 16px; background: ${CONFIG.primaryColor}; color: #fff;
                    border: none; border-radius: 20px; font-size: 13px; font-weight: 500; cursor: pointer;
                    transition: background 0.2s; white-space: nowrap;
                }
                #chatbot-widget .chatbot-input-area button:hover { background: #5a52d5; }
                @media (max-width:500px) {
                    #chatbot-widget .chatbot-window { bottom:0; right:0; width:100%; height:100%; max-height:100vh; border-radius:0; }
                    #chatbot-widget .chatbot-button { bottom:16px; right:16px; width:56px; height:56px; font-size:24px; }
                }
            </style>

            <button class="chatbot-button" id="chatbot-toggle">${CONFIG.botAvatar}</button>

            <div class="chatbot-window" id="chatbot-window">
                <div class="chatbot-header">
                    <span class="bot-icon">${CONFIG.botAvatar}</span>
                    <span class="bot-name">${CONFIG.botName}</span>
                    <div class="session-selector">
                        <select id="session-select" title="Switch chat session"></select>
                    </div>
                    <div class="header-actions">
                        <button class="header-btn" id="new-chat-btn" title="New chat">➕</button>
                        <button class="header-btn" id="chatbot-clear" title="Clear current chat">🗑</button>
                        <button class="header-btn" id="chatbot-close" title="Close">✕</button>
                    </div>
                </div>

                <div class="chatbot-messages" id="chatbot-messages">
                    <div class="chatbot-message bot">
                        <div class="avatar">${CONFIG.botAvatar}</div>
                        <div class="bubble">${CONFIG.greeting}</div>
                    </div>
                </div>

                <div class="chatbot-input-area">
                    <input id="chatbot-input" placeholder="Ask a question..." autofocus>
                    <button id="chatbot-send">Send</button>
                </div>
            </div>
        `;
        document.body.appendChild(widget);

        if (!hasApiKey()) {
            const msgs = document.getElementById('chatbot-messages');
            msgs.innerHTML = `
                <div class="chatbot-message bot">
                    <div class="avatar">${CONFIG.botAvatar}</div>
                    <div class="bubble">❌ Missing API key. Please set window.CHATBOT_API_KEY.</div>
                </div>
            `;
            document.getElementById('chatbot-input').disabled = true;
            document.getElementById('chatbot-send').disabled = true;
            document.getElementById('session-select').disabled = true;
        } else {
            document.getElementById('chatbot-input').disabled = false;
            document.getElementById('chatbot-send').disabled = false;
            // Load sessions and history
            loadSessions();
        }

        // ── Event listeners ──
        document.getElementById('chatbot-toggle').addEventListener('click', toggleChat);
        document.getElementById('chatbot-close').addEventListener('click', closeChat);
        document.getElementById('chatbot-clear').addEventListener('click', clearCurrentChat);
        document.getElementById('new-chat-btn').addEventListener('click', newChat);
        document.getElementById('chatbot-send').addEventListener('click', sendMessage);
        document.getElementById('chatbot-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        document.getElementById('session-select').addEventListener('change', onSessionChange);
    }

    // ── Load sessions ──
    async function loadSessions() {
        if (!hasApiKey()) return;
        try {
            const res = await fetch(`${CONFIG.apiUrl}/api/sessions`, {
                headers: { 'X-API-Key': CONFIG.apiKey }
            });
            if (!res.ok) return;
            const data = await res.json();
            allSessions = data.sessions || [];
            populateSessionDropdown();
            // If current session is not in the list, add it (but it may not have history yet)
            if (!allSessions.includes(sessionId)) {
                allSessions.push(sessionId);
            }
            // Select current session in dropdown
            document.getElementById('session-select').value = sessionId;
            // Load history for current session
            loadHistory(sessionId);
        } catch (e) {
            console.warn('Could not load sessions:', e);
        }
    }

    function populateSessionDropdown() {
        const select = document.getElementById('session-select');
        select.innerHTML = '';
        // Show sessions in reverse chronological order (newest first) – we sort by timestamp if available
        // For simplicity, we just display them as is.
        allSessions.forEach(sid => {
            const option = document.createElement('option');
            option.value = sid;
            // Display a shortened label
            const label = sid.replace('session_', 'Chat ');
            option.textContent = label.length > 15 ? label.slice(0, 12) + '…' : label;
            select.appendChild(option);
        });
        if (allSessions.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No chats';
            select.appendChild(option);
        }
    }

    function onSessionChange() {
        const select = document.getElementById('session-select');
        const newSession = select.value;
        if (newSession && newSession !== sessionId) {
            sessionId = newSession;
            localStorage.setItem('chatbot_session', sessionId);
            loadHistory(sessionId);
        }
    }

    // ── Load history ──
    async function loadHistory(sid) {
        if (!hasApiKey() || !sid) return;
        try {
            const res = await fetch(`${CONFIG.apiUrl}/api/history?session_id=${sid}&last_n=30`, {
                headers: { 'X-API-Key': CONFIG.apiKey }
            });
            if (!res.ok) return;
            const data = await res.json();
            const container = document.getElementById('chatbot-messages');
            container.innerHTML = '';
            if (data.history && data.history.length > 0) {
                data.history.forEach(msg => {
                    addMessage(msg.content, msg.role, []);
                });
            } else {
                // No history, show greeting
                addMessage(CONFIG.greeting, 'bot', []);
            }
            container.scrollTop = container.scrollHeight;
        } catch (e) {
            console.warn('Could not load history:', e);
        }
    }

    // ── Toggle chat ──
    function toggleChat() {
        if (!hasApiKey()) {
            alert('Please set window.CHATBOT_API_KEY to use the widget.');
            return;
        }
        isOpen = !isOpen;
        const win = document.getElementById('chatbot-window');
        const btn = document.getElementById('chatbot-toggle');
        if (isOpen) {
            win.classList.add('open');
            btn.classList.add('hidden');
            setTimeout(() => document.getElementById('chatbot-input').focus(), 200);
            // Refresh sessions and history
            loadSessions();
        } else {
            win.classList.remove('open');
            btn.classList.remove('hidden');
        }
    }

    function closeChat() {
        isOpen = false;
        document.getElementById('chatbot-window').classList.remove('open');
        document.getElementById('chatbot-toggle').classList.remove('hidden');
    }

    // ── New Chat ──
    async function newChat() {
        // Generate new session_id
        sessionId = 'session_' + Date.now();
        localStorage.setItem('chatbot_session', sessionId);
        // Add to sessions list and dropdown
        allSessions.push(sessionId);
        populateSessionDropdown();
        document.getElementById('session-select').value = sessionId;
        // Clear UI and show greeting
        const container = document.getElementById('chatbot-messages');
        container.innerHTML = '';
        addMessage(CONFIG.greeting, 'bot', []);
        document.getElementById('chatbot-input').focus();
    }

    // ── Clear current chat ──
    async function clearCurrentChat() {
        if (!sessionId) return;
        try {
            await fetch(`${CONFIG.apiUrl}/api/conversation/${sessionId}`, {
                method: 'DELETE',
                headers: { 'X-API-Key': CONFIG.apiKey }
            });
        } catch(e) {}
        // Remove from sessions list
        const index = allSessions.indexOf(sessionId);
        if (index > -1) allSessions.splice(index, 1);
        populateSessionDropdown();
        // If no sessions left, create a new one
        if (allSessions.length === 0) {
            newChat();
        } else {
            // Select the first session
            sessionId = allSessions[0];
            localStorage.setItem('chatbot_session', sessionId);
            document.getElementById('session-select').value = sessionId;
            loadHistory(sessionId);
        }
    }

    // ── Add message ──
    function addMessage(text, role, sources = []) {
        const container = document.getElementById('chatbot-messages');
        const div = document.createElement('div');
        div.className = `chatbot-message ${role}`;
        const avatar = role === 'user' ? '👤' : CONFIG.botAvatar;
        let html = `<div class="avatar">${avatar}</div>`;
        html += `<div class="bubble">${text.replace(/\\n/g, '<br>')}`;
        if (sources && sources.length > 0) {
            html += `<div class="sources">`;
            const seen = new Set();
            sources.forEach(s => {
                if (!seen.has(s.document)) {
                    seen.add(s.document);
                    html += `<span>📄 ${s.document}</span>`;
                }
            });
            html += `</div>`;
        }
        html += `</div>`;
        div.innerHTML = html;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }

    // ── Typing indicator ──
    function showTyping() {
        const container = document.getElementById('chatbot-messages');
        const div = document.createElement('div');
        div.className = 'chatbot-message bot';
        div.id = 'chatbot-typing';
        div.innerHTML = `
            <div class="avatar">${CONFIG.botAvatar}</div>
            <div class="bubble"><div class="chatbot-typing"><span></span><span></span><span></span></div></div>
        `;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }

    function hideTyping() {
        const el = document.getElementById('chatbot-typing');
        if (el) el.remove();
    }

    // ── Send message ──
    async function sendMessage() {
        const input = document.getElementById('chatbot-input');
        const question = input.value.trim();
        if (!question || isLoading) return;
        if (!hasApiKey()) {
            alert('Missing API key.');
            return;
        }

        isLoading = true;
        addMessage(question, 'user', []);
        input.value = '';
        showTyping();

        try {
            const res = await fetch(`${CONFIG.apiUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': CONFIG.apiKey
                },
                body: JSON.stringify({ question, session_id: sessionId })
            });

            const data = await res.json();

            if (res.status === 401) {
                hideTyping();
                addMessage('🔐 Invalid API key. Please check your configuration.', 'bot');
                document.getElementById('chatbot-input').disabled = true;
                document.getElementById('chatbot-send').disabled = true;
                isLoading = false;
                return;
            }

            hideTyping();
            if (data.answer) {
                addMessage(data.answer, 'bot', data.sources || []);
            } else {
                addMessage('⚠️ Could not process your question.', 'bot');
            }
        } catch (e) {
            console.error('Widget error:', e);
            hideTyping();
            addMessage('⚠️ Network error. Please try again.', 'bot');
        }
        isLoading = false;
    }

    // ── Initialize ──
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createWidget);
    } else {
        createWidget();
    }
})();