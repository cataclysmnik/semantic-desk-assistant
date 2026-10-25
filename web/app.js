// Desk Assistant - Mobile Monitor Client Logic
(function () {
  const API_STATUS = '/api/status';
  const API_IMAGE = '/api/image';
  
  let currentSession = null;
  let sessionStartTime = Date.now();
  let pollInterval = null;
  let isConnected = true;

  // DOM Elements
  const sessionCard = document.getElementById('session-card');
  const sessionName = document.getElementById('session-name');
  const sessionDesc = document.getElementById('session-desc');
  const sessionTimer = document.getElementById('session-timer');
  const sessionGlow = document.getElementById('session-glow');

  const scoreVal = document.getElementById('score-val');
  const gradeBadge = document.getElementById('grade-badge');
  const gaugeBar = document.getElementById('gauge-bar');

  const timeWork = document.getElementById('time-work');
  const timeStudy = document.getElementById('time-study');
  const timeBreak = document.getElementById('time-break');
  const timeIdle = document.getElementById('time-idle');

  const camTimestamp = document.getElementById('cam-timestamp');
  const feedImg = document.getElementById('feed-img');
  const refreshImgBtn = document.getElementById('refresh-img-btn');

  const objectCountBadge = document.getElementById('object-count-badge');
  const objectsContainer = document.getElementById('objects-container');

  const eventsContainer = document.getElementById('events-container');
  const lastSyncTag = document.getElementById('last-sync-tag');
  const connectionDot = document.getElementById('connection-dot');
  const liveIndicator = document.getElementById('live-indicator');
  const footerIp = document.getElementById('footer-ip');

  // Tabs
  const tabBtnLive = document.getElementById('tab-btn-live');
  const tabBtnHistory = document.getElementById('tab-btn-history');
  const paneLive = document.getElementById('pane-live');
  const paneHistory = document.getElementById('pane-history');

  function switchTab(activeBtn, activePane, inactiveBtn, inactivePane) {
    if (!activeBtn || !activePane) return;
    activeBtn.classList.add('active');
    inactiveBtn.classList.remove('active');
    activePane.classList.add('active');
    inactivePane.classList.remove('active');
  }

  if (tabBtnLive && tabBtnHistory) {
    tabBtnLive.addEventListener('click', () => switchTab(tabBtnLive, paneLive, tabBtnHistory, paneHistory));
    tabBtnHistory.addEventListener('click', () => switchTab(tabBtnHistory, paneHistory, tabBtnLive, paneLive));
  }

  // Descriptions for each session in Nothing OS style
  const SESSION_INFO = {
    WORK: { desc: '// FOCUSED ON COMPUTER / WORKSPACE', theme: 'theme-work', color: '#E50000' },
    STUDY: { desc: '// DEEP READING / NOTEBOOK ACTIVE', theme: 'theme-study', color: '#ffffff' },
    BREAK: { desc: '// RECESS / ACTIVE BREAK DETECTED', theme: 'theme-break', color: '#FF5500' },
    IDLE: { desc: '// DESK AREA UNOCCUPIED', theme: 'theme-idle', color: '#777777' }
  };

  // Format seconds to "Xm Ys" or "Xh Ym"
  function formatDuration(seconds) {
    seconds = Math.max(0, Math.floor(seconds || 0));
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m ${s < 10 ? '0' : ''}${s}s`;
  }

  // Format seconds to HH:MM:SS stopwatch
  function formatTimer(ms) {
    const totalSecs = Math.max(0, Math.floor(ms / 1000));
    const h = Math.floor(totalSecs / 3600);
    const m = Math.floor((totalSecs % 3600) / 60);
    const s = totalSecs % 60;
    return `${h < 10 ? '0' : ''}${h}:${m < 10 ? '0' : ''}${m}:${s < 10 ? '0' : ''}${s}`;
  }

  // Update Radial Gauge Ring
  function updateGauge(score) {
    score = Math.max(0, Math.min(100, score || 0));
    scoreVal.textContent = `${score}%`;
    
    // Circle circumference for r=48 is 2 * PI * 48 = ~301.59
    const circumference = 301.59;
    const offset = circumference - (score / 100) * circumference;
    gaugeBar.style.strokeDashoffset = offset;

    // Nothing OS Color Scheme
    if (score >= 75) gaugeBar.style.stroke = '#ffffff'; // Nothing White
    else if (score >= 50) gaugeBar.style.stroke = '#ffffff';
    else if (score >= 25) gaugeBar.style.stroke = '#FF5500'; // Nothing Amber
    else gaugeBar.style.stroke = '#E50000'; // Nothing Red
  }

  // Update UI with incoming status data
  function renderStatus(data) {
    if (!data) return;

    // 1. Session State
    const session = (data.session || 'IDLE').toUpperCase();
    if (session !== currentSession) {
      currentSession = session;
      sessionStartTime = Date.now();
      
      // Update theme classes
      sessionCard.className = 'card hero-card ' + (SESSION_INFO[session]?.theme || 'theme-idle');
    }

    sessionName.textContent = session;
    sessionDesc.textContent = SESSION_INFO[session]?.desc || 'Monitoring desk';

    // 2. Productivity Score & Durations
    const score = data.score !== undefined ? data.score : 0;
    updateGauge(score);
    gradeBadge.textContent = data.grade || '—';
    if (score >= 75) {
      gradeBadge.style.color = '#ffffff';
      gradeBadge.style.borderColor = '#ffffff';
    } else if (score >= 50) {
      gradeBadge.style.color = '#ffffff';
      gradeBadge.style.borderColor = '#888888';
    } else if (score >= 25) {
      gradeBadge.style.color = '#FF5500';
      gradeBadge.style.borderColor = '#FF5500';
    } else {
      gradeBadge.style.color = '#E50000';
      gradeBadge.style.borderColor = '#E50000';
    }

    const durations = data.durations || {};
    timeWork.textContent = formatDuration(durations.WORK);
    timeStudy.textContent = formatDuration(durations.STUDY);
    timeBreak.textContent = formatDuration(durations.BREAK);
    timeIdle.textContent = formatDuration(durations.IDLE);

    // 3. Objects List
    const objects = data.objects || {};
    const objKeys = Object.keys(objects);
    let totalItems = 0;
    
    if (objKeys.length === 0) {
      objectsContainer.innerHTML = '<div class="empty-state">No objects currently on desk</div>';
      objectCountBadge.textContent = '0 items';
    } else {
      let html = '';
      objKeys.forEach(name => {
        const count = objects[name];
        totalItems += count;
        html += `
          <div class="object-item">
            <span class="object-name">${name}</span>
            <span class="object-count">×${count}</span>
          </div>
        `;
      });
      objectsContainer.innerHTML = html;
      objectCountBadge.textContent = `${totalItems} ${totalItems === 1 ? 'item' : 'items'}`;
    }

    // 4. Events Log
    const events = data.recent_events || [];
    if (events.length === 0) {
      eventsContainer.innerHTML = '<div class="empty-state">No recent activity logged</div>';
    } else {
      let html = '';
      // Reverse to show most recent at the top
      events.slice(-15).reverse().forEach(evStr => {
        // Expected format: "[HH:MM:SS] Message..."
        const match = evStr.match(/^\[(.*?)\]\s*(.*)$/);
        let timePart = '';
        let msgPart = evStr;
        if (match) {
          timePart = match[1];
          msgPart = match[2];
        }

        const isHighlight = msgPart.toLowerCase().includes('started') || msgPart.toLowerCase().includes('detected:');
        html += `
          <div class="event-entry ${isHighlight ? 'highlight' : ''}">
            <span class="event-time">${timePart}</span>
            <span class="event-msg">${msgPart}</span>
          </div>
        `;
      });
      eventsContainer.innerHTML = html;
    }

    // 5. Host IP and Timestamp
    if (data.ip) {
      footerIp.textContent = `${data.ip}:${data.port || 8080}`;
    }
    const now = new Date();
    lastSyncTag.textContent = now.toLocaleTimeString();
    camTimestamp.textContent = data.last_capture_time || now.toLocaleTimeString();

    // 6. Refresh camera image if filename changed
    if (data.latest_image) {
      feedImg.src = `${API_IMAGE}?t=${encodeURIComponent(data.latest_image)}&_r=${Date.now()}`;
    }

    // 7. Past Sessions Tab Rendering
    const pastSessions = data.past_sessions || [];
    const historyCounter = document.getElementById('history-counter');
    const historyTotalBadge = document.getElementById('history-total-badge');
    const statTotalSessions = document.getElementById('stat-total-sessions');
    const statFocusTime = document.getElementById('stat-focus-time');
    const statLongestSession = document.getElementById('stat-longest-session');
    const timelineStrip = document.getElementById('timeline-strip');
    const pastSessionsList = document.getElementById('past-sessions-list');

    if (historyCounter) historyCounter.textContent = pastSessions.length;
    if (historyTotalBadge) historyTotalBadge.textContent = `${pastSessions.length} RECORDED`;
    if (statTotalSessions) statTotalSessions.textContent = pastSessions.length;

    let focusSecs = 0;
    let longestSecs = 0;

    pastSessions.forEach(s => {
      const dur = s.duration_seconds || 0;
      if (s.session === 'WORK' || s.session === 'STUDY') {
        focusSecs += dur;
      }
      if (dur > longestSecs) {
        longestSecs = dur;
      }
    });

    if (statFocusTime) statFocusTime.textContent = formatDuration(focusSecs);
    if (statLongestSession) {
      statLongestSession.textContent = longestSecs > 0 ? formatDuration(longestSecs) : '0m';
    }

    // Render Visual Timeline Strip
    if (timelineStrip) {
      if (pastSessions.length === 0) {
        timelineStrip.innerHTML = '<div style="flex:1; background:#111111; display:flex; align-items:center; justify-content:center; font-size:10px; color:#555;">NO DATA YET</div>';
      } else {
        let stripHtml = '';
        pastSessions.forEach(s => {
          const typeClass = (s.session || 'idle').toLowerCase();
          const dur = Math.max(1, s.duration_seconds || 1);
          const flexGrow = Math.max(1, Math.round(dur / 5));
          const tooltip = `[${s.session}] ${s.start_time} - ${s.end_time} (${formatDuration(dur)})`;
          stripHtml += `<div class="timeline-segment ${typeClass}" style="flex:${flexGrow}" title="${tooltip}"></div>`;
        });
        timelineStrip.innerHTML = stripHtml;
      }
    }

    // Render Past Sessions List (Latest first)
    if (pastSessionsList) {
      if (pastSessions.length === 0) {
        pastSessionsList.innerHTML = '<div class="empty-state">NO PREVIOUS SESSIONS RECORDED YET</div>';
      } else {
        let listHtml = '';
        [...pastSessions].reverse().forEach(s => {
          const type = (s.session || 'IDLE').toUpperCase();
          const typeClass = 'type-' + type.toLowerCase();
          const dur = s.duration_seconds || 0;
          const range = `${s.start_time || '--:--'} &rarr; ${s.end_time || '--:--'}`;
          const date = s.date || '';

          listHtml += `
            <div class="past-session-item ${typeClass}">
              <div class="past-sess-left">
                <span class="past-sess-tag">[${type}]</span>
                <span class="past-sess-times">${range}</span>
              </div>
              <div class="past-sess-right">
                <span class="past-sess-duration dot-font">${formatDuration(dur)}</span>
                <span class="past-sess-date">${date}</span>
              </div>
            </div>
          `;
        });
        pastSessionsList.innerHTML = listHtml;
      }
    }
  }

  // Refresh feed image manually
  function refreshImage() {
    feedImg.src = `${API_IMAGE}?_r=${Date.now()}`;
  }

  if (refreshImgBtn) {
    refreshImgBtn.addEventListener('click', refreshImage);
  }

  // Fetch status from server
  async function fetchStatus() {
    try {
      const res = await fetch(API_STATUS, { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      
      if (!isConnected) {
        isConnected = true;
        connectionDot.className = 'radar-dot active';
        liveIndicator.querySelector('.live-text').textContent = 'ONLINE';
      }
      
      renderStatus(data);
    } catch (err) {
      if (isConnected) {
        isConnected = false;
        connectionDot.className = 'radar-dot offline';
        liveIndicator.querySelector('.live-text').textContent = 'RETRYING';
      }
    }
  }

  // Local Stopwatch Loop for current session
  setInterval(() => {
    if (sessionStartTime) {
      const elapsed = Date.now() - sessionStartTime;
      sessionTimer.textContent = formatTimer(elapsed);
    }
  }, 1000);

  // Poll server every 2.5 seconds
  fetchStatus();
  pollInterval = setInterval(fetchStatus, 2500);

  // Refresh image every 5 seconds independently
  setInterval(refreshImage, 5000);
})();
