/* ═══════════════════════════════════════════════════════════
   PREDICTOR
═══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {

    const team1Select      = document.getElementById('team1');
    const team2Select      = document.getElementById('team2');
    const tossWinnerSelect = document.getElementById('toss_winner');
    const form             = document.getElementById('prediction-form');
    const resultContainer  = document.getElementById('result-container');
    const winnerName       = document.getElementById('winner-name');
    const team1Name        = document.getElementById('team1-name');
    const team2Name        = document.getElementById('team2-name');
    const team1Prob        = document.getElementById('team1-prob');
    const team2Prob        = document.getElementById('team2-prob');
    const probBar          = document.getElementById('prob-bar-fill');
    const confidenceText   = document.getElementById('confidence-text');
    const predictBtn       = document.getElementById('predict-btn');

    /* Sync toss dropdown to selected teams */
    const updateTossOptions = () => {
        const t1 = team1Select.value;
        const t2 = team2Select.value;
        tossWinnerSelect.innerHTML = '<option value="" disabled selected>Who won the toss?</option>';
        if (t1) tossWinnerSelect.innerHTML += `<option value="${t1}">${t1}</option>`;
        if (t2) tossWinnerSelect.innerHTML += `<option value="${t2}">${t2}</option>`;
    };

    team1Select.addEventListener('change', () => {
        if (team1Select.value === team2Select.value && team1Select.value !== '') {
            team2Select.value = '';
        }
        updateTossOptions();
    });

    team2Select.addEventListener('change', () => {
        if (team1Select.value === team2Select.value && team2Select.value !== '') {
            team1Select.value = '';
        }
        updateTossOptions();
    });

    /* Confidence label */
    const getConfidence = (p1, p2) => {
        const diff = Math.abs(p1 - p2);
        if (diff > 40) return { text: 'High Confidence',   cls: 'conf-high' };
        if (diff > 20) return { text: 'Medium Confidence', cls: 'conf-medium' };
        return             { text: 'Low Confidence',    cls: 'conf-low' };
    };

    /* Submit */
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const payload = {
            team1:         team1Select.value,
            team2:         team2Select.value,
            venue:         document.getElementById('venue').value,
            toss_winner:   tossWinnerSelect.value,
            toss_decision: document.querySelector('input[name="toss_decision"]:checked').value
        };

        predictBtn.textContent  = 'Analyzing Matchup…';
        predictBtn.style.opacity = '0.7';
        predictBtn.disabled      = true;

        try {
            const res  = await fetch('/predict', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify(payload)
            });
            const data = await res.json();

            if (data.success) {
                winnerName.textContent  = data.winner;
                team1Name.textContent   = data.team1;
                team2Name.textContent   = data.team2;
                team1Prob.textContent   = `${data.team1_prob}%`;
                team2Prob.textContent   = `${data.team2_prob}%`;

                setTimeout(() => { probBar.style.width = `${data.team1_prob}%`; }, 100);

                const conf = getConfidence(data.team1_prob, data.team2_prob);
                confidenceText.textContent = conf.text;
                confidenceText.className   = conf.cls;

                resultContainer.classList.remove('hidden');
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

            } else {
                alert('Prediction failed: ' + data.error);
            }

        } catch {
            alert('Error connecting to prediction server.');
        } finally {
            predictBtn.textContent   = 'Predict Match Winner';
            predictBtn.style.opacity = '1';
            predictBtn.disabled      = false;
        }
    });
});


/* ═══════════════════════════════════════════════════════════
   TAB SWITCHING
═══════════════════════════════════════════════════════════ */
(() => {
    const tabs      = document.querySelectorAll('.tab-btn');
    const panels    = document.querySelectorAll('.panel');
    const container = document.querySelector('.container');

    tabs.forEach(btn => {
        btn.addEventListener('click', () => {
            tabs.forEach(t   => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById('panel-' + btn.dataset.tab).classList.add('active');

            /* Widen the card when on the season tracker tab */
            if (btn.dataset.tab === 'season') {
                container.classList.add('wide');
                if (!window._seasonLoaded) {
                    loadSeasonResults();
                    window._seasonLoaded = true;
                }
            } else {
                container.classList.remove('wide');
            }
        });
    });
})();


/* ═══════════════════════════════════════════════════════════
   SEASON TRACKER
═══════════════════════════════════════════════════════════ */
(() => {
    let allMatches   = [];
    let activeFilter = 'all';

    /* ── Fetch from Flask API ─────────────────────────── */
    window.loadSeasonResults = async function () {
        try {
            const res  = await fetch('/season-results');
            const data = await res.json();

            /* Stats cards */
            document.getElementById('stat-total').textContent    = data.total_matches;
            document.getElementById('stat-correct').textContent  = data.correct;
            document.getElementById('stat-wrong').textContent    = data.incorrect;
            document.getElementById('stat-accuracy').textContent = data.accuracy + '%';

            /* Accuracy bar */
            document.getElementById('bar-accuracy-label').textContent = data.accuracy + '%';
            setTimeout(() => {
                document.getElementById('accuracy-bar').style.width = data.accuracy + '%';
            }, 150);

            /* Last updated */
            if (data.last_updated) {
                document.getElementById('last-updated').textContent =
                    'Last updated: ' + data.last_updated;
            }

            allMatches = data.matches || [];
            renderTable(activeFilter);

        } catch {
            document.getElementById('table-body-wrapper').innerHTML =
                `<div class="empty-state">
                    <div class="emoji">⚠️</div>
                    <p>Could not load results. Run evaluate_season.py first.</p>
                </div>`;
        }
    };

    /* ── Build table rows ─────────────────────────────── */
    function renderTable(filter) {
        const wrapper  = document.getElementById('table-body-wrapper');

        const filtered = allMatches.filter(m => {
            if (filter === 'correct') return m.correct === true;
            if (filter === 'wrong')   return m.correct === false;
            return true;
        });

        if (filtered.length === 0) {
            wrapper.innerHTML =
                `<div class="empty-state">
                    <div class="emoji">🏏</div>
                    <p>${allMatches.length === 0
                        ? 'No completed matches yet. Results will appear once evaluate_season.py runs.'
                        : 'No matches match this filter.'
                    }</p>
                </div>`;
            return;
        }

        const rows = filtered.map(m => {
            const badge = m.correct
                ? `<span class="badge badge-correct">✓</span>`
                : `<span class="badge badge-wrong">✗</span>`;

            const actualClass = m.correct ? 'td-correct' : 'td-wrong';
            const pw          = m.team1_prob ?? 50;

            return `
            <tr>
                <td class="match-cell">
                    <div class="match-teams">${m.team1} <span style="font-weight:400;color:var(--text-muted)">vs</span> ${m.team2}</div>
                    <div class="match-venue">${m.venue}</div>
                </td>
                <td class="td-predicted">${m.predicted_winner}</td>
                <td class="${actualClass}">${m.actual_winner}</td>
                <td>
                    <span class="prob-nums">${m.team1_prob}% · ${m.team2_prob}%</span>
                    <div class="mini-bar-track">
                        <div class="mini-bar-fill" style="width:${pw}%"></div>
                    </div>
                </td>
                <td>${badge}</td>
            </tr>`;
        }).join('');

        wrapper.innerHTML = `
            <table>
                <colgroup>
                    <col class="col-match">
                    <col class="col-pred">
                    <col class="col-actual">
                    <col class="col-prob">
                    <col class="col-result">
                </colgroup>
                <thead>
                    <tr>
                        <th>Match</th>
                        <th>Predicted Winner</th>
                        <th>Actual Winner</th>
                        <th>Win Prob</th>
                        <th>Result</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>`;
    }

    /* ── Filter tab clicks ────────────────────────────── */
    document.querySelectorAll('.filter-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeFilter = btn.dataset.filter;
            renderTable(activeFilter);
        });
    });
})();