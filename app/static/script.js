document.addEventListener('DOMContentLoaded', () => {
    const team1Select = document.getElementById('team1');
    const team2Select = document.getElementById('team2');
    const tossWinnerSelect = document.getElementById('toss_winner');
    const form = document.getElementById('prediction-form');

    const resultContainer = document.getElementById('result-container');
    const winnerName = document.getElementById('winner-name');

    const team1Name = document.getElementById('team1-name');
    const team2Name = document.getElementById('team2-name');
    const team1Prob = document.getElementById('team1-prob');
    const team2Prob = document.getElementById('team2-prob');
    const probBar = document.getElementById('prob-bar-fill');
    const confidenceText = document.getElementById('confidence-text');

    const predictBtn = document.getElementById('predict-btn');

    const updateTossOptions = () => {
        const t1 = team1Select.value;
        const t2 = team2Select.value;

        tossWinnerSelect.innerHTML = '<option value="" disabled selected>Who won the toss?</option>';

        if (t1) tossWinnerSelect.innerHTML += `<option value="${t1}">${t1}</option>`;
        if (t2) tossWinnerSelect.innerHTML += `<option value="${t2}">${t2}</option>`;
    };

    team1Select.addEventListener('change', () => {
        if (team1Select.value === team2Select.value && team1Select.value !== "") {
            team2Select.value = "";
        }
        updateTossOptions();
    });

    team2Select.addEventListener('change', () => {
        if (team1Select.value === team2Select.value && team2Select.value !== "") {
            team1Select.value = "";
        }
        updateTossOptions();
    });

    const getConfidence = (p1, p2) => {
        const diff = Math.abs(p1 - p2);

        if (diff > 40) return { text: "High Confidence", class: "conf-high" };
        if (diff > 20) return { text: "Medium Confidence", class: "conf-medium" };
        return { text: "Low Confidence", class: "conf-low" };
    };

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const payload = {
            team1: team1Select.value,
            team2: team2Select.value,
            venue: document.getElementById('venue').value,
            toss_winner: tossWinnerSelect.value,
            toss_decision: document.querySelector('input[name="toss_decision"]:checked').value
        };

        // Disable button
        predictBtn.textContent = "Analyzing Matchup...";
        predictBtn.style.opacity = "0.7";
        predictBtn.disabled = true;

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();

            if (data.success) {
                // 🎉 Winner
                winnerName.textContent = data.winner;

                // 📊 Probabilities
                team1Name.textContent = data.team1;
                team2Name.textContent = data.team2;

                team1Prob.textContent = `${data.team1_prob}%`;
                team2Prob.textContent = `${data.team2_prob}%`;

                // 📈 Animate bar
                setTimeout(() => {
                    probBar.style.width = `${data.team1_prob}%`;
                }, 100);

                // 🧠 Confidence
                const conf = getConfidence(data.team1_prob, data.team2_prob);
                confidenceText.textContent = conf.text;
                confidenceText.className = conf.class;

                // Show result
                resultContainer.classList.remove('hidden');
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

            } else {
                alert("Prediction failed: " + data.error);
            }

        } catch (err) {
            alert("Error connecting to prediction server.");
        } finally {
            predictBtn.textContent = "Predict Match Winner";
            predictBtn.style.opacity = "1";
            predictBtn.disabled = false;
        }
    });
});