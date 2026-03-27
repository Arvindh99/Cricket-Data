document.addEventListener('DOMContentLoaded', () => {
    const team1Select = document.getElementById('team1');
    const team2Select = document.getElementById('team2');
    const tossWinnerSelect = document.getElementById('toss_winner');
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const winnerName = document.getElementById('winner-name');
    const predictBtn = document.getElementById('predict-btn');

    // Update toss winner options based on selected teams
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
            alert("Team 2 must differ from Team 1");
        }
        updateTossOptions();
    });

    team2Select.addEventListener('change', () => {
        if (team1Select.value === team2Select.value && team2Select.value !== "") {
            team1Select.value = "";
            alert("Team 1 must differ from Team 2");
        }
        updateTossOptions();
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const payload = {
            team1: team1Select.value,
            team2: team2Select.value,
            venue: document.getElementById('venue').value,
            toss_winner: tossWinnerSelect.value,
            toss_decision: document.querySelector('input[name="toss_decision"]:checked').value
        };

        predictBtn.textContent = "Analyzing Matchup...";
        predictBtn.style.opacity = "0.7";

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if (data.success) {
                winnerName.textContent = data.winner;
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
        }
    });
});
