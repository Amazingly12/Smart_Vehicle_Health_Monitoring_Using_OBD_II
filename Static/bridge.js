let selectedDatasetId = null;

// Dataset Choice
document.querySelectorAll('input[name="dataset-selection"]').forEach(input => {
    input.addEventListener('change', function () {
        selectedDatasetId = this.value;
        console.log("Selected Dataset:", selectedDatasetId); // Debugging
    });
});

// Model Choice
document.querySelectorAll('input[name="Model-selection"]').forEach(input => {
    input.addEventListener('change', async function () {
        if (!selectedDatasetId) {
            alert("Please select a Dataset first (Section 2)!");
            this.checked = false;
            return;
        }

        const resultsSection = document.getElementById('results-section');
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });

        if (this.value === "2") {
            document.getElementById('dynamic-title').innerText = "Vehicle Health Results";
            await handleLSTM();
        } else {
            document.getElementById('dynamic-title').innerText = "Compare & Check Variables";
            await handleVisualization();
        }
    });
});

async function handleLSTM() {
    const loading = document.getElementById('loading-spinner');
    const results = document.getElementById('lstm-results');
    const vizResults = document.getElementById('viz-results');
    const plotlyContainer = document.getElementById('plotly-chart');

    loading.style.display = 'block';
    results.style.display = 'none';
    vizResults.style.display = 'none';
    try {
        const response = await fetch(`http://127.0.0.1:8000/analyze/${selectedDatasetId}`);
        const result = await response.json();

        if (result.status === "success") {
            loading.style.display = 'none';
            document.getElementById('lstm-results').style.display = 'block';
            document.getElementById('plotly-chart').style.display = 'block';

            document.getElementById('res-score').innerText = result.health_score + "%";
            document.getElementById('res-faults').innerText = result.fault_events;

            const messageElem = document.getElementById('res-message');
            messageElem.innerText = result.condition;

            if (result.health_score >= 75) messageElem.style.color = "#28a745"; // Green
            else if (result.health_score >= 50) messageElem.style.color = "#ffc107"; // Yellow
            else messageElem.style.color = "#E40046"; // Red

            Plotly.newPlot('future-error-chart', JSON.parse(result.future_error_graph).data, JSON.parse(result.future_error_graph).layout, { responsive: true });
            Plotly.newPlot('health-decline-chart', JSON.parse(result.health_decline_graph).data, JSON.parse(result.health_decline_graph).layout, { responsive: true });
        }
    } catch (err) {
        console.error("LSTM Error:", err);
        loading.innerHTML = `<p style="color:red;">Error: Backend processing failed.</p>`;
    }
}

async function handleVisualization() {
    const vizResults = document.getElementById('viz-results');
    const loading = document.getElementById('loading-spinner');

    vizResults.style.display = 'none';
    loading.style.display = 'block';
    try {
        const response = await fetch(`http://127.0.0.1:8000/get-sensors/${selectedDatasetId}`);
        const data = await response.json();

        const s1 = document.getElementById('sensor1-select');
        const s2 = document.getElementById('sensor2-select');

        const options = data.sensors.map(s => `<option value="${s}">${s}</option>`).join('');
        s1.innerHTML = s2.innerHTML = options;

        console.log("Sensors loaded successfully");

        loading.style.display = 'none';
        vizResults.style.display = 'block';

    } catch (err) {
        console.error("Failed to load sensors:", err);
        loading.innerHTML = `<p style="color:red;">Error: Backend unreachable.</p>`;
    }
}

async function runVisualization() {
    const s1 = document.getElementById('sensor1-select').value;
    const s2 = document.getElementById('sensor2-select').value;
    const chartDiv = document.getElementById('plotly-chart');

    try {
        const response = await fetch(`http://127.0.0.1:8000/visualize/${selectedDatasetId}?sensor1=${s1}&sensor2=${s2}`);
        const result = await response.json();

        if (result.status === "success") {
            document.getElementById('corr-val').innerText = result.correlation;

            const graphData = JSON.parse(result.graph);

            console.log("Graph Data received:", graphData.data);

            chartDiv.style.display = 'block';
            Plotly.newPlot('plotly-chart', graphData.data, graphData.layout, { responsive: true });
        } else {
            alert("Error from server: " + result.detail);
        }
    } catch (err) {
        console.error("Visualization error:", err);
        alert("Failed to connect to the server. Is main.py running?");
    }
}