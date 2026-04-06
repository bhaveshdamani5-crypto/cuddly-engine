// Telemetry Chart Setup
const ctx = document.getElementById('telemetryChart').getContext('2d');
const telemetryChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            { label: 'CO2 (ppm)', borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', data: [], tension: 0.4, fill: true, yAxisID: 'y' },
            { label: 'PM2.5', borderColor: '#8b5cf6', backgroundColor: 'rgba(139, 92, 246, 0.1)', data: [], tension: 0.4, fill: true, yAxisID: 'y1' }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        plugins: {
            legend: { labels: { color: '#f0f4f8' } }
        },
        scales: {
            x: { display: false },
            y: { position: 'left', grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#94a3b8' } },
            y1: { position: 'right', grid: { display: false }, ticks: { color: '#94a3b8' } }
        }
    }
});

// The error chart has been removed as it was for autoencoders. We will use numerical values for actions.
// Internal State
let currentAnomaly = 'normal';
let baseTemp = 24.5;
let baseHum = 45.0;
let baseCo2 = 410;
let basePm = 15;

function injectAnomaly(type) {
    currentAnomaly = type;
}

// Generate Sensor Data based on state
function generateSensorData() {
    let t = baseTemp + (Math.random() * 0.5 - 0.25);
    let h = baseHum + (Math.random() * 2 - 1);
    let c = baseCo2 + (Math.random() * 10 - 5);
    let p = basePm + (Math.random() * 2 - 1);

    if (currentAnomaly === 'fire') {
        t += 15 + Math.random() * 5;
        c += 400 + Math.random() * 100;
        p += 50 + Math.random() * 20;
    } else if (currentAnomaly === 'leak') {
        c += 800 + Math.random() * 200;
    }

    return { temperature: t, humidity: h, co2: c, pm25: p };
}

// Update DOM
function updateDashboard(data, prediction) {
    // Sensor values
    document.getElementById('val-temp').innerText = data.temperature.toFixed(1) + ' °C';
    document.getElementById('val-hum').innerText = data.humidity.toFixed(1) + ' %';
    document.getElementById('val-co2').innerText = data.co2.toFixed(0) + ' ppm';
    document.getElementById('val-pm').innerText = data.pm25.toFixed(0) + ' µg/m³';

    // Telemetry Chart
    const time = new Date().toLocaleTimeString();
    telemetryChart.data.labels.push(time);
    telemetryChart.data.datasets[0].data.push(data.co2);
    telemetryChart.data.datasets[1].data.push(data.pm25);
    
    if (telemetryChart.data.labels.length > 30) {
        telemetryChart.data.labels.shift();
        telemetryChart.data.datasets[0].data.shift();
        telemetryChart.data.datasets[1].data.shift();
    }
    telemetryChart.update();

    // AI RL Protocol
    const riskPanel = document.getElementById('rl-panel');
    const actionId = document.getElementById('action-id');
    const actionName = document.getElementById('action-name');
    const actionDesc = document.getElementById('action-desc');
    const llamaReport = document.getElementById('llama-report');
    const rlReward = document.getElementById('rl-reward');

    riskPanel.classList.remove('status-critical', 'status-high');
    
    let color = '#10b981';

    if (prediction.action_id === 3) {
        riskPanel.classList.add('status-critical');
        actionDesc.innerText = "CRITICAL ANOMALY. Shutting down systems immediately.";
        rlReward.innerText = "-100.00";
        rlReward.style.color = "#ef4444";
    } else if (prediction.action_id === 1 || prediction.action_id === 2) {
        riskPanel.classList.add('status-high');
        actionDesc.innerText = "Mitigation systems active to restore baseline.";
        rlReward.innerText = "-1.00";
        rlReward.style.color = "#f59e0b";
    } else {
        actionDesc.innerText = "Environment is stable.";
        rlReward.innerText = "+2.00";
        rlReward.style.color = "#10b981";
    }

    actionId.innerText = prediction.action_id.toString();
    actionName.innerText = prediction.action_name;
    
    if (prediction.llama_assessment) {
        llamaReport.innerText = prediction.llama_assessment;
    }
}

// Main Loop
setInterval(async () => {
    const sensorData = generateSensorData();

    try {
        const response = await fetch('/predict_action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(sensorData)
        });
        
        if (response.ok) {
            const prediction = await response.json();
            updateDashboard(sensorData, prediction);
        }
    } catch (err) {
        console.error("API Error", err);
    }
}, 1000);
