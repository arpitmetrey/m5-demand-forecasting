// JavaScript logic for M5 Forecasting Dashboard

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initCharts();
  initSimulator();
});

// Navigation Handling
function initNavigation() {
  const links = document.querySelectorAll('.nav-link');
  const panels = document.querySelectorAll('.section-panel');

  links.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const targetId = link.getAttribute('data-section');

      links.forEach(l => l.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));

      link.classList.add('active');
      document.getElementById(targetId).classList.add('active');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  });
}

// Chart Initializations
function initCharts() {
  // Chart 1: Sales Distribution & Zero inflation
  const ctxEDA = document.getElementById('chartEDA')?.getContext('2d');
  if (ctxEDA) {
    new Chart(ctxEDA, {
      type: 'doughnut',
      data: {
        labels: ['Zero Sales Days (58.8%)', 'Positive Sales Days (41.2%)'],
        datasets: [{
          data: [58.8, 41.2],
          backgroundColor: ['#cbd5e1', '#2563eb']
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'bottom' },
          title: { display: true, text: 'M5 Sales Sparsity Breakdown (1.52M Rows)' }
        }
      }
    });
  }

  // Chart 2: Model Benchmark (Table 1 from Paper)
  const ctxBench = document.getElementById('chartBenchmark')?.getContext('2d');
  if (ctxBench) {
    new Chart(ctxBench, {
      type: 'bar',
      data: {
        labels: ['Zero Forecast', 'Seasonal Naive', '4-Wk Weekday Mean', 'Ridge (All Features)', 'Boosted Trees (All)'],
        datasets: [
          {
            label: 'Test MAE (Lower is Better)',
            data: [1.556, 1.292, 1.157, 1.115, 1.114],
            backgroundColor: '#3b82f6'
          },
          {
            label: 'Test RMSE (Lower is Better)',
            data: [3.539, 2.466, 2.130, 1.966, 1.948],
            backgroundColor: '#0f172a'
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'top' },
          title: { display: true, text: 'Test Set Error Comparison (28-Day Horizon)' }
        }
      }
    });
  }

  // Chart 3: Volume Tier Error Breakdown (Figure 1 from Paper)
  const ctxVol = document.getElementById('chartVolume')?.getContext('2d');
  if (ctxVol) {
    new Chart(ctxVol, {
      type: 'bar',
      data: {
        labels: ['<0.25 units/day (n=251)', '0.25-1 units/day (n=373)', '1-3 units/day (n=238)', '>=3 units/day (n=138)'],
        datasets: [
          {
            label: '4-Wk Weekday Mean',
            data: [0.318, 0.706, 1.467, 3.365],
            backgroundColor: '#94a3b8'
          },
          {
            label: 'Ridge Regression',
            data: [0.361, 0.716, 1.382, 3.108],
            backgroundColor: '#2563eb'
          },
          {
            label: 'Boosted Trees (GBT)',
            data: [0.377, 0.718, 1.376, 3.075],
            backgroundColor: '#dc2626'
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'top' },
          title: { display: true, text: 'Test MAE by Product Sales Volume Tier' }
        }
      }
    });
  }

  // Chart 4: Multi-Window Robust Summary
  const ctxRobust = document.getElementById('chartRobust')?.getContext('2d');
  if (ctxRobust) {
    new Chart(ctxRobust, {
      type: 'bar',
      data: {
        labels: ['Recent 28-day Mean', 'Ridge Regression', 'Price-aware Ridge', 'Random Forest', 'Historical Mean', 'Seasonal Naive'],
        datasets: [{
          label: 'Mean MAE across 3 Validation Windows',
          data: [1.1404, 1.1485, 1.1801, 1.1825, 1.1941, 1.4055],
          backgroundColor: ['#059669', '#2563eb', '#d97706', '#dc2626', '#64748b', '#94a3b8']
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'Multi-Window Robust Evaluation (Mean MAE over d_1802-d_1913)' }
        }
      }
    });
  }
}

// Interactive What-If Forecast Simulator
function initSimulator() {
  const volSelect = document.getElementById('simVol');
  const snapCheck = document.getElementById('simSnap');
  const priceSlider = document.getElementById('simPrice');
  const priceVal = document.getElementById('simPriceVal');
  const eventCheck = document.getElementById('simEvent');

  if (!volSelect) return;

  const ctxSim = document.getElementById('chartSimulator')?.getContext('2d');
  let simChart;

  function updateSimulation() {
    const vol = parseFloat(volSelect.value);
    const snap = snapCheck.checked ? 1.08 : 1.0;
    const priceRatio = parseFloat(priceSlider.value);
    const isEvent = eventCheck.checked ? 1.15 : 1.0;
    priceVal.textContent = priceRatio.toFixed(2) + 'x';

    // Generate 28-day simulated trajectory
    const days = Array.from({ length: 28 }, (_, i) => `Day ${i + 1}`);
    const actual = [];
    const predBaseline = [];
    const predRidge = [];
    const predGBT = [];

    for (let d = 1; d <= 28; d++) {
      const weekdayMult = (d % 7 === 6 || d % 7 === 0) ? 1.3 : 0.9;
      const noise = (Math.random() - 0.5) * 0.4;
      const baseDemand = vol * weekdayMult * snap * isEvent;
      
      actual.push(Math.max(0, baseDemand + noise));
      predBaseline.push(baseDemand * 1.05); // slightly over-predicts
      predRidge.push(baseDemand * (1 - (priceRatio - 1) * 0.1));
      predGBT.push(baseDemand * (1 - (priceRatio - 1) * 0.12) + (isEvent > 1 ? 0.2 : 0));
    }

    if (simChart) simChart.destroy();

    simChart = new Chart(ctxSim, {
      type: 'line',
      data: {
        labels: days,
        datasets: [
          { label: 'Simulated Actual Demand', data: actual, borderColor: '#0f172a', borderDash: [5, 5], fill: false },
          { label: '4-Wk Weekday Mean Baseline', data: predBaseline, borderColor: '#94a3b8', fill: false },
          { label: 'Ridge Forecast', data: predRidge, borderColor: '#2563eb', fill: false },
          { label: 'Boosted Trees (GBT) Forecast', data: predGBT, borderColor: '#dc2626', fill: false }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          title: { display: true, text: 'Live 28-Day Direct Demand Trajectory' }
        }
      }
    });

    // Update risk alert text
    const alertBox = document.getElementById('simRiskAlert');
    if (vol < 0.25) {
      alertBox.className = 'sim-alert warning';
      alertBox.innerHTML = '<strong>Deployment Risk Warning:</strong> On slow-moving items (<0.25 units/day), learned ML models suffer from zero-inflation loss mismatch. The baseline achieves lower MAE here.';
    } else {
      alertBox.className = 'sim-alert info';
      alertBox.innerHTML = '<strong>Optimal Deployment Domain:</strong> On moderate to high volume items, Ridge and GBT outperform baselines by reducing peak RMSE and capturing promotional lift.';
    }
  }

  [volSelect, snapCheck, priceSlider, eventCheck].forEach(el => {
    el.addEventListener('input', updateSimulation);
  });

  updateSimulation();
}
