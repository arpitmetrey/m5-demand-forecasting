// JavaScript logic for M5 Forecasting Dashboard

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initCharts();
  initExplorer();
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

  // Chart 4: Test MAE by forecast week (weekly blocks of the 28-day horizon)
  const ctxRobust = document.getElementById('chartRobust')?.getContext('2d');
  if (ctxRobust) {
    new Chart(ctxRobust, {
      type: 'bar',
      data: {
        labels: ['Days 1-7', 'Days 8-14', 'Days 15-21', 'Days 22-28'],
        datasets: [
          { label: '4-Wk Weekday Mean', data: [1.1010, 1.1853, 1.1890, 1.1515], backgroundColor: '#94a3b8' },
          { label: 'Ridge Regression', data: [1.0709, 1.1541, 1.1532, 1.0835], backgroundColor: '#2563eb' },
          { label: 'Boosted Trees (GBT)', data: [1.0645, 1.1509, 1.1431, 1.0989], backgroundColor: '#dc2626' }
        ]
      },
      options: {
        responsive: true,
        scales: { y: { beginAtZero: false, title: { display: true, text: 'Test MAE' } } },
        plugins: {
          legend: { position: 'top' },
          title: { display: true, text: 'Test MAE by Forecast Week (no growth with horizon)' }
        }
      }
    });
  }
}

// Results Explorer: shows measured test results only (no model runs in the browser)
const ABLATION = {
  // key = feature groups added to static encodings: H = history, C = calendar, P = price
  '':    { ridge: [1.771, 3.097], gbt: [1.766, 3.091] },
  'H':   { ridge: [1.144, 1.975], gbt: [1.136, 1.981] },
  'C':   { ridge: [1.743, 3.090], gbt: [1.732, 3.080] },
  'P':   { ridge: [1.741, 3.045], gbt: [1.413, 2.434] },
  'HC':  { ridge: [1.117, 1.967], gbt: [1.113, 1.959] },
  'HP':  { ridge: [1.142, 1.974], gbt: [1.132, 1.980] },
  'HCP': { ridge: [1.115, 1.966], gbt: [1.114, 1.948] }
  // 'CP' (calendar + price without history) was not run
};
const ZERO_MAE = 1.556;
const MEAN4_MAE = 1.157;

const TIERS = [
  { name: '<0.25 units/day', mean4: 0.318, ridge: 0.361, gbt: 0.377 },
  { name: '0.25-1 units/day', mean4: 0.706, ridge: 0.716, gbt: 0.718 },
  { name: '1-3 units/day', mean4: 1.467, ridge: 1.382, gbt: 1.376 },
  { name: '>=3 units/day', mean4: 3.365, ridge: 3.108, gbt: 3.075 }
];

function initExplorer() {
  const hist = document.getElementById('exHistory');
  const cal = document.getElementById('exCalendar');
  const price = document.getElementById('exPrice');
  const tierSelect = document.getElementById('exTier');
  if (!hist || !cal || !price || !tierSelect) return;

  const ctxAbl = document.getElementById('chartAblation')?.getContext('2d');
  const ctxTier = document.getElementById('chartTier')?.getContext('2d');
  const ablText = document.getElementById('exAblationText');
  const tierAlert = document.getElementById('exTierAlert');
  let ablChart;
  let tierChart;

  function pct(value, ref) {
    const d = (value - ref) / ref * 100;
    return (d >= 0 ? '+' : '') + d.toFixed(1) + '%';
  }

  function updateAblation() {
    const key = (hist.checked ? 'H' : '') + (cal.checked ? 'C' : '') + (price.checked ? 'P' : '');
    const row = ABLATION[key];
    if (!row) {
      ablText.className = 'sim-alert warning';
      ablText.innerHTML = '<strong>Not evaluated:</strong> calendar + price without history was not run in this project. Pick another combination.';
      if (ablChart) { ablChart.destroy(); ablChart = null; }
      return;
    }
    const label = key === '' ? 'static encodings only' : 'static + ' + key.split('').map(c => ({ H: 'history', C: 'calendar', P: 'price' }[c])).join(' + ');
    const beats = row.gbt[0] < MEAN4_MAE && row.ridge[0] < MEAN4_MAE;
    ablText.className = 'sim-alert ' + (beats ? 'info' : 'warning');
    ablText.innerHTML = '<strong>' + label.charAt(0).toUpperCase() + label.slice(1) + ':</strong> Ridge MAE ' + row.ridge[0].toFixed(3) + ' (RMSE ' + row.ridge[1].toFixed(3) + '), GBT MAE ' + row.gbt[0].toFixed(3) + ' (RMSE ' + row.gbt[1].toFixed(3) + '). Versus the 4-week weekday mean (MAE 1.157): Ridge ' + pct(row.ridge[0], MEAN4_MAE) + ', GBT ' + pct(row.gbt[0], MEAN4_MAE) + '.';

    if (ablChart) ablChart.destroy();
    ablChart = new Chart(ctxAbl, {
      type: 'bar',
      data: {
        labels: ['Zero forecast', '4-wk weekday mean', 'Ridge', 'Boosted trees'],
        datasets: [{
          label: 'Test MAE',
          data: [ZERO_MAE, MEAN4_MAE, row.ridge[0], row.gbt[0]],
          backgroundColor: ['#cbd5e1', '#94a3b8', '#2563eb', '#dc2626']
        }]
      },
      options: {
        responsive: true,
        scales: { y: { beginAtZero: true, title: { display: true, text: 'Test MAE (lower is better)' } } },
        plugins: { legend: { display: false } }
      }
    });
  }

  function updateTier() {
    const t = TIERS[parseInt(tierSelect.value, 10)];
    const modelsBeat = t.ridge < t.mean4 && t.gbt < t.mean4;
    const tierName = t.name.replace('<', '&lt;').replace('>', '&gt;');
    tierAlert.className = 'sim-alert ' + (modelsBeat ? 'info' : 'warning');
    if (modelsBeat) {
      tierAlert.innerHTML = '<strong>' + tierName + ':</strong> both learned models have lower MAE than the 4-week weekday mean (Ridge ' + t.ridge.toFixed(3) + ', GBT ' + t.gbt.toFixed(3) + ' vs ' + t.mean4.toFixed(3) + ').';
    } else {
      tierAlert.innerHTML = '<strong>' + tierName + ':</strong> the 4-week weekday mean has the lowest MAE (' + t.mean4.toFixed(3) + ' vs Ridge ' + t.ridge.toFixed(3) + ', GBT ' + t.gbt.toFixed(3) + '). Squared-error models predict means, while MAE favours medians on intermittent demand.';
    }

    if (tierChart) tierChart.destroy();
    tierChart = new Chart(ctxTier, {
      type: 'bar',
      data: {
        labels: ['4-wk weekday mean', 'Ridge', 'Boosted trees'],
        datasets: [{
          label: 'Test MAE',
          data: [t.mean4, t.ridge, t.gbt],
          backgroundColor: ['#94a3b8', '#2563eb', '#dc2626']
        }]
      },
      options: {
        responsive: true,
        scales: { y: { beginAtZero: true, title: { display: true, text: 'Test MAE (lower is better)' } } },
        plugins: { legend: { display: false } }
      }
    });
  }

  [hist, cal, price].forEach(el => el.addEventListener('input', updateAblation));
  tierSelect.addEventListener('input', updateTier);

  updateAblation();
  updateTier();
}
