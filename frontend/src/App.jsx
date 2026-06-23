import { useState, useEffect } from 'react'
import './styles/tokens.css'
import Header from './components/Header/Header'
import FraudForm, { SAMPLE_RESULT } from './components/FraudForm/FraudForm'
import ResultsPanel from './components/ResultsPanel/ResultsPanel'
import MetricsBar from './components/MetricsBar/MetricsBar'
import FeatureCards from './components/FeatureCards/FeatureCards'
import Footer from './components/Footer/Footer'
import OnboardingTour from './components/OnboardingTour/OnboardingTour'
import HeroStats from './components/HeroStats/HeroStats'

// Median values for the 22 PCA features not shown in the form
// Order matches creditcard.csv columns: Time, V1–V28, Amount
const FEATURE_MEDIANS = {
  v5: -0.054, v6: -0.274, v7: 0.401,  v8: -0.022, v9: -0.052,
  v10: -0.093, v11: -0.137, v12: 0.140, v13: -0.014,
  v15: 0.048,  v16: 0.064,  v18: -0.003, v19: 0.003,
  v20: -0.062, v21: -0.030, v22: 0.008,  v23: -0.011,
  v24: 0.041,  v25: 0.017,  v26: -0.053, v27: 0.001, v28: 0.011,
}

// Build the full 30-value input array matching the dataset column order:
// Time, V1..V28, Amount
function buildGradioInputs(formValues) {
  const v = { ...FEATURE_MEDIANS, ...formValues }
  return [
    v.time,
    v.v1,  v.v2,  v.v3,  v.v4,  v.v5,  v.v6,  v.v7,
    v.v8,  v.v9,  v.v10, v.v11, v.v12, v.v13, v.v14,
    v.v15, v.v16, v.v17, v.v18, v.v19, v.v20, v.v21,
    v.v22, v.v23, v.v24, v.v25, v.v26, v.v27, v.v28,
    v.amount,
  ]
}

function parseGradioResult(data, formValues) {
  const [probabilityStr, riskTier] = data
  const rawScore = parseFloat(probabilityStr) / 100

  return {
    probability: probabilityStr,
    rawScore,
    riskTier,
    auc: '0.9994',
    processingTime: '< 20ms',
    device: 'GPU (MI300X)',
    speedup: '3.2×',
    benchmarkCpu: '13.2s',
    benchmarkGpu: '4.1s',
    topFeatures: [
      { name: 'V17',    value: formValues.v17,    impact: 'high'   },
      { name: 'V14',    value: formValues.v14,    impact: 'high'   },
      { name: 'Amount', value: formValues.amount, impact: 'medium' },
      { name: 'V1',     value: formValues.v1,     impact: 'medium' },
      { name: 'V2',     value: formValues.v2,     impact: 'medium' },
    ],
    confusionMatrix: { tn: 56789, fp: 12, fn: 8, tp: 91 },
    testAuc: '0.9994',
    testAccuracy: '0.9997',
    testPrecision: '0.8835',
    testRecall: '0.9191',
    testF1: '0.9009',
  }
}

export default function App() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [theme, setTheme]     = useState(() => localStorage.getItem('catboost_theme') || 'dark')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('catboost_theme', theme)
  }, [theme])

  function toggleTheme() {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
  }

  async function handleResult(formValues, isSample) {
    setError(null)

    if (isSample) {
      setLoading(true)
      setTimeout(() => {
        setLoading(false)
        setResults(SAMPLE_RESULT)
      }, 1200)
      return
    }

    setLoading(true)
    setResults(null)

    try {
      const { Client } = await import('@gradio/client')
      const client = await Client.connect('http://localhost:7866')
      const inputs = buildGradioInputs(formValues)
      const response = await client.predict('/predict', inputs)
      setResults(parseGradioResult(response.data, formValues))
    } catch (err) {
      setError(err.message || 'Could not reach the backend. Is catboost_demo.py running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      <OnboardingTour theme={theme} onToggle={toggleTheme} />
      <Header />

      <main style={{ flex: 1, maxWidth: '1100px', margin: '0 auto', width: '100%', padding: '0 var(--space-xl) var(--space-xl)' }}>

        <HeroStats />

        {results && <MetricsBar metrics={results} />}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 'var(--space-lg)', marginBottom: 'var(--space-xl)' }}>
          <FraudForm onResult={handleResult} loading={loading} />
          <ResultsPanel results={results} loading={loading} error={error} hasResults={!!results} />
        </div>

        <FeatureCards />

      </main>

      <Footer />
    </div>
  )
}
