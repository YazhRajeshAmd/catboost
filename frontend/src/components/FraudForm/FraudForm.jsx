import { useState } from 'react'
import styles from './FraudForm.module.css'

const SUSPICIOUS_SAMPLE = {
  amount: 2125.87,
  time: 406,
  v1: -3.043541,
  v2: -3.157307,
  v3: 1.088463,
  v4: 2.288644,
  v14: -9.499020,
  v17: -18.683715,
}

const NORMAL_SAMPLE = {
  amount: 49.99,
  time: 86400,
  v1: 1.191857,
  v2: 0.266151,
  v3: 0.166480,
  v4: 0.448154,
  v14: -0.311169,
  v17: -0.753917,
}

const SAMPLE_RESULT = {
  probability: '87.34%',
  rawScore: 0.8734,
  riskTier: 'HIGH',
  auc: '0.9994',
  processingTime: '12ms',
  device: 'GPU (MI300X)',
  topFeatures: [
    { name: 'V17', value: -18.68, impact: 'high' },
    { name: 'V14', value: -9.50,  impact: 'high' },
    { name: 'Amount', value: 2125.87, impact: 'medium' },
    { name: 'V1', value: -3.04, impact: 'medium' },
    { name: 'V2', value: -3.16, impact: 'medium' },
  ],
  benchmarkCpu: '13.2s',
  benchmarkGpu: '4.1s',
  speedup: '3.2×',
  confusionMatrix: { tn: 56789, fp: 12, fn: 8, tp: 91 },
  testAuc: '0.9994',
  testAccuracy: '0.9997',
  testPrecision: ' 0.8835',
  testRecall: '0.9191',
  testF1: '0.9009',
}

const FIELDS = [
  { key: 'amount', label: 'Transaction Amount (USD)', min: 0, max: 25000, step: 0.01, defaultVal: 100.00, help: 'Dollar value of the transaction' },
  { key: 'time',   label: 'Time (seconds elapsed)',   min: 0, max: 172800, step: 1,    defaultVal: 86400,  help: 'Seconds since first transaction in dataset' },
  { key: 'v1',     label: 'V1 — PCA Feature 1',       min: -10, max: 5, step: 0.001, defaultVal: 1.192,  help: 'PCA-transformed feature (most predictive)' },
  { key: 'v2',     label: 'V2 — PCA Feature 2',       min: -10, max: 10, step: 0.001, defaultVal: 0.266,  help: 'PCA-transformed feature' },
  { key: 'v3',     label: 'V3 — PCA Feature 3',       min: -10, max: 10, step: 0.001, defaultVal: 0.166,  help: 'PCA-transformed feature' },
  { key: 'v4',     label: 'V4 — PCA Feature 4',       min: -5,  max: 10, step: 0.001, defaultVal: 0.448,  help: 'PCA-transformed feature' },
  { key: 'v14',    label: 'V14 — PCA Feature 14',     min: -20, max: 5,  step: 0.001, defaultVal: -0.311, help: 'High-impact fraud signal' },
  { key: 'v17',    label: 'V17 — PCA Feature 17',     min: -25, max: 10, step: 0.001, defaultVal: -0.754, help: 'High-impact fraud signal' },
]

export default function FraudForm({ onResult, loading }) {
  const [values, setValues] = useState(() =>
    Object.fromEntries(FIELDS.map((f) => [f.key, f.defaultVal]))
  )

  function handleChange(key, val) {
    setValues((prev) => ({ ...prev, [key]: parseFloat(val) || 0 }))
  }

  function handleSubmit(e) {
    e.preventDefault()
    onResult(values, false)
  }

  function loadSample(sample) {
    setValues(sample)
    onResult(null, true)
  }

  return (
    <form className={styles.form} onSubmit={handleSubmit} data-tour="form">
      <div className={styles.formHeader}>
        <h2 className={styles.formTitle}>Transaction Analyzer</h2>
        <p className={styles.formSub}>Enter transaction features to predict fraud risk</p>
      </div>

      <div className={styles.fields}>
        {FIELDS.map((f) => (
          <div key={f.key} className={styles.field}>
            <label className={styles.label} htmlFor={f.key}>{f.label}</label>
            <input
              id={f.key}
              type="number"
              className={styles.input}
              value={values[f.key]}
              step={f.step}
              min={f.min}
              max={f.max}
              onChange={(e) => handleChange(f.key, e.target.value)}
            />
            <span className={styles.help}>{f.help}</span>
          </div>
        ))}
      </div>

      <p className={styles.note}>
        Remaining 22 PCA features (V5–V13, V15–V16, V18–V28) use dataset median values.
      </p>

      <button
        type="submit"
        className={styles.submitBtn}
        disabled={loading}
        data-tour="submit"
      >
        {loading ? 'Analyzing...' : 'Analyze Transaction'}
      </button>

      <div className={styles.sampleRow}>
        <button type="button" className={styles.sampleBtn} onClick={() => loadSample(SUSPICIOUS_SAMPLE)} disabled={loading}>
          Load Suspicious Sample
        </button>
        <button type="button" className={styles.sampleBtn} onClick={() => loadSample(NORMAL_SAMPLE)} disabled={loading}>
          Load Normal Sample
        </button>
      </div>

      <div className={styles.sampleResultHint}>
        <span className={styles.hintIcon}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        </span>
        Sample data uses a pre-loaded result. Start the backend for live inference.
      </div>
    </form>
  )
}

export { SAMPLE_RESULT, SUSPICIOUS_SAMPLE, NORMAL_SAMPLE }
