import { useState } from 'react'
import styles from './FraudForm.module.css'

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
    onResult(values)
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
    </form>
  )
}
