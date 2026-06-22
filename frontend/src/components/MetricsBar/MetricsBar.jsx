import { useCountUp } from '../../lib/useCountUp'
import styles from './MetricsBar.module.css'

function Metric({ label, value, suffix = '', prefix = '', decimals = 0 }) {
  const numeric = parseFloat(value)
  const counted = useCountUp(isNaN(numeric) ? 0 : numeric, { duration: 1400 })

  const display = isNaN(numeric)
    ? value
    : `${prefix}${counted.toFixed(decimals)}${suffix}`

  return (
    <div className={styles.metric}>
      <span className={styles.value}>{display}</span>
      <span className={styles.label}>{label}</span>
    </div>
  )
}

export default function MetricsBar({ metrics }) {
  if (!metrics) return null

  return (
    <div className={styles.bar}>
      <Metric label="Fraud Probability"  value={metrics.rawScore * 100} suffix="%" decimals={1} />
      <div className={styles.divider} />
      <Metric label="Model AUC"          value={metrics.auc} decimals={4} />
      <div className={styles.divider} />
      <Metric label="Processing Time"    value={metrics.processingTime} />
      <div className={styles.divider} />
      <Metric label="GPU Speedup"        value={metrics.speedup} />
      <div className={styles.divider} />
      <div className={`${styles.metric} ${styles.riskMetric}`}>
        <span className={`${styles.tier} ${styles[`tier${metrics.riskTier}`]}`}>{metrics.riskTier}</span>
        <span className={styles.label}>Risk Tier</span>
      </div>
    </div>
  )
}
