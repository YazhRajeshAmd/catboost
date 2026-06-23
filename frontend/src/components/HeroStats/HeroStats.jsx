import { motion } from 'framer-motion'
import { useCountUp } from '../../lib/useCountUp'
import styles from './HeroStats.module.css'

const STATS = [
  { value: 99.94, suffix: '%',  label: 'Model ROC-AUC',          decimals: 2 },
  { value: 284,   suffix: 'K',  label: 'Training Transactions',  decimals: 0 },
  { value: 3.2,   suffix: '×',  label: 'GPU Speedup vs CPU',     decimals: 1 },
  { value: 20,    prefix: '<',  suffix: 'ms', label: 'Inference Latency', decimals: 0 },
]

function StatItem({ value, suffix = '', prefix = '', label, decimals, delay }) {
  const counted = useCountUp(value, { duration: 1600 })
  return (
    <motion.div
      className={styles.stat}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: 'easeOut' }}
    >
      <span className={styles.value}>
        {prefix}{counted.toFixed(decimals)}{suffix}
      </span>
      <span className={styles.label}>{label}</span>
    </motion.div>
  )
}

export default function HeroStats() {
  return (
    <div className={styles.wrap}>
      {STATS.map((s, i) => (
        <StatItem key={s.label} {...s} delay={i * 0.1} />
      ))}
    </div>
  )
}
