import styles from './AMDLoader.module.css'

export default function AMDLoader({ label = 'Analyzing with AMD MI300X...' }) {
  return (
    <div className={styles.wrapper}>
      <svg className={styles.svg} viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="shimmer" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#00C2DE" />
            <stop offset="50%"  stopColor="#C1A968" />
            <stop offset="100%" stopColor="#00C2DE" />
            <animateTransform
              attributeName="gradientTransform"
              type="translate"
              from="-1 0"
              to="2 0"
              dur="1.6s"
              repeatCount="indefinite"
            />
          </linearGradient>
        </defs>
        <rect x="8" y="8" width="64" height="64" rx="8" stroke="url(#shimmer)" strokeWidth="2.5" fill="none" />
        <rect x="20" y="20" width="40" height="40" rx="4" stroke="url(#shimmer)" strokeWidth="1.5" fill="none" />
        <rect x="32" y="32" width="16" height="16" rx="2" fill="url(#shimmer)" opacity="0.7" />
        <line x1="32" y1="8"  x2="32" y2="20" stroke="url(#shimmer)" strokeWidth="1.5" />
        <line x1="48" y1="8"  x2="48" y2="20" stroke="url(#shimmer)" strokeWidth="1.5" />
        <line x1="32" y1="60" x2="32" y2="72" stroke="url(#shimmer)" strokeWidth="1.5" />
        <line x1="48" y1="60" x2="48" y2="72" stroke="url(#shimmer)" strokeWidth="1.5" />
        <line x1="8"  y1="32" x2="20" y2="32" stroke="url(#shimmer)" strokeWidth="1.5" />
        <line x1="8"  y1="48" x2="20" y2="48" stroke="url(#shimmer)" strokeWidth="1.5" />
        <line x1="60" y1="32" x2="72" y2="32" stroke="url(#shimmer)" strokeWidth="1.5" />
        <line x1="60" y1="48" x2="72" y2="48" stroke="url(#shimmer)" strokeWidth="1.5" />
      </svg>
      <p className={styles.label}>{label}</p>
      <div className={styles.dots}>
        <span className={styles.dot} />
        <span className={styles.dot} />
        <span className={styles.dot} />
      </div>
    </div>
  )
}
