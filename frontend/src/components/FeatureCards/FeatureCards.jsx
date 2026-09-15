import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import styles from './FeatureCards.module.css'

const FEATURES = [
  {
    eyebrow: 'Hardware',
    title: 'AMD Instinct MI300X',
    description: '192GB HBM3 GPU memory delivering unmatched throughput for large-scale fraud detection workloads.',
    detail: 'The AMD Instinct MI300X accelerates CatBoost training with 192GB HBM3 memory and 5.3 TB/s memory bandwidth. Its CDNA3 architecture is optimized for the parallel computation demands of gradient boosting on large financial datasets with hundreds of thousands of transactions.',
    links: [
      { label: 'MI300X product page', href: 'https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html' },
      { label: 'ROCm documentation',  href: 'https://rocm.docs.amd.com' },
    ],
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" />
        <line x1="9" y1="1" x2="9" y2="4" /><line x1="15" y1="1" x2="15" y2="4" />
        <line x1="9" y1="20" x2="9" y2="23" /><line x1="15" y1="20" x2="15" y2="23" />
        <line x1="20" y1="9" x2="23" y2="9" /><line x1="20" y1="14" x2="23" y2="14" />
        <line x1="1" y1="9" x2="4" y2="9" /><line x1="1" y1="14" x2="4" y2="14" />
      </svg>
    ),
  },
  {
    eyebrow: 'Software',
    title: 'ROCm + CatBoost',
    description: 'Open-source ROCm stack enables GPU-accelerated CatBoost training with full explainability.',
    detail: 'ROCm provides the open-source GPU compute platform that enables CatBoost to run on AMD GPUs. CatBoost\'s GPU task type leverages ROCm for parallel tree building, dramatically reducing training time on large transaction datasets while preserving full SHAP-based explainability.',
    links: [
      { label: 'ROCm documentation', href: 'https://rocm.docs.amd.com' },
      { label: 'ROCm on GitHub',      href: 'https://github.com/ROCm/ROCm' },
    ],
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2" />
        <polyline points="2 17 12 22 22 17" />
        <polyline points="2 12 12 17 22 12" />
      </svg>
    ),
  },
  {
    eyebrow: 'Performance',
    title: '3.2× GPU Speedup',
    description: 'AMD Instinct GPU training runs 3.2× faster than CPU on 284K transaction datasets.',
    detail: 'Benchmarks on the Credit Card Fraud Detection dataset (284,807 transactions) show the AMD Instinct MI300X completing 1,000 CatBoost iterations in ~4.1s versus ~13.2s on CPU — a 3.2× speedup enabling faster model iteration, real-time retraining, and lower infrastructure cost per inference.',
    links: [
      { label: 'AMD Instinct performance', href: 'https://www.amd.com/en/products/accelerators/instinct.html' },
    ],
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="13 2 13 9 19 9" />
        <path d="M19 2H13L3 12h7l-2 10 13-13h-7l5-9z" />
      </svg>
    ),
  },
  {
    eyebrow: 'Capability',
    title: 'Explainable Fraud Detection',
    description: 'SHAP-powered feature importance reveals which transaction signals drive each fraud prediction.',
    detail: 'CatBoost\'s built-in SHAP values and recursive feature selection identify the top predictive signals in PCA-transformed transaction data. The model achieves 99.94% ROC-AUC on the Kaggle Credit Card Fraud dataset, with full transparency into per-prediction decisions for regulatory compliance.',
    links: [],
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6"  y1="20" x2="6"  y2="14" />
        <line x1="2"  y1="20" x2="22" y2="20" />
      </svg>
    ),
  },
]

const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.1 } },
}

const cardVariants = {
  hidden:  { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
}

export default function FeatureCards() {
  const [active, setActive] = useState(null)
  const feature = FEATURES.find((f) => f.title === active)

  return (
    <>
      <AnimatePresence>
        {active && (
          <motion.div
            className={styles.overlay}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={() => setActive(null)}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {active && feature && (
          <motion.aside
            className={styles.panel}
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className={styles.panelHeader}>
              <div className={styles.panelHeaderLeft}>
                <div className={styles.panelIconBox}>{feature.icon}</div>
                <div>
                  <div className={styles.panelEyebrow}>{feature.eyebrow}</div>
                  <div className={styles.panelName}>{feature.title}</div>
                </div>
              </div>
              <button className={styles.panelClose} onClick={() => setActive(null)} aria-label="Close panel">
                &times;
              </button>
            </div>
            <div className={styles.panelBody}>
              <p className={styles.panelDetail}>{feature.detail}</p>
              {feature.links.length > 0 && (
                <div className={styles.panelLinks}>
                  <span className={styles.panelLinksLabel}>Learn more</span>
                  {feature.links.map((l) => (
                    <a key={l.href} href={l.href} target="_blank" rel="noreferrer" className={styles.panelLink}>
                      {l.label}
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M7 17L17 7M17 7H7M17 7v10" />
                      </svg>
                    </a>
                  ))}
                </div>
              )}
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      <motion.section
        className={styles.section}
        data-tour="cards"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {FEATURES.map((f) => (
          <motion.div
            key={f.title}
            className={`${styles.card} ${active === f.title ? styles.cardActive : ''}`}
            variants={cardVariants}
            onClick={() => setActive((prev) => (prev === f.title ? null : f.title))}
            role="button"
            aria-expanded={active === f.title}
            aria-label={`${f.title} — ${f.eyebrow}. Click to ${active === f.title ? 'close' : 'learn more'}`}
          >
            <div className={styles.header}>
              <span className={styles.accent} aria-hidden="true" />
              <span className={styles.icon} aria-hidden="true">{f.icon}</span>
            </div>
            <h3 className={styles.title}>{f.title}</h3>
            <p className={styles.description}>{f.description}</p>
          </motion.div>
        ))}
      </motion.section>
    </>
  )
}
