import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import AMDLoader from '../AMDLoader/AMDLoader'
import styles from './ResultsPanel.module.css'

const TABS = ['Risk Assessment', 'Executive Dashboard', 'GPU vs CPU', 'Explainability', 'Test Evaluation']

function RiskMeter({ score }) {
  const pct = Math.round(score * 100)
  const color = score > 0.6 ? '#ef4444' : score > 0.3 ? '#f59e0b' : '#00C2DE'
  return (
    <div className={styles.meterWrap}>
      <div className={styles.meterTrack}>
        <motion.div
          className={styles.meterFill}
          style={{ background: color }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.9, ease: 'easeOut' }}
        />
      </div>
      <div className={styles.meterLabels}>
        <span>0%</span>
        <span style={{ color }}>Risk: {pct}%</span>
        <span>100%</span>
      </div>
    </div>
  )
}

function RiskBadge({ tier }) {
  const cls = tier === 'HIGH' ? styles.badgeHigh : tier === 'MEDIUM' ? styles.badgeMedium : styles.badgeLow
  return <span className={`${styles.badge} ${cls}`}>{tier} RISK</span>
}

function TabRiskAssessment({ results }) {
  return (
    <div className={styles.tabContent}>
      <div className={styles.riskHero}>
        <div className={styles.riskScore}>
          <span className={styles.scoreLabel}>Fraud Probability</span>
          <span className={styles.scoreValue}>{results.probability}</span>
          <RiskBadge tier={results.riskTier} />
        </div>
        <RiskMeter score={results.rawScore} />
      </div>

      <div className={styles.section}>
        <h4 className={styles.sectionTitle}>Top Feature Signals</h4>
        <div className={styles.featureList}>
          {results.topFeatures.map((f) => (
            <div key={f.name} className={styles.featureRow}>
              <span className={styles.featureName}>{f.name}</span>
              <div className={styles.featureBar}>
                <motion.div
                  className={`${styles.featureBarFill} ${f.impact === 'high' ? styles.featureHigh : styles.featureMed}`}
                  initial={{ width: 0 }}
                  animate={{ width: f.impact === 'high' ? '80%' : '45%' }}
                  transition={{ duration: 0.6, ease: 'easeOut' }}
                />
              </div>
              <span className={styles.featureVal}>{f.value}</span>
            </div>
          ))}
        </div>
      </div>

      <div className={styles.infoRow}>
        <div className={styles.infoChip}>
          <span className={styles.infoKey}>Device</span>
          <span className={styles.infoVal}>{results.device}</span>
        </div>
        <div className={styles.infoChip}>
          <span className={styles.infoKey}>Inference</span>
          <span className={styles.infoVal}>{results.processingTime}</span>
        </div>
        <div className={styles.infoChip}>
          <span className={styles.infoKey}>Model AUC</span>
          <span className={styles.infoVal}>{results.auc}</span>
        </div>
      </div>
    </div>
  )
}

function TabDashboard({ results }) {
  const kpis = [
    { label: 'Model AUC Score',          value: results.auc },
    { label: 'GPU Training Speedup',     value: results.speedup },
    { label: 'GPU Training Time',        value: results.benchmarkGpu },
    { label: 'CPU Training Time',        value: results.benchmarkCpu },
    { label: 'Test Accuracy',            value: results.testAccuracy },
    { label: 'Test F1 Score',            value: results.testF1 },
  ]
  return (
    <div className={styles.tabContent}>
      <p className={styles.tabIntro}>Executive summary of model performance on the Kaggle Credit Card Fraud dataset (284,807 transactions, 80/20 train/test split).</p>
      <div className={styles.kpiGrid}>
        {kpis.map((k) => (
          <div key={k.label} className={styles.kpiCard}>
            <span className={styles.kpiValue}>{k.value}</span>
            <span className={styles.kpiLabel}>{k.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function TabBenchmark({ results }) {
  const speedup = parseFloat(results.speedup)
  const gpuWidth = 100 / speedup
  return (
    <div className={styles.tabContent}>
      <p className={styles.tabIntro}>CatBoost gradient boosting trained on 284K transactions. Higher is faster.</p>

      <div className={styles.benchmarkRow}>
        <span className={styles.benchmarkDevice}>CPU</span>
        <div className={styles.benchmarkTrack}>
          <div className={styles.benchmarkBarCpu} style={{ width: '100%' }} />
        </div>
        <span className={styles.benchmarkTime}>{results.benchmarkCpu}</span>
      </div>

      <div className={styles.benchmarkRow}>
        <span className={styles.benchmarkDevice}>GPU</span>
        <div className={styles.benchmarkTrack}>
          <motion.div
            className={styles.benchmarkBarGpu}
            initial={{ width: 0 }}
            animate={{ width: `${gpuWidth}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          />
        </div>
        <span className={styles.benchmarkTime}>{results.benchmarkGpu}</span>
      </div>

      <div className={styles.speedupBadge}>
        <span className={styles.speedupNum}>{results.speedup}</span>
        <span className={styles.speedupLabel}>AMD Instinct MI300X speedup over CPU</span>
      </div>
    </div>
  )
}

function TabExplainability({ results }) {
  return (
    <div className={styles.tabContent}>
      <p className={styles.tabIntro}>CatBoost feature importance via SHAP values + recursive selection. Higher importance = stronger fraud signal.</p>
      <div className={styles.importanceTable}>
        <div className={styles.tableHeader}>
          <span>Feature</span>
          <span>Importance</span>
          <span>Relative</span>
        </div>
        {results.topFeatures.map((f, i) => {
          const pct = f.impact === 'high' ? 80 - i * 10 : 35 - i * 5
          return (
            <div key={f.name} className={styles.tableRow}>
              <span className={styles.tableFeature}>{f.name}</span>
              <span className={styles.tableScore}>{Math.abs(f.value).toFixed(3)}</span>
              <div className={styles.tableBarWrap}>
                <motion.div
                  className={styles.tableBar}
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.5, delay: i * 0.07, ease: 'easeOut' }}
                />
              </div>
            </div>
          )
        })}
      </div>
      <p className={styles.tabNote}>V14 and V17 are consistently the strongest fraud signals across training runs.</p>
    </div>
  )
}

function TabTestEval({ results }) {
  const { confusionMatrix: cm } = results
  const metrics = [
    { label: 'ROC-AUC',   value: results.testAuc },
    { label: 'Accuracy',  value: results.testAccuracy },
    { label: 'Precision', value: results.testPrecision },
    { label: 'Recall',    value: results.testRecall },
    { label: 'F1 Score',  value: results.testF1 },
  ]
  return (
    <div className={styles.tabContent}>
      <p className={styles.tabIntro}>Evaluation on 20% hold-out test set ({(cm.tn + cm.fp + cm.fn + cm.tp).toLocaleString()} samples).</p>
      <div className={styles.evalMetrics}>
        {metrics.map((m) => (
          <div key={m.label} className={styles.evalChip}>
            <span className={styles.evalValue}>{m.value}</span>
            <span className={styles.evalLabel}>{m.label}</span>
          </div>
        ))}
      </div>
      <h4 className={styles.sectionTitle}>Confusion Matrix</h4>
      <div className={styles.confMatrix}>
        <div className={`${styles.cell} ${styles.cellTn}`}>
          <span className={styles.cellNum}>{cm.tn.toLocaleString()}</span>
          <span className={styles.cellLbl}>True Negative</span>
        </div>
        <div className={`${styles.cell} ${styles.cellFp}`}>
          <span className={styles.cellNum}>{cm.fp}</span>
          <span className={styles.cellLbl}>False Positive</span>
        </div>
        <div className={`${styles.cell} ${styles.cellFn}`}>
          <span className={styles.cellNum}>{cm.fn}</span>
          <span className={styles.cellLbl}>False Negative</span>
        </div>
        <div className={`${styles.cell} ${styles.cellTp}`}>
          <span className={styles.cellNum}>{cm.tp}</span>
          <span className={styles.cellLbl}>True Positive</span>
        </div>
      </div>
    </div>
  )
}

export default function ResultsPanel({ results, loading, error, hasResults }) {
  const [activeTab, setActiveTab] = useState(0)

  return (
    <div className={styles.panel} data-tour="results">
      <div className={styles.tabBar}>
        {TABS.map((tab, i) => (
          <button
            key={tab}
            className={`${styles.tab} ${activeTab === i ? styles.tabActive : ''}`}
            onClick={() => setActiveTab(i)}
            disabled={!hasResults}
          >
            {tab}
            {activeTab === i && hasResults && (
              <motion.div className={styles.tabIndicator} layoutId="tab-indicator" />
            )}
          </button>
        ))}
      </div>

      <div className={styles.body}>
        {loading && <AMDLoader label="Running CatBoost inference on AMD MI300X..." />}

        {error && !loading && (
          <div className={styles.errorState}>
            <p className={styles.errorText}>{error}</p>
            <p className={styles.errorHint}>Make sure the backend is running: <code>python3 catboost_demo.py</code></p>
          </div>
        )}

        {!loading && !error && !hasResults && (
          <div className={styles.emptyState}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" className={styles.emptyIcon}>
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
            <p className={styles.emptyTitle}>No analysis yet</p>
            <p className={styles.emptyBody}>Enter transaction details and click Analyze, or load a sample to preview results.</p>
          </div>
        )}

        {!loading && !error && hasResults && results && (
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.18, ease: 'easeOut' }}
            >
              {activeTab === 0 && <TabRiskAssessment results={results} />}
              {activeTab === 1 && <TabDashboard results={results} />}
              {activeTab === 2 && <TabBenchmark results={results} />}
              {activeTab === 3 && <TabExplainability results={results} />}
              {activeTab === 4 && <TabTestEval results={results} />}
            </motion.div>
          </AnimatePresence>
        )}
      </div>
    </div>
  )
}
