import { useState, useEffect, useLayoutEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import styles from './OnboardingTour.module.css'

const TOUR_KEY = 'catboost_tour_done'
const PAD = 10
const TOOLTIP_W = 300
const GAP = 16

const STEPS = [
  {
    target: '[data-tour="header"]',
    title: 'AMD CatBoost Fraud Detection Demo',
    body: 'This demo runs on AMD Instinct MI300X hardware with the ROCm open-source software stack — delivering GPU-accelerated CatBoost fraud detection at enterprise scale.',
    position: 'bottom',
  },
  {
    target: '[data-tour="form"]',
    title: 'Configure Transaction',
    body: 'Enter transaction details to analyze fraud risk. Key PCA-transformed features and the transaction amount drive the CatBoost model\'s fraud probability score.',
    position: 'right',
  },
  {
    target: '[data-tour="submit"]',
    title: 'Run Fraud Analysis',
    body: 'Click Analyze Transaction to run the trained CatBoost model. The GPU-accelerated model returns a fraud probability and LOW/MEDIUM/HIGH risk tier in milliseconds.',
    position: 'top',
  },
  {
    target: '[data-tour="results"]',
    title: 'Risk Assessment Results',
    body: 'View the fraud probability score, risk tier classification, feature importance rankings, model performance metrics, and GPU vs CPU benchmark comparisons.',
    position: 'left',
  },
  {
    target: '[data-tour="cards"]',
    title: 'Explore the Tech Stack',
    body: 'Click any card to learn more about the hardware and software powering this demo — from MI300X specs to ROCm documentation.',
    position: 'top',
    scrollBlock: 'end',
  },
]

function getTooltipPos(rect, position) {
  const centreX = Math.max(12, Math.min(rect.left + rect.width / 2 - TOOLTIP_W / 2, window.innerWidth - TOOLTIP_W - 12))
  // For left/right, vertically center the tooltip on the element and clamp to viewport
  const midY = Math.max(GAP + 60, Math.min(rect.top + rect.height / 2 - 90, window.innerHeight - 220))

  if (position === 'bottom') return { top: rect.bottom + GAP, left: centreX, width: TOOLTIP_W }
  if (position === 'top')    return { bottom: window.innerHeight - rect.top + GAP, left: centreX, width: TOOLTIP_W }
  if (position === 'right')  return { top: midY, left: rect.right + GAP, width: TOOLTIP_W }
  if (position === 'left')   return { top: midY, left: rect.left - GAP - TOOLTIP_W, width: TOOLTIP_W }
}

export default function OnboardingTour({ theme = 'dark', onToggle }) {
  const [step, setStep]       = useState(0)
  const [rect, setRect]       = useState(null)
  const [visible, setVisible] = useState(() => !localStorage.getItem(TOUR_KEY))

  function dismiss() {
    localStorage.setItem(TOUR_KEY, '1')
    setRect(null)
    setVisible(false)
  }

  function restart() {
    setStep(0)
    setRect(null)
    setVisible(true)
  }

  function next() {
    if (step < STEPS.length - 1) setStep((s) => s + 1)
    else dismiss()
  }

  const measureTarget = useCallback(() => {
    const el = document.querySelector(STEPS[step].target)
    if (!el) return
    // Always scroll the target to the center of the viewport so the
    // spotlight appears at a consistent central position each step.
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setTimeout(() => setRect(el.getBoundingClientRect()), 380)
  }, [step])

  useLayoutEffect(() => {
    if (!visible) return
    measureTarget()
  }, [step, visible, measureTarget])

  useEffect(() => {
    if (!visible) return
    window.addEventListener('resize', measureTarget)
    return () => window.removeEventListener('resize', measureTarget)
  }, [visible, measureTarget])

  // Block user scrolling during the tour so fixed tooltips stay aligned with
  // their targets. The tour's own scrollIntoView (programmatic) still works.
  useEffect(() => {
    if (!visible) return
    const scrollKeys = new Set(['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '])
    const prevent = (e) => e.preventDefault()
    const preventKeys = (e) => {
      const tag = e.target.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'BUTTON') return
      if (scrollKeys.has(e.key)) e.preventDefault()
    }
    window.addEventListener('wheel', prevent, { passive: false })
    window.addEventListener('touchmove', prevent, { passive: false })
    window.addEventListener('keydown', preventKeys, { passive: false })
    return () => {
      window.removeEventListener('wheel', prevent)
      window.removeEventListener('touchmove', prevent)
      window.removeEventListener('keydown', preventKeys)
    }
  }, [visible])

  const showTour   = visible && rect
  const current    = STEPS[step]
  const tooltipPos = rect ? getTooltipPos(rect, current.position) : null
  const isDark     = theme === 'dark'

  return (
    <div className={styles.root}>
      <button className={styles.themeBtn} onClick={onToggle} title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}>
        {isDark ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="5" />
            <line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" />
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
            <line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" />
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
          </svg>
        )}
        {isDark ? 'Light' : 'Dark'}
      </button>

      {!visible && (
        <button className={styles.demoBtn} onClick={restart} title="Replay tour">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          Tour
        </button>
      )}

      {showTour && (
        <div
          className={styles.overlay}
          onClick={dismiss}
          style={{
            '--sx': `${rect.left   - PAD}px`,
            '--sy': `${rect.top    - PAD}px`,
            '--sw': `${rect.width  + PAD * 2}px`,
            '--sh': `${rect.height + PAD * 2}px`,
          }}
        />
      )}

      {showTour && (
        <motion.div
          className={styles.spotlight}
          animate={{
            left:    rect.left   - PAD,
            top:     rect.top    - PAD,
            width:   rect.width  + PAD * 2,
            height:  rect.height + PAD * 2,
            opacity: 1,
          }}
          initial={{
            left:    rect.left   - PAD,
            top:     rect.top    - PAD,
            width:   rect.width  + PAD * 2,
            height:  rect.height + PAD * 2,
            opacity: 0,
          }}
          transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
        />
      )}

      <AnimatePresence mode="wait">
        {showTour && tooltipPos && (
          <motion.div
            key={step}
            className={styles.tooltip}
            style={tooltipPos}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{    opacity: 0, y: -6 }}
            transition={{ duration: 0.28, ease: 'easeOut' }}
          >
            <div className={styles.tooltipStep}>Step {step + 1} of {STEPS.length}</div>
            <div className={styles.tooltipTitle}>{current.title}</div>
            <p className={styles.tooltipBody}>{current.body}</p>
            <div className={styles.tooltipFooter}>
              <button className={styles.skipBtn} onClick={dismiss}>Skip</button>
              <button className={styles.nextBtn} onClick={next}>
                {step < STEPS.length - 1 ? 'Next →' : 'Get Started'}
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
