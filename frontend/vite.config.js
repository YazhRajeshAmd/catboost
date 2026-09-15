import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/run':    'http://localhost:7866',
      '/api':    'http://localhost:7866',
      '/queue':  'http://localhost:7866',
      '/upload': 'http://localhost:7866',
    }
  }
})
