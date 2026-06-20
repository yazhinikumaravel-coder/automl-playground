# ML Explorer v3 🚀

ML Explorer v3 is a feature-rich, high-performance **AutoML & Interactive Predictive Modeling Platform** built on a full-stack architecture (React, Vite, Tailwind CSS, and Express/TypeScript). It enables both beginners and seasoned practitioners to upload datasets, analyze feature correlations, train predictive models, adjust hyperparameters in an "Algorithm Lab," and run real-time inference using a dynamic probability-based prediction engine.

---

## ✨ Features

- **📊 Dynamic Data Analysis**: Extract statistics, auto-identify target characteristics, and view interactive correlation metrics, distribution histograms, and multi-variable charts.
- **🤖 AutoML Training Engine**: Train multiple supervised models (Random Forest, Decision Tree, Support Vector Machine, Linear/Logistic Regression) simultaneously.
- **⚙️ Interactive Algorithm Lab**: Tune hyperparameters (like max depth, learning rate, and estimators) interactively and watch evaluation metrics (Loss, Accuracy, Precision, Recall, F1-Score) update dynamically.
- **🔮 Real-Time Inference Predictor**: Try out models via a live simulation dashboard. Features are parsed dynamically as continuous sliders or categorical dropdown menus, computing sigmoidal probability outputs based on dynamic Pearson correlation analysis.
- **🌌 Sleek Cybernetic Theme**: A highly refined UI featuring custom interactive scales, styled tooltips, and a gorgeous dark galactic visual identity.

---

## 🛠️ Tech Stack

- **Frontend**: React 18, Vite, Tailwind CSS, Recharts / D3, Lucide Icons, Frame Motion.
- **Backend**: Node.js Express server configured with tsx/esbuild for robust math calculations, input mapping, and Pearson correlation matrices.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have **Node.js (v18 or higher)** installed.

### Installation

1. Clone or download the repository contents.
2. In the root directory, install all required dependencies:
   ```bash
   npm install
   ```

### Running Locally

To start the development server on `http://localhost:3000`:
```bash
npm run dev
```

### Production Build

Compile the production-ready clients and bundle the server into a standalone executable:
```bash
npm run build
```

To run the compiled production application:
```bash
npm start
```

---

## 🌐 Project Structure

- `/src/App.tsx` - Main frontend client workspace and navigation logic.
- `/server.ts` - High-fidelity mathematical prediction routing and Express API.
- `/index.html` - Primary entry point adorned with custom SVG favicon and metadata.
- `/src/index.css` - Custom global typography variables (Inter, Space Grotesk, JetBrains Mono).
- `package.json` - Complete dependency scripts and bundling configurations.
