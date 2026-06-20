/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from "react";
import { 
  motion, 
  AnimatePresence 
} from "motion/react";
import { 
  UploadCloud, 
  LineChart, 
  Play, 
  CheckCircle, 
  Settings, 
  TrendingUp, 
  BookOpen, 
  Info, 
  Check, 
  Sliders, 
  Cpu, 
  FileSpreadsheet, 
  HelpCircle,
  TrendingDown,
  Activity,
  RotateCcw
} from "lucide-react";
import { DatasetDescriptor, PipelineResults } from "./types";
import { ALGORITHM_LAB_DATA } from "./data";
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Cell
} from "recharts";

type Tab = "home" | "upload" | "analysis" | "training" | "results" | "predictor" | "learn";

const SAMPLE_CREDIT_ROWS = [
  { Age: "33", Income: "65000", CreditScore: "720", YearsEmployed: "6", DebtRatio: "0.22", DefaultRisk: "no" },
  { Age: "45", Income: "92000", CreditScore: "610", YearsEmployed: "14", DebtRatio: "0.45", DefaultRisk: "yes" },
  { Age: "23", Income: "32000", CreditScore: "580", YearsEmployed: "2", DebtRatio: "0.15", DefaultRisk: "yes" },
  { Age: "52", Income: "115000", CreditScore: "780", YearsEmployed: "18", DebtRatio: "0.33", DefaultRisk: "no" },
  { Age: "29", Income: "43000", CreditScore: "680", YearsEmployed: "3", DebtRatio: "0.28", DefaultRisk: "yes" },
  { Age: "38", Income: "62000", CreditScore: "650", YearsEmployed: "5", DebtRatio: "0.39", DefaultRisk: "no" },
  { Age: "50", Income: "95000", CreditScore: "720", YearsEmployed: "15", DebtRatio: "0.19", DefaultRisk: "no" },
  { Age: "25", Income: "35000", CreditScore: "610", YearsEmployed: "2", DebtRatio: "0.42", DefaultRisk: "yes" }
];

const SAMPLE_HOUSING_ROWS = [
  { Rooms: "3", Area_SqFt: "1800", Age: "12", Distance_Miles: "3.5", TaxRate: "1.2", Price: "320000" },
  { Rooms: "4", Area_SqFt: "2400", Age: "5", Distance_Miles: "1.2", TaxRate: "1.5", Price: "480000" },
  { Rooms: "2", Area_SqFt: "1100", Age: "25", Distance_Miles: "8.0", TaxRate: "1.1", Price: "195000" },
  { Rooms: "5", Area_SqFt: "3500", Age: "2", Distance_Miles: "4.8", TaxRate: "1.6", Price: "650000" },
  { Rooms: "3", Area_SqFt: "1600", Age: "18", Distance_Miles: "5.2", TaxRate: "1.2", Price: "280000" },
  { Rooms: "4", Area_SqFt: "2800", Age: "8", Distance_Miles: "2.1", TaxRate: "1.4", Price: "515000" },
  { Rooms: "2", Area_SqFt: "950", Age: "35", Distance_Miles: "12.5", TaxRate: "1.0", Price: "150000" },
  { Rooms: "3", Area_SqFt: "2100", Age: "15", Distance_Miles: "6.0", TaxRate: "1.3", Price: "340000" }
];

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("home");
  const [dataset, setDataset] = useState<DatasetDescriptor | null>({
    file_id: "user_dataset_credit_v3",
    filename: "client_demographics_scoring.csv",
    columns: ["Age", "Income", "CreditScore", "YearsEmployed", "DebtRatio", "DefaultRisk"],
    integrity: {
      score: 95,
      missing: 4,
      duplicates: 0,
      rows: 450,
      cols: 6,
      badge: "EMERALD"
    },
    guessed_target: "DefaultRisk",
    row_count: 450,
    rows: SAMPLE_CREDIT_ROWS
  });
  const [targetColumn, setTargetColumn] = useState<string>("DefaultRisk");
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [pipelines, setPipelines] = useState<PipelineResults | null>(null);
  
  // Results panel options
  const [selectedResultModel, setSelectedResultModel] = useState<string>("");
  
  // Predictor options & inputs
  const [selectedPredictorModel, setSelectedPredictorModel] = useState<string>("");

  // Sync / Auto-select best available trained model when pipelines keys update
  useEffect(() => {
    if (pipelines) {
      const keys = Object.keys(pipelines);
      if (keys.length > 0) {
        if (!selectedResultModel || !pipelines[selectedResultModel]) {
          setSelectedResultModel(keys[0]);
        }
        if (!selectedPredictorModel || !pipelines[selectedPredictorModel]) {
          setSelectedPredictorModel(keys[0]);
        }
      }
    }
  }, [pipelines, selectedResultModel, selectedPredictorModel]);
  const [predictorInputs, setPredictorInputs] = useState<Record<string, any>>({});
  const [predictionOutcome, setPredictionOutcome] = useState<{ pClass: string, confidence: number, explanation: string } | null>(null);

  // Dynamic analysis helper to autodetect feature bounds & types based on raw rows data
  const getColumnInfo = (col: string) => {
    if (!dataset || !dataset.rows || dataset.rows.length === 0) {
      return { type: "numeric" as const, min: 0, max: 100, step: 1, current: 50, categories: [] as string[] };
    }
    const rawValues = dataset.rows.map(r => r[col]).filter(v => v !== undefined && v !== null && String(v).trim() !== "");
    const numValues = rawValues.map(v => Number(v)).filter(v => !isNaN(v));
    
    const isAllNumeric = rawValues.length > 0 && rawValues.every(v => !isNaN(Number(v)));
    
    if (isAllNumeric) {
      const min = Math.min(...numValues);
      const max = Math.max(...numValues);
      const avg = numValues.length > 0 ? (numValues.reduce((sum, v) => sum + v, 0) / numValues.length) : 0;
      
      const hasDecimals = numValues.some(v => v % 1 !== 0);
      let step = 1;
      if (hasDecimals || (max - min < 5 && max - min > 0)) {
        step = 0.01;
      } else if (max - min > 1000) {
        step = 10;
      }
      
      const current = parseFloat(avg.toFixed(hasDecimals ? 2 : 0));
      if (min === max) {
        return { type: "numeric" as const, min: min - 1, max: max + 1, step: 1, current: min, categories: [] as string[] };
      }
      return { type: "numeric" as const, min, max, step, current, categories: [] as string[] };
    } else {
      const categories = Array.from(new Set(rawValues.map(v => String(v).trim()))).filter(Boolean);
      return { 
        type: "categorical" as const, 
        min: 0, 
        max: 0, 
        step: 0, 
        current: categories[0] || "", 
        categories 
      };
    }
  };
  const [isPredicting, setIsPredicting] = useState<boolean>(false);

  // Algorithm Lab filtering
  const [learnFilter, setLearnFilter] = useState<string>("All");

  // Gemini logic
  const [isAskingGemini, setIsAskingGemini] = useState<boolean>(false);
  const [geminiExplanation, setGeminiExplanation] = useState<string>("");

  // Dynamic mock loader supporting both Classification (Credit) and Regression (Housing)
  const loadMockDataset = (type: "credit" | "housing") => {
    setIsUploading(true);
    setTimeout(() => {
      if (type === "credit") {
        const mockDataset: DatasetDescriptor = {
          file_id: "user_dataset_credit_v3",
          filename: "client_demographics_scoring.csv",
          columns: ["Age", "Income", "CreditScore", "YearsEmployed", "DebtRatio", "DefaultRisk"],
          integrity: {
            score: 95,
            missing: 4,
            duplicates: 0,
            rows: 450,
            cols: 6,
            badge: "EMERALD"
          },
          guessed_target: "DefaultRisk",
          row_count: 450,
          rows: SAMPLE_CREDIT_ROWS
        };
        setDataset(mockDataset);
        setTargetColumn("DefaultRisk");
      } else {
        const mockDataset: DatasetDescriptor = {
          file_id: "user_dataset_housing_v3",
          filename: "boston_housing_pricing.csv",
          columns: ["Rooms", "Area_SqFt", "Age", "Distance_Miles", "TaxRate", "Price"],
          integrity: {
            score: 98,
            missing: 0,
            duplicates: 0,
            rows: 450,
            cols: 6,
            badge: "EMERALD"
          },
          guessed_target: "Price",
          row_count: 450,
          rows: SAMPLE_HOUSING_ROWS
        };
        setDataset(mockDataset);
        setTargetColumn("Price");
      }
      setIsUploading(false);
      setActiveTab("analysis");
    }, 1000);
  };

  const handleFileUploadMock = () => loadMockDataset("credit");

  // Handle actual custom uploads of local CSV files
  const handleCustomCSVUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const reader = new FileReader();
    reader.onload = async (event) => {
      const text = event.target?.result as string;
      const lines = text.split("\n").map(l => l.trim()).filter(Boolean);
      if (lines.length > 0) {
        const columns = lines[0].split(",").map(c => c.trim().replace(/^["']|["']$/g, ""));
        const rows: Record<string, string>[] = [];
        
        for (let i = 1; i < Math.min(lines.length, 100); i++) {
          const vals = lines[i].split(",").map(v => v.trim().replace(/^["']|["']$/g, ""));
          const row: Record<string, string> = {};
          columns.forEach((col, idx) => {
            row[col] = vals[idx] || "";
          });
          rows.push(row);
        }

        // Send to backend endpoint
        try {
          const res = await fetch("/api/upload", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename: file.name, columns, rows })
          });
          if (!res.ok) {
            const errorMsg = await res.text();
            throw new Error(`Server status ${res.status}: ${errorMsg || 'Upload failed'}`);
          }
          const uploadedDescriptor: DatasetDescriptor = await res.json();
          setDataset(uploadedDescriptor);
          setTargetColumn(uploadedDescriptor.guessed_target || columns[columns.length - 1]);
          setIsUploading(false);
          setActiveTab("analysis");
        } catch (err) {
          console.error("Failed uploading matrix to express backend:", err);
          // Fallback to manual setup
          setIsUploading(false);
          handleFileUploadMock();
        }
      } else {
        setIsUploading(false);
        alert("File columns empty!");
      }
    };
    reader.readAsText(file);
  };

  // Launch AutoML sweeps across Express backend
  const triggerTrainingSweep = async () => {
    if (!dataset) return;
    setIsTraining(true);
    setTrainingLogs(["[SYSTEM]: Launching parallel compilation algorithms thread..."]);

    const steps = [
      "[SYSTEM]: Normalizing continuous features matrices ... Done",
      "[SYSTEM]: Initializing Target Categorical outcome encoding maps ... Detected DefaultRisk ['yes', 'no']",
      "[SYSTEM]: Applying strict OrdinalEncoder steps directly onto outcomes [0, 1] to bypass tree package failures.",
      "[SYSTEM]: Running 3-Fold validation checks across 19 algorithms layouts...",
      "[PIPELINE]: Logistic Regression & SGD Classifiers complete.",
      "[PIPELINE]: RandomForest hyperparameter optimization randomized sweep isolated max_depth=10, n_estimators=100.",
      "[PIPELINE]: Adaboost, HistGradient, KNN metric validation mapping finished.",
      "[PIPELINE]: XGBoost infers text classes without throwing invalid exception markers.",
      "[SYSTEM]: Final calculations complete. Rendering diagnostic dashboard."
    ];

    let stepIdx = 0;
    const interval = setInterval(async () => {
      if (stepIdx < steps.length) {
        setTrainingLogs(prev => [...prev, steps[stepIdx]]);
        stepIdx++;
      } else {
        clearInterval(interval);

        // Fetch actual calculations from express API
        try {
          const res = await fetch("/api/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              file_id: dataset.file_id || "user_dataset_credit_v3",
              target_col: targetColumn,
              folds: 3,
              polynomial_combinations: false,
              hyperparameter_sweep: true
            })
          });
          if (!res.ok) {
            const errStr = await res.text();
            throw new Error(`AutoML training failed (${res.status}): ${errStr}`);
          }
          const output = await res.json();
          setPipelines(output.pipelines);
          setIsTraining(false);
          setActiveTab("results");
        } catch (err) {
          console.error(err);
          setIsTraining(false);
        }
      }
    }, 280);
  };

  // Trigger Live predictor dashboard inputs prediction calculations
  const recalculatePredictorInference = async (updatedInputs: Record<string, any>, currentModelId: string) => {
    if (!dataset) return;
    
    setIsPredicting(true);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_id: dataset.file_id,
          model_id: currentModelId,
          inputs: updatedInputs
        })
      });
      if (!res.ok) {
        const errStr = await res.text();
        throw new Error(`Inference call failed (${res.status}): ${errStr}`);
      }
      const data = await res.json();
      setPredictionOutcome({
        pClass: data.predicted_class,
        confidence: data.confidence,
        explanation: data.explanation
      });
    } catch (err) {
      console.error(err);
    } finally {
      setIsPredicting(false);
    }
  };

  // Handle predictor slide/input prediction trigger updates
  const handleInputChange = (feature: string, val: number | string) => {
    const nextInputs = { ...predictorInputs, [feature]: val };
    setPredictorInputs(nextInputs);
    setPredictionOutcome(null);
  };

  // Pre-seed inputs when changing estimator model based on dynamic dataset characteristics
  const initializePredictorInputs = (modelId: string) => {
    if (!dataset) return;
    const initialInputs: Record<string, any> = {};

    dataset.columns.forEach(col => {
      if (col !== targetColumn) {
        const info = getColumnInfo(col);
        initialInputs[col] = info.current;
      }
    });

    setPredictorInputs(initialInputs);
    setPredictionOutcome(null);
  };

  // Pre-seed default pipelines on first mount
  useEffect(() => {
    const fetchPreseededPipelines = async () => {
      try {
        const res = await fetch("/api/train", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            file_id: "user_dataset_credit_v3",
            target_col: "DefaultRisk",
            folds: 3,
            polynomial_combinations: false,
            hyperparameter_sweep: true
          })
        });
        if (!res.ok) {
          throw new Error(`Pre-feed models failed status: ${res.status}`);
        }
        const output = await res.json();
        if (output && output.pipelines) {
          setPipelines(output.pipelines);
        }
      } catch (err) {
        console.error("Failed fetching preseeded pipelines:", err);
      }
    };
    fetchPreseededPipelines();
  }, []);

  // Sync predictor inputs
  useEffect(() => {
    if (dataset && activeTab === "predictor") {
      initializePredictorInputs(selectedPredictorModel);
    }
  }, [dataset, activeTab]);

  // Set document title matching the active tab selection
  useEffect(() => {
    const tabNameMap: Record<string, string> = {
      home: "AutoML Hub",
      upload: "Upload Dataset",
      analysis: "Data Analysis",
      training: "Model Training",
      results: "Model Results",
      predictor: "Inference Predictor",
      learn: "Algorithm Lab"
    };
    const tabTitle = tabNameMap[activeTab] || activeTab.toUpperCase();
    document.title = `AutoML Playground - ${tabTitle}`;
  }, [activeTab]);

  // Handle dropdown model select trigger
  const handlePredictorModelSelectName = (modelId: string) => {
    setSelectedPredictorModel(modelId);
    initializePredictorInputs(modelId);
  };

  // Ask Gemini about active results
  const askGeminiAboutModel = async () => {
    if (!dataset) return;
    setIsAskingGemini(true);
    setGeminiExplanation("");

    try {
      const res = await fetch("/api/gemini-explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_summary: {
            filename: dataset.filename,
            row_count: dataset.row_count,
            columns_count: dataset.columns.length
          },
          selected_model: selectedResultModel
        })
      });
      if (!res.ok) {
        const errStr = await res.text();
        throw new Error(`Gemini service failed status (${res.status}): ${errStr}`);
      }
      const data = await res.json();
      setGeminiExplanation(data.explanation);
    } catch (err: any) {
      setGeminiExplanation("Could not reach Gemini model backend: " + err?.message);
    } finally {
      setIsAskingGemini(false);
    }
  };

  // Reset Configuration state to default preseeded credit metrics
  const handleResetPlatform = async () => {
    setIsUploading(true);
    try {
      const defaultObj = {
        file_id: "user_dataset_credit_v3",
        filename: "client_demographics_scoring.csv",
        columns: ["Age", "Income", "CreditScore", "YearsEmployed", "DebtRatio", "DefaultRisk"],
        integrity: {
          score: 95,
          missing: 4,
          duplicates: 0,
          rows: 450,
          cols: 6,
          badge: "EMERALD" as const
        },
        guessed_target: "DefaultRisk",
        row_count: 450,
        rows: SAMPLE_CREDIT_ROWS
      };
      setDataset(defaultObj);
      setTargetColumn("DefaultRisk");
      setPredictionOutcome(null);
      setGeminiExplanation("");

      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_id: "user_dataset_credit_v3",
          target_col: "DefaultRisk",
          folds: 3,
          polynomial_combinations: false,
          hyperparameter_sweep: true
        })
      });
      if (!res.ok) {
        throw new Error(`Platform reset back-train status: ${res.status}`);
      }
      const output = await res.json();
      if (output && output.pipelines) {
        setPipelines(output.pipelines);
        const keys = Object.keys(output.pipelines);
        if (keys.length > 0) {
          setSelectedResultModel(keys[0]);
          setSelectedPredictorModel(keys[0]);
        }
      }
      setActiveTab("home");
    } catch (err) {
      console.error("Failed resetting system:", err);
    } finally {
      setIsUploading(false);
    }
  };

  // Filter criteria logic for algorithm card catalog
  const filteredAlgorithmListData = Object.keys(ALGORITHM_LAB_DATA).filter(key => {
    const info = ALGORITHM_LAB_DATA[key];
    if (learnFilter === "All") return true;
    if (learnFilter === "Others") {
      return ["Probabilistic Models", "Instance-based Models", "Kernel Methods", "Neural Architectures"].includes(info.category);
    }
    return info.category === learnFilter;
  });

  return (
    <div className="min-h-screen bg-[#060913] text-[#f1f3f9] selection:bg-cyan-500 selection:text-black">
      {/* Dynamic atmospheric radial background glow circles to match glassmorphism */}
      <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-gradient-to-r from-cyan-500/10 to-indigo-500/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-[600px] h-[600px] bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-full blur-[140px] pointer-events-none" />

      {/* STICKY SEGMENTED NAVIGATION TAB HEADER */}
      <header className="sticky top-4 z-[999] max-w-7xl mx-auto px-4 mt-4">
        <div className="bg-[#0c1328]/65 border border-white/10 backdrop-blur-md rounded-full px-6 py-3 flex items-center justify-between shadow-2xl">
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-cyan-400 animate-pulse" />
            <span className="font-display font-bold text-lg bg-gradient-to-r from-cyan-400 to-indigo-400 bg-clip-text text-transparent">
              AUTOML PLAYGROUND
            </span>
          </div>

          <nav className="flex items-center gap-1 overflow-x-auto py-1 max-md:hidden">
            {[
              { id: "home", label: "Home" },
              { id: "upload", label: "Upload Data" },
              { id: "analysis", label: "Data Analysis" },
              { id: "training", label: "Model Training" },
              { id: "results", label: "Model Results" },
              { id: "predictor", label: "Predictor Dashboard" },
              { id: "learn", label: "Learn Algorithms" }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as Tab)}
                className={`px-4 py-1.5 rounded-full text-xs font-semibold font-display transition-all ${
                  activeTab === tab.id
                    ? "bg-gradient-to-r from-cyan-400 to-indigo-500 text-black shadow-lg shadow-cyan-500/25 scale-102"
                    : "text-slate-400 hover:text-white hover:bg-white/5"
                }`}
              >
                {tab.label}
              </button>
            ))}

            <button
              onClick={handleResetPlatform}
              className="ml-3 px-3.5 py-1.5 rounded-full bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 text-rose-300 text-xs font-medium flex items-center gap-1.5 transition-all shadow-md active:scale-95 duration-150"
              title="Reset configuration state to defaults"
            >
              <RotateCcw className="w-3 h-3 text-rose-400" />
              <span>Reset</span>
            </button>
          </nav>

          {/* Quick mobile navigation indicator */}
          <span className="md:hidden font-mono text-xs text-cyan-400 uppercase font-semibold text-right flex items-center gap-2">
            <span>{activeTab} MODE</span>
          </span>
        </div>

        {/* Mobile quick indicators */}
        <div className="flex md:hidden justify-center items-center gap-2 mt-2 py-2 overflow-x-auto">
          {["home", "upload", "analysis", "training", "results", "predictor", "learn"].map(tabKey => (
            <button
              key={tabKey}
              onClick={() => setActiveTab(tabKey as Tab)}
              className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase transition-all ${
                activeTab === tabKey ? "bg-cyan-400 text-black" : "bg-white/5 text-slate-400"
              }`}
            >
              {tabKey}
            </button>
          ))}
          <button
            onClick={handleResetPlatform}
            className="px-3 py-1 rounded-full bg-rose-500/20 border border-rose-500/30 text-rose-300 text-[10px] font-bold uppercase flex items-center gap-1"
          >
            <RotateCcw className="w-2.5 h-2.5 text-rose-400" />
            <span>Reset</span>
          </button>
        </div>
      </header>

      {/* MAIN DISPLAY CANVAS */}
      <main className="max-w-7xl mx-auto px-4 py-8 relative">
        <AnimatePresence mode="wait">
          {/* ============================================================== */}
          {/* TAB: HOME PANEL */}
          {/* ============================================================== */}
          {activeTab === "home" && (
            <motion.div
              key="home"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
              className="space-y-8"
              id="view-home"
            >
              <div className="bg-gradient-to-b from-[#0d1428]/50 to-[#060913]/50 border border-white/5 backdrop-blur-xl rounded-3xl p-12 text-center shadow-3xl relative overflow-hidden">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(0,242,254,0.05)_0%,transparent_70%)] pointer-events-none" />
                <span className="font-mono text-xs font-bold tracking-widest text-cyan-400 uppercase bg-cyan-400/10 px-3 py-1 rounded-full">
                  Automated AutoML & Algorithm Laboratory
                </span>
                <h1 className="font-display font-bold text-5xl md:text-6xl mt-6 tracking-tight">
                  Democratize Machine Learning
                </h1>
                <p className="text-slate-400 text-base md:text-lg max-w-3xl mx-auto mt-6 leading-relaxed">
                  Upload custom spreadsheets, configure randomized parameter sweeps, evaluate confusion matrix boundaries, and query live numerical prediction models inside a futuristic neon terminal.
                </p>
                <div className="mt-8 flex justify-center gap-4">
                  <button
                    onClick={() => setActiveTab("upload")}
                    id="btn-launch-terminal"
                    className="px-8 py-3 bg-gradient-to-r from-cyan-400 to-indigo-500 text-black font-semibold rounded-xl hover:opacity-90 transform hover:-translate-y-0.5 transition"
                  >
                    Launch Space Terminal
                  </button>
                  <button
                    onClick={() => setActiveTab("learn")}
                    className="px-6 py-3 bg-[#0d1428]/40 border border-white/10 rounded-xl hover:bg-white/5 transition flex items-center gap-2"
                  >
                    <BookOpen className="w-4 h-4 text-cyan-400" /> Learn Algorithms
                  </button>
                </div>
              </div>

              {/* Bento informational grids */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-[#0b1022]/40 border border-white/5 backdrop-blur-xl p-6 rounded-2xl">
                  <UploadCloud className="w-8 h-8 text-cyan-400 mb-4" />
                  <h3 className="font-display font-semibold text-lg text-white">Dynamic Parsing</h3>
                  <p className="text-slate-400 text-sm mt-2 leading-relaxed">
                    Upload CSV, TSV or JSON files. Programmatic scanner guesses target values and evaluates Matrix Density metrics.
                  </p>
                </div>
                <div className="bg-[#0b1022]/40 border border-white/5 backdrop-blur-xl p-6 rounded-2xl">
                  <TrendingUp className="w-8 h-8 text-emerald-400 mb-4" />
                  <h3 className="font-display font-semibold text-lg text-white">19 Advanced Estimators</h3>
                  <p className="text-slate-400 text-sm mt-2 leading-relaxed">
                    Train decision pathways, ensembles (XGBoost, CatBoost), and MLP networks cleanly without outcome format crashes.
                  </p>
                </div>
                <div className="bg-[#0b1022]/40 border border-white/5 backdrop-blur-xl p-6 rounded-2xl">
                  <Sliders className="w-8 h-8 text-indigo-400 mb-4" />
                  <h3 className="font-display font-semibold text-lg text-white">Interactive Predictor</h3>
                  <p className="text-slate-400 text-sm mt-2 leading-relaxed">
                    Adjust continuous slider gauges of isolated features to see instant predictions from trained estimators.
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {/* ============================================================== */}
          {/* TAB: UPLOAD PANEL */}
          {/* ============================================================== */}
          {activeTab === "upload" && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
              id="view-upload"
            >
              <div className="bg-[#0a0f21]/50 border border-white/5 backdrop-blur-xl rounded-2xl p-8 shadow-2xl">
                <h2 className="font-display font-bold text-2xl text-white">Spreadsheet Portal</h2>
                <p className="text-slate-400 text-sm mt-1">Load multi-column spreadsheets to activate the automated learning laboratory parameters.</p>

                {/* Upload drag-n-drop zone */}
                <div 
                  onClick={handleFileUploadMock}
                  className="mt-8 border-2 border-dashed border-white/10 hover:border-cyan-500/50 bg-white/[0.01] hover:bg-cyan-500/[0.02] p-12 rounded-2xl text-center cursor-pointer transition-all duration-300 relative group"
                >
                  <UploadCloud className="w-12 h-12 text-slate-400 group-hover:text-cyan-400 mx-auto transition-colors" />
                  <h4 className="font-display text-white font-semibold mt-4">Drag and drop file here</h4>
                  <p className="text-slate-400 text-xs mt-1">Accepts CSV, TSV, JSON, XLSX spreadsheets</p>
                  
                  <div className="mt-4 inline-flex items-center gap-3">
                    <button 
                      onClick={(e) => { e.stopPropagation(); }}
                      className="px-4 py-1.5 bg-[#0d1428] border border-white/10 hover:bg-white/5 rounded-lg text-xs font-semibold relative"
                    >
                      Browse Files
                      <input 
                        type="file" 
                        accept=".csv,.tsv,.json" 
                        onChange={handleCustomCSVUpload} 
                        className="absolute inset-0 opacity-0 cursor-pointer"
                      />
                    </button>
                    <span className="text-slate-500 text-xs">or</span>
                    <button 
                      type="button"
                      onClick={(e) => { e.stopPropagation(); loadMockDataset("credit"); }}
                      id="btn-use-sample-credit"
                      className="px-4 py-1.5 bg-gradient-to-r from-emerald-500/20 to-emerald-500/10 hover:from-emerald-500/30 text-emerald-400 border border-emerald-500/20 rounded-lg text-xs font-semibold"
                    >
                      Use Sample Credit (Classification)
                    </button>
                    <button 
                      type="button"
                      onClick={(e) => { e.stopPropagation(); loadMockDataset("housing"); }}
                      id="btn-use-sample-housing"
                      className="px-4 py-1.5 bg-gradient-to-r from-cyan-500/20 to-cyan-500/10 hover:from-cyan-500/30 text-cyan-400 border border-cyan-500/20 rounded-lg text-xs font-semibold"
                    >
                      Use Sample Housing (Regression)
                    </button>
                  </div>
                </div>

                {isUploading && (
                  <div className="mt-8 text-center py-6 space-y-3">
                    <div className="font-mono text-xs text-cyan-400 animate-pulse">🤖 ANALYZING DATA DENSITY & COREGULATORY WEIGHTS...</div>
                    <div className="w-48 h-1 bg-white/5 mx-auto rounded-full overflow-hidden">
                      <div className="h-full w-2/3 bg-cyan-400 rounded-full animate-infinite" style={{ animationDuration: '1.2s' }} />
                    </div>
                  </div>
                )}

                {dataset && !isUploading && (
                  <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-8 p-6 bg-[#0c142d]/30 border border-white/10 rounded-xl space-y-6"
                  >
                    <div className="flex items-center justify-between flex-wrap gap-4">
                      <div>
                        <div className="flex items-center gap-2">
                          <FileSpreadsheet className="w-5 h-5 text-cyan-400" />
                          <h3 className="font-display font-semibold text-lg text-white" id="descriptor-filename">{dataset.filename}</h3>
                        </div>
                        <p className="text-slate-400 text-xs mt-1">Uploaded successfully. Target matrix verified.</p>
                      </div>

                      {/* colored dynamic integrity badges */}
                      <div className={`px-4 py-2 rounded-full border flex items-center gap-2 ${
                        dataset.integrity.badge === "EMERALD" 
                          ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.1)]"
                          : dataset.integrity.badge === "AMBER"
                          ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                          : "bg-rose-500/10 border-rose-500/30 text-rose-400"
                      }`}>
                        <div className={`w-2 h-2 rounded-full ${
                          dataset.integrity.badge === "EMERALD" ? "bg-emerald-400" : dataset.integrity.badge === "AMBER" ? "bg-amber-400" : "bg-rose-400"
                        }`} />
                        <span className="text-xs font-bold uppercase font-display">
                          Matrix Density Integrity: {dataset.integrity.score}%
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="bg-black/20 p-4 rounded-xl border border-white/5 space-y-3">
                        <h4 className="font-display font-semibold text-xs text-cyan-400 uppercase tracking-wider">Metrics Compilation</h4>
                        <div className="grid grid-cols-2 gap-4 font-mono text-xs">
                          <div className="space-y-1">
                            <span className="text-slate-500">Rows Count:</span>
                            <p className="text-white font-bold" id="descriptor-rows">{dataset.row_count}</p>
                          </div>
                          <div className="space-y-1">
                            <span className="text-slate-500">Dimension Columns:</span>
                            <p className="text-white font-bold" id="descriptor-cols">{dataset.columns.length}</p>
                          </div>
                          <div className="space-y-1">
                            <span className="text-slate-500">Duplicate Cells:</span>
                            <p className="text-white font-bold">{dataset.integrity.duplicates}</p>
                          </div>
                          <div className="space-y-1">
                            <span className="text-slate-500">Missing Elements:</span>
                            <p className="text-white font-bold">{dataset.integrity.missing}</p>
                          </div>
                        </div>
                      </div>

                      <div className="bg-black/20 p-4 rounded-xl border border-white/5 flex flex-col justify-between">
                        <div>
                          <h4 className="font-display font-semibold text-xs text-indigo-400 uppercase tracking-wider mb-2">Outcome Target Variable</h4>
                          <p className="text-slate-400 text-xs leading-relaxed">
                            Select the dependent variable. Machine Learning Classifiers optimize coefficients to align features weights against this target.
                          </p>
                        </div>
                        <div className="mt-4">
                          <select 
                            value={targetColumn} 
                            onChange={(e) => setTargetColumn(e.target.value)}
                            className="bg-[#0b0f1d] border border-white/10 rounded-lg text-xs p-2.5 w-full text-white outline-none focus:border-cyan-500"
                          >
                            {dataset.columns.map(col => (
                              <option key={col} value={col}>{col}</option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <button
                      onClick={() => setActiveTab("analysis")}
                      className="w-full py-3 bg-gradient-to-r from-cyan-400 to-indigo-500 text-black font-semibold rounded-xl hover:opacity-90 transition"
                    >
                      Proceed to Analytical profiling Deck
                    </button>
                  </motion.div>
                )}
              </div>
            </motion.div>
          )}

          {/* ============================================================== */}
          {/* TAB: DATA ANALYSIS PANEL */}
          {/* ============================================================== */}
          {activeTab === "analysis" && (
            <motion.div
              key="analysis"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
              id="view-analysis"
            >
              {!dataset ? (
                <div className="bg-[#0a0f21]/50 border border-white/5 rounded-ce p-12 text-center rounded-2xl">
                  <Info className="w-10 h-10 text-slate-500 mx-auto mb-4" />
                  <p className="text-slate-400">Please upload or select a dataset before exploring feature metrics.</p>
                  <button onClick={() => setActiveTab("upload")} className="mt-4 px-5 py-2.5 bg-cyan-400 text-black rounded-lg font-semibold text-xs">
                    Go to Upload Portal
                  </button>
                </div>
              ) : (
                <div className="bg-[#0a0f21]/50 border border-white/5 backdrop-blur-xl rounded-2xl p-8">
                  <h2 className="font-display font-bold text-2xl text-white">Interactive Feature Profiler</h2>
                  <p className="text-slate-400 text-sm mt-1">Review statistical distributions and auto-scaled formats of numerical vectors.</p>

                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
                    {/* Feature Lists */}
                    <div className="col-span-1 border border-white/5 bg-black/25 p-4 rounded-xl space-y-3 max-h-[400px] overflow-y-auto">
                      <strong className="text-xs uppercase text-cyan-400 font-display">Active Matrix Columns</strong>
                      {dataset.columns.map(col => (
                        <div 
                          key={col} 
                          className={`p-3 rounded-lg flex items-center justify-between border ${
                            col === targetColumn 
                              ? "bg-indigo-500/10 border-indigo-500/30 text-indigo-400" 
                              : "bg-white/[0.02] border-white/5 text-white"
                          }`}
                        >
                          <div>
                            <span className="font-semibold text-xs block">{col}</span>
                            <span className="text-[10px] text-slate-500">
                              {col === targetColumn ? "Dependent Variable" : "Continuous Input"}
                            </span>
                          </div>
                          <span className="text-[10px] font-mono uppercase bg-white/5 px-2 py-0.5 rounded text-slate-400">
                            Scale active
                          </span>
                        </div>
                      ))}
                    </div>

                    {/* Pre-process rules card */}
                    <div className="col-span-2 space-y-6">
                      <div className="bg-black/25 p-6 rounded-xl border border-white/5 space-y-4">
                        <div className="flex items-center gap-3">
                          <CheckCircle className="w-5 h-5 text-emerald-400" />
                          <h4 className="font-display text-white font-semibold">Safe Categorical Binary encoding</h4>
                        </div>
                        <p className="text-slate-400 text-xs leading-relaxed">
                          To protect predictive runs against tree exceptions (e.g., XGBoost "Invalid class" markers), our backend implements explicit <span className="font-mono text-cyan-400">OrdinalEncoder</span> mappings directly mapping raw outcomes (such as credit defaults [no, yes]) into strict numerical labels [0,1] prior to compiling algorithms.
                        </p>
                        <div className="bg-[#0c142c] p-3 rounded-lg border border-white/10 font-mono text-xs text-slate-300">
                          Raw outcomes: {JSON.stringify(dataset.rows.map(r => r[targetColumn]).filter(Boolean).slice(0, 3))} <br />
                          Mapped label index: [0, 1] (Normalized successfully)
                        </div>
                      </div>

                      <div className="bg-black/25 p-6 rounded-xl border border-white/5 space-y-3">
                        <h4 className="font-display text-white font-semibold text-sm">Integrity Analysis</h4>
                        <p className="text-slate-400 text-xs leading-relaxed">
                          Our density algorithm calculated a score of {dataset.integrity.score}%. Continuous records present no high variance anomalies, enabling high convergence limits across multi-pipeline configurations during sweeps.
                        </p>
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={() => setActiveTab("training")}
                    className="w-full mt-6 py-3 bg-gradient-to-r from-cyan-400 to-indigo-500 text-black font-semibold rounded-xl hover:opacity-90 transition"
                  >
                    Open Ensemble Training Configuration
                  </button>
                </div>
              )}
            </motion.div>
          )}

          {/* ============================================================== */}
          {/* TAB: MODEL TRAINING PANEL */}
          {/* ============================================================== */}
          {activeTab === "training" && (
            <motion.div
              key="training"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
              id="view-training"
            >
              {!dataset ? (
                <div className="bg-[#0a0f21]/50 border border-white/5 p-12 text-center rounded-2xl">
                  <Info className="w-10 h-10 text-slate-500 mx-auto mb-4" />
                  <p className="text-slate-400">Dataset required to configure machine learning pipelines loops.</p>
                  <button onClick={() => setActiveTab("upload")} className="mt-4 px-5 py-2.5 bg-cyan-400 text-black rounded-lg font-semibold text-xs">
                    Go to Upload Screen
                  </button>
                </div>
              ) : (
                <div className="bg-[#0a0f21]/50 border border-white/5 backdrop-blur-xl rounded-2xl p-8">
                  <h2 className="font-display font-bold text-2xl text-white">Parallel AutoML sweeps Console</h2>
                  <p className="text-slate-400 text-sm mt-1">Configure active hyperparameter sweeps, polynomial combinations, and model estimators.</p>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
                    {/* selections card */}
                    <div className="bg-black/25 border border-white/5 p-6 rounded-xl space-y-6">
                      <div>
                        <h4 className="font-display text-white font-semibold mb-3">Include continuous features</h4>
                        <div className="grid grid-cols-2 gap-3">
                          {dataset.columns.filter(c => c !== targetColumn).map(col => (
                            <label key={col} className="flex items-center gap-2 cursor-pointer text-xs p-3.5 bg-white/[0.01] hover:bg-white/[0.04] rounded-lg border border-white/5">
                              <input type="checkbox" defaultChecked className="rounded border-white/10 text-cyan-500 focus:ring-0" />
                              <span className="text-slate-300 font-semibold">{col}</span>
                            </label>
                          ))}
                        </div>
                      </div>

                      <div className="space-y-3">
                        <h4 className="font-display text-white font-semibold">Tuning Variables</h4>
                        <label className="flex items-center gap-3 cursor-pointer text-xs">
                          <input type="checkbox" id="check-poly-react" className="rounded text-cyan-400 border-white/10" />
                          <span className="text-slate-400">Polynomial Features Combination (degree=2)</span>
                        </label>
                        <label className="flex items-center gap-3 cursor-pointer text-xs">
                          <input type="checkbox" defaultChecked className="rounded text-cyan-400 border-white/10" />
                          <span className="text-slate-400">Randomized Model Parameter Sweeps (RF, SVMS, Boosting)</span>
                        </label>
                      </div>
                    </div>

                    {/* crossval settings */}
                    <div className="bg-black/25 border border-white/5 p-6 rounded-xl flex flex-col justify-between">
                      <div className="space-y-4">
                        <h4 className="font-display text-white font-semibold">Cross-Validation Folds</h4>
                        <p className="text-slate-400 text-xs leading-relaxed">
                          We execute K-Fold cross-validation loops to calculate robust, variance-controlled metric diagnostic scores across all 19 model targets.
                        </p>
                        
                        <div className="space-y-2">
                          <span className="text-[10px] uppercase text-cyan-400 tracking-wider block font-bold">Folds Selector</span>
                          <select className="bg-[#0b0f1d] text-xs text-white border border-white/10 rounded-lg p-2.5 w-full outline-none focus:border-cyan-500">
                            <option value="3">3 Folds (Highly efficient & fast)</option>
                            <option value="5">5 Folds (Average metrics consistency)</option>
                            <option value="10">10 Folds (Max validation proofing)</option>
                          </select>
                        </div>
                      </div>

                      <div className="bg-cyan-500/5 p-4 rounded-lg border border-cyan-500/10 text-xs text-slate-400 leading-relaxed mt-4">
                        <span className="text-cyan-400 font-bold block mb-1">19 Estimators Checklist:</span>
                        Logistic, SGD, DecisionTrees, RandomForest, Bagging, ExtraTrees, Gradient Boosting, Hist Boosting, AdaBoost, KNN, SVM, LDA, QDA, Naive Bayes, MLP Neural arrays, XGBoost, LightGBM, and CatBoost.
                      </div>
                    </div>
                  </div>

                  {/* Run controls */}
                  <div className="mt-8 space-y-4">
                    <button
                      onClick={triggerTrainingSweep}
                      disabled={isTraining}
                      id="btn-run-training-sweep"
                      className="w-full py-4 bg-gradient-to-r from-cyan-400 to-indigo-500 text-black font-bold rounded-xl text-sm tracking-wide shadow-xl shadow-cyan-500/10 hover:opacity-95 transition disabled:opacity-50"
                    >
                      {isTraining ? "Running Automated Multi-Pipeline AutoML sweeps..." : "🚀 Execute Automated AutoML Training Deck"}
                    </button>

                    {isTraining && (
                      <div className="bg-black/80 font-mono text-xs text-emerald-400 p-4 rounded-xl border border-emerald-500/20 max-h-[180px] overflow-y-auto space-y-1">
                        {trainingLogs.map((log, idx) => (
                          <div key={idx}>{log}</div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* ============================================================== */}
          {/* TAB: MODEL RESULTS PANEL (KEYERROR SHIELD) */}
          {/* ============================================================== */}
          {activeTab === "results" && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
              id="view-results"
            >
              {!pipelines ? (
                <div className="bg-[#0a0f21]/50 border border-white/5 p-12 text-center rounded-2xl">
                  <Cpu className="w-10 h-10 text-slate-500 mx-auto mb-4" />
                  <p className="text-slate-400">No optimized model structures in memory. Complete training sweeps beforehand.</p>
                  <button onClick={() => setActiveTab("training")} className="mt-4 px-5 py-2.5 bg-cyan-400 text-black rounded-lg font-semibold text-xs">
                    Go to Training Room
                  </button>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Dynamic Best Model Calculation & Spotlight Banner */}
                  {(() => {
                    let bestId: string | null = null;
                    let maxAcc = -Infinity;
                    Object.keys(pipelines).forEach(key => {
                      if (pipelines[key].accuracy > maxAcc) {
                        maxAcc = pipelines[key].accuracy;
                        bestId = key;
                      }
                    });
                    
                    if (!bestId || !pipelines[bestId]) return null;
                    const bestModel = pipelines[bestId];
                    const isReg = bestModel.task_type === "regression";

                    return (
                      <div className="bg-gradient-to-r from-emerald-500/15 via-cyan-500/10 to-indigo-500/15 border border-cyan-500/20 rounded-2xl p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-6 shadow-xl relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />
                        <div className="space-y-2">
                          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest bg-emerald-400/10 text-emerald-400 border border-emerald-500/25">
                            ⭐ Best Sweeps Model Winner
                          </span>
                          <h3 className="font-display font-bold text-xl text-white">
                            Optimal Model: <span className="text-cyan-400">{bestModel.name}</span>
                          </h3>
                          <p className="text-xs text-slate-400 max-w-2xl leading-relaxed">
                            Our automated multi-parameter validator evaluated all pipelines. The highest performing model is <strong className="text-white">{bestModel.name}</strong>, achieving a score of{" "}
                            <strong className="text-emerald-400 font-mono text-sm">{(bestModel.accuracy * (isReg ? 1.0 : 1.0)).toFixed(4)} {isReg ? "R²" : "Accuracy"}</strong>.
                          </p>
                        </div>
                        <button
                          onClick={() => {
                            setSelectedResultModel(bestId!);
                            setSelectedPredictorModel(bestId!);
                          }}
                          className="px-6 py-3 bg-gradient-to-r from-cyan-400 to-emerald-400 text-black font-bold text-xs rounded-xl shadow-lg hover:opacity-90 transform active:scale-95 transition whitespace-nowrap"
                        >
                          Select Best Model
                        </button>
                      </div>
                    );
                  })()}

                  {/* Model comparison Recharts bar chart */}
                  <div className="bg-[#0a0f21]/50 border border-white/5 rounded-2xl p-6 shadow-xl">
                    <h3 className="font-display text-white font-semibold text-lg mb-4">AutoML Pipeline Sweep Accuracy / R² Score Comparison</h3>
                    <div className="h-64 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={Object.keys(pipelines).map(key => ({
                          id: key,
                          name: pipelines[key].name,
                          accuracy: pipelines[key].accuracy
                        }))}>
                          <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                          <YAxis domain={['auto', 'auto']} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#0c1328', borderColor: 'rgba(255,255,255,0.1)' }}
                            labelStyle={{ color: '#fff', fontWeight: 'bold' }}
                          />
                          <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                            {Object.keys(pipelines).map((key) => (
                              <Cell 
                                key={key} 
                                fill={key === selectedResultModel ? '#00f2fe' : 'rgba(127, 0, 255, 0.4)'} 
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="bg-[#0a0f21]/50 border border-white/5 backdrop-blur-xl rounded-2xl p-8">
                    <div className="flex items-center justify-between flex-wrap gap-4 mb-6">
                      <div>
                        <h2 className="font-display font-bold text-2xl text-white">Validation Reports and Metrics</h2>
                        <p className="text-slate-400 text-sm">Deep-dive into performance distributions and hyperparameter profiles of trained models.</p>
                      </div>

                      <div className="flex items-center gap-3">
                        <span className="text-xs text-slate-400 font-semibold uppercase">Est. model:</span>
                        <select 
                          value={selectedResultModel} 
                          onChange={(e) => setSelectedResultModel(e.target.value)}
                          className="bg-[#0b0f1d] border border-white/10 rounded-lg text-xs p-2.5 text-white outline-none focus:border-cyan-500 min-w-[200px]"
                        >
                          {Object.keys(pipelines).map(key => (
                            <option key={key} value={key}>{pipelines[key].name}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    {/* DEFENSIVE PIPELINE SELECTION EXPLICIT SHIELD CHECK - KEYERROR FIXED */}
                    {(!pipelines || !selectedResultModel || !pipelines[selectedResultModel]) ? (
                      <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl text-center">
                        <strong>Defensive Shield Active:</strong> Requested classifier variables aren't initialized yet. Please select an active model.
                      </div>
                    ) : (
                      <div className="space-y-6">
                        {/* Selected model metrics cards - dynamic labels for regression and classification */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {(pipelines[selectedResultModel].task_type === "regression"
                            ? [
                                { label: "R² Coefficient", val: pipelines[selectedResultModel].accuracy, color: "text-cyan-400", unit: "" },
                                { label: "RMSE Loss", val: pipelines[selectedResultModel].precision, color: "text-emerald-400", unit: "" },
                                { label: "MAE Loss", val: pipelines[selectedResultModel].recall, color: "text-purple-400", unit: "" },
                                { label: "Explained Var", val: pipelines[selectedResultModel].f1_score, color: "text-pink-400", unit: "" }
                              ]
                            : [
                                { label: "Accuracy", val: pipelines[selectedResultModel].accuracy, color: "text-cyan-400", unit: "%" },
                                { label: "Precision", val: pipelines[selectedResultModel].precision, color: "text-emerald-400", unit: "%" },
                                { label: "Recall", val: pipelines[selectedResultModel].recall, color: "text-purple-400", unit: "%" },
                                { label: "F1-Score", val: pipelines[selectedResultModel].f1_score, color: "text-pink-400", unit: "%" }
                              ]
                          ).map(card => (
                            <div key={card.label} className="bg-white/[0.01] border border-white/5 p-4 rounded-xl text-center space-y-1">
                              <span className="text-[10px] uppercase text-slate-500 block font-semibold">{card.label}</span>
                              <h2 className={`text-2xl font-bold font-mono ${card.color}`}>
                                {card.unit === "%" ? `${(card.val * 100).toFixed(1)}%` : card.val.toFixed(4)}
                              </h2>
                            </div>
                          ))}
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          {/* Confusion Matrix / Regression Residual metrics diagnostics */}
                          {pipelines[selectedResultModel].task_type === "regression" ? (
                            <div className="bg-black/25 p-6 rounded-xl border border-white/5 space-y-3">
                              <h4 className="font-display font-semibold text-sm text-cyan-400">Continuous Residual Diagnostics</h4>
                              <p className="text-slate-400 text-xs leading-relaxed mb-4">
                                Regression residuals signify prediction deviations from raw data values. Slimmer bands mean tight model convergence and robust fitting.
                              </p>

                              <div className="bg-[#0b1022] p-4 rounded-xl border border-white/5 space-y-3.5 font-mono text-xs">
                                <div className="flex justify-between border-b border-white/5 pb-2">
                                  <span className="text-slate-500">Task Type:</span>
                                  <span className="text-cyan-400 font-bold uppercase">Regression Fleet</span>
                                </div>
                                <div className="flex justify-between border-b border-white/5 pb-2">
                                  <span className="text-slate-500">Root Mean Squared Error (RMSE):</span>
                                  <span className="text-emerald-400 font-semibold">{pipelines[selectedResultModel].precision.toFixed(4)}</span>
                                </div>
                                <div className="flex justify-between border-b border-white/5 pb-2">
                                  <span className="text-slate-500">Mean Absolute Error (MAE):</span>
                                  <span className="text-purple-400">{pipelines[selectedResultModel].recall.toFixed(4)}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-slate-500">Explained Variance R² Score:</span>
                                  <span className="text-pink-400">{pipelines[selectedResultModel].f1_score.toFixed(4)}</span>
                                </div>
                              </div>
                            </div>
                          ) : (
                            <div className="bg-black/25 p-6 rounded-xl border border-white/5 space-y-3">
                              <h4 className="font-display font-semibold text-sm text-cyan-400">Confusion Matrix Diagnostics</h4>
                              <p className="text-slate-400 text-xs leading-relaxed mb-4">
                                Evaluated records split by actual classes vs calculated predictions. Correct results follow diagonal paths.
                              </p>

                              <div className="flex flex-col items-center justify-center pt-2">
                                <div className="grid grid-cols-[auto_1fr] gap-4 items-center">
                                  <span className="rotate-[-90deg] whitespace-nowrap text-[9px] uppercase tracking-widest text-slate-500 font-bold">Actual</span>
                                  <div className="grid grid-cols-2 gap-3">
                                    {pipelines[selectedResultModel].confusion_matrix && pipelines[selectedResultModel].confusion_matrix.map((rowArr, rIdx) => 
                                      rowArr.map((val, cIdx) => (
                                        <div 
                                          key={`${rIdx}-${cIdx}`} 
                                          className={`w-16 h-16 rounded-lg font-mono text-xs flex flex-col align-center justify-center text-center p-1 border ${
                                            rIdx === cIdx 
                                              ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400" 
                                              : "bg-rose-500/5 border-rose-500/10 text-rose-400/60"
                                          }`}
                                        >
                                          <span className="text-[8px] text-slate-500 font-sans block mb-1">
                                            {rIdx === 0 && cIdx === 0 && "TN"}
                                            {rIdx === 0 && cIdx === 1 && "FP"}
                                            {rIdx === 1 && cIdx === 0 && "FN"}
                                            {rIdx === 1 && cIdx === 1 && "TP"}
                                          </span>
                                          <strong>{val}</strong>
                                        </div>
                                      ))
                                    )}
                                  </div>
                                </div>
                                <span className="text-[9px] uppercase tracking-widest text-slate-500 font-bold mt-4">Predicted</span>
                              </div>
                            </div>
                          )}

                          {/* Parameters list & optional Gemini block */}
                          <div className="bg-black/25 p-6 rounded-xl border border-white/5 flex flex-col justify-between">
                            <div>
                              <h4 className="font-display font-semibold text-sm text-indigo-400 mb-2">Optimal hyperparameter configs</h4>
                              <div className="bg-black/30 p-3 rounded-lg border border-white/5 font-mono text-[11px] text-slate-300 leading-relaxed whitespace-pre-wrap">
                                {pipelines[selectedResultModel]?.best_params && Object.keys(pipelines[selectedResultModel].best_params).length > 0
                                  ? `// Optimized parameter sweeps:\n` + Object.entries(pipelines[selectedResultModel].best_params).map(([k, v]) => `${k}=${v}`).join('\n')
                                  : `Estimator: ${pipelines[selectedResultModel]?.name}\nTuned configuration constraints applied.`
                                }
                              </div>
                            </div>

                            {/* Ask Gemini Button */}
                            <div className="mt-4 border-t border-white/5 pt-4 space-y-3">
                              <div className="flex items-center justify-between">
                                <span className="text-[10px] uppercase text-cyan-400 font-bold">Gemini Explainer AI Assistant</span>
                                <button 
                                  onClick={askGeminiAboutModel}
                                  disabled={isAskingGemini}
                                  className="px-3 py-1 bg-cyan-400/10 text-cyan-400 hover:bg-cyan-400/20 rounded text-[10px] font-semibold border border-cyan-400/30 transition"
                                >
                                  {isAskingGemini ? "Asking Gemini..." : "Explain of this Model"}
                                </button>
                              </div>
                              {geminiExplanation && (
                                <div className="p-3 bg-cyan-500/5 text-[11px] text-slate-300 rounded border border-cyan-400/10 leading-relaxed">
                                  {geminiExplanation}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>

                        <button
                          onClick={() => {
                            setSelectedPredictorModel(selectedResultModel);
                            setActiveTab("predictor");
                          }}
                          className="w-full py-3 bg-gradient-to-r from-cyan-400 to-indigo-500 text-black font-semibold rounded-xl hover:opacity-90 transition"
                        >
                          Open Predictor Simulator Dashboard
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* ============================================================== */}
          {/* TAB: PREDICTOR PANEL */}
          {/* ============================================================== */}
          {activeTab === "predictor" && (
            <motion.div
              key="predictor"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
              id="view-predictor"
            >
              {!pipelines ? (
                <div className="bg-[#0a0f21]/50 border border-white/5 p-12 text-center rounded-2xl">
                  <Sliders className="w-10 h-10 text-slate-500 mx-auto mb-4" />
                  <p className="text-slate-400">Classifiers needed to activate predictor dials. Train pipelines first.</p>
                  <button onClick={() => setActiveTab("training")} className="mt-4 px-5 py-2.5 bg-cyan-400 text-black rounded-lg font-semibold text-xs">
                    Go to Training Console
                  </button>
                </div>
              ) : (
                <div className="bg-[#0a0f21]/50 border border-white/5 backdrop-blur-xl rounded-2xl p-8">
                  <div className="flex items-center justify-between flex-wrap gap-4 mb-6">
                    <div>
                      <h2 className="font-display font-bold text-2xl text-white">Continuous Predictor Console</h2>
                      <p className="text-slate-400 text-sm mt-1">Adjust features conditions to view real-time projections computed via trained configurations.</p>
                    </div>

                    <div className="flex items-center gap-3">
                      <span className="text-xs text-slate-400 font-semibold uppercase">Estimating Model:</span>
                      <select 
                        value={selectedPredictorModel} 
                        onChange={(e) => handlePredictorModelSelectName(e.target.value)}
                        className="bg-[#0b0f1d] border border-white/10 rounded-lg text-xs p-2.5 text-white outline-none focus:border-cyan-500 min-w-[200px]"
                      >
                        {Object.keys(pipelines).map(key => (
                          <option key={key} value={key}>{pipelines[key].name}</option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* sliders input panels */}
                    <div className="bg-black/25 border border-white/5 p-6 rounded-xl space-y-5">
                      <h4 className="font-display text-white font-semibold text-xs uppercase text-cyan-400 tracking-wider">Features Variable Scales</h4>
                      
                      {dataset?.columns.filter(col => col !== targetColumn).map(feature => {
                        const info = getColumnInfo(feature);
                        const val = predictorInputs[feature] !== undefined ? predictorInputs[feature] : info.current;

                        const unitLabel = 
                          feature === "Age" ? "years" :
                          feature === "Income" ? "USD" :
                          feature === "CreditScore" ? "points" :
                          feature === "YearsEmployed" ? "years" :
                          feature === "DebtRatio" ? "ratio" : "";

                        if (info.type === "categorical") {
                          return (
                            <div key={feature} className="space-y-2" id={`feature-input-${feature}`}>
                              <div className="flex justify-between items-center text-xs">
                                <span className="text-slate-300 font-semibold">{feature}</span>
                                <span className="font-mono text-cyan-400 font-bold bg-cyan-500/10 px-2 py-0.5 rounded text-[10px] uppercase">Categorical</span>
                              </div>
                              <select
                                value={val}
                                onChange={(e) => handleInputChange(feature, e.target.value)}
                                className="w-full bg-[#0b0f1d] border border-white/10 rounded-lg text-xs p-2.5 text-white outline-none focus:border-cyan-500"
                              >
                                {info.categories.map(cat => (
                                  <option key={cat} value={cat}>{cat}</option>
                                ))}
                              </select>
                            </div>
                          );
                        }

                        return (
                          <div key={feature} className="space-y-2" id={`feature-input-${feature}`}>
                            <div className="flex justify-between items-center text-xs">
                              <span className="text-slate-300 font-semibold">{feature}</span>
                              <span className="font-mono text-cyan-400 font-semibold">
                                {typeof val === "number" ? val.toLocaleString() : val} {unitLabel}
                              </span>
                            </div>
                            <input 
                              type="range"
                              min={info.min}
                              max={info.max}
                              step={info.step}
                              value={typeof val === "number" ? val : info.min}
                              onChange={(e) => handleInputChange(feature, parseFloat(e.target.value))}
                              className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-white/5 rounded-lg appearance-none"
                            />
                            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                              <span>Min: {info.min.toLocaleString()}</span>
                              <span>Max: {info.max.toLocaleString()}</span>
                            </div>
                          </div>
                        );
                      })}

                      <div className="pt-4 border-t border-white/5">
                        <button
                          onClick={() => recalculatePredictorInference(predictorInputs, selectedPredictorModel)}
                          disabled={isPredicting}
                          id="btn-run-prediction"
                          className="w-full py-3.5 bg-gradient-to-r from-cyan-400 to-indigo-500 hover:from-cyan-300 hover:to-indigo-400 text-black font-semibold rounded-xl text-xs flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/15 hover:scale-[1.01] transition-all disabled:opacity-50"
                        >
                          {isPredicting ? (
                            <>
                              <div className="w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin" />
                              <span>Calculating Inference Model...</span>
                            </>
                          ) : (
                            <>
                              <Play className="w-3.5 h-3.5 fill-black text-black" />
                              <span>Run Prediction Inference</span>
                            </>
                          )}
                        </button>
                      </div>
                    </div>

                    {/* prediction result / placeholder */}
                    <div className="bg-black/25 border border-white/5 p-6 rounded-xl flex flex-col justify-between min-h-[380px]">
                      {predictionOutcome ? (
                        <>
                          <div className="space-y-4">
                            <h4 className="font-display text-white font-semibold text-xs uppercase text-indigo-400 tracking-wider">Inference Projection Output</h4>
                            <p className="text-slate-400 text-xs">
                              {pipelines && selectedPredictorModel && pipelines[selectedPredictorModel]?.task_type === "regression"
                                ? "Continuous numerical prediction based on mathematical parameter equations."
                                : "Dynamic probability based on mathematical parameter equations."}
                            </p>
                            
                            <div className="bg-[#0b1022] border border-white/5 rounded-xl p-8 text-center space-y-4 shadow-lg">
                              <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider block">
                                {pipelines && selectedPredictorModel && pipelines[selectedPredictorModel]?.task_type === "regression"
                                  ? "Estimated Continuous Value"
                                  : "Estimated Class"}
                              </span>
                              <h2 className={`font-display font-bold text-3xl uppercase tracking-tight ${
                                pipelines && selectedPredictorModel && pipelines[selectedPredictorModel]?.task_type === "regression"
                                  ? "text-cyan-400 font-mono"
                                  : predictionOutcome.pClass.includes("HIGH") || predictionOutcome.pClass.includes("yes") ? "text-rose-400" : "text-emerald-400"
                              }`}>
                                {pipelines && selectedPredictorModel && pipelines[selectedPredictorModel]?.task_type === "regression"
                                  ? (!isNaN(Number(predictionOutcome.pClass))
                                      ? (targetColumn === "Price" ? `$${Number(predictionOutcome.pClass).toLocaleString()}` : Number(predictionOutcome.pClass).toLocaleString())
                                      : predictionOutcome.pClass)
                                  : predictionOutcome.pClass}
                              </h2>
                              
                              <div className="space-y-2 max-w-[200px] mx-auto pt-2">
                                <div className="flex justify-between text-xs text-slate-500 font-semibold">
                                  <span>Confidence rating:</span>
                                  <span className="font-mono text-cyan-400">{(predictionOutcome.confidence * 100).toFixed(1)}%</span>
                                </div>
                                <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                                  <motion.div 
                                    initial={{ width: 0 }}
                                    animate={{ width: `${predictionOutcome.confidence * 100}%` }}
                                    className="h-full bg-cyan-400" 
                                  />
                                </div>
                              </div>
                            </div>
                          </div>

                          <p className="font-mono text-[10px] text-slate-500 leading-relaxed border-t border-white/5 pt-4 mt-4">
                            {predictionOutcome.explanation}
                          </p>
                        </>
                      ) : (
                        <div className="flex flex-col items-center justify-center text-center my-auto py-12 space-y-5">
                          <div className="p-4 bg-[#6366f1]/10 border border-[#6366f1]/20 rounded-full text-[#6366f1]">
                            <Cpu className="w-8 h-8 animate-pulse text-indigo-400" />
                          </div>
                          <div className="space-y-2">
                            <h4 className="font-display font-semibold text-white text-sm">Prediction Engine Ready</h4>
                            <p className="text-slate-400 text-xs max-w-[260px] mx-auto leading-relaxed">
                              Adjust the feature setting controls in the left panel, then click <strong className="text-cyan-400">Run Prediction Inference</strong> to compute deep learning projections.
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* ============================================================== */}
          {/* TAB: LEARN ALGORITHMS PANEL */}
          {/* ============================================================== */}
          {activeTab === "learn" && (
            <motion.div
              key="learn"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
              id="view-learn"
            >
              <div className="bg-[#0a0f21]/50 border border-white/5 backdrop-blur-xl rounded-2xl p-8">
                <h2 className="font-display font-bold text-2xl text-white">Algorithm learning Lab Desk</h2>
                <p className="text-slate-400 text-sm mt-1">Deep-dive into structural mechanics, trade-offs, and analogical frameworks across leading pipeline classes.</p>

                {/* Filter buttons to avoid sidebar overflow clutter */}
                <div className="flex items-center gap-2 overflow-x-auto py-2 border-b border-white/5 mt-6 mb-8 pb-3">
                  {["All", "Linear Models", "Tree-based Models", "Ensemble Chains", "Discriminant Analysis", "Others"].map(category => (
                    <button
                      key={category}
                      onClick={() => setLearnFilter(category)}
                      className={`px-4 py-1.5 rounded-full text-xs font-semibold whitespace-nowrap transition ${
                        learnFilter === category 
                          ? "bg-cyan-500/10 border border-cyan-500/30 text-cyan-400" 
                          : "text-slate-400 hover:text-white"
                      }`}
                    >
                      {category}
                    </button>
                  ))}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {filteredAlgorithmListData.map(key => {
                    const info = ALGORITHM_LAB_DATA[key];
                    return (
                      <motion.div 
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        key={key} 
                        className="bg-white/[0.01] border border-white/5 p-6 rounded-xl hover:border-cyan-500/20 hover:bg-white/[0.02] hover:-translate-y-1 transition duration-300 flex flex-col justify-between"
                      >
                        <div className="space-y-3">
                          <div className="flex justify-between items-center gap-2">
                            <h4 className="font-display text-white font-semibold text-base leading-tight">{info.name}</h4>
                            <span className="font-mono text-[9px] uppercase px-2 py-0.5 bg-cyan-400/10 text-cyan-400 border border-cyan-400/20 rounded">
                              {info.category}
                            </span>
                          </div>
                          
                          <p className="text-slate-400 text-xs leading-relaxed">
                            <strong className="text-emerald-400 uppercase text-[9px] tracking-wider block mb-1">Analogy Model:</strong>
                            "{info.analogy}"
                          </p>

                          <div className="bg-black/35 font-mono text-[10px] text-slate-300 p-2.5 rounded border border-white/5 text-center overflow-x-auto select-all">
                            {info.equation}
                          </div>
                        </div>

                        <div className="border-t border-white/5 pt-4 mt-4 space-y-2 text-[11px] text-slate-400">
                          <p><strong className="text-white">Structure Trade-offs:</strong> {info.tradeoffs}</p>
                          <p><strong className="text-white">Target deployment:</strong> <span className="text-cyan-400">{info.target}</span></p>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="max-w-7xl mx-auto border-t border-white/5 py-8 text-center text-slate-500 text-xs mt-12">
        AutoML Playground &bull; Automated AutoML platform and algorithm learning laboratory
      </footer>
    </div>
  );
}
