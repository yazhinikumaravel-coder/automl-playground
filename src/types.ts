/**
 * AutoML Playground - Applet Shared Type Definitions
 */

export interface IntegrityMetrics {
  score: number;
  missing: number;
  duplicates: number;
  rows: number;
  cols: number;
  badge: 'EMERALD' | 'AMBER' | 'ROSE';
}

export interface DatasetDescriptor {
  file_id: string;
  filename: string;
  columns: string[];
  integrity: IntegrityMetrics;
  guessed_target: string | null;
  row_count: number;
  rows: Record<string, string>[];
}

export interface ModelMetrics {
  name: string;
  category: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  best_params: Record<string, any>;
  confusion_matrix: number[][];
  features_used: string[];
  target_states: string[];
}

export interface PipelineResults {
  [model_id: string]: ModelMetrics;
}

export interface AlgorithmInfo {
  name: string;
  category: string;
  analogy: string;
  equation: string;
  tradeoffs: string;
  target: string;
}

export interface AlgorithmLab {
  [algorithm_id: string]: AlgorithmInfo;
}
