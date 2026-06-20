import { AlgorithmLab } from "./types";

export const ALGORITHM_LAB_DATA: AlgorithmLab = {
  // === CLASSIFICATION MODELS ===
  logistic_regression: {
    name: "Logistic Regression",
    category: "Classification",
    analogy: "A line drawn on a chalkboard dividing apples and oranges based on weight. Items near the line have uncertain probabilities, while ones far away are high-confidence.",
    equation: "p(x) = 1 / (1 + e^(-\\beta^T x))",
    tradeoffs: "Extremely fast, easy to interpret coefficients, but completely unable to capture non-linear relationships unless manual feature engineering is applied.",
    target: "Credit scoring, click-through-rate predictions, diagnostic baselines."
  },
  decision_tree: {
    name: "Decision Tree Classifier",
    category: "Classification",
    analogy: "The game of '20 Questions'. Is it an animal? Does it fly? Does it live in water? Each split divides the data into increasingly pure buckets.",
    equation: "Gini = 1 - \\sum p_i^2",
    tradeoffs: "Superb transparency, handles mixed types instantly, but prone to high variance and memory explosion by expanding trees too deep.",
    target: "Rule-based systems, medical diagnostic tree triage, customer churn pathways."
  },
  random_forest: {
    name: "Random Forest Classifier",
    category: "Classification",
    analogy: "A democratic committee of diverse, slightly random decision trees. Instead of trusting one expert, we average the opinions of a thousand specialists.",
    equation: "f(x) = (1/B) \\sum T_b(x)",
    tradeoffs: "Incredibly robust, resists overfitting via bagging, but slow to predict in multi-level architectures with millions of trees.",
    target: "General classification, default tabular champion, feature importance extraction."
  },
  svm: {
    name: "Support Vector Machine (SVM)",
    category: "Classification",
    analogy: "Two warring factions. The SVM places a wide concrete wall (the margin) exactly in the middle. If they are tangled, the SVM warps space (kernel trick) to split them.",
    equation: "f(x) = sign(\\sum \\alpha_i y_i K(x_i, x) + b)",
    tradeoffs: "Excellent high-dimensional handling, resists overfitting, but scales poorly on datasets with hundreds of thousands of rows.",
    target: "Text classification, bioinformatics, image search categorizations."
  },
  knn: {
    name: "K-Nearest Neighbors (KNN)",
    category: "Classification",
    analogy: "'Birds of a feather flock together'. To classify an unknown star, look at the cluster of its 5 closest stellar neighbors and take the majority vote.",
    equation: "d(p, q) = \\sqrt{\\sum (q_i - p_i)^2}",
    tradeoffs: "Zero training time, highly intuitive decision boundaries, but extremely slow inference because it has to calculate distances to every training data point on every query.",
    target: "Recommender systems, handwriting matching, niche anomaly clusters."
  },
  naive_bayes: {
    name: "Naive Bayes Classifier",
    category: "Classification",
    analogy: "Predicting if a message is spam by counting triggers like 'winner' or 'free', assuming each word's presence is completely independent of other words.",
    equation: "P(y|x) = \\frac{P(x|y)P(y)}{P(x)}",
    tradeoffs: "Extremely fast, requires minimal training data, but the independence assumption is almost always false in real-world scenarios.",
    target: "Spam filtering, simple sentiment categorization, real-time categorizer baselines."
  },
  gradient_boosting: {
    name: "Gradient Boosting Classifier",
    category: "Classification",
    analogy: "A mentor-apprentice chain. The first model makes predictions, the second trains solely on the first model's errors, and each successor corrects the cumulative leftovers.",
    equation: "F_m(x) = F_{m-1}(x) + \\gamma_m h_m(x)",
    tradeoffs: "Extreme precision and predictive capabilities on tabular data, but highly sequential (difficult to parallelize during training).",
    target: "Highly optimized search and ranking frameworks, consumer behavior conversion forecasting."
  },
  xgboost: {
    name: "XGBoost Classifier",
    category: "Classification",
    analogy: "A precision-engineered gradient booster with built-in regularizations, tree-pruning, and extreme hardware optimization. The absolute King of Kaggle.",
    equation: "Obj = \\sum L(y, \\hat{y}) + \\Omega(f)",
    tradeoffs: "Incredibly high predictive accuracy, handles missing data elegantly, but hyperparameter configuration is extremely vast and complex.",
    target: "Kaggle tournaments, algorithmic ranking, financial risk assessment."
  },

  // === REGRESSION MODELS ===
  linear_regression: {
    name: "Linear Regression",
    category: "Regression",
    analogy: "Fitting a rigid steel rod through a swarm of floating helium balloons to capture the general upwards or downwards trend mathematically.",
    equation: "y = \\beta_0 + \\sum \\beta_i x_i + \\epsilon",
    tradeoffs: "Blazing fast and instantly interpretable coefficients, but entirely blind to complex non-linear relations and highly sensitive to outliers.",
    target: "Price estimations, sales forecasting, baseline prediction trends."
  },
  ridge_regression: {
    name: "Ridge Regression",
    category: "Regression",
    analogy: "Standard linear regression, but with elastic bands pulling the regression line down, preventing individual sparse features from skewing the predictions too severely.",
    equation: "L = RSS + \\lambda \\sum \\beta_j^2",
    tradeoffs: "Solves model variance and multicollinearity gracefully, but does not perform feature selection (does not zero out column coefficients).",
    target: "Highly correlated dataset predictors, economic trend evaluations."
  },
  decision_tree_regressor: {
    name: "Decision Tree Regressor",
    category: "Regression",
    analogy: "Carving a landscape of scattered points into rectangular terraces, predicting any new point's value as the average height of its terrace.",
    equation: "MSE = (1/N) \\sum (y_i - \\bar{y})^2",
    tradeoffs: "Captures highly complex, step-wise non-linear features automatically without scaling, but suffers from severe overfitting and hard boundary jumps.",
    target: "Hierarchical process controls, non-linear continuous physical boundaries."
  },
  random_forest_regressor: {
    name: "Random Forest Regressor",
    category: "Regression",
    analogy: "Consulting a large crowd of real estate appraisers who all look at slightly different aspects of a house, then averaging their individual estimates under a democratic consensus.",
    equation: "f_RF(x) = (1/B) \\sum Tree_b(x)",
    tradeoffs: "Incredibly stable and highly accurate general baseline on continuous variables, but cannot extrapolate beyond the limits of its training range.",
    target: "Property appraisal estimates, algorithmic inventory forecasting."
  },
  svr: {
    name: "Support Vector Regressor (SVR)",
    category: "Regression",
    analogy: "Paving a straight, wide highway through a scatter of points. The SVR tries to fit as many data points as possible INSIDE the highway lanes, ignoring small errors.",
    equation: "L_\\epsilon = max(0, |y - f(x)| - \\epsilon)",
    tradeoffs: "Excellent for multi-dimensional data and highly robust to noise outside the lane margin, but computationally intensive on heavy row scales.",
    target: "Financial derivatives pricing, chemical concentration projections."
  },
  knn_regressor: {
    name: "K-Nearest Neighbors Regressor",
    category: "Regression",
    analogy: "Estimating the value of a house by looking up the sold prices of its 5 closest geographical neighbors and calculating the simple mathematical average.",
    equation: "f(x) = (1/k) \\sum_{x_i \\in N_k(x)} y_i",
    tradeoffs: "Zero initial training time and naturally adapts to localized variations, but prediction loops require scanning all historical rows, making it scale terribly.",
    target: "Local commodity valuations, spatial distance regressions."
  },
  gradient_boosting_regressor: {
    name: "Gradient Boosting Regressor",
    category: "Regression",
    analogy: "Sculpting a statue. The first stage chisels a rough outline, and each successive junior model gently scrapes off the remaining clay lumps (residuals) on every pass.",
    equation: "F_m(x) = F_{m-1}(x) + \\gamma_m h_m(x)",
    tradeoffs: "Superb accuracy on tabular data sheets and handles missing values natively, but training is strictly sequential and slow to compute.",
    target: "Corporate demand planning, high-stakes metric projections."
  },
  xgboost_regressor: {
    name: "XGBoost Regressor",
    category: "Regression",
    analogy: "A state-of-the-art continuous predictor armed with hardware-aware cash optimizations and regularizations, carving perfect pathways for tabular predictions.",
    equation: "Obj = \\sum l(y_i, \\hat{y}_i) + \\sum \\Omega(f_k)",
    tradeoffs: "Tremendously high continuous accuracy and execution speed, but requires comprehensive configuration sweeps to avoid overfitting.",
    target: "Surgical inventory controls, dynamic prices engine benchmarks."
  }
};
