import { useState, useEffect, useCallback, useRef } from "react";

// ── KaTeX LaTeX Renderer ──────────────────────────────────────────────────────
let katexLoaded = false;
let katexLoadPromise = null;

function loadKatex() {
  if (katexLoaded) return Promise.resolve();
  if (katexLoadPromise) return katexLoadPromise;
  katexLoadPromise = new Promise((resolve, reject) => {
    // Load CSS
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css";
    document.head.appendChild(link);
    // Load JS
    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js";
    script.onload = () => { katexLoaded = true; resolve(); };
    script.onerror = reject;
    document.head.appendChild(script);
  });
  return katexLoadPromise;
}

// Splits text into plain and LaTeX segments, renders each appropriately
function LatexRenderer({ text, style = {}, serif = false }) {
  const [ready, setReady] = useState(katexLoaded);
  useEffect(() => { if (!ready) loadKatex().then(() => setReady(true)).catch(() => setReady(true)); }, []);

  if (!ready) return <span style={style}>{text}</span>;

  // Parse $$...$$ (display) and $...$ (inline) segments
  const segments = [];
  const re = /(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) segments.push({ type: "text", content: text.slice(last, m.index) });
    const isDisplay = m[0].startsWith("$$");
    const inner = isDisplay ? m[0].slice(2, -2) : m[0].slice(1, -1);
    segments.push({ type: "latex", display: isDisplay, content: inner });
    last = m.index + m[0].length;
  }
  if (last < text.length) segments.push({ type: "text", content: text.slice(last) });

  return (
    <span style={style}>
      {segments.map((seg, i) => {
        if (seg.type === "text") {
          return (
            <span key={i} style={{ whiteSpace: "pre-wrap", fontFamily: serif ? "'Cormorant Garamond', serif" : "inherit" }}>
              {seg.content}
            </span>
          );
        }
        try {
          const html = window.katex.renderToString(seg.content, {
            displayMode: seg.display,
            throwOnError: false,
            output: "html",
          });
          return (
            <span key={i}
              style={{ display: seg.display ? "block" : "inline", textAlign: seg.display ? "center" : "inherit", margin: seg.display ? "10px 0" : "0 2px" }}
              dangerouslySetInnerHTML={{ __html: html }}
            />
          );
        } catch {
          return <span key={i}>{seg.content}</span>;
        }
      })}
    </span>
  );
}

// ── SM-2 ─────────────────────────────────────────────────────────────────────
function sm2(card, quality) {
  let { ef, interval, repetitions } = card;
  if (quality < 1) { repetitions = 0; interval = 1; }
  else {
    if (repetitions === 0) interval = 1;
    else if (repetitions === 1) interval = 6;
    else interval = Math.round(interval * ef);
    repetitions += 1;
    ef = Math.max(1.3, ef + 0.1 - (3 - quality) * (0.08 + (3 - quality) * 0.02));
  }
  return { ef, interval, repetitions, nextReview: Date.now() + interval * 86400000 };
}

function initCard(card) {
  return { ...card, ef: 2.5, interval: 1, repetitions: 0, nextReview: Date.now() };
}

// ── Storage helpers ───────────────────────────────────────────────────────────
// Adapter: use window.storage (artifact env) or localStorage (local dev)
const storage = window.storage ?? {
  get: async (key) => { const v = localStorage.getItem(key); return v ? { value: v } : null; },
  set: async (key, value) => { localStorage.setItem(key, value); },
};

async function loadDecks() {
  try {
    const r = await storage.get("mlcards:decks");
    return r ? JSON.parse(r.value) : null;
  } catch { return null; }
}
async function saveDecks(decks) {
  try { await storage.set("mlcards:decks", JSON.stringify(decks)); } catch {}
}

// ── Distractor Cache ──────────────────────────────────────────────────────────
// Shape: { [cardId]: string[] }  (3 distractor strings per card)
async function loadDistractorCache() {
  // Load bundled seed distractors (committed to repo, no API call needed)
  let bundled = {};
  try {
    const res = await fetch("/distractors.json");
    if (res.ok) bundled = await res.json();
  } catch {}
  // Merge with user's persisted cache; user entries take priority
  try {
    const r = await storage.get("mlcards:distractors");
    const saved = r ? JSON.parse(r.value) : {};
    return { ...bundled, ...saved };
  } catch { return bundled; }
}
async function saveDistractorCache(cache) {
  try { await storage.set("mlcards:distractors", JSON.stringify(cache)); } catch {}
}
function exportDistractorCache(cache) {
  const json = JSON.stringify(cache, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "mlcards-distractors.json"; a.click();
  URL.revokeObjectURL(url);
}
function importDistractorFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => {
      try { resolve(JSON.parse(e.target.result)); }
      catch { reject(new Error("Invalid JSON")); }
    };
    reader.onerror = () => reject(new Error("Read failed"));
    reader.readAsText(file);
  });
}

// ── Seed data ─────────────────────────────────────────────────────────────────
const SEED_DECKS = [
  {
    id: "deck-1", name: "ML Fundamentals", color: "#0ea5e9", icon: "⬡", created: Date.now(),
    cards: [
      { id: "c1", front: "What is the bias-variance tradeoff?", back: "Bias error comes from wrong assumptions in the learning algorithm (underfitting). Variance error comes from sensitivity to small fluctuations in training data (overfitting). Increasing model complexity reduces bias but increases variance. The goal is to find the sweet spot that minimizes total error." },
      { id: "c2", front: "Explain L1 vs L2 regularization.", back: "L1 (Lasso) adds $|w|$ penalty — produces sparse solutions by driving some weights to exactly zero; useful for feature selection. L2 (Ridge) adds $w^2$ penalty — shrinks weights toward zero but rarely exactly; handles multicollinearity well. The combined Elastic Net objective is:\n$$L = L_0 + \\lambda_1 \\|w\\|_1 + \\lambda_2 \\|w\\|_2^2$$" },
      { id: "c3", front: "What is the kernel trick in SVMs?", back: "The kernel trick implicitly maps data into a high-dimensional feature space by replacing dot products with $K(x, x') = \\phi(x) \\cdot \\phi(x')$. This allows SVMs to find non-linear decision boundaries without explicitly computing $\\phi$. Common kernels: RBF $K(x,x')=\\exp(-\\gamma\\|x-x'\\|^2)$, polynomial, and sigmoid." },
    ].map(initCard)
  },
  {
    id: "deck-2", name: "Deep Learning", color: "#f472b6", icon: "∇", created: Date.now(),
    cards: [
      { id: "c4", front: "What is batch normalization and why does it help?", back: "Batch normalization normalizes layer inputs to have zero mean and unit variance per mini-batch, then applies learnable γ (scale) and β (shift). Benefits: reduces internal covariate shift, allows higher learning rates, acts as slight regularization, reduces sensitivity to initialization." },
      { id: "c5", front: "Explain the attention mechanism in transformers.", back: "Attention computes a weighted sum of values based on query-key compatibility:\n$$\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\nThe $\\sqrt{d_k}$ scaling prevents dot products from growing large in high dimensions. Multi-head attention runs this in parallel across $h$ subspaces, then concatenates and projects results." },
      { id: "c6", front: "Derive $\\partial L / \\partial z_j$ for cross-entropy loss with softmax.", back: "For true label $y$ (one-hot) and softmax output $p_i = e^{z_i}/\\sum_k e^{z_k}$:\n$$\\frac{\\partial L}{\\partial z_j} = p_j - y_j$$\nThis elegant result comes from the chain rule where $\\partial p_i/\\partial z_j = p_i(\\delta_{ij} - p_j)$. The softmax Jacobian and log loss cancel beautifully." },
      { id: "dl-1", front: "What is the vanishing gradient problem and how do residual connections address it?", back: "In deep networks, gradients diminish exponentially during backpropagation: $\\frac{\\partial L}{\\partial x} = \\prod_{i} \\frac{\\partial f_i}{\\partial x_{i-1}}$. When these Jacobians have spectral radius < 1, the product vanishes, making early layers nearly untrainable. Residual connections add a skip path $\\mathbf{y} = F(\\mathbf{x}) + \\mathbf{x}$, ensuring the gradient includes an identity term: $\\frac{\\partial L}{\\partial \\mathbf{x}} = \\frac{\\partial L}{\\partial \\mathbf{y}}\\!\\left(1 + \\frac{\\partial F}{\\partial \\mathbf{x}}\\right)$, which never vanishes." },
      { id: "dl-2", front: "How does dropout work and why does it reduce overfitting?", back: "During training, dropout randomly zeroes each neuron's output with probability $p$, then scales remaining activations by $\\frac{1}{1-p}$ to maintain expected values. This prevents co-adaptation: neurons cannot rely on specific others always being present, forcing each to learn more independent, robust features. At inference all neurons are active and no scaling is needed. Dropout approximates training an ensemble of $2^n$ thinned networks." },
      { id: "dl-3", front: "What are the key differences between LSTMs and GRUs?", back: "LSTMs have three gates (input, forget, output) and a separate cell state $c_t$ carrying long-term memory: $c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t$, $h_t = o_t \\odot \\tanh(c_t)$. GRUs simplify this with two gates (reset $r_t$, update $z_t$) and merge cell and hidden state: $h_t = (1-z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t$. GRUs have fewer parameters, train faster, and perform comparably to LSTMs on most tasks." },
      { id: "dl-4", front: "Why do transformers need positional encodings, and how do sinusoidal encodings work?", back: "Self-attention is permutation-invariant: shuffling the input sequence produces the same attention weights. Positional encodings inject order information by adding a fixed vector to each token embedding. The original sinusoidal encoding uses:\n$$PE_{(pos,2i)} = \\sin\\!\\left(\\frac{pos}{10000^{2i/d}}\\right), \\quad PE_{(pos,2i+1)} = \\cos\\!\\left(\\frac{pos}{10000^{2i/d}}\\right)$$\nDifferent frequencies allow attention to relative and absolute positions; relative positions can be recovered via linear combinations of encodings." },
      { id: "dl-5", front: "What is knowledge distillation and how do soft targets improve learning?", back: "Knowledge distillation trains a small student network to mimic a large teacher's output distribution rather than hard labels. The student minimizes:\n$$L = \\alpha\\,L_{CE}(y,\\sigma(z_s)) + (1-\\alpha)\\,T^2\\,L_{CE}\\!\\left(\\sigma(z_t/T),\\sigma(z_s/T)\\right)$$\nwhere $T$ is temperature. High $T$ softens the teacher's distribution, exposing inter-class similarities that one-hot labels discard. This richer signal provides more gradient information per example, enabling a compact model to approach teacher-level accuracy." },
      { id: "dl-6", front: "What is the difference between generative and discriminative models?", back: "Discriminative models learn $P(y \\mid x)$ directly — the decision boundary separating classes (e.g., logistic regression, SVMs, neural classifiers). Generative models learn the joint $P(x, y) = P(x \\mid y)P(y)$, modeling how data is produced (e.g., naive Bayes, VAEs, GANs). Via Bayes' rule, $P(y \\mid x) = P(x \\mid y)P(y)/P(x)$, so generative models can classify, sample new data, handle missing features, and do density estimation — at the cost of modeling the often high-dimensional $P(x)$." },
      { id: "dl-7", front: "What is the reparameterization trick in Variational Autoencoders?", back: "VAEs encode input $x$ as a distribution $q_{\\phi}(z \\mid x) = \\mathcal{N}(\\mu, \\sigma^2)$ and must backpropagate through the sampling step. Direct sampling $z \\sim \\mathcal{N}(\\mu,\\sigma^2)$ is non-differentiable. The reparameterization trick writes $z = \\mu + \\sigma \\odot \\epsilon$, $\\epsilon \\sim \\mathcal{N}(0,I)$, moving randomness into a fixed distribution. Gradients now flow through $\\mu$ and $\\sigma$ normally, enabling end-to-end training of the ELBO:\n$$\\mathcal{L} = \\mathbb{E}_{q}[\\log p(x|z)] - D_{KL}(q_{\\phi}(z|x) \\| p(z))$$" },
      { id: "dl-8", front: "How does layer normalization differ from batch normalization, and when is each preferred?", back: "Batch norm normalizes across the batch dimension per feature: $\\hat{x}_i = (x_i - \\mu_B)/\\sigma_B$, requiring large batch sizes for stable statistics and using running stats at inference. Layer norm normalizes across the feature dimension per example: $\\hat{x} = (x - \\mu_L)/\\sigma_L$, making it batch-size independent. Layer norm is preferred for transformers and RNNs (variable sequence length, small batches); batch norm excels in CNNs processing fixed-size inputs with large batches." },
      { id: "dl-9", front: "What is gradient clipping and why is it used in training deep networks?", back: "Gradient clipping caps the gradient norm before the optimizer step. Global norm clipping rescales all gradients when their $\\ell_2$ norm exceeds threshold $\\tau$:\n$$g \\leftarrow g \\cdot \\frac{\\min(\\|g\\|,\\,\\tau)}{\\|g\\|}$$\nThis prevents the exploding gradient problem, where large Jacobian products cause updates to overshoot and destabilize training. It is especially important for RNNs and transformers on long sequences. Value clipping (clamping each element independently) is an alternative but can distort gradient direction." },
      { id: "dl-10", front: "Explain transfer learning and common strategies for fine-tuning pretrained models.", back: "Transfer learning reuses a model pretrained on a large source task for a smaller target task. Pretrained weights encode general features (edges, textures, syntax) transferable across domains. Common strategies: (1) **Feature extraction** — freeze all pretrained layers, train only a new head; (2) **Full fine-tuning** — update all weights on the target task; (3) **Gradual unfreezing** — unfreeze layers top-to-bottom progressively; (4) **PEFT** — add small trainable modules (LoRA, adapters) while keeping base weights frozen. Use a lower learning rate ($10^{-4}$ to $10^{-5}$) to preserve pretrained knowledge." },
    ].map(initCard)
  },
  {
    id: "deck-3", name: "Classical ML", color: "#4ade80", icon: "∑", created: Date.now(),
    cards: [
      { id: "cml-1", front: "Explain the Expectation-Maximization (EM) algorithm.", back: "EM is an iterative algorithm for maximum likelihood estimation with latent variables. Given observed data $X$ and latent variables $Z$, it maximizes $\\log P(X;\\theta)$ by alternating: **E-step** — compute posterior $Q(Z) = P(Z \\mid X, \\theta^{\\text{old}})$; **M-step** — maximize $\\mathbb{E}_{Q}[\\log P(X,Z;\\theta)]$ over $\\theta$. Each iteration is guaranteed to non-decrease the log-likelihood because the ELBO is a tight lower bound at the E-step. Classic applications: Gaussian Mixture Models, HMMs, missing-data imputation." },
      { id: "cml-2", front: "How does the random forest algorithm work and what makes it effective?", back: "Random forests build an ensemble of $T$ decision trees, each trained on a bootstrap sample of the data. At each split only a random subset of $m \\approx \\sqrt{p}$ features is considered, decorrelating the trees. Predictions are aggregated by majority vote (classification) or averaging (regression). The ensemble variance is $\\rho\\sigma^2 + \\frac{1-\\rho}{T}\\sigma^2$, where $\\rho$ is pairwise tree correlation. Feature subsampling reduces $\\rho$, so adding more trees reduces variance while bias stays constant." },
      { id: "cml-3", front: "How does gradient boosting differ from bagging, and what does it optimize?", back: "Bagging trains trees independently on bootstrap samples and averages predictions — primarily a variance-reduction technique. Gradient boosting trains trees **sequentially**, each fitting the negative gradient (pseudo-residuals) of the loss w.r.t. the current ensemble:\n$$F_m(x) = F_{m-1}(x) + \\eta\\,h_m(x)$$\nwhere $h_m \\approx -\\partial L/\\partial F_{m-1}$. This is functional gradient descent in prediction space. Boosting primarily reduces bias, can optimize arbitrary differentiable losses, and is more prone to overfitting than bagging." },
      { id: "cml-4", front: "Explain Principal Component Analysis (PCA) and its relationship to the SVD.", back: "PCA finds orthogonal directions of maximum variance. For centered data $X \\in \\mathbb{R}^{n \\times p}$, the covariance is $C = \\frac{1}{n-1}X^TX$. Principal components are the eigenvectors of $C$, equivalently the right singular vectors of $X$: via SVD $X = U\\Sigma V^T$, the PCs are columns of $V$ and projected coordinates are $U\\Sigma$. Selecting the top $k$ components minimizes reconstruction error $\\|X - X_k\\|_F^2 = \\sum_{i>k}\\sigma_i^2$ (Eckart-Young theorem). Kernel PCA generalizes PCA to non-linear manifolds." },
      { id: "cml-5", front: "Describe the k-nearest neighbors algorithm and analyze its computational properties.", back: "KNN classifies a query $x$ by finding its $k$ nearest training points by distance and taking a majority vote. There is no training phase — KNN is a lazy learner storing all data. Prediction cost is $O(np)$ per query for $n$ examples and $p$ features. Decision boundaries can approximate any shape, but performance degrades in high dimensions as distances concentrate (curse of dimensionality). Optimal $k$ balances bias (large $k$, smooth boundary) and variance (small $k$, complex boundary)." },
      { id: "cml-6", front: "Compare Gini impurity and information gain as decision tree splitting criteria.", back: "Both measure node impurity for class distribution $\\{p_k\\}$. **Gini impurity**: $G = 1 - \\sum_k p_k^2$ (probability of misclassifying a random sample). **Information gain**: $\\Delta H = H(\\text{parent}) - \\sum_j w_j H(\\text{child}_j)$ where $H = -\\sum_k p_k \\log p_k$. Gini favors the most frequent class, is faster to compute (no log), and tends to isolate the largest class. Entropy is more sensitive to probability changes near 0 and 1. In practice they produce very similar trees; CART uses Gini, C4.5/ID3 use entropy." },
      { id: "cml-7", front: "What is the curse of dimensionality and how does it affect machine learning?", back: "In high dimensions data becomes exponentially sparse: the volume of an inscribed hypersphere relative to the unit hypercube shrinks to zero as $d \\to \\infty$. Almost all volume lies near the surface, so random points concentrate at similar distances — distance metrics lose discriminative power. Practically: $O(r^{-d})$ samples are needed to maintain density $r$; distance-based methods degrade; irrelevant features add noise. Remedies: dimensionality reduction (PCA), feature selection, regularization, sparse models." },
      { id: "cml-8", front: "Explain the naive Bayes classifier and when its independence assumption is acceptable.", back: "Naive Bayes applies Bayes' rule assuming all features are conditionally independent given the class:\n$$P(y \\mid x_1,\\ldots,x_p) \\propto P(y)\\prod_{j=1}^p P(x_j \\mid y)$$\nThis makes estimation tractable ($O(pc)$ parameters for $c$ classes) and training a single pass. Despite frequent violations in practice (e.g., word co-occurrences), the classifier can still rank classes correctly even with miscalibrated probabilities. Works well for text classification (multinomial NB) and spam filtering." },
      { id: "cml-9", front: "What is k-fold cross-validation and how should it be used for model selection?", back: "K-fold CV partitions data into $k$ equal folds. For each fold $i$, train on the other $k-1$ folds and evaluate on fold $i$; the estimate is $\\hat{R} = \\frac{1}{k}\\sum_{i=1}^k L_i$. This gives a nearly unbiased estimate of generalization error with lower variance than a single split. For model selection: (1) use CV to compare hyperparameters; (2) retrain the selected model on the **full** training set; (3) evaluate on a held-out test set. Conflating model selection and final evaluation leads to optimistic bias." },
      { id: "cml-10", front: "Describe the AdaBoost algorithm and its connection to exponential loss minimization.", back: "AdaBoost maintains a weight distribution over training examples, initially uniform. At each round $t$: (1) train weak learner $h_t$ minimizing weighted error $\\epsilon_t$; (2) compute $\\alpha_t = \\frac{1}{2}\\ln\\frac{1-\\epsilon_t}{\\epsilon_t}$; (3) reweight: $w_i \\leftarrow w_i \\exp(-\\alpha_t y_i h_t(x_i))$, normalized. Final classifier: $H(x) = \\text{sign}\\!\\left(\\sum_t \\alpha_t h_t(x)\\right)$. Friedman et al. showed AdaBoost is coordinate descent on the exponential loss $L = \\sum_i \\exp(-y_i F(x_i))$." },
    ].map(initCard)
  },
  {
    id: "deck-4", name: "Probability & Statistics", color: "#a78bfa", icon: "◈", created: Date.now(),
    cards: [
      { id: "ps-1", front: "State Bayes' theorem and explain its components in the context of machine learning.", back: "Bayes' theorem:\n$$P(\\theta \\mid D) = \\frac{P(D \\mid \\theta)\\,P(\\theta)}{P(D)}$$\n$P(\\theta)$ is the **prior** (belief before data); $P(D \\mid \\theta)$ is the **likelihood**; $P(D) = \\int P(D \\mid \\theta)P(\\theta)\\,d\\theta$ is the marginal likelihood (normalizing constant); $P(\\theta \\mid D)$ is the **posterior**. In ML: the prior encodes regularization, MAP estimation maximizes the posterior, and full Bayesian inference averages predictions over the posterior." },
      { id: "ps-2", front: "State the Central Limit Theorem and explain its relevance to machine learning.", back: "For i.i.d. variables $X_1,\\ldots,X_n$ with mean $\\mu$ and finite variance $\\sigma^2$, the standardized sample mean converges in distribution:\n$$\\frac{\\bar{X}_n - \\mu}{\\sigma/\\sqrt{n}} \\xrightarrow{d} \\mathcal{N}(0,1)$$\nIn ML: justifies Gaussian approximations for parameter estimates, motivates confidence intervals for evaluation metrics, explains why mini-batch gradient estimates have approximately normal noise, and underpins the validity of hypothesis tests comparing model performance." },
      { id: "ps-3", front: "What is maximum likelihood estimation (MLE) and how does it relate to minimizing cross-entropy?", back: "MLE finds parameters maximizing the probability of observed data:\n$$\\hat{\\theta} = \\arg\\max_\\theta \\sum_i \\log P(x_i;\\theta)$$\nFor a classifier with softmax output, $\\log P(y \\mid x;\\theta) = \\sum_k y_k \\log p_k$, so maximizing log-likelihood over the dataset is identical to minimizing the average cross-entropy $-\\frac{1}{n}\\sum_i\\sum_k y_{ik}\\log p_{ik}$. Under model misspecification, MLE converges to the parameter minimizing $D_{KL}(P_{\\text{true}} \\| P_{\\theta})$." },
      { id: "ps-4", front: "Define KL divergence and explain its key properties and asymmetry.", back: "KL divergence measures how much $Q$ differs from reference $P$:\n$$D_{KL}(P \\| Q) = \\mathbb{E}_P\\!\\left[\\log\\frac{P(x)}{Q(x)}\\right] \\geq 0$$\nKey properties: (1) $D_{KL} \\geq 0$ with equality iff $P = Q$ (Gibbs' inequality); (2) **asymmetric**: $D_{KL}(P\\|Q) \\neq D_{KL}(Q\\|P)$ in general; (3) not a metric. Forward KL $D_{KL}(P_{\\text{true}}\\|Q)$ is zero-avoiding (covers all modes); reverse KL $D_{KL}(Q\\|P_{\\text{true}})$ is zero-forcing (mode-seeking). VAEs minimize reverse KL; MLE minimizes forward KL." },
      { id: "ps-5", front: "State the law of total expectation and give an ML application.", back: "The law of iterated expectations states:\n$$\\mathbb{E}[X] = \\mathbb{E}_Y[\\mathbb{E}[X \\mid Y]]$$\nML applications: (1) decomposing generalization error by data partition; (2) deriving the bias-variance decomposition $\\mathbb{E}[(\\hat{f}-f)^2] = \\text{Bias}^2 + \\text{Var}$; (3) showing the Bayes optimal predictor is $f^*(x) = \\mathbb{E}[Y \\mid X=x]$; (4) analyzing EM convergence via the expected complete-data log-likelihood." },
      { id: "ps-6", front: "What is the fundamental difference between frequentist and Bayesian inference?", back: "**Frequentist**: probability is the long-run frequency of events; parameters $\\theta$ are fixed unknowns. Inference produces point estimates (MLE) and confidence intervals: in 95% of repeated experiments the interval contains the true $\\theta$. **Bayesian**: probability quantifies degree of belief; $\\theta$ is a random variable. Inference produces the posterior $P(\\theta \\mid D)$ and credible intervals: $P(\\theta \\in I \\mid D) = 0.95$. Predictions integrate over uncertainty: $P(x^* \\mid D) = \\int P(x^* \\mid \\theta)P(\\theta \\mid D)\\,d\\theta$." },
      { id: "ps-7", front: "What is a sufficient statistic and why is it important?", back: "A statistic $T(X)$ is sufficient for $\\theta$ if the conditional distribution of data given $T$ does not depend on $\\theta$: $P(X \\mid T(X),\\theta) = P(X \\mid T(X))$. By the Fisher-Neyman factorization theorem, $T$ is sufficient iff $P(X;\\theta) = g(T(X),\\theta)\\cdot h(X)$. A sufficient statistic captures all information in the data relevant to estimating $\\theta$. For a Gaussian with known variance, $\\bar{X}$ is sufficient for $\\mu$; for Bernoulli, $\\sum X_i$ is sufficient for $p$." },
      { id: "ps-8", front: "Derive why minimizing cross-entropy loss is equivalent to MLE for classification.", back: "For dataset $\\{(x_i,y_i)\\}$ and model $P(y\\mid x;\\theta)$, MLE maximizes:\n$$\\log\\mathcal{L}(\\theta) = \\sum_i \\log P(y_i \\mid x_i;\\theta)$$\nFor a $C$-class classifier with softmax output $p_k$ and one-hot $y_i$:\n$$\\log P(y_i \\mid x_i;\\theta) = \\sum_k y_{ik}\\log p_k$$\nMaximizing $\\sum_i \\log P(y_i \\mid x_i;\\theta)$ equals minimizing $-\\frac{1}{n}\\sum_i\\sum_k y_{ik}\\log p_{ik}$, the average cross-entropy $H(y,p)$. Cross-entropy training is exactly MLE under the categorical likelihood." },
      { id: "ps-9", front: "What is the Fisher information matrix and what does it quantify?", back: "The Fisher information matrix quantifies how much information data carries about $\\theta$:\n$$\\mathcal{I}(\\theta) = \\mathbb{E}\\left[\\nabla_\\theta \\log P(X;\\theta)\\,\\nabla_\\theta \\log P(X;\\theta)^T\\right] = -\\mathbb{E}\\left[\\nabla^2_\\theta \\log P(X;\\theta)\\right]$$\nThe Cramér-Rao bound states any unbiased estimator satisfies $\\text{Cov}(\\hat{\\theta}) \\geq \\mathcal{I}(\\theta)^{-1}$. In ML: $\\mathcal{I}^{-1}$ approximates the MLE covariance; the natural gradient $\\mathcal{I}^{-1}\\nabla L$ performs steepest descent in distribution space (used in natural gradient descent and TRPO)." },
      { id: "ps-10", front: "Explain the Monte Carlo method for computing expectations and why it scales well.", back: "Monte Carlo approximates intractable expectations by sampling:\n$$\\mathbb{E}_{p(x)}[f(x)] \\approx \\frac{1}{N}\\sum_{i=1}^N f(x_i), \\quad x_i \\sim p$$\nBy the law of large numbers the estimate converges almost surely; by CLT the error is $O(1/\\sqrt{N})$, **independent of dimension** — unlike quadrature rules that scale as $O(N^{-k/d})$. ML applications: Bayesian predictive distributions, VAE training via reparameterization, policy gradient estimation in RL (REINFORCE), and dropout as approximate model averaging." },
    ].map(initCard)
  },
];

// ── AI Generation ─────────────────────────────────────────────────────────────
function anthropicHeaders() {
  const key = typeof import.meta !== "undefined" && import.meta.env?.VITE_ANTHROPIC_API_KEY;
  return {
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
    "anthropic-dangerous-direct-browser-access": "true",
    ...(key ? { "x-api-key": key } : {}),
  };
}
async function generateCards(topic, count, deckName) {
  const prompt = `You are an expert ML educator. Generate exactly ${count} high-quality flashcards about "${topic}" for the deck "${deckName}".

Return ONLY valid JSON in this exact format, no other text:
{
  "cards": [
    {
      "front": "Question text here",
      "back": "Detailed answer here. Use \\n for line breaks. Include equations if relevant."
    }
  ]
}

Requirements:
- Mix of conceptual, mathematical, and practical questions
- Answers should be 2-6 sentences, precise and educational
- Use proper ML terminology
- Use LaTeX for ALL mathematical expressions: inline math with $...$ and display/block equations with $$...$$
- Example inline: "The update rule is $w \\leftarrow w - \\alpha \\nabla L$"
- Example display: "$$\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$"
- Always use LaTeX for Greek letters, fractions, subscripts, superscripts, summations`;

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: anthropicHeaders(),
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      messages: [{ role: "user", content: prompt }]
    })
  });
  const data = await response.json();
  const text = data.content.map(b => b.text || "").join("");
  const clean = text.replace(/```json|```/g, "").trim();
  const parsed = JSON.parse(clean);
  return parsed.cards.map((c, i) => initCard({ id: `ai-${Date.now()}-${i}`, ...c }));
}

// ── Multiple Choice Distractor Generation ────────────────────────────────────
async function generateDistractors(card, allCards) {
  // Use up to 5 other cards' backs as context so distractors are plausible but wrong
  const siblings = allCards
    .filter(c => c.id !== card.id)
    .sort(() => Math.random() - 0.5)
    .slice(0, 5)
    .map(c => c.back);

  const prompt = `You are an expert ML educator creating a multiple choice question.

Question: ${card.front}
Correct answer: ${card.back}

Generate exactly 3 plausible but INCORRECT answer choices for this question.
They must be:
- Realistic enough to challenge someone who partially understands the topic
- Clearly wrong to someone who truly knows the answer
- Similar in length and style to the correct answer
- For math questions, use LaTeX with $...$ and $$...$$ notation just like the correct answer
${siblings.length ? `\nContext (other cards in this deck for distractor inspiration):\n${siblings.map((s,i) => `${i+1}. ${s}`).join('\n')}` : ''}

Return ONLY valid JSON, no other text:
{"distractors": ["wrong answer 1", "wrong answer 2", "wrong answer 3"]}`;

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: anthropicHeaders(),
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      messages: [{ role: "user", content: prompt }]
    })
  });
  const data = await response.json();
  const text = data.content.map(b => b.text || "").join("");
  const clean = text.replace(/```json|```/g, "").trim();
  const parsed = JSON.parse(clean);
  return parsed.distractors;
}

// Shuffle array helper
function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ── Palette ───────────────────────────────────────────────────────────────────
const DECK_COLORS = ["#0ea5e9","#4ade80","#f472b6","#fb923c","#a78bfa","#34d399","#f59e0b","#60a5fa"];
const DECK_ICONS = ["⬡","∇","∑","◈","⊕","⟁","⊗","◉"];
const QUALITY_BTNS = [
  { label: "Blackout", sub: "No recall", q: 0, color: "#ef4444" },
  { label: "Hard", sub: "Barely", q: 1, color: "#f97316" },
  { label: "Good", sub: "With effort", q: 2, color: "#eab308" },
  { label: "Easy", sub: "Instantly", q: 3, color: "#4ade80" },
];

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [decks, setDecks] = useState(null);
  const [view, setView] = useState("home"); // home | deck | review | generate
  const [reviewMode, setReviewMode] = useState("flip"); // flip | mc
  const [activeDeck, setActiveDeck] = useState(null);
  const [queue, setQueue] = useState([]);
  const [qIdx, setQIdx] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [sessionStats, setSessionStats] = useState({ reviewed: 0, easy: 0, hard: 0, correct: 0 });
  const [doneAnim, setDoneAnim] = useState(false);
  // MC state: null | { choices: [{text,correct}], selected: number|null, loading: bool }
  const [mcState, setMcState] = useState(null);
  // Distractor cache: { [cardId]: string[] } — persisted to JSON file
  const [distractorCache, setDistractorCache] = useState({});
  const importFileRef = useRef(null);
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState("");
  const [genTopic, setGenTopic] = useState("");
  const [genCount, setGenCount] = useState(5);
  const [genDeckId, setGenDeckId] = useState("new");
  const [genDeckName, setGenDeckName] = useState("");
  const [showNewDeckForm, setShowNewDeckForm] = useState(false);
  const [newDeckName, setNewDeckName] = useState("");
  const [newDeckColor, setNewDeckColor] = useState(DECK_COLORS[0]);
  const [newDeckIcon, setNewDeckIcon] = useState(DECK_ICONS[0]);
  const [deleteConfirm, setDeleteConfirm] = useState(null);

  // Load persisted decks + distractor cache
  useEffect(() => {
    loadDecks().then(d => setDecks(d || SEED_DECKS));
    loadDistractorCache().then(c => setDistractorCache(c));
  }, []);

  // Persist on change
  useEffect(() => {
    if (decks) saveDecks(decks);
  }, [decks]);

  const updateDecks = useCallback((updater) => setDecks(prev => {
    const next = updater(prev);
    return next;
  }), []);

  function getDeck(id) { return decks?.find(d => d.id === id); }

  function startReview(deckId, mode = "flip") {
    const deck = getDeck(deckId);
    if (!deck) return;
    const due = deck.cards.filter(c => c.nextReview <= Date.now());
    if (!due.length) return;
    const sorted = [...due].sort((a, b) => a.nextReview - b.nextReview);
    setActiveDeck(deckId);
    setQueue(sorted);
    setQIdx(0); setFlipped(false);
    setReviewMode(mode);
    setSessionStats({ reviewed: 0, easy: 0, hard: 0, correct: 0 });
    setMcState(null);
    setView("review");
    if (mode === "mc") loadMcChoices(sorted[0], deck.cards);
  }

  async function loadMcChoices(card, allCards, forceRegenerate = false) {
    // Check cache first (unless forced regen)
    const cached = !forceRegenerate && distractorCache[card.id];
    if (cached) {
      const choices = shuffle([
        { text: card.back, correct: true },
        ...cached.map(d => ({ text: d, correct: false }))
      ]);
      setMcState({ choices, selected: null, loading: false, cardId: card.id });
      return;
    }
    setMcState({ choices: null, selected: null, loading: true, cardId: card.id });
    try {
      const distractors = await generateDistractors(card, allCards);
      // Persist to cache
      const newCache = { ...distractorCache, [card.id]: distractors };
      setDistractorCache(newCache);
      saveDistractorCache(newCache);
      const choices = shuffle([
        { text: card.back, correct: true },
        ...distractors.map(d => ({ text: d, correct: false }))
      ]);
      setMcState({ choices, selected: null, loading: false, cardId: card.id });
    } catch {
      setMcState({ choices: null, selected: null, loading: false, error: true, cardId: card.id });
    }
  }

  function handleMcSelect(idx) {
    setMcState(prev => prev.selected !== null ? prev : { ...prev, selected: idx });
  }

  function handleMcNext(wasCorrect) {
    const card = queue[qIdx];
    // Correct = Easy (3), Wrong = Hard (1)
    const quality = wasCorrect ? 3 : 1;
    const updated = sm2(card, quality);
    updateDecks(prev => prev.map(d =>
      d.id === activeDeck
        ? { ...d, cards: d.cards.map(c => c.id === card.id ? { ...c, ...updated } : c) }
        : d
    ));
    setSessionStats(s => ({
      reviewed: s.reviewed + 1,
      easy: s.easy + (wasCorrect ? 1 : 0),
      hard: s.hard + (wasCorrect ? 0 : 1),
      correct: s.correct + (wasCorrect ? 1 : 0),
    }));
    if (qIdx + 1 >= queue.length) {
      setDoneAnim(true);
      setTimeout(() => { setDoneAnim(false); setView("deck"); }, 1800);
    } else {
      const nextCard = queue[qIdx + 1];
      const deck = getDeck(activeDeck);
      setMcState(null);
      setQIdx(i => i + 1);
      loadMcChoices(nextCard, deck.cards, false);
    }
  }

  function handleRate(quality) {
    const card = queue[qIdx];
    const updated = sm2(card, quality);
    updateDecks(prev => prev.map(d =>
      d.id === activeDeck
        ? { ...d, cards: d.cards.map(c => c.id === card.id ? { ...c, ...updated } : c) }
        : d
    ));
    setSessionStats(s => ({ reviewed: s.reviewed + 1, easy: s.easy + (quality >= 2 ? 1 : 0), hard: s.hard + (quality < 2 ? 1 : 0) }));
    if (qIdx + 1 >= queue.length) {
      setDoneAnim(true);
      setTimeout(() => { setDoneAnim(false); setView("deck"); }, 1800);
    } else {
      setFlipped(false);
      setTimeout(() => setQIdx(i => i + 1), 50);
    }
  }

  async function handleGenerate() {
    if (!genTopic.trim()) return;
    setGenerating(true); setGenError("");
    try {
      const targetDeckName = genDeckId === "new" ? (genDeckName.trim() || genTopic) : getDeck(genDeckId)?.name;
      const newCards = await generateCards(genTopic, genCount, targetDeckName);
      if (genDeckId === "new") {
        const newDeck = {
          id: `deck-${Date.now()}`,
          name: genDeckName.trim() || genTopic,
          color: DECK_COLORS[Math.floor(Math.random() * DECK_COLORS.length)],
          icon: DECK_ICONS[Math.floor(Math.random() * DECK_ICONS.length)],
          created: Date.now(),
          cards: newCards,
        };
        updateDecks(prev => [...prev, newDeck]);
        setActiveDeck(newDeck.id);
        setView("deck");
      } else {
        updateDecks(prev => prev.map(d =>
          d.id === genDeckId ? { ...d, cards: [...d.cards, ...newCards] } : d
        ));
        setView("deck");
      }
      setGenTopic(""); setGenDeckName(""); setGenDeckId("new");
    } catch (e) {
      setGenError("Generation failed. Check your API connection and try again.");
    } finally { setGenerating(false); }
  }

  function createDeck() {
    if (!newDeckName.trim()) return;
    const deck = { id: `deck-${Date.now()}`, name: newDeckName.trim(), color: newDeckColor, icon: newDeckIcon, created: Date.now(), cards: [] };
    updateDecks(prev => [...prev, deck]);
    setNewDeckName(""); setShowNewDeckForm(false);
    setActiveDeck(deck.id); setView("deck");
  }

  function deleteDeck(id) {
    updateDecks(prev => prev.filter(d => d.id !== id));
    setDeleteConfirm(null);
    if (activeDeck === id) { setActiveDeck(null); setView("home"); }
  }

  const current = queue[qIdx];
  const progress = queue.length > 0 ? (qIdx / queue.length) * 100 : 0;
  const activeDeckData = activeDeck ? getDeck(activeDeck) : null;

  if (!decks) return (
    <div style={{ minHeight: "100vh", background: "#060a10", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'DM Mono', monospace", color: "#475569" }}>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 32, marginBottom: 12, animation: "spin 1.5s linear infinite", display: "inline-block" }}>◉</div>
        <div style={{ fontSize: 12, letterSpacing: "0.1em" }}>loading your decks...</div>
      </div>
    </div>
  );

  return (
    <div style={{ minHeight: "100vh", background: "#060a10", fontFamily: "'DM Mono', monospace", color: "#e2e8f0" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 3px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 2px; }
        input, textarea, select { outline: none; }
        .card-scene { perspective: 1400px; cursor: pointer; }
        .card-inner { position: relative; width: 100%; height: 100%; transition: transform 0.6s cubic-bezier(0.4,0,0.2,1); transform-style: preserve-3d; }
        .card-inner.flipped { transform: rotateY(180deg); }
        .card-face { position: absolute; inset: 0; backface-visibility: hidden; -webkit-backface-visibility: hidden; border-radius: 16px; padding: 36px; display: flex; flex-direction: column; }
        .card-back-face { transform: rotateY(180deg); }
        .btn-primary { background: linear-gradient(135deg, #1d4ed8, #0ea5e9); border: none; color: white; padding: 11px 28px; border-radius: 8px; font-family: 'DM Mono', monospace; font-size: 13px; cursor: pointer; transition: all 0.2s; letter-spacing: 0.04em; }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 6px 24px rgba(14,165,233,0.25); }
        .btn-primary:disabled { opacity: 0.35; cursor: not-allowed; transform: none; box-shadow: none; }
        .btn-ghost { background: transparent; border: 1px solid #1e2d4a; color: #64748b; padding: 8px 18px; border-radius: 7px; font-family: 'DM Mono', monospace; font-size: 12px; cursor: pointer; transition: all 0.2s; }
        .btn-ghost:hover { border-color: #2d4a6e; color: #94a3b8; background: #0d1321; }
        .btn-danger { background: transparent; border: 1px solid #ef444430; color: #ef4444; padding: 7px 16px; border-radius: 7px; font-family: 'DM Mono', monospace; font-size: 11px; cursor: pointer; transition: all 0.2s; }
        .btn-danger:hover { background: #ef444415; }
        .input-field { background: #0a1220; border: 1px solid #1e2d4a; border-radius: 8px; padding: 10px 14px; color: #e2e8f0; font-family: 'DM Mono', monospace; font-size: 13px; transition: border-color 0.2s; width: 100%; }
        .input-field:focus { border-color: #2d4a6e; }
        .input-field::placeholder { color: #334155; }
        .rate-btn { border: none; border-radius: 8px; padding: 10px 0; cursor: pointer; font-family: 'DM Mono', monospace; font-size: 12px; font-weight: 500; transition: transform 0.15s, filter 0.15s; }
        .rate-btn:hover { transform: translateY(-2px); filter: brightness(1.2); }
        .rate-btn:active { transform: scale(0.97); }
        .deck-card { background: #0a1220; border: 1px solid #1a2540; border-radius: 14px; padding: 22px; cursor: pointer; transition: all 0.22s; position: relative; overflow: hidden; }
        .deck-card:hover { border-color: #2d4a6e; transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); }
        .grid-bg { position: fixed; inset: 0; pointer-events: none; background-image: linear-gradient(rgba(30,45,74,0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(30,45,74,0.15) 1px, transparent 1px); background-size: 48px 48px; }
        .glow-orb { position: fixed; border-radius: 50%; pointer-events: none; }
        .nav-item { background: transparent; border: none; color: #475569; padding: 7px 16px; border-radius: 6px; font-family: 'DM Mono', monospace; font-size: 11px; cursor: pointer; transition: all 0.2s; letter-spacing: 0.06em; text-transform: uppercase; }
        .nav-item:hover { color: #94a3b8; background: #0d1321; }
        .nav-item.active { color: #0ea5e9; background: #0d1a2e; }
        .done-overlay { position: fixed; inset: 0; background: rgba(6,10,16,0.96); display: flex; align-items: center; justify-content: center; z-index: 200; animation: fadeIn 0.3s; }
        .color-swatch { width: 28px; height: 28px; border-radius: 50%; cursor: pointer; transition: transform 0.15s; border: 2px solid transparent; flex-shrink: 0; }
        .color-swatch:hover { transform: scale(1.15); }
        .color-swatch.selected { border-color: white; transform: scale(1.1); }
        .icon-swatch { width: 34px; height: 34px; border-radius: 8px; background: #0a1220; border: 1px solid #1e2d4a; display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 16px; transition: all 0.15s; }
        .icon-swatch:hover { border-color: #2d4a6e; background: #0d1a2e; }
        .icon-swatch.selected { border-color: #0ea5e9; background: #0a1e38; }
        .mc-choice:not([disabled]):hover { filter: brightness(1.1); transform: translateX(3px); }
        .progress-bar { height: 2px; background: #1e2d4a; border-radius: 1px; overflow: hidden; }
        .progress-fill { height: 100%; border-radius: 1px; transition: width 0.5s ease; }
        .tag { border-radius: 5px; padding: 3px 10px; font-size: 10px; letter-spacing: 0.07em; text-transform: uppercase; }
        .katex { color: inherit; font-size: 1.05em; }
        .katex-display { overflow-x: auto; overflow-y: hidden; padding: 4px 0; }
        .katex-display > .katex { text-align: center; }
        @keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }
        @keyframes popIn { from { transform: scale(0) rotate(-10deg); opacity: 0 } to { transform: scale(1) rotate(0); opacity: 1 } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(16px) } to { opacity: 1; transform: translateY(0) } }
        @keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }
        @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.4 } }
        .slide-up { animation: slideUp 0.35s ease forwards; }
        .spin { animation: spin 1.2s linear infinite; display: inline-block; }
        .select-field { background: #0a1220; border: 1px solid #1e2d4a; border-radius: 8px; padding: 10px 14px; color: #e2e8f0; font-family: 'DM Mono', monospace; font-size: 13px; width: 100%; cursor: pointer; appearance: none; }
        .select-field:focus { border-color: #2d4a6e; outline: none; }
        .breadcrumb { font-size: 11px; color: #334155; letter-spacing: 0.05em; display: flex; align-items: center; gap: 8px; margin-bottom: 28px; }
        .breadcrumb span { cursor: pointer; transition: color 0.2s; } .breadcrumb span:hover { color: #60a5fa; }
        .card-list-item { background: #0a1220; border: 1px solid #1a2540; border-radius: 8px; padding: 12px 16px; display: flex; gap: 12px; align-items: flex-start; }
        .tooltip { position: relative; } .tooltip:hover::after { content: attr(data-tip); position: absolute; bottom: calc(100% + 6px); left: 50%; transform: translateX(-50%); background: #1e2d4a; color: #94a3b8; font-size: 10px; padding: 4px 8px; border-radius: 4px; white-space: nowrap; pointer-events: none; }
      `}</style>

      <div className="grid-bg" />
      <div className="glow-orb" style={{ width: 500, height: 500, background: "radial-gradient(circle, rgba(14,165,233,0.05) 0%, transparent 70%)", top: -200, right: -100 }} />
      <div className="glow-orb" style={{ width: 400, height: 400, background: "radial-gradient(circle, rgba(244,114,182,0.04) 0%, transparent 70%)", bottom: -100, left: -100 }} />

      {/* Done overlay */}
      {doneAnim && (
        <div className="done-overlay">
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 56, animation: "popIn 0.5s cubic-bezier(0.34,1.56,0.64,1)", color: "#4ade80" }}>✓</div>
            <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 32, marginTop: 16 }}>Session Complete</div>
            <div style={{ color: "#475569", fontSize: 12, marginTop: 8 }}>
              {sessionStats.reviewed} reviewed · {sessionStats.easy} easy · {sessionStats.hard} hard
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div style={{ borderBottom: "1px solid #0f1a2e", padding: "14px 28px", display: "flex", alignItems: "center", justifyContent: "space-between", backdropFilter: "blur(16px)", position: "sticky", top: 0, zIndex: 50, background: "rgba(6,10,16,0.9)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }} onClick={() => setView("home")}>
          <div style={{ width: 26, height: 26, background: "linear-gradient(135deg, #1d4ed8, #0ea5e9)", borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13 }}>∇</div>
          <span style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 20, letterSpacing: "-0.01em" }}>ml<span style={{ color: "#0ea5e9", fontStyle: "italic" }}>cards</span></span>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          <button className={`nav-item ${view === "home" ? "active" : ""}`} onClick={() => setView("home")}>decks</button>
          <button className={`nav-item ${view === "generate" ? "active" : ""}`} onClick={() => { setActiveDeck(null); setView("generate"); }}>+ generate</button>
        </div>
      </div>

      <div style={{ maxWidth: 800, margin: "0 auto", padding: "36px 24px", position: "relative" }}>

        {/* ── HOME ── */}
        {view === "home" && (
          <div className="slide-up">
            <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginBottom: 36 }}>
              <div>
                <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 42, lineHeight: 1.1, marginBottom: 8 }}>
                  Your <span style={{ fontStyle: "italic", color: "#0ea5e9" }}>Decks</span>
                </div>
                <div style={{ color: "#334155", fontSize: 12 }}>
                  {decks.length} deck{decks.length !== 1 ? "s" : ""} · {decks.reduce((s, d) => s + d.cards.filter(c => c.nextReview <= Date.now()).length, 0)} cards due
                </div>
              </div>
              <button className="btn-ghost" onClick={() => setShowNewDeckForm(f => !f)}>+ new deck</button>
            </div>

            {/* New deck form */}
            {showNewDeckForm && (
              <div style={{ background: "#0a1220", border: "1px solid #1e2d4a", borderRadius: 12, padding: 24, marginBottom: 28 }}>
                <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 20 }}>create deck</div>
                <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
                  <input className="input-field" placeholder="Deck name..." value={newDeckName} onChange={e => setNewDeckName(e.target.value)} onKeyDown={e => e.key === "Enter" && createDeck()} style={{ flex: 1 }} />
                </div>
                <div style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 10, color: "#334155", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 10 }}>color</div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {DECK_COLORS.map(c => <div key={c} className={`color-swatch ${newDeckColor === c ? "selected" : ""}`} style={{ background: c }} onClick={() => setNewDeckColor(c)} />)}
                  </div>
                </div>
                <div style={{ marginBottom: 20 }}>
                  <div style={{ fontSize: 10, color: "#334155", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 10 }}>icon</div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {DECK_ICONS.map(ic => <div key={ic} className={`icon-swatch ${newDeckIcon === ic ? "selected" : ""}`} onClick={() => setNewDeckIcon(ic)}>{ic}</div>)}
                  </div>
                </div>
                <div style={{ display: "flex", gap: 10 }}>
                  <button className="btn-primary" onClick={createDeck} disabled={!newDeckName.trim()}>create</button>
                  <button className="btn-ghost" onClick={() => setShowNewDeckForm(false)}>cancel</button>
                </div>
              </div>
            )}

            {decks.length === 0 ? (
              <div style={{ textAlign: "center", padding: "60px 0", color: "#334155" }}>
                <div style={{ fontSize: 36, marginBottom: 12 }}>◉</div>
                <div style={{ fontSize: 13 }}>No decks yet. Create one or generate cards with AI.</div>
              </div>
            ) : (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
                {decks.map(deck => {
                  const due = deck.cards.filter(c => c.nextReview <= Date.now()).length;
                  const mastered = deck.cards.filter(c => c.repetitions >= 3 && c.ef > 2.2).length;
                  return (
                    <div key={deck.id} className="deck-card" onClick={() => { setActiveDeck(deck.id); setView("deck"); }}>
                      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, ${deck.color}, transparent)`, borderRadius: "14px 14px 0 0" }} />
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                        <div style={{ width: 40, height: 40, borderRadius: 10, background: `${deck.color}18`, border: `1px solid ${deck.color}30`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20, color: deck.color }}>
                          {deck.icon}
                        </div>
                        {due > 0 && <span className="tag" style={{ background: "#f9731618", color: "#fb923c", border: "1px solid #f9731630" }}>{due} due</span>}
                      </div>
                      <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 20, marginBottom: 6, color: "#e2e8f0" }}>{deck.name}</div>
                      <div style={{ fontSize: 11, color: "#334155", marginBottom: 16 }}>{deck.cards.length} cards · {mastered} mastered</div>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: deck.cards.length ? `${(mastered / deck.cards.length) * 100}%` : "0%", background: deck.color }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* ── DECK VIEW ── */}
        {view === "deck" && activeDeckData && (
          <div className="slide-up">
            <div className="breadcrumb">
              <span onClick={() => setView("home")}>decks</span>
              <span style={{ color: "#1e2d4a" }}>›</span>
              <span style={{ color: "#94a3b8", cursor: "default" }}>{activeDeckData.name}</span>
            </div>

            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 32, gap: 16 }}>
              <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
                <div style={{ width: 52, height: 52, borderRadius: 14, background: `${activeDeckData.color}18`, border: `1px solid ${activeDeckData.color}30`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 26, color: activeDeckData.color, flexShrink: 0 }}>
                  {activeDeckData.icon}
                </div>
                <div>
                  <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 32, lineHeight: 1 }}>{activeDeckData.name}</div>
                  <div style={{ color: "#334155", fontSize: 11, marginTop: 5 }}>
                    {activeDeckData.cards.length} cards · {activeDeckData.cards.filter(c => c.nextReview <= Date.now()).length} due · {activeDeckData.cards.filter(c => c.repetitions >= 3 && c.ef > 2.2).length} mastered
                  </div>
                </div>
              </div>
              <div style={{ display: "flex", gap: 8, flexShrink: 0, flexWrap: "wrap", justifyContent: "flex-end" }}>
                <button className="btn-ghost" style={{ fontSize: 11 }} onClick={() => { setGenDeckId(activeDeckData.id); setView("generate"); }}>+ generate cards</button>
                {(() => {
                  const dueCount = activeDeckData.cards.filter(c => c.nextReview <= Date.now()).length;
                  const disabled = !dueCount;
                  return (<>
                    <button className="btn-ghost" style={{ fontSize: 12 }} onClick={() => startReview(activeDeckData.id, "flip")} disabled={disabled}>
                      flip {dueCount > 0 ? `(${dueCount})` : ""}
                    </button>
                    <button className="btn-primary" style={{ fontSize: 12 }} onClick={() => startReview(activeDeckData.id, "mc")} disabled={disabled}>
                      quiz {dueCount > 0 ? `(${dueCount})` : ""}
                    </button>
                  </>);
                })()}
              </div>
            </div>

            {/* Card list */}
            {activeDeckData.cards.length === 0 ? (
              <div style={{ textAlign: "center", padding: "48px 0", color: "#334155" }}>
                <div style={{ fontSize: 28, marginBottom: 12 }}>◈</div>
                <div style={{ fontSize: 13, marginBottom: 20 }}>This deck is empty.</div>
                <button className="btn-primary" onClick={() => { setGenDeckId(activeDeckData.id); setView("generate"); }}>generate cards with AI</button>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {activeDeckData.cards.map((card, i) => {
                  const due = card.nextReview <= Date.now();
                  const mastered = card.repetitions >= 3 && card.ef > 2.2;
                  const daysUntil = Math.ceil((card.nextReview - Date.now()) / 86400000);
                  return (
                    <div key={card.id} className="card-list-item">
                      <div style={{ width: 8, height: 8, borderRadius: "50%", background: mastered ? "#4ade80" : due ? "#f97316" : "#1e2d4a", marginTop: 5, flexShrink: 0 }} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{card.front}</div>
                        <div style={{ fontSize: 10, color: "#334155", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{card.back}</div>
                      </div>
                      <div style={{ flexShrink: 0, textAlign: "right", display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4 }}>
                        <div style={{ fontSize: 10, color: due ? "#f97316" : "#334155" }}>{due ? "due" : `in ${daysUntil}d`}</div>
                        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                          {distractorCache[card.id] && (
                            <span title="distractors cached" style={{ fontSize: 9, color: "#4ade8060", letterSpacing: "0.05em" }}>◆ mc</span>
                          )}
                          <span style={{ fontSize: 10, color: "#1e2d4a" }}>ef {card.ef.toFixed(1)}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            <div style={{ marginTop: 32, paddingTop: 24, borderTop: "1px solid #0f1a2e" }}>
              {/* Distractor cache management */}
              {(() => {
                const deckCards = activeDeckData.cards;
                const cachedCount = deckCards.filter(c => distractorCache[c.id]).length;
                const totalCount = deckCards.length;
                return totalCount > 0 ? (
                  <div style={{ marginBottom: 20 }}>
                    <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12 }}>quiz distractors</div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                      <span style={{ fontSize: 11, color: cachedCount === totalCount ? "#4ade80" : "#f97316" }}>
                        {cachedCount}/{totalCount} cached
                      </span>
                      <button className="btn-ghost" style={{ fontSize: 11 }}
                        onClick={() => exportDistractorCache(distractorCache)}>
                        ↓ export distractors.json
                      </button>
                      <button className="btn-ghost" style={{ fontSize: 11 }}
                        onClick={() => importFileRef.current?.click()}>
                        ↑ import distractors.json
                      </button>
                      <input ref={importFileRef} type="file" accept=".json" style={{ display: "none" }}
                        onChange={async e => {
                          const file = e.target.files?.[0];
                          if (!file) return;
                          try {
                            const imported = await importDistractorFile(file);
                            const merged = { ...distractorCache, ...imported };
                            setDistractorCache(merged);
                            saveDistractorCache(merged);
                          } catch { alert("Failed to import: invalid JSON file"); }
                          e.target.value = "";
                        }} />
                    </div>
                  </div>
                ) : null;
              })()}

              {deleteConfirm === activeDeckData.id ? (
                <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                  <span style={{ fontSize: 12, color: "#ef4444" }}>Delete this deck?</span>
                  <button className="btn-danger" onClick={() => deleteDeck(activeDeckData.id)}>yes, delete</button>
                  <button className="btn-ghost" style={{ fontSize: 11 }} onClick={() => setDeleteConfirm(null)}>cancel</button>
                </div>
              ) : (
                <button className="btn-danger" onClick={() => setDeleteConfirm(activeDeckData.id)}>delete deck</button>
              )}
            </div>
          </div>
        )}

        {/* ── GENERATE ── */}
        {view === "generate" && (
          <div className="slide-up">
            <div className="breadcrumb">
              <span onClick={() => setView("home")}>decks</span>
              <span style={{ color: "#1e2d4a" }}>›</span>
              <span style={{ color: "#94a3b8", cursor: "default" }}>generate cards</span>
            </div>

            <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 38, marginBottom: 8 }}>
              AI <span style={{ fontStyle: "italic", color: "#0ea5e9" }}>Card Generator</span>
            </div>
            <div style={{ color: "#334155", fontSize: 12, marginBottom: 36 }}>Describe a topic and Claude will generate high-quality flashcards.</div>

            <div style={{ background: "#0a1220", border: "1px solid #1e2d4a", borderRadius: 14, padding: 28 }}>
              <div style={{ marginBottom: 20 }}>
                <label style={{ fontSize: 10, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>topic</label>
                <input className="input-field" placeholder="e.g. Transformer architecture, Bayesian inference, RLHF..." value={genTopic} onChange={e => setGenTopic(e.target.value)} onKeyDown={e => e.key === "Enter" && !generating && handleGenerate()} />
                <div style={{ fontSize: 10, color: "#1e2d4a", marginTop: 6 }}>Be specific for better cards — "attention mechanism in transformers" beats "deep learning"</div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
                <div>
                  <label style={{ fontSize: 10, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>number of cards</label>
                  <div style={{ display: "flex", gap: 6 }}>
                    {[3, 5, 8, 10].map(n => (
                      <button key={n} onClick={() => setGenCount(n)} style={{ flex: 1, padding: "8px 0", borderRadius: 6, background: genCount === n ? "#0d1a2e" : "#060a10", border: `1px solid ${genCount === n ? "#2d4a6e" : "#1a2540"}`, color: genCount === n ? "#60a5fa" : "#475569", cursor: "pointer", fontFamily: "'DM Mono', monospace", fontSize: 12, transition: "all 0.15s" }}>{n}</button>
                    ))}
                  </div>
                </div>

                <div>
                  <label style={{ fontSize: 10, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>add to deck</label>
                  <select className="select-field" value={genDeckId} onChange={e => setGenDeckId(e.target.value)}>
                    <option value="new">+ create new deck</option>
                    {decks.map(d => <option key={d.id} value={d.id}>{d.icon} {d.name}</option>)}
                  </select>
                </div>
              </div>

              {genDeckId === "new" && (
                <div style={{ marginBottom: 20 }}>
                  <label style={{ fontSize: 10, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>new deck name <span style={{ color: "#334155" }}>(optional, defaults to topic)</span></label>
                  <input className="input-field" placeholder={genTopic || "Deck name..."} value={genDeckName} onChange={e => setGenDeckName(e.target.value)} />
                </div>
              )}

              {genError && <div style={{ fontSize: 12, color: "#ef4444", background: "#ef444410", border: "1px solid #ef444430", borderRadius: 6, padding: "10px 14px", marginBottom: 16 }}>{genError}</div>}

              <button className="btn-primary" onClick={handleGenerate} disabled={!genTopic.trim() || generating} style={{ width: "100%", padding: "13px", fontSize: 13 }}>
                {generating ? <><span className="spin" style={{ marginRight: 8 }}>◉</span>generating {genCount} cards...</> : `generate ${genCount} cards →`}
              </button>
            </div>

            {/* Examples */}
            <div style={{ marginTop: 28 }}>
              <div style={{ fontSize: 10, color: "#334155", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 14 }}>example topics</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {["Gradient descent variants", "Convolutional neural networks", "Bayesian inference", "RLHF & alignment", "Graph neural networks", "Diffusion models", "Mixture of Experts", "Contrastive learning"].map(t => (
                  <button key={t} onClick={() => setGenTopic(t)} style={{ background: "#0a1220", border: "1px solid #1a2540", borderRadius: 20, padding: "6px 14px", color: "#475569", fontSize: 11, cursor: "pointer", fontFamily: "'DM Mono', monospace", transition: "all 0.15s" }}
                    onMouseOver={e => { e.target.style.borderColor = "#2d4a6e"; e.target.style.color = "#94a3b8"; }}
                    onMouseOut={e => { e.target.style.borderColor = "#1a2540"; e.target.style.color = "#475569"; }}>
                    {t}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── REVIEW ── */}
        {view === "review" && current && activeDeckData && (
          <div className="slide-up">
            <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 28 }}>
              <div style={{ flex: 1 }}>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${progress}%`, background: activeDeckData.color }} />
                </div>
              </div>
              <div style={{ fontSize: 11, color: "#334155", flexShrink: 0 }}>{qIdx + 1} / {queue.length}</div>
              <button style={{ background: "transparent", border: "none", color: "#334155", cursor: "pointer", fontSize: 14, padding: "2px 6px", transition: "color 0.2s" }} onClick={() => setView("deck")} onMouseOver={e => e.target.style.color="#94a3b8"} onMouseOut={e => e.target.style.color="#334155"}>✕</button>
            </div>

            <div style={{ marginBottom: 18, display: "flex", alignItems: "center", gap: 8 }}>
              <span className="tag" style={{ background: `${activeDeckData.color}15`, color: activeDeckData.color, border: `1px solid ${activeDeckData.color}25` }}>
                {activeDeckData.icon} {activeDeckData.name}
              </span>
              <span className="tag" style={{ background: reviewMode === "mc" ? "#a78bfa18" : "#0ea5e918", color: reviewMode === "mc" ? "#a78bfa" : "#0ea5e9", border: reviewMode === "mc" ? "1px solid #a78bfa30" : "1px solid #0ea5e930" }}>
                {reviewMode === "mc" ? "quiz" : "flip"}
              </span>
            </div>

            {reviewMode === "flip" ? (
              <>
                {/* Flip card */}
                <div className="card-scene" style={{ height: 300, marginBottom: 20 }} onClick={() => setFlipped(f => !f)}>
                  <div className={`card-inner ${flipped ? "flipped" : ""}`}>
                    <div className="card-face" style={{ background: "#0a1220", border: "1px solid #1a2540", justifyContent: "space-between" }}>
                      <div style={{ fontSize: 10, color: "#1e2d4a", letterSpacing: "0.12em", textTransform: "uppercase" }}>question</div>
                      <div style={{ fontSize: 22, lineHeight: 1.6, color: "#e2e8f0", flex: 1, display: "flex", alignItems: "center", padding: "16px 0" }}>
                        <LatexRenderer text={current.front} serif={true} />
                      </div>
                      <div style={{ fontSize: 10, color: "#1e2d4a", letterSpacing: "0.08em" }}>tap to reveal ↗</div>
                    </div>
                    <div className="card-face card-back-face" style={{ background: "#080f1e", border: `1px solid ${activeDeckData.color}30`, justifyContent: "space-between" }}>
                      <div style={{ fontSize: 10, color: activeDeckData.color, letterSpacing: "0.12em", textTransform: "uppercase", opacity: 0.7 }}>answer</div>
                      <div style={{ fontSize: 13, lineHeight: 1.9, color: "#b0bec5", flex: 1, overflowY: "auto", padding: "14px 0" }}>
                        <LatexRenderer text={current.back} />
                      </div>
                      <div style={{ fontSize: 10, color: "#1e2d4a" }}>interval {current.interval}d · ef {current.ef.toFixed(2)} · rep {current.repetitions}</div>
                    </div>
                  </div>
                </div>
                {flipped ? (
                  <div>
                    <div style={{ fontSize: 10, color: "#334155", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12, textAlign: "center" }}>how well did you recall?</div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8 }}>
                      {QUALITY_BTNS.map(btn => (
                        <button key={btn.q} className="rate-btn" onClick={() => handleRate(btn.q)} style={{ background: `${btn.color}15`, border: `1px solid ${btn.color}35`, color: btn.color }}>
                          <div style={{ fontWeight: 500 }}>{btn.label}</div>
                          <div style={{ fontSize: 10, opacity: 0.6, marginTop: 3 }}>{btn.sub}</div>
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div style={{ textAlign: "center", color: "#1e2d4a", fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase", padding: "16px 0" }}>
                    recall your answer, then tap the card
                  </div>
                )}
              </>
            ) : (
              /* ── Multiple Choice Mode ── */
              <div>
                {/* Question */}
                <div style={{ background: "#0a1220", border: "1px solid #1a2540", borderRadius: 14, padding: "28px 32px", marginBottom: 20, minHeight: 120, display: "flex", alignItems: "center" }}>
                  <div style={{ fontSize: 20, lineHeight: 1.6, color: "#e2e8f0", width: "100%" }}>
                    <LatexRenderer text={current.front} serif={true} />
                  </div>
                </div>

                {/* Choices */}
                {mcState?.loading ? (
                  <div style={{ textAlign: "center", padding: "32px 0", color: "#334155" }}>
                    <span className="spin" style={{ fontSize: 20, marginRight: 10 }}>◉</span>
                    <span style={{ fontSize: 12, letterSpacing: "0.08em" }}>generating choices...</span>
                  </div>
                ) : mcState?.error ? (
                  <div style={{ textAlign: "center", padding: "24px", color: "#ef4444", fontSize: 12 }}>
                    Failed to generate choices.
                    <button className="btn-ghost" style={{ marginLeft: 12, fontSize: 11 }} onClick={() => loadMcChoices(current, getDeck(activeDeck).cards, true)}>retry</button>
                  </div>
                ) : mcState?.choices ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {mcState.choices.map((choice, idx) => {
                      const isSelected = mcState.selected === idx;
                      const isRevealed = mcState.selected !== null;
                      const isCorrect = choice.correct;
                      let bg = "#0a1220", border = "#1a2540", color = "#94a3b8";
                      if (isRevealed && isCorrect) { bg = "#4ade8015"; border = "#4ade8050"; color = "#4ade80"; }
                      else if (isRevealed && isSelected && !isCorrect) { bg = "#ef444415"; border = "#ef444450"; color = "#ef4444"; }
                      return (
                        <button key={idx} onClick={() => handleMcSelect(idx)}
                          style={{ background: bg, border: `1px solid ${border}`, borderRadius: 10, padding: "16px 20px", textAlign: "left", cursor: isRevealed ? "default" : "pointer", transition: "all 0.2s", fontFamily: "'DM Mono', monospace", color, width: "100%" }}>
                          <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                            <span style={{ width: 22, height: 22, borderRadius: "50%", border: `1px solid ${border}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, flexShrink: 0, marginTop: 2, color }}>
                              {isRevealed && isCorrect ? "✓" : isRevealed && isSelected ? "✗" : String.fromCharCode(65 + idx)}
                            </span>
                            <div style={{ fontSize: 13, lineHeight: 1.7 }}>
                              <LatexRenderer text={choice.text} />
                            </div>
                          </div>
                        </button>
                      );
                    })}
                    {mcState.selected !== null && (
                      <div style={{ marginTop: 8, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span style={{ fontSize: 12, color: mcState.choices[mcState.selected].correct ? "#4ade80" : "#ef4444" }}>
                          {mcState.choices[mcState.selected].correct ? "✓ Correct" : "✗ Incorrect"}
                          <span style={{ color: "#334155", marginLeft: 8 }}>
                            {sessionStats.reviewed + 1 > 0 ? `${sessionStats.correct + (mcState.choices[mcState.selected].correct ? 1 : 0)}/${sessionStats.reviewed + 1} this session` : ""}
                          </span>
                        </span>
                        <div style={{ display: "flex", gap: 8 }}>
                          <button className="btn-ghost" style={{ fontSize: 11 }}
                            onClick={() => loadMcChoices(current, getDeck(activeDeck).cards, true)}
                            title="Regenerate distractors for this card">
                            ↺ regen
                          </button>
                          <button className="btn-primary" style={{ fontSize: 12, padding: "8px 22px" }}
                            onClick={() => handleMcNext(mcState.choices[mcState.selected].correct)}>
                            {qIdx + 1 >= queue.length ? "finish" : "next →"}
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                ) : null}
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
