# Regime-Aware-Stock-Trading-using-Reinforcement-Learning 

## 1.G-Learning for Stock Trading with Curriculum and KL-Regularization
- This project implements a state-of-the-art G-learning reinforcement learning algorithm with an advanced multi-phase curriculum training strategy designed for robust and risk-aware stock trading. The agent progressively learns from short-term tactical trading to long-term strategic decision-making, using risk-adjusted rewards, policy regularization, and adaptive exploration control.
- The notebook demonstrates an iterative design and learning process
### Algorithm Summary Table

 | **Algorithm**   | **Variant Name**                          | **Description**                                                                                                                                                                                                                                      | **Key Outcome**                                                                                                                                                                     |
|-----------------|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Algorithm 1     | *Vanilla G‑Learning*                       | Initial G-Learning setup using fixed 21-step episodes to simulate one month of trading. Agent uses softmax-based policy derived from G-values. KL-divergence penalty applied to keep the policy near a safe uniform prior. No curriculum, no risk adjustment. | Poor performance due to lack of long-term context and unstable reward signals. Policy failed to capture delayed returns.                                                            |
| Algorithm 2     | *Greedy Testing Variant*                   | Same structure as Algorithm 1, but with greedy evaluation after training. Training remained stochastic (softmax), while testing used `argmax`. Intended to measure policy sharpness, but showed minimal improvement due to shallow learned strategies.     | Evaluation highlighted overfitting to noise and lack of generalization. Greedy testing didn't reveal any strategic shift in behavior.                                               |
| Algorithm 3     | *Circular Curriculum G‑Learning*           | Introduced multi-phase curriculum: 21-step training for fast feedback in early episodes (0–199), followed by 60-step training for longer-term strategy (200+). Used KL annealing to gradually reduce regularization pressure, enabling confident policy divergence from the prior. | Major breakthrough in learning quality. Early phase encouraged short-term reactivity, while late phase built strategic foresight. KL-annealing allowed the agent to gradually specialize. |
| Algorithm 4     | *Sharpe-Scaled + Risk-Aware G‑Learning*    | Refined reward design using Sharpe-like scaling (`σ × √T`) for volatility-adjusted returns. Replaced raw profit with log-returns over phase baseline. Integrated persistence bonuses for consistent performance, and capped volatility to reduce noise. KL term used adaptively. | Final version achieved smooth, high-confidence trading strategies. Balanced profitability with risk, showing robustness across different trading regimes.                          |
| Algorithm 5     | *Structured Curriculum with 4 Phases*      | Expanded the curriculum to four progressive phases, each increasing in difficulty and planning horizon: window sizes `[21, 45, 60, 90]`, baseline targets `[0.98, 1.02, 1.05, 1.10]`, volatility caps `[0.2, 0.25, 0.3, 0.4]`. Sharpe-like scaling applied from Phase 2 onward. | Enabled gradual adaptation to complex trading regimes. Agent showed strong reward smoothing, risk sensitivity, and phase-wise generalization. Progressively optimized for both short-term momentum and long-term mean-reversion. |
| Algorithm 6     | *Dynamic Curriculum Loop + Adaptive KL*    | Fully modular training loop with automated curriculum transitions via list-based phase control. KL penalty scaled per phase using a custom annealing function, allowing precise control over exploration pressure. Each phase acts as a stage in an RL syllabus, building cumulative capability. | Final agent demonstrates stateful, risk-adjusted decision making across multiple regimes. Learns to handle rising profit targets and volatility while adjusting confidence through adaptive regularization. Achieved consistent reward growth across curriculum. |

 ###  Key Improvements & Their Benefits 
 KL Penalty Annealing
Smoothly reduces the KL-divergence penalty over episodes, allowing the agent to begin with safe, exploratory behavior and gradually shift toward confident, optimized policies. Improves learning stability and adaptivity over time.

Greedy Policy in Final Episodes
After episode 800, action selection becomes greedy (argmax) instead of softmax sampling, shifting the focus toward exploitation and policy certainty—critical for deployment scenarios.

Reward Smoothing + Floor + Sharpe Scaling
Logarithmic returns and Sharpe-like scaling (σ × √T) normalize rewards and reduce noise from volatile environments. A minimum reward floor prevents destabilizing penalties during early training.

Final Profit Bonus
Grants a small bonus for ending episodes without losses. Encourages breakeven preservation and discourages overly risky strategies.

Greedy Evaluation Phase Every 50 Episodes
Every 50 episodes, the agent is evaluated under a deterministic (greedy) policy. This separates training noise from actual strategy quality and tracks real-world performance readiness.

Improved Logging & Diagnostics
Diagnostic printouts highlight key metrics like log returns, volatility, and reward shaping terms. Logging is focused, clear, and useful for debugging and tracking learning dynamics.

Safe Softmax with Clipping
Clamps extreme values during softmax computation to prevent overflow or NaNs. Ensures stable action probabilities even with diverging G-values.

Decaying Temperature (Tau)
The softmax temperature τ decays over time, starting with high exploration and shifting toward deterministic decision-making—mimicking natural learning progression.

Enriched State Encoding
Includes technical indicators like momentum, MACD, RSI, and Bollinger Band width. These features help the agent detect trends and adapt actions accordingly.

Adaptive Learning Rate (alpha_t)
Increases the learning rate temporarily during poor performance to accelerate recovery from bad strategies.

Bonus Rewards for Surpassing Local Highs
Provides additional rewards when new equity peaks are achieved. Encourages upward performance trends and discourages stagnation.



## 2. `Research Paper Implementation of CERL.ipynb
### Certainty Equivalent Reinforcement Learning (CERL) – *In Progress*
- This notebook explores and implements key ideas from the J.P. Morgan AI Research paper:
> **“Idiosyncrasies and Challenges of Data-Driven Learning in Electronic Trading”**  
> *V. Bacoyannis, V. Glukhov, T. Jin, J. Kochems, D. Song (2018)*  
> [arXiv:1811.09549](https://arxiv.org/abs/1811.09549)

---
- This notebook is an ongoing research implementation focused on

1. Modeling Certainty Equivalent Reinforcement Learning (CERL)

2. Building a risk-aware reward structure (inspired the the financial exponential utility functions proposed in the paper

3. also draws inspiration from other J.P. Morgan AI research and follow-up papers in the same series.

