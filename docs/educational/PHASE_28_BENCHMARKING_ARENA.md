# Phase 28: Continuous Benchmarking & Automated Model Arena (Elo & Win Rate)

## 1. WHAT
Phase 28 delivers a first-principles **Continuous Benchmarking & Automated Model Arena** for Project Libra:
- **Bradley-Terry Elo Rating Engine**: Calculates dynamic pairwise Elo ratings, 95% confidence intervals, win rates, and category rankings (Coding, Reasoning, Factual, Instruction).
- **Automated Referee with Position-Bias Mitigation**: Dual-pass evaluation mechanism evaluating `(Candidate A, Candidate B)` and swapped `(Candidate B, Candidate A)`, canceling out the systematic LLM bias toward the first completion.
- **Automated Round-Robin Tournament Runner**: Evaluates model pairs across curated benchmark categories without requiring manual human oversight.
- **Interactive Arena UI & REST APIs**: Web UI providing side-by-side metric comparison, human voting, automated referee judging, and live Elo leaderboards.

---

## 2. WHY
In LLM evaluation, standard perplexity or single-prompt benchmarking often fails to capture comparative nuances or suffers from severe bias:
1. **Subjectivity & Incommensurability**: Single-point score cards fail to establish clear relative skill between checkpoints. Pairwise head-to-head comparisons (as popularized by LMSYS Chatbot Arena) provide a globally consistent ordering.
2. **Position Bias in LLM Evaluators**: When an LLM judge evaluates two completions, it systematically favors Candidate 1 due to primacy effects. Without position swapping, automated referee ratings are skewed.
3. **Statistical Uncertainty**: Small sample sizes make raw win rates misleading; tracking 95% confidence intervals ensures statistical rigor before declaring an architecture superior.

---

## 3. HOW (The Math & Architecture)

### Bradley-Terry Logistic Formulation
The probability that Model $A$ defeats Model $B$ given their respective ratings $R_A$ and $R_B$ is modeled as:
$$E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$
$$E_B = 1 - E_A$$

When ratings are equal ($R_A = R_B$), $E_A = 0.5$. A 400-point difference implies a 10:1 expected odds ratio ($E_A \approx 0.909$).

### Rating Update & Conservation
Given match outcome $S_A \in \{1.0, 0.5, 0.0\}$ and $S_B = 1.0 - S_A$:
$$\Delta R_A = K \cdot (S_A - E_A)$$
$$\Delta R_B = K \cdot (S_B - E_B) = - \Delta R_A$$

Rating conservation is strictly guaranteed: $\Delta R_A + \Delta R_B = 0$.

### Dynamic K-Factor & Asymptotic Confidence Intervals
- **Dynamic $K$**: For models with fewer than 10 matches, $K = 48.0$ accelerates rating convergence; thereafter, $K = 32.0$ ensures stability.
- **Confidence Interval**: The standard error under Bradley-Terry asymptotic variance is estimated as:
  $$\text{SE} \approx \frac{400}{\sqrt{N}}$$
  $$\text{CI}_{95\%} = \left[ R - 1.96 \cdot \text{SE}, \; R + 1.96 \cdot \text{SE} \right]$$

### Dual-Pass Position-Bias Mitigation
Let $f(P, X, Y)$ denote the judge's scoring function on prompt $P$ with candidates in order $(X, Y)$:
- **Pass 1**: $s_1 = f(P, A, B) \rightarrow (s_{1,A}, s_{1,B})$
- **Pass 2**: $s_2 = f(P, B, A) \rightarrow (s_{2,B}, s_{2,A})$
- **Final Reconciled Score**:
  $$S_A = \frac{s_{1,A} + s_{2,A}}{2}, \quad S_B = \frac{s_{1,B} + s_{2,B}}{2}$$
Any systematic positional additive term $\epsilon_{\text{pos}}$ satisfies:
$$S_A = \frac{(v_A + \epsilon) + v_A}{2} = v_A + \frac{\epsilon}{2}, \quad S_B = \frac{v_B + (v_B + \epsilon)}{2} = v_B + \frac{\epsilon}{2}$$
$$S_A - S_B = v_A - v_B$$
The position bias cancels out entirely from the competitive margin.

---

## 4. TEST
- **Unit Tests (`tests/evaluation/test_elo.py`)**:
  - Validated Bradley-Terry symmetry ($E_A + E_B = 1.0$).
  - Validated rating conservation ($\Delta_A + \Delta_B = 0$).
  - Verified 95% confidence interval narrowing with sample size ($N=1$ vs $N=10$ vs $N=100$).
  - Tested dual-pass automated referee scoring and tie detection.
  - Verified persistence and reload of `EloLeaderboard`.
- **API Tests (`tests/api/test_arena_elo.py`)**:
  - `GET /api/v1/arena/leaderboard`: returns structured rankings and confidence bounds.
  - `POST /api/v1/arena/vote`: human voting updates ratings correctly.
  - `POST /api/v1/arena/battle`: automated referee head-to-head match execution.
  - `POST /api/v1/arena/tournament`: automated multi-domain round-robin execution.
- **Frontend Build**: Verified zero-error compilation with Next.js App Router and Tailwind CSS.

---

## 5. NEXT
Phase 29: Speculative Decoding & Draft Model Verification Engine (first-principles draft-verify accept/reject algorithm for CPU acceleration).
