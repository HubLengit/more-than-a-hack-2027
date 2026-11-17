# Technical Report: Controlled Touch & Kinematic Inertia Possession Modeling

**Competition:** More than a Hack 2027 — Phase 1: Ball Possession  
**Organizers:** FC Barcelona, Metrica Sports, Mobile World Capital Barcelona  
**Match:** FC Barcelona vs Athletic Club (LaLiga EA Sports 2024-25, Matchday 2)  
**Deliverable:** Technical Document (Deliverable #3)

---

## 1. Executive Summary & Core Results

The challenge requires predicting the frame-by-frame possession label (`H`, `A`, `D`, `X`) across 199,319 video frames (25 fps) of a full professional football match. The official ranking metric is **`spellF1`**, which evaluates temporal Intersection over Union (IoU $\in [0.3, 0.7]$) over contiguous possession spells, with **`macroF1`** as the primary tie-breaker.

The official reference floor (`baseline_smooth50`) scores **`spellF1 = 0.2433`** on the full match (**0.3256** on the public 5-minute window). Naive methods based on the nearest player to the ball suffer catastrophic temporal fragmentation (`spellF1 = 0.0378`, 131 predicted spells vs 17 ground truth) due to passing flight physics and missing ball frames (41% missing during graded play).

Our proposed system, **Controlled Touch & Kinematic Inertia (v3)**, addresses the underlying physics and tactical structure of football:
- **`spellF1`:** **0.5212** (+19.6 percentage points over the baseline_smooth50 public score).
- **`macroF1`:** **0.8222** (+20.15 percentage points over the baseline_smooth50 public score).
- **Possession Spells Count:** **16** predicted vs **17** ground truth.
- **Mean Spell Duration:** **15.5 s** vs **14.2 s** ground truth.
- **Temporal Fragmentation Ratio:** **0.95** (baseline was 1.43, raw was 6.52).
- **Dead-Ball Discrimination (`alive_dead_f1`):** **0.9542** (D recall increased from 21% to 80%).

```
Official Leaderboard Comparison (Public Ground Truth):
rank  submission                macroF1  spellF1   poss     dur  transF1  lat_f  shareE  frag
----  ------------------------  -------  -------  -----  ------  -------  -----  ------  ----
   1  controlled_inertia_v3      0.8222   0.5212     16    15.5   0.2051   -2.2   0.022  0.95
   2  baseline_smooth125         0.5812   0.4242     16    18.5   0.1667   -7.7   0.052  0.81
   3  baseline_smooth50          0.6207   0.3256     26    10.8   0.0816   +4.0   0.048  1.43 (Official Baseline)
   4  baseline_smooth25          0.6200   0.2868     36     7.6   0.2034   +8.2   0.056  1.90
   5  baseline_raw               0.5972   0.0378    131     2.0   0.1410   -3.1   0.046  6.52
```

---

## 2. Approach: Why Naive Models Fail & What We Built

### 2.1 The Nearest-Player Fallacy & Passing Physics
In a naive model, whoever is closest to the ball is assigned possession. We empirically measured the error rate of this heuristic on the match data:
- When ball speed $v_{ball} < 2\text{ m/s}$, the nearest player matches true possession in **91.1%** of frames.
- When $v_{ball} \in [10, 20]\text{ m/s}$ (passes, clearances), the nearest player belongs to the *opposing* team in **33.6%** of frames.
- When $v_{ball} > 20\text{ m/s}$, the nearest player belongs to the *opposing* team in **61.2%** of frames!

In professional football, long passes fly over or past defenders. A defending player being momentarily 1.5 meters from a ball traveling at 60 km/h does not possess the ball. The baseline switches labels during flight, fragmenting single possessions into multiple micro-spells.

### 2.2 Controlled Touch Inertia Principle
Possession is governed by **control**, not proximity:
1. **Inertia:** When Team $T$ has the ball, possession remains with Team $T$ during passes, aerial balls, rebounds, and occlusions.
2. **Turnover Condition:** Possession only passes to the opponent Team $O$ when an opponent establishes sustained controlled contact:
   $$d_O < d_{control} \quad \text{and} \quad d_O < d_T - \Delta_{margin} \quad \text{and} \quad v_{ball} < v_{control\_max}$$
   sustained continuously for $K$ consecutive frames ($K = 10\text{ frames} = 0.4\text{ s}$).

### 2.3 Collective Kinematic Dead-Ball Detector
Dead-ball stoppages (`D`) represent over 54% of match frames. While the baseline only outputs `D` when the tracker loses the ball, real stoppages have clear kinematic signatures:
- In live play, player speeds average $2.1 - 2.4\text{ m/s}$.
- During stoppages, the collective speed of all 22 players collapses to $\mu = 1.32\text{ m/s}$ (median $1.11\text{ m/s}$ — collective walking).
- Ball coordinates leaving the calibrated pitch polygon $[0, 105.3]\text{ m} \times [0, 68.0]\text{ m}$ instantly trigger dead ball.
- Multi-pass temporal regularization eliminates spurious micro-stoppages shorter than 4.0 seconds (100 frames), matching the reality that football restarts never take under 4 seconds.

---

## 3. Assets Used & Integration Details

| Asset | Role in Our Pipeline | Key Finding / Utilization |
|---|---|---|
| **ATD Tracking (Tactical)** | Core player & ball coordinates | Extracted frame-level min distances ($d_H, d_A$), ball velocities $(v_x, v_y)$ by finite differences, collective velocities, and team centroids. |
| **Smart Tagging XML** | Semantic restart anchors | 57 out of 59 `SET PIECES` were geometrically resolved and used to anchor restarts with zero latency. |
| **Homography (Tactical)** | Metric pitch boundaries | Validated 100% usable tactical homography to convert normalized $[0, 1]$ coordinates to real meters ($105.3\text{m} \times 68.0\text{m}$) and detect ball out-of-bounds. |
| **Broadcast Video & ATD** | Explored and dropped | Tactical video and ATD already provide 100% homography coverage and continuous timeline without commercial cuts or zooms; cross-syncing broadcast was redundant. |

### 3.1 Resolving the 32 "Team: N/A" Set Pieces
In Metrica's Smart Tagging export, 32 instances are tagged `Team: N/A`, including every throw-in. We resolved 96.6% of all set pieces automatically:
- **Throw-ins (26 instances):** During each throw-in window, we inspect touchline proximity ($y \le 0\text{m}$ or $y \ge 68\text{m}$). The throwing player stands outside the pitch holding the ball ($d < 0.20\text{m}$). This uniquely identified the throwing team for all 26 throw-ins (15 Barcelona, 11 Athletic).
- **Match Kickoffs (2 instances):** Frame 11185 (1st half) initialized to Athletic Club ($d_A = 0.42\text{m}$); frame 113742 (2nd half) initialized to Barcelona ($d_H = 1.46\text{m}$). Pre-kickoff frames (110808..113741) locked to dead ball `D`.
- **Kickoffs after Goal (4 instances):** Conceding team kicks off from center circle (frame 49631 Athletic, frame 62012 Barca, frame 75457 Barca, frame 163000 Athletic).

---

## 4. Training Data & Compliance

- **No hand labels were used on the provided game**, strictly adhering to Rule 1 and Rule 2.
- No human-annotated external event data (Opta, StatsBomb, Wyscout) for this match was used.
- All parameters ($d_{control} = 1.8\text{m}$, $K = 10\text{ frames}$, $v_{walk} = 1.30\text{ m/s}$, $t_{min\_dead} = 4.0\text{ s}$) were derived from first-principles kinematics and verified for stability across a wide grid search to prevent overfitting to the 5-minute public ground truth.

---

## 5. Limitations & Future Work

1. **Deflections & Aerial Duels:** In high-congestion penalty box scrambles, multiple players contest the ball within 1 meter. A graph neural network (GNN) modeling player-ball interaction edges could further refine touch attribution.
2. **Computer Vision Ball Re-detection:** For the 41% ball-missing frames in open play, a lightweight TrackNetV2 / YOLO model running on the tactical video crops could visually reconstruct occluded ball coordinates.

---

## 6. How to Reproduce

Everything runs in standard Python 3.10+ using only `numpy` and `pandas` (no GPU or external framework required):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Extract features (fast streaming parser)
python src/features.py

# 3. Generate the submission file (controlled_inertia_v3.csv)
python src/model_v3.py

# 4. Validate file format against official scorer rules
python evaluation/validate_submission.py evaluation/results/controlled_inertia_v3.csv

# 5. Run official evaluation
python evaluation/run_evaluation.py
```
Execution time from raw tracking to final submission: **~12 seconds**.
