"""Controlled Touch & Inertia Model (v1).

A physics- and football-informed state machine:
1. Dead Ball Detection:
   - Ball out of pitch bounds (from homography / ATD)
   - Collective low speed (mean speed across all players < threshold for > min_dead frames)
2. Controlled Possession & Turnover:
   - Possession stays with current team during ball flight / passes / clearances / duels
   - A change of possession requires sustained proximity (< dist_control for >= k_frames)
     and reasonable ball speed (not passing by at 20 m/s).
3. Temporal Regularization:
   - Minimum dead ball duration (merging unrealistic micro-stoppages)
   - Minimum possession spell duration
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from starter.baseline import write_submission
from src.features import extract_features, HOME, AWAY, NOBODY

LABELS = np.array(["H", "A", "D", "X"])


def predict_possession(
    features,
    k_frames_control=15,
    dist_control=1.8,
    margin=0.4,
    walk_speed=1.35,
    speed_window=75,
    min_dead_frames=100,
    min_poss_frames=25,
):
    n_frames = int(features["n_frames"])
    ball_vis = features["ball_vis"]
    ball_out = features["ball_out"]
    ball_speed = features["ball_speed"]
    min_dist_H = features["min_dist_H"]
    min_dist_A = features["min_dist_A"]
    mean_speed_all = features["mean_speed_all"]

    # Smooth player speeds over window
    kernel = np.ones(speed_window, dtype=np.float32) / speed_window
    smooth_speed = np.convolve(mean_speed_all, kernel, mode="same")

    preds = np.full(n_frames, 2, dtype=np.int8)  # 2 = D
    current_poss = HOME
    candidate_poss = HOME
    candidate_count = 0

    for f in range(n_frames):
        # Dead ball condition
        if ball_out[f] or smooth_speed[f] < walk_speed:
            preds[f] = 2  # D
            candidate_count = 0
        else:
            if ball_vis[f]:
                dH = min_dist_H[f]
                dA = min_dist_A[f]
                b_sp = ball_speed[f] if not np.isnan(ball_speed[f]) else 5.0

                h_controls = (dH < dist_control) and (dH < dA - margin) and (b_sp < 12.0)
                a_controls = (dA < dist_control) and (dA < dH - margin) and (b_sp < 12.0)

                if current_poss == HOME:
                    if a_controls:
                        if candidate_poss == AWAY:
                            candidate_count += 1
                            if candidate_count >= k_frames_control:
                                current_poss = AWAY
                                candidate_count = 0
                        else:
                            candidate_poss = AWAY
                            candidate_count = 1
                    else:
                        candidate_count = 0
                else:  # current_poss == AWAY
                    if h_controls:
                        if candidate_poss == HOME:
                            candidate_count += 1
                            if candidate_count >= k_frames_control:
                                current_poss = HOME
                                candidate_count = 0
                        else:
                            candidate_poss = HOME
                            candidate_count = 1
                    else:
                        candidate_count = 0

            preds[f] = current_poss

    # Multi-pass run-length temporal cleanup
    for _ in range(4):
        diffs = np.concatenate(([1], (preds[1:] != preds[:-1]).astype(int), [1]))
        starts = np.flatnonzero(diffs == 1)[:-1]
        ends = np.flatnonzero(diffs == 1)[1:]
        lens = ends - starts
        changed = False

        for s, e, l in zip(starts, ends, lens):
            lbl = preds[s]
            if lbl == 2 and l < min_dead_frames:
                # Short dead ball -> merge with neighboring possession
                prev_lbl = preds[s - 1] if s > 0 else (preds[e] if e < n_frames else HOME)
                preds[s:e] = prev_lbl if prev_lbl != 2 else HOME
                changed = True
            elif lbl in (HOME, AWAY) and l < min_poss_frames:
                # Short possession spell -> merge with neighbor
                prev_lbl = preds[s - 1] if s > 0 else (preds[e] if e < n_frames else 2)
                preds[s:e] = prev_lbl
                changed = True

        if not changed:
            break

    return preds


def main():
    cache_path = os.path.join(ROOT, "data", "features_cache.npz")
    features = extract_features(cache_path=cache_path)

    preds = predict_possession(features)
    labels = LABELS[preds]

    out_csv = os.path.join(ROOT, "evaluation", "results", "controlled_inertia_v1.csv")
    write_submission(out_csv, labels)
    print(f"Generated submission: {out_csv}")


if __name__ == "__main__":
    main()
