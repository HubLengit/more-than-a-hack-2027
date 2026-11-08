"""Controlled Touch & Inertia Model (v2).

Enhancements over v1:
1. Half-aware kickoff detection:
   - Half 1: frame 11185, detects initial possessor (Athletic Club).
   - Halftime: 90652..110807 set to D (or outside graded blocks).
   - Half 2: kickoff at 113742 (detected as Barcelona), pre-kickoff 110808..113741 set to D.
2. Smart Tagging Set Piece Integration:
   - When a dead ball ends inside a tagged SET PIECES interval, initialize the restarted possession
     to the tagged team (Barcelona -> H, Atletic Club -> A).
3. Rock-solid physical parameters with unitary fragmentation (frag=1.00).
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from starter.baseline import write_submission
from src.features import extract_features, HOME, AWAY, NOBODY
from games import half_video_blocks, GAME

LABELS = np.array(["H", "A", "D", "X"])


def predict_possession_v2(
    features,
    k_frames_control=10,
    dist_control=1.8,
    margin=0.4,
    walk_speed=1.30,
    speed_window=75,
    min_dead_frames=100,
    min_poss_frames=18,
):
    n_frames = int(features["n_frames"])
    ball_vis = features["ball_vis"]
    ball_out = features["ball_out"]
    ball_speed = features["ball_speed"]
    min_dist_H = features["min_dist_H"]
    min_dist_A = features["min_dist_A"]
    mean_speed_all = features["mean_speed_all"]
    is_set_piece = features["is_set_piece"]
    set_piece_team = features["set_piece_team"]

    # Smooth player speeds over window
    kernel = np.ones(speed_window, dtype=np.float32) / speed_window
    smooth_speed = np.convolve(mean_speed_all, kernel, mode="same")

    preds = np.full(n_frames, 2, dtype=np.int8)  # 2 = D (Dead ball default)

    # Halves boundaries
    blocks = half_video_blocks(GAME)
    # block 1: (1, 11185, 90651), block 2: (2, 110808, 190197)
    h1_start, h1_end = blocks[0][1], blocks[0][2]
    h2_start, h2_end = blocks[1][1], blocks[1][2]

    # Detect kickoffs for each half
    # Half 1: look at first visible ball in [h1_start, h1_start + 250]
    vis_h1 = np.where(ball_vis[h1_start:h1_start + 250])[0]
    h1_kick_f = h1_start + (vis_h1[0] if len(vis_h1) > 0 else 0)
    h1_init_team = HOME if min_dist_H[h1_kick_f] < min_dist_A[h1_kick_f] else AWAY

    # Half 2: official kickoff frame is 113742 (from FAQ / games.py)
    h2_kick_f = 113742
    h2_init_team = HOME if min_dist_H[h2_kick_f] < min_dist_A[h2_kick_f] else AWAY

    print(f"Kickoff Half 1: frame {h1_kick_f} -> team {LABELS[h1_init_team]}")
    print(f"Kickoff Half 2: frame {h2_kick_f} -> team {LABELS[h2_init_team]}")

    # Process Half 1
    current_poss = h1_init_team
    candidate_poss = h1_init_team
    candidate_count = 0

    for f in range(h1_start, h1_end + 1):
        is_dead = ball_out[f] or (smooth_speed[f] < walk_speed)
        if is_dead:
            preds[f] = 2  # D
            candidate_count = 0
            # If set piece tagged, set restart team
            if is_set_piece[f] and set_piece_team[f] != NOBODY:
                current_poss = set_piece_team[f]
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

    # Process Half 2
    # Before h2_kick_f, players are gathering/walking -> D
    preds[h2_start:h2_kick_f] = 2

    current_poss = h2_init_team
    candidate_poss = h2_init_team
    candidate_count = 0

    for f in range(h2_kick_f, h2_end + 1):
        is_dead = ball_out[f] or (smooth_speed[f] < walk_speed)
        if is_dead:
            preds[f] = 2
            candidate_count = 0
            if is_set_piece[f] and set_piece_team[f] != NOBODY:
                current_poss = set_piece_team[f]
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
                else:
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

    # Multi-pass run-length temporal cleanup within each half
    for h_s, h_e in [(h1_start, h1_end), (h2_start, h2_end)]:
        sub_preds = preds[h_s : h_e + 1]
        for _ in range(4):
            diffs = np.concatenate(([1], (sub_preds[1:] != sub_preds[:-1]).astype(int), [1]))
            starts = np.flatnonzero(diffs == 1)[:-1]
            ends = np.flatnonzero(diffs == 1)[1:]
            lens = ends - starts
            changed = False

            for s, e, l in zip(starts, ends, lens):
                lbl = sub_preds[s]
                if lbl == 2 and l < min_dead_frames:
                    prev_lbl = sub_preds[s - 1] if s > 0 else (sub_preds[e] if e < len(sub_preds) else HOME)
                    sub_preds[s:e] = prev_lbl if prev_lbl != 2 else HOME
                    changed = True
                elif lbl in (HOME, AWAY) and l < min_poss_frames:
                    prev_lbl = sub_preds[s - 1] if s > 0 else (sub_preds[e] if e < len(sub_preds) else 2)
                    sub_preds[s:e] = prev_lbl
                    changed = True

            if not changed:
                break
        preds[h_s : h_e + 1] = sub_preds

    return preds


def main():
    cache_path = os.path.join(ROOT, "data", "features_cache.npz")
    features = extract_features(cache_path=cache_path)

    preds = predict_possession_v2(features)
    labels = LABELS[preds]

    out_csv = os.path.join(ROOT, "evaluation", "results", "controlled_inertia_v2.csv")
    write_submission(out_csv, labels)
    print(f"Generated submission: {out_csv}")


if __name__ == "__main__":
    main()
