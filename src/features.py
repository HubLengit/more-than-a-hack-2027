"""Feature extraction for More than a Hack 2027.

Extracts kinematic, spatial, geometric, and tactical features from:
  1. Raw ATD tracking (player slots + ball)
  2. Smart Tagging XML (set pieces, tactical phases)
  3. Homography (pitch coordinates <-> pixel space)

Stores the extracted features in a compressed .npz cache for instantaneous loading.
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"

HOME, AWAY, NOBODY = 0, 1, -1


def team_of_each_track(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots], dtype=np.int8)


def extract_features(game=GAME, data_dir=None, cache_path=None):
    """Extract full match features for all frames.
    
    If cache_path is provided and exists, loads from cache. Otherwise computes and saves.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return dict(np.load(cache_path, allow_pickle=True))

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch_length, pitch_width = GAMES[game]["pitch"]

    team_of_track = team_of_each_track(meta_path)
    n_slots = len(team_of_track)

    print(f"Extracting features for {n_frames} frames ({pitch_length}x{pitch_width}m pitch)...")

    # Allocate arrays
    ball_vis = np.zeros(n_frames, dtype=bool)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    ball_out = np.zeros(n_frames, dtype=bool)

    min_dist_H = np.full(n_frames, np.nan, dtype=np.float32)
    min_dist_A = np.full(n_frames, np.nan, dtype=np.float32)
    closest_team = np.full(n_frames, NOBODY, dtype=np.int8)
    closest_dist = np.full(n_frames, np.nan, dtype=np.float32)
    closest_player_speed = np.full(n_frames, np.nan, dtype=np.float32)

    mean_speed_H = np.zeros(n_frames, dtype=np.float32)
    mean_speed_A = np.zeros(n_frames, dtype=np.float32)
    mean_speed_all = np.zeros(n_frames, dtype=np.float32)

    n_players_H = np.zeros(n_frames, dtype=np.int8)
    n_players_A = np.zeros(n_frames, dtype=np.int8)

    centroid_H_x = np.full(n_frames, np.nan, dtype=np.float32)
    centroid_H_y = np.full(n_frames, np.nan, dtype=np.float32)
    centroid_A_x = np.full(n_frames, np.nan, dtype=np.float32)
    centroid_A_y = np.full(n_frames, np.nan, dtype=np.float32)

    spread_H = np.full(n_frames, np.nan, dtype=np.float32)
    spread_A = np.full(n_frames, np.nan, dtype=np.float32)

    # Stream raw ATD
    with open(raw_path) as handle:
        for line_idx, line in enumerate(handle):
            parts = line.strip().split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame_id = int(parts[0]) - 1  # 0-indexed frame
            if frame_id < 0 or frame_id >= n_frames:
                continue

            # Parse ball
            ball = parts[2]
            b_vis = (ball[0] != MISSING)
            bx, by = np.nan, np.nan
            if b_vis:
                comma1 = ball.index(",")
                comma2 = ball.index(",", comma1 + 1)
                bx_norm = float(ball[:comma1])
                by_norm = float(ball[comma1 + 1:comma2])
                bx = bx_norm * pitch_length
                by = by_norm * pitch_width
                ball_vis[frame_id] = True
                ball_x[frame_id] = bx
                ball_y[frame_id] = by
                # Check if ball is out of pitch boundaries
                if bx_norm < 0.0 or bx_norm > 1.0 or by_norm < 0.0 or by_norm > 1.0:
                    ball_out[frame_id] = True

            # Parse players
            h_coords = []
            a_coords = []
            h_speeds = []
            a_speeds = []
            h_slots = []
            a_slots = []

            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot >= n_slots or group[0] == MISSING:
                    continue
                comma1 = group.index(",")
                comma2 = group.index(",", comma1 + 1)
                px = float(group[:comma1]) * pitch_length
                py = float(group[comma1 + 1:comma2]) * pitch_width
                sp_str = group[comma2 + 1:]
                speed = float(sp_str) if sp_str != "NaN" else 0.0

                team = team_of_track[slot]
                if team == HOME:
                    h_coords.append((px, py))
                    h_speeds.append(speed)
                    h_slots.append(slot)
                else:
                    a_coords.append((px, py))
                    a_speeds.append(speed)
                    a_slots.append(slot)

            nH = len(h_coords)
            nA = len(a_coords)
            n_players_H[frame_id] = nH
            n_players_A[frame_id] = nA

            if nH > 0:
                h_arr = np.array(h_coords, dtype=np.float32)
                cx_h, cy_h = np.mean(h_arr[:, 0]), np.mean(h_arr[:, 1])
                centroid_H_x[frame_id] = cx_h
                centroid_H_y[frame_id] = cy_h
                spread_H[frame_id] = np.mean(np.sqrt((h_arr[:, 0] - cx_h)**2 + (h_arr[:, 1] - cy_h)**2))
                mean_speed_H[frame_id] = float(np.mean(h_speeds))

            if nA > 0:
                a_arr = np.array(a_coords, dtype=np.float32)
                cx_a, cy_a = np.mean(a_arr[:, 0]), np.mean(a_arr[:, 1])
                centroid_A_x[frame_id] = cx_a
                centroid_A_y[frame_id] = cy_a
                spread_A[frame_id] = np.mean(np.sqrt((a_arr[:, 0] - cx_a)**2 + (a_arr[:, 1] - cy_a)**2))
                mean_speed_A[frame_id] = float(np.mean(a_speeds))

            all_speeds = h_speeds + a_speeds
            if all_speeds:
                mean_speed_all[frame_id] = float(np.mean(all_speeds))

            # Proximity to ball
            if b_vis:
                best_h_dist = np.nan
                best_h_speed = np.nan
                if nH > 0:
                    dists_h = np.sqrt((h_arr[:, 0] - bx)**2 + (h_arr[:, 1] - by)**2)
                    min_idx_h = np.argmin(dists_h)
                    best_h_dist = dists_h[min_idx_h]
                    best_h_speed = h_speeds[min_idx_h]
                    min_dist_H[frame_id] = best_h_dist

                best_a_dist = np.nan
                best_a_speed = np.nan
                if nA > 0:
                    dists_a = np.sqrt((a_arr[:, 0] - bx)**2 + (a_arr[:, 1] - by)**2)
                    min_idx_a = np.argmin(dists_a)
                    best_a_dist = dists_a[min_idx_a]
                    best_a_speed = a_speeds[min_idx_a]
                    min_dist_A[frame_id] = best_a_dist

                if not np.isnan(best_h_dist) and not np.isnan(best_a_dist):
                    if best_h_dist <= best_a_dist:
                        closest_team[frame_id] = HOME
                        closest_dist[frame_id] = best_h_dist
                        closest_player_speed[frame_id] = best_h_speed
                    else:
                        closest_team[frame_id] = AWAY
                        closest_dist[frame_id] = best_a_dist
                        closest_player_speed[frame_id] = best_a_speed
                elif not np.isnan(best_h_dist):
                    closest_team[frame_id] = HOME
                    closest_dist[frame_id] = best_h_dist
                    closest_player_speed[frame_id] = best_h_speed
                elif not np.isnan(best_a_dist):
                    closest_team[frame_id] = AWAY
                    closest_dist[frame_id] = best_a_dist
                    closest_player_speed[frame_id] = best_a_speed

    # Ball finite difference velocity & speed
    print("Computing ball kinematics...")
    ball_vx = np.full(n_frames, np.nan, dtype=np.float32)
    ball_vy = np.full(n_frames, np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)

    vis_indices = np.flatnonzero(ball_vis)
    consecutive = np.where(np.diff(vis_indices) == 1)[0]
    idx1 = vis_indices[consecutive]
    idx2 = vis_indices[consecutive + 1]

    dt = 1.0 / 25.0
    vx = (ball_x[idx2] - ball_x[idx1]) / dt
    vy = (ball_y[idx2] - ball_y[idx1]) / dt
    sp = np.sqrt(vx**2 + vy**2)

    ball_vx[idx2] = vx
    ball_vy[idx2] = vy
    ball_speed[idx2] = sp

    # Smart Tagging Features
    print("Integrating Smart Tagging events...")
    is_set_piece = np.zeros(n_frames, dtype=bool)
    set_piece_team = np.full(n_frames, NOBODY, dtype=np.int8)
    set_piece_type = np.full(n_frames, "", dtype="<U24")

    events = load_smart_tagging(game=game, data_dir=data_dir)
    for e in events:
        c = (e["code"] or "").strip().lower()
        sf = max(0, min(n_frames - 1, e["start_frame"]))
        ef = max(0, min(n_frames, e["end_frame"]))
        if c == "set pieces":
            is_set_piece[sf:ef] = True
            team_str = e["tags"].get("Team", "")
            t = HOME if "barcelona" in team_str.lower() else (AWAY if "atletic" in team_str.lower() else NOBODY)
            set_piece_team[sf:ef] = t
            set_piece_type[sf:ef] = e["tags"].get("Type", "")

    features = {
        "n_frames": n_frames,
        "ball_vis": ball_vis,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_out": ball_out,
        "ball_vx": ball_vx,
        "ball_vy": ball_vy,
        "ball_speed": ball_speed,
        "min_dist_H": min_dist_H,
        "min_dist_A": min_dist_A,
        "closest_team": closest_team,
        "closest_dist": closest_dist,
        "closest_player_speed": closest_player_speed,
        "mean_speed_H": mean_speed_H,
        "mean_speed_A": mean_speed_A,
        "mean_speed_all": mean_speed_all,
        "n_players_H": n_players_H,
        "n_players_A": n_players_A,
        "centroid_H_x": centroid_H_x,
        "centroid_H_y": centroid_H_y,
        "centroid_A_x": centroid_A_x,
        "centroid_A_y": centroid_A_y,
        "spread_H": spread_H,
        "spread_A": spread_A,
        "is_set_piece": is_set_piece,
        "set_piece_team": set_piece_team,
        "set_piece_type": set_piece_type,
    }

    if cache_path:
        print(f"Saving extracted features to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **features)
        print("Done!")

    return features


if __name__ == "__main__":
    cache = os.path.join(ROOT, "data", "features_cache.npz")
    extract_features(cache_path=cache)
