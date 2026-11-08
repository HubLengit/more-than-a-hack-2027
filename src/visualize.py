"""Match Video & Prediction Visualizer.

Overlays predictions, player tracks, ball tracking, and a 2D pitch radar
directly onto the tactical 1080p match video using the provided homography.

Usage:
  python src/visualize.py --start 11185 --frames 300 --out output_kickoff.mp4
  python src/visualize.py --start 12660 --frames 2000 --out output_transition.mp4
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import require, GAME, GAMES
from starter.baseline import team_of_each_track
from starter.load_homography import load as load_homography, project
from evaluation import possession_metrics as pm

# Colors (BGR for OpenCV)
BARCA_COLOR = (140, 20, 20)       # Deep Blue / Navy (BGR)
BARCA_SECONDARY = (60, 20, 160)   # Blaugrana Red
ATHLETIC_COLOR = (40, 40, 220)    # Athletic Red (BGR)
ATHLETIC_WHITE = (240, 240, 240)  # White
BALL_COLOR = (255, 255, 255)      # White
BALL_RING = (0, 215, 255)         # Gold / Fluorescent Amber
DEAD_COLOR = (80, 80, 80)         # Gray
PITCH_GREEN = (34, 120, 50)       # 2D Pitch Green
LINE_WHITE = (220, 220, 220)

HOME, AWAY, NOBODY = 0, 1, -1


def create_radar(width=320, height=207):
    """Draws a base 2D football pitch image in BGR."""
    radar = np.full((height, width, 3), PITCH_GREEN, dtype=np.uint8)
    
    # Outer pitch boundary
    margin = 8
    pw = width - 2 * margin
    ph = height - 2 * margin
    cv2.rectangle(radar, (margin, margin), (width - margin, height - margin), LINE_WHITE, 1)
    
    # Halfway line
    mid_x = width // 2
    cv2.line(radar, (mid_x, margin), (mid_x, height - margin), LINE_WHITE, 1)
    
    # Center circle
    center_y = height // 2
    circle_r = int(9.15 / 105.0 * pw)
    cv2.circle(radar, (mid_x, center_y), circle_r, LINE_WHITE, 1)
    cv2.circle(radar, (mid_x, center_y), 2, LINE_WHITE, -1)
    
    # Penalty boxes (16.5m depth, 40.3m width)
    box_w = int(16.5 / 105.0 * pw)
    box_h = int(40.3 / 68.0 * ph)
    box_y1 = (height - box_h) // 2
    box_y2 = box_y1 + box_h
    
    # Left box
    cv2.rectangle(radar, (margin, box_y1), (margin + box_w, box_y2), LINE_WHITE, 1)
    # Right box
    cv2.rectangle(radar, (width - margin - box_w, box_y1), (width - margin, box_y2), LINE_WHITE, 1)
    
    return radar, margin, pw, ph


def render_visualization(
    start_frame=11185,
    n_frames=250,
    pred_csv="evaluation/results/controlled_inertia_v3.csv",
    out_path="match_visualization.mp4",
    show_radar=True,
    fps=25.0,
):
    print(f"Loading match assets for frames {start_frame}..{start_frame + n_frames}...")
    
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    video_path = require(["video"])[0]
    team_of_track = team_of_each_track(meta_path)
    
    # Load Homography
    matrices, _, valid = load_homography()
    
    # Load Predictions
    pred_path = os.path.join(ROOT, pred_csv)
    preds_df = pd.read_csv(pred_path)
    pred_labels = preds_df["label"].values
    
    # Public Ground Truth (if available for comparison)
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt_labels = None
    if os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path)
        gt_labels = dict(zip(gt_df["frame"], gt_df["label"]))
        
    # Open Video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))
    
    # Pre-read ATD lines for the requested frame window
    atd_frames = {}
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            fid = int(parts[0]) - 1
            if start_frame <= fid < start_frame + n_frames:
                atd_frames[fid] = (parts[1], parts[2])
            elif fid >= start_frame + n_frames:
                break
                
    base_radar, r_margin, r_pw, r_ph = create_radar(320, 207)
    
    print(f"Rendering {n_frames} frames to {out_path}...")
    
    ball_history = []
    
    for f_idx in range(n_frames):
        curr_frame = start_frame + f_idx
        ret, frame = cap.read()
        if not ret:
            break
            
        H = matrices[curr_frame] if valid[curr_frame] else None
        pred_label = pred_labels[curr_frame] if curr_frame < len(pred_labels) else "X"
        gt_label = gt_labels.get(curr_frame, "X") if gt_labels else "X"
        
        # 2D Radar copy
        radar = base_radar.copy() if show_radar else None
        
        # Parse tracking
        atd = atd_frames.get(curr_frame)
        if atd and H is not None:
            players_str, ball_str = atd
            
            # Draw players
            for slot, group in enumerate(players_str.split(";")):
                if group[0] == "N":
                    continue
                comma1 = group.index(",")
                comma2 = group.index(",", comma1 + 1)
                px_norm = float(group[:comma1])
                py_norm = float(group[comma1 + 1:comma2])
                
                # Project to video pixels
                vx, vy = project(H, px_norm, py_norm)
                if not np.isnan(vx) and not np.isnan(vy):
                    ix, iy = int(round(vx)), int(round(vy))
                    if 0 <= ix < frame_w and 0 <= iy < frame_h:
                        team = team_of_track[slot]
                        color = BARCA_COLOR if team == HOME else ATHLETIC_COLOR
                        inner_color = BARCA_SECONDARY if team == HOME else ATHLETIC_WHITE
                        
                        # Ellipse / circle at feet
                        cv2.ellipse(frame, (ix, iy), (12, 6), 0, 0, 360, color, 3)
                        cv2.circle(frame, (ix, iy), 3, inner_color, -1)
                        
                # Draw on 2D Radar
                if radar is not None:
                    rx = int(r_margin + px_norm * r_pw)
                    ry = int(r_margin + py_norm * r_ph)
                    if 0 <= rx < radar.shape[1] and 0 <= ry < radar.shape[0]:
                        team = team_of_track[slot]
                        r_color = (200, 30, 30) if team == HOME else (30, 30, 220)
                        cv2.circle(radar, (rx, ry), 4, r_color, -1)
                        cv2.circle(radar, (rx, ry), 5, (0, 0, 0), 1)

            # Draw Ball
            if ball_str[0] != "N":
                b_comma1 = ball_str.index(",")
                b_comma2 = ball_str.index(",", b_comma1 + 1)
                bx_norm = float(ball_str[:b_comma1])
                by_norm = float(ball_str[b_comma1 + 1:b_comma2])
                
                bvx, bvy = project(H, bx_norm, by_norm)
                if not np.isnan(bvx) and not np.isnan(bvy):
                    ibx, iby = int(round(bvx)), int(round(bvy))
                    if 0 <= ibx < frame_w and 0 <= iby < frame_h:
                        ball_history.append((ibx, iby))
                        if len(ball_history) > 12:
                            ball_history.pop(0)
                            
                        # Trajectory tail
                        for h_i in range(1, len(ball_history)):
                            alpha = h_i / len(ball_history)
                            thickness = max(1, int(alpha * 3))
                            cv2.line(frame, ball_history[h_i - 1], ball_history[h_i], (0, 255, 255), thickness)
                            
                        # Ball spot
                        cv2.circle(frame, (ibx, iby), 10, BALL_RING, 2)
                        cv2.circle(frame, (ibx, iby), 6, BALL_COLOR, -1)
                        
                if radar is not None:
                    rbx = int(r_margin + bx_norm * r_pw)
                    rby = int(r_margin + by_norm * r_ph)
                    if 0 <= rbx < radar.shape[1] and 0 <= rby < radar.shape[0]:
                        cv2.circle(radar, (rbx, rby), 4, (255, 255, 255), -1)
                        cv2.circle(radar, (rbx, rby), 5, (0, 0, 0), 1)
            else:
                ball_history.clear()
                
        # Top HUD Dashboard
        hud_h = 70
        hud_overlay = frame[:hud_h, :].copy()
        cv2.rectangle(hud_overlay, (0, 0), (frame_w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(hud_overlay, 0.75, frame[:hud_h, :], 0.25, 0, frame[:hud_h, :])
        
        # Match time
        seconds_from_start = curr_frame / fps
        mins = int(seconds_from_start // 60)
        secs = int(seconds_from_start % 60)
        time_text = f"{mins:02d}:{secs:02d} (Frame {curr_frame})"
        cv2.putText(frame, time_text, (25, 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, (230, 230, 230), 2)
        
        # Prediction Pill
        pred_x = 450
        if pred_label == "H":
            pill_text = "PREDICTION: BARCELONA IN POSSESSION"
            pill_color = (180, 40, 40)  # Blue
            text_color = (255, 255, 255)
        elif pred_label == "A":
            pill_text = "PREDICTION: ATHLETIC CLUB IN POSSESSION"
            pill_color = (40, 40, 200)  # Red
            text_color = (255, 255, 255)
        else:
            pill_text = "PREDICTION: DEAD BALL (STOPPAGE)"
            pill_color = (60, 60, 60)
            text_color = (200, 200, 200)
            
        cv2.rectangle(frame, (pred_x, 15), (pred_x + 580, 55), pill_color, -1)
        cv2.rectangle(frame, (pred_x, 15), (pred_x + 580, 55), (255, 255, 255), 1)
        cv2.putText(frame, pill_text, (pred_x + 20, 43), cv2.FONT_HERSHEY_DUPLEX, 0.75, text_color, 2)
        
        # Ground Truth Tag if available
        if gt_label in ("H", "A", "D"):
            gt_text = f"GT: {gt_label}"
            match_color = (0, 200, 0) if gt_label == pred_label else (0, 0, 220)
            status_text = "MATCH" if gt_label == pred_label else "MISMATCH"
            cv2.putText(frame, f"{gt_text} [{status_text}]", (1100, 45), cv2.FONT_HERSHEY_DUPLEX, 0.75, match_color, 2)
            
        # Draw 2D Radar in bottom-right corner
        if radar is not None:
            rw, rh = radar.shape[1], radar.shape[0]
            rx1 = frame_w - rw - 30
            ry1 = frame_h - rh - 30
            
            # Semi-transparent border
            cv2.rectangle(frame, (rx1 - 4, ry1 - 24), (rx1 + rw + 4, ry1 + rh + 4), (10, 10, 10), -1)
            cv2.putText(frame, "2D TACTICAL RADAR", (rx1 + 10, ry1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            frame[ry1:ry1 + rh, rx1:rx1 + rw] = radar
            
        out.write(frame)
        if (f_idx + 1) % 50 == 0:
            print(f"  Rendered {f_idx + 1}/{n_frames} frames...")
            
    cap.release()
    out.release()
    print(f"Visualization video saved successfully to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=11185, help="start video frame (default: 11185 = kickoff)")
    parser.add_argument("--frames", type=int, default=300, help="number of frames to render (default: 300 = 12s)")
    parser.add_argument("--pred", default="evaluation/results/controlled_inertia_v3.csv", help="prediction CSV")
    parser.add_argument("--out", default="match_visualization.mp4", help="output video path")
    args = parser.parse_args()
    
    render_visualization(
        start_frame=args.start,
        n_frames=args.frames,
        pred_csv=args.pred,
        out_path=args.out
    )
