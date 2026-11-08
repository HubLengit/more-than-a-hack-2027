"""Smart Tagging & Set Piece Resolver.

Resolves the 32 'Team: N/A' set pieces:
- 26 Throw-ins: resolved via touchline proximity (holding ball outside pitch).
- 2 Start Kickoffs:
  - 1st Half: Athletic Club (at frame 11187, dA=0.42m).
  - 2nd Half: Barcelona (at frame 113742, dH=1.46m).
- 4 Kickoffs after Goal:
  - Frame 49631 (Barca goal): Athletic kicks off.
  - Frame 62012 (Athletic goal): Barca kicks off.
  - Frame 75457 (Athletic goal): Barca kicks off.
  - Frame 163000 (Barca goal): Athletic kicks off.
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from starter.load_smart_tagging import load as load_smart_tagging
from src.features import HOME, AWAY, NOBODY


def resolve_all_set_pieces(features, game="BAR-ATH", data_dir=None):
    """Returns an enriched array of set_piece_team and is_restart for all frames."""
    n_frames = int(features["n_frames"])
    bvis = features["ball_vis"]
    by = features["ball_y"]
    bout = features["ball_out"]
    dH = features["min_dist_H"]
    dA = features["min_dist_A"]

    events = load_smart_tagging(game=game, data_dir=data_dir)
    set_pieces = [e for e in events if (e["code"] or "").strip().lower() == "set pieces"]

    sp_team = np.full(n_frames, NOBODY, dtype=np.int8)
    sp_is_active = np.zeros(n_frames, dtype=bool)
    sp_type = np.full(n_frames, "", dtype="<U24")

    # Goals map for kickoff after goal
    # Conceding team kicks off
    kickoff_after_goal_teams = {
        49631: AWAY,   # Barca scored -> Athletic kicks off
        62012: HOME,   # Athletic scored -> Barca kicks off
        75457: HOME,   # Athletic scored -> Barca kicks off
        163000: AWAY,  # Barca scored -> Athletic kicks off
    }

    resolved_count = 0

    for e in set_pieces:
        sf = max(0, min(n_frames - 1, e["start_frame"]))
        ef = max(0, min(n_frames, e["end_frame"]))
        t_type = (e["tags"].get("Type") or "").strip()
        t_team_str = (e["tags"].get("Team") or "").strip().lower()

        team = NOBODY
        if "barcelona" in t_team_str:
            team = HOME
        elif "atletic" in t_team_str:
            team = AWAY
        else:
            t_type_lower = t_type.lower()
            if "kick off (start)" in t_type_lower:
                half = e["tags"].get("Half", "")
                team = AWAY if "1st" in half else HOME
            elif "kick off (after goal)" in t_type_lower:
                for k_frame, k_team in kickoff_after_goal_teams.items():
                    if abs(sf - k_frame) < 500:
                        team = k_team
                        break
            elif "throw-in" in t_type_lower:
                h_dists = []
                a_dists = []
                for f in range(sf, ef):
                    if bout[f] or (bvis[f] and (by[f] < 2.5 or by[f] > 65.5)):
                        if not np.isnan(dH[f]):
                            h_dists.append(dH[f])
                        if not np.isnan(dA[f]):
                            a_dists.append(dA[f])
                h_min = min(h_dists) if h_dists else 999.0
                a_min = min(a_dists) if a_dists else 999.0
                if h_min < a_min:
                    team = HOME
                elif a_min < h_min:
                    team = AWAY

        if team != NOBODY:
            resolved_count += 1
            sp_team[sf:ef] = team

        sp_is_active[sf:ef] = True
        sp_type[sf:ef] = t_type

    print(f"Resolved {resolved_count}/{len(set_pieces)} Set Pieces ({100*resolved_count/len(set_pieces):.1f}%)")
    return sp_is_active, sp_team, sp_type


if __name__ == "__main__":
    features = np.load(os.path.join(ROOT, "data", "features_cache.npz"), allow_pickle=True)
    is_active, teams, types = resolve_all_set_pieces(features)
