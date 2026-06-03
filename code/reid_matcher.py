import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def run_reid_merge(threshold=0.75, min_zone_overlap=0.5):
    session_root = "static"
    feature_data = []

    for folder in os.listdir(session_root):
        session_path = os.path.join(session_root, folder)
        json_path = os.path.join(session_path, "visitor_features.json")
        if os.path.isfile(json_path):
            with open(json_path) as f:
                session_features = json.load(f)
                for entry in session_features:
                    if entry["total_time"] < 10:
                        continue  # 🚫 Skip very short detections
                    entry["session"] = folder
                    entry["camera"] = entry["visitor_id"].split("_")[0]
                    feature_data.append(entry)

    if not feature_data:
        raise ValueError("No visitor_features.json files found.")

    print(f"📄 Loaded {len(feature_data)} feature entries")

    vectors = np.array([v["feature"] for v in feature_data])
    vectors = normalize(vectors)
    similarity = cosine_similarity(vectors)

    def jaccard(a, b):
        set_a = set(a)
        set_b = set(b)
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union > 0 else 0

    matched = [-1] * len(feature_data)
    group_id = 0

    for i in range(len(feature_data)):
        if matched[i] != -1:
            continue
        matched[i] = group_id
        for j in range(i + 1, len(feature_data)):
            if matched[j] == -1:
                same_camera = feature_data[i]["camera"] == feature_data[j]["camera"]
                zone_overlap = jaccard(feature_data[i].get("zone_path", []), feature_data[j].get("zone_path", []))
                if not same_camera and similarity[i, j] >= threshold and zone_overlap >= min_zone_overlap:
                    print(f"🧠 Merging {feature_data[i]['visitor_id']} + {feature_data[j]['visitor_id']} | Sim: {similarity[i,j]:.2f}, Overlap: {zone_overlap:.2f}")
                    matched[j] = group_id
        group_id += 1

    grouped_data = {}
    for i, m_id in enumerate(matched):
        grouped_data.setdefault(m_id, []).append(feature_data[i])

    # Stage 1 merged visitors
    stage1_groups = list(grouped_data.values())

    # Create cluster features and zone paths for Stage 2
    cluster_features = []
    cluster_paths = []
    cluster_ids = []
    for gid, visitors in enumerate(stage1_groups):
        cluster_ids.append(gid)
        features = [v["feature"] for v in visitors]
        avg_feat = np.mean(features, axis=0)
        cluster_features.append(avg_feat)

        zones = []
        for v in visitors:
            zones.extend(v.get("zone_path", []))
        cluster_paths.append(zones)

    cluster_features = normalize(np.array(cluster_features))
    sim2 = cosine_similarity(cluster_features)
    merged2 = [-1] * len(stage1_groups)
    final_id = 0

    for i in range(len(stage1_groups)):
        if merged2[i] != -1:
            continue
        merged2[i] = final_id
        for j in range(i + 1, len(stage1_groups)):
            if merged2[j] == -1:
                overlap = jaccard(cluster_paths[i], cluster_paths[j])
                if sim2[i, j] >= threshold and overlap >= min_zone_overlap:
                    print(f"🔁 Stage 2 Merging Group {i} + {j} | Sim: {sim2[i,j]:.2f}, Overlap: {overlap:.2f}")
                    merged2[j] = final_id
        final_id += 1

    final_groups = {}
    for i, new_gid in enumerate(merged2):
        final_groups.setdefault(new_gid, []).extend(stage1_groups[i])

    merged_rows = []
    for gid, visitors in final_groups.items():
        sessions = [v["session"] for v in visitors]
        ids = [v["visitor_id"] for v in visitors]
        total_time = sum(v["total_time"] for v in visitors)

        row = {
            "Global ID": gid,
            "Session": ", ".join(sessions),
            "Original ID": ", ".join(ids),
            "Total Time": round(total_time, 2)
        }

        zone_times = {}
        for v in visitors:
            for zone, t in v["zones"].items():
                zone_times[zone] = zone_times.get(zone, 0) + t

        row.update(zone_times)
        merged_rows.append(row)

    df = pd.DataFrame(merged_rows)
    out_path = os.path.join("static", "merged_global_visitors.csv")
    df.to_csv(out_path, index=False)
    print(f"📄 merged_global_visitors.csv saved at: {out_path}")
    print(f"✅ Final merged into {len(merged_rows)} global visitors (after Stage 2)")
    return out_path
