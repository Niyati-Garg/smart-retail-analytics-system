import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, Counter
from datetime import timedelta
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from generate_global_heatmaps import generate_global_heatmaps
import torch
import torchvision.transforms as T

# Load global layout config
with open("config.json", "r") as f:
    layout_config = json.load(f)

ZONE_ROWS = layout_config["zone_rows"]
ZONE_COLS = layout_config["zone_cols"]
FPS_SAMPLE = layout_config.get("fps_sample", 5)

# Load camera config
with open("camera_config.json", "r") as f:
    camera_config = json.load(f)

# Load ReID model
import torchreid
reid_model = torchreid.models.build_model("osnet_x1_0", num_classes=1000, pretrained=True)
reid_model.eval()
reid_model.to("cpu")

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def is_valid_crop(image):
    h, w = image.shape[:2]
    if h < 50 or w < 20:
        return False
    aspect_ratio = w / h
    return 0.3 <= aspect_ratio <= 0.75

def extract_reid_feature(image):
    if not is_valid_crop(image):
        return None
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to("cpu")
    with torch.no_grad():
        feature = reid_model(img)
    return feature.squeeze(0).cpu().numpy()

def run_customer_analysis_multi(video_paths, output_folder):
    yolo = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30)

    visitor_data = {}
    zone_heatmap = np.zeros((ZONE_ROWS, ZONE_COLS))
    visitor_features = []

    for video_path in video_paths:
        cam_id = None
        for cid, cfg in camera_config.items():
            if os.path.basename(video_path) == cfg["video"]:
                cam_id = cid
                break
        if not cam_id:
            print("Skipping video (not in config):", video_path)
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        zone_w = width // ZONE_COLS
        zone_h = height // ZONE_ROWS

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % FPS_SAMPLE != 0:
                frame_idx += 1
                continue

            results = yolo(frame, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append(([x1, y1, x2 - x1, y2 - y1], 0.99, "person"))

            tracks = []
            if detections:
                tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = f"{cam_id}_{track.track_id}"
                l, t, r, b = track.to_ltrb()
                cx = int((l + r) / 2)
                cy = int((t + b) / 2)
                if cx < 0 or cy < 0 or cx >= width or cy >= height:
                    continue
                z_row = min(max(cy // zone_h, 0), ZONE_ROWS - 1)
                z_col = min(max(cx // zone_w, 0), ZONE_COLS - 1)
                zone_key = f"Zone_{z_row}_{z_col}"
                timestamp = str(timedelta(seconds=frame_idx / fps))

                visitor = visitor_data.setdefault(tid, {
                    'entry': frame_idx / fps,
                    'zone_time': defaultdict(int),
                    'zone_timestamps': defaultdict(list),
                    'frames': 0,
                    'features': [],
                    'zone_map': np.zeros((ZONE_ROWS, ZONE_COLS), dtype=int),
                    'zone_path': []
                })
                visitor['zone_time'][zone_key] += 1
                visitor['zone_timestamps'][zone_key].append(timestamp)
                visitor['zone_map'][z_row, z_col] += 1
                visitor['frames'] += 1
                if not visitor['zone_path'] or visitor['zone_path'][-1] != zone_key:
                    visitor['zone_path'].append(zone_key)

                person_crop = frame[int(t):int(b), int(l):int(r)]
                if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                    feature = extract_reid_feature(person_crop)
                    if feature is not None:
                        visitor['features'].append(feature)

                zone_heatmap[z_row, z_col] += 1

            frame_idx += 1

        cap.release()

    os.makedirs(output_folder, exist_ok=True)
    rows = []
    for tid, info in visitor_data.items():
        total_time = info['frames'] / fps * FPS_SAMPLE
        row = {'Visitor ID': tid, 'Total Time (s)': round(total_time)}
        zone_times = {}
        for zone, frames in info['zone_time'].items():
            seconds = frames / fps * FPS_SAMPLE
            row[zone] = round(seconds, 2)
            zone_times[zone] = round(seconds, 2)
        for zone, times in info['zone_timestamps'].items():
            row[f"{zone}_Timestamps"] = ", ".join(times)
        rows.append(row)

        if info['features']:
            feature_array = np.mean(info['features'], axis=0)
            norm_feature = normalize([feature_array])[0].tolist()
            visitor_features.append({
                "visitor_id": tid,
                "feature": norm_feature,
                "zones": {z: round(t, 2) for z, t in zone_times.items()},
                "total_time": round(total_time, 2),
                "zone_path": info['zone_path']
            })

        zone_path = os.path.join(output_folder, f"{tid}_zonemap.npy")
        np.save(zone_path, info['zone_map'])

    df_zone_times = pd.DataFrame(rows)
    df_zone_times.to_csv(os.path.join(output_folder, "zone_times_by_visitor.csv"), index=False)

    zone_df = pd.DataFrame(zone_heatmap.astype(int))
    plt.figure(figsize=(10, 6))
    sns.heatmap(zone_df, cmap="YlOrRd", annot=True, linewidths=0.5, cbar=True)
    plt.title("Zone-based Customer Heatmap")
    plt.xlabel("Zone Columns")
    plt.ylabel("Zone Rows")
    plt.savefig(os.path.join(output_folder, "zone_heatmap.jpg"))

    if not visitor_features:
        print("⚠️ Warning: No features saved — possible tracking issue.")
    else:
        with open(os.path.join(output_folder, "visitor_features.json"), "w") as f:
            json.dump(visitor_features, f, indent=2)
        print(f"✅ visitor_features.json saved with {len(visitor_features)} visitors")

    total_visitors = len(visitor_data)
    total_seconds = sum(v['frames'] for v in visitor_data.values()) / fps * FPS_SAMPLE
    avg_time = total_seconds / total_visitors if total_visitors else 0
    report = f"Total Visitors: {total_visitors}\nAverage Time per Visitor: {int(avg_time // 60)}m {int(avg_time % 60)}s"

    with open(os.path.join(output_folder, "summary_report.txt"), "w") as f:
        f.write(report)

    print("✅ Combined analysis complete for:", video_paths)

    try:
        from reid_matcher import run_reid_merge
        run_reid_merge()
    except Exception as e:
        print("⚠️ ReID merge failed:", e)

    try:
        generate_global_heatmaps()
    except Exception as e:
        print("⚠️ Global heatmap generation failed:", e)

    return output_folder
