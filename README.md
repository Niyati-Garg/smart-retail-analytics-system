# Smart Retail Analytics System

An AI-powered retail analytics system that transforms CCTV footage into actionable customer behavior insights using computer vision and multi-camera tracking techniques.

The system detects and tracks customers across video streams, performs cross-camera customer re-identification (ReID), analyzes movement patterns, and generates analytics such as footfall, dwell time, zone popularity, and heatmaps.

## Key Features

- Customer detection using YOLOv8
- Multi-object tracking using DeepSORT
- Cross-camera customer re-identification (ReID) using OSNet embeddings
- Visitor merging using cosine similarity and zone-path analysis
- Zone-wise movement tracking and dwell-time analysis
- Customer session recording and behavioral analytics
- Heatmap generation for store activity visualization
- Dashboard-based analytics reporting

- ## System Architecture

```text
CCTV Video Streams
        ↓
Customer Detection (YOLOv8)
        ↓
Multi-Object Tracking (DeepSORT)
        ↓
Feature Extraction (OSNet)
        ↓
Cross-Camera Re-Identification
        ↓
Visitor Merging & Identity Resolution
        ↓
Zone-Based Movement Analysis
        ↓
Analytics & Heatmap Generation
        ↓
Dashboard Visualization

```
## Tech Stack

**Programming Language**
- Python

**Computer Vision & Machine Learning**
- YOLOv8
- DeepSORT
- TorchReID (OSNet)
- OpenCV
- NumPy
- Scikit-learn

**Data Analysis & Visualization**
- Pandas
- Matplotlib
- Seaborn

**Web Framework**
- Flask
