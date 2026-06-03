# Smart Retail Analytics System

An AI-powered retail analytics system that transforms CCTV footage into actionable customer behavior insights using computer vision and multi-camera tracking techniques.

The system detects and tracks customers across video streams, performs cross-camera customer re-identification (ReID), analyzes movement patterns, and generates analytics such as footfall, dwell time, zone popularity, and heatmaps.

---

## Problem Statement

Retail stores often use CCTV systems solely for surveillance purposes, leaving valuable customer behavior information unexplored. This project converts CCTV footage into actionable business intelligence by tracking customer movement, analyzing dwell time, identifying high-traffic zones, and generating visual analytics to support data-driven retail decisions.

---

## Key Features

* Customer detection using YOLOv8
* Multi-object tracking using DeepSORT
* Cross-camera customer re-identification (ReID) using OSNet embeddings
* Visitor merging using cosine similarity and zone-path analysis
* Zone-wise movement tracking and dwell-time analysis
* Customer session recording and behavioral analytics
* Heatmap generation for store activity visualization
* Dashboard-based analytics reporting

---

## System Architecture

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

---

## Methodology

### 1. Video Upload and Processing

Users upload one or more CCTV video streams through a Flask-based interface. Uploaded videos are passed to the analytics pipeline for processing.

### 2. Customer Detection

YOLOv8 is used to detect people in each video frame. Only person-class detections are retained for further analysis.

### 3. Multi-Object Tracking

DeepSORT assigns unique IDs to detected customers and tracks them across frames.

### 4. Zone-Based Analysis

The store is divided into predefined zones. Customer positions are mapped to zones to analyze movement patterns and dwell time.

### 5. Cross-Camera Re-Identification

OSNet feature embeddings are extracted for tracked customers. Similarity matching is used to identify the same customer across multiple camera views.

### 6. Visitor Merging

Matched visitors are merged into global identities to reduce duplicate counting and improve analytics accuracy.

### 7. Analytics Generation

Customer paths, dwell times, visitor counts, and zone statistics are aggregated.

### 8. Visualization

Heatmaps, summary reports, and dashboard charts are generated to support retail decision-making.

---

## My Contribution

This project was developed as part of a team-based Applications of AI course project.

My primary responsibility was the development of the customer re-identification (ReID) module used for cross-camera customer tracking.

Key contributions:

* Implemented embedding-based customer matching using OSNet feature vectors.
* Applied feature normalization and cosine similarity scoring.
* Designed a zone-path overlap validation mechanism using Jaccard similarity.
* Developed threshold-based visitor matching logic.
* Implemented a two-stage clustering and visitor-merging pipeline.
* Generated global customer identities across multiple camera streams.

### ReID Workflow

```text
Tracked Customer
       ↓
OSNet Feature Extraction
       ↓
Feature Vector Generation
       ↓
Cosine Similarity Matching
       ↓
Zone Overlap Validation
       ↓
Visitor Matching
       ↓
Stage 1 Merge
       ↓
Stage 2 Cluster Merge
       ↓
Global Visitor Identity
```

---

## Tech Stack

### Programming Language

* Python

### Computer Vision & Machine Learning

* YOLOv8
* DeepSORT
* TorchReID (OSNet)
* OpenCV
* NumPy
* Scikit-learn

### Data Analysis & Visualization

* Pandas
* Matplotlib
* Seaborn

### Web Framework

* Flask

---

## Results & Outputs

### Upload Interface

![Upload Interface](screenshots/upload_page.png)

### Processing Stage

![Processing Screen](screenshots/processing.png)

### Analytics Dashboard

![Dashboard](screenshots/dashboard.png)

### Visitor Time Analysis

![Visitor Time Analysis](screenshots/visitor_time.png)

### Zone Activity Analysis

![Zone Activity](screenshots/zone_analysis.png)

### Zone-Based Heatmap

![Heatmap](screenshots/heatmap.png)

---

## Generated Analytics

The system generates:

* Visitor counts
* Customer dwell time analysis
* Zone-wise engagement metrics
* Customer movement paths
* Summary reports
* Store activity heatmaps
* Dashboard-based visual analytics

---

## Future Improvements

* Real-time video stream processing
* Multi-store analytics support
* Cloud-based deployment
* Live occupancy monitoring
* Advanced customer behavior prediction
* Fine-tuned person re-identification models

---

## Project Information

* Academic team project developed as part of the Applications of AI course.
* Repository contains core implementation files and project documentation.
* Full project report is included in the repository for reference.

```
```
