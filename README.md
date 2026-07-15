# Tire Bead Inspection System

A modular computer vision solution for automated industrial tire bead inspection (specifically steel wire bundles). This system leverages YOLO-based object detection to perform precise, real-time measurements across different camera perspectives (Front, Head, and Tail), ensuring high-quality control in manufacturing pipelines.

## Project Architecture
The system is built with a Modular Design pattern, separating logic for different inspection points to ensure scalability and ease of maintenance.
```bash
bead-inspection-system/
├── camera_front/
│   ├── models/
│   └── inference_front.py
├── camera_head/
│   ├── models/
│   └── inference_head.py
├── camera_tail/
│   ├── models/
│   └── inference_tail.py
├── performance/
│   ├── front/
│   ├── head/
│   └── tail/
├── .gitignore
├── README.md
└── requirements.txt
```

## Quick Start & Usage
Each module operates independently. To run the inference pipeline for a specific camera, use the following commands:

### Run Head Camera (Vertical Inspection)
```python camera_head/inference_head.py --weights camera_head/models/best.pt```

### Run Front Camera (Geometric & Angular Inspection)
```python camera_front/inference_front.py --weights camera_front/models/best.pt```

### Run Tail Camera (Bundle Termination Inspection)
```python camera_tail/inference_tail.py --weights camera_tail/models/best.pt```

## Performance & Results
The system has been tested on real industrial bead footage. The performance/ folder contains sample output images from each module.
### Performance Summary

| Camera | Focus Area                        | Classes | mAP50   | mAP50-95 | Status         |
|--------|-----------------------------------|---------|---------|----------|----------------|
| Front  | Geometric + Angular Analysis      | 5       | ~0.99   | ~0.88    | Excellent      |
| Head   | Vertical Alignment + Regions      | 5       | ~1.00   | ~0.92    | Outstanding    |
| Tail   | Wire Termination + Counting       | 3       | ~0.99   | ~0.82    | Very Strong    |

Detailed model information → MODEL_CARD.md

### 1. Front Camera (Geometric & Angular Inspection)

Accurate bead center & radius detection
Smart clamp filtering and angular measurements
Strict 4-clamp overlap validation

(See performance/front/ for result images)
### 2. Head Camera (Vertical Alignment)

Vertical perpendicular distance measurement
Bundle height consistency tracking
Region-based structural analysis

(See performance/head/ for result images)
### 3. Tail Camera (Termination Quality)

X-axis double-detection suppression (NMS)
Accurate wire counting
Clamp-to-wire termination gap measurement

(See performance/tail/ for result images)

## Technical Pipeline & Module Functionality

### 1. Head Camera (Vertical Inspection)
**Role:** Evaluates the vertical alignment and structural integrity of the bead head wires.
**Core Logic:**
* Vertical Perpendicular Measurements: Calculates the perpendicular distance between the wire bundle and the clamp (Region 1) on the Y-axis.
* Bundle Height Tracking: Tracks the vertical distance between the shortest and longest wire to ensure cluster consistency.
* Dynamic Telemetry: Monitors Region 3 height and real-time wire counting.
**Key Insight:** This module prevents "bundle scattering" by ensuring the vertical variance of the wires remains within strict tolerances.

### 2. Front Camera (Geometric & Angular Inspection)
**Role:** The most complex module responsible for bead circularity, clamp positioning, and head/tail angular boundaries.
**Core Logic:**
* Geometric Modeling: Identifies the bead center and calculates the radius to establish a reference coordinate system.
* Smart Clamp Filtering: Implements a proximity and angular threshold to suppress duplicate detections and noise, ensuring only valid clamps are processed.
* Angular Overlap Logic: Detects Head and Tail positions and calculates the arc length between them.
* Overlap Constraint: Enforces a strict 4-clamp maximum constraint within the overlap zone for precise quality control.
**Key Insight:** This module ensures that components are placed at precise angular intervals, which is critical for electrical conductivity and seal integrity.

### 3. Tail Camera (Bundle Termination Inspection)
**Role:** Validates the final termination point of the bead and the wire distribution.
**Core Logic:**
* X-Axis NMS (Double-Detection Filter): Employs horizontal suppression logic to filter out overlapping wire detections, ensuring each wire is counted exactly once.
* Bundle Verticality: Calculates the vertical height of the tail wire bundle to verify proper termination.
* Clamp-to-Wire Distance: Measures the distance from the clamp to the bottom-most wire to ensure the termination gap is within factory specifications.
**Key Insight:** By filtering out "phantom" detections along the X-axis, this module provides the most accurate wire count in the final stage.

## Technologies Used
* Python: Core programming language.
* OpenCV: Image processing and dynamic UI visualization.
* Ultralytics YOLO: Deep learning-based object detection.

## Privacy & Data Policy
This repository strictly contains no video or image assets from the original source. Source video files are excluded from version control to ensure data security. Only processed result images (without sensitive content) are included in the performance/ folder.
