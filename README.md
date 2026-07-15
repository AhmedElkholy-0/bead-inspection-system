# Bead Inspection System

A modular computer vision solution for automated industrial bead inspection. This system leverages YOLO-based object detection to perform precise, real-time measurements across different camera perspectives (Front, Head, and Tail), ensuring high-quality control in manufacturing pipelines.

##  Project Architecture
The system is built with a Modular Design pattern, separating logic for different inspection points to ensure scalability and ease of maintenance.

/bead-inspection-system
├── /camera_front    # Geometric & Angular Inspection
├── /camera_head     # Vertical Inspection
├── /camera_tail     # Bundle Termination Inspection
├── requirements.txt
└── README.md

##  Quick Start & Usage
Each module operates independently. To run the inference pipeline for a specific camera, use the following commands:

# Run Head Camera (Vertical Inspection)
python camera_head/inference_head.py --weights camera_head/models/best.pt

# Run Front Camera (Geometric & Angular Inspection)
python camera_front/inference_front.py --weights camera_front/models/best.pt

# Run Tail Camera (Bundle Termination Inspection)
python camera_tail/inference_tail.py --weights camera_tail/models/best.pt

## Technical Pipeline & Module Functionality

### 1. Head Camera (Vertical Inspection)
**Role:** Evaluates the vertical alignment and structural integrity of the bead head wires.
**Core Logic:**
* Vertical Perpendicular Measurements: Calculates the perpendicular distance between the wire bundle and the clamp (Region 1) on the Y-axis[cite: 1].
* Bundle Height Tracking: Tracks the vertical distance between the shortest and longest wire to ensure cluster consistency[cite: 1].
* Dynamic Telemetry: Monitors Region 3 height and real-time wire counting[cite: 1].
**Key Insight:** This module prevents "bundle scattering" by ensuring the vertical variance of the wires remains within strict tolerances[cite: 1].

### 2. Front Camera (Geometric & Angular Inspection)
**Role:** The most complex module responsible for bead circularity, clamp positioning, and head/tail angular boundaries.
**Core Logic:**
* Geometric Modeling: Identifies the bead center and calculates the radius to establish a reference coordinate system[cite: 2].
* Smart Clamp Filtering: Implements a proximity and angular threshold to suppress duplicate detections and noise, ensuring only valid clamps are processed[cite: 2].
* Angular Overlap Logic: Detects Head and Tail positions and calculates the arc length between them[cite: 2].
* Overlap Constraint: Enforces a strict 4-clamp maximum constraint within the overlap zone for precise quality control[cite: 2].
**Key Insight:** This module ensures that components are placed at precise angular intervals, which is critical for electrical conductivity and seal integrity[cite: 2].

### 3. Tail Camera (Bundle Termination Inspection)
**Role:** Validates the final termination point of the bead and the wire distribution.
**Core Logic:**
* X-Axis NMS (Double-Detection Filter): Employs horizontal suppression logic to filter out overlapping wire detections, ensuring each wire is counted exactly once[cite: 3].
* Bundle Verticality: Calculates the vertical height of the tail wire bundle to verify proper termination[cite: 3].
* Clamp-to-Wire Distance: Measures the distance from the clamp to the bottom-most wire to ensure the termination gap is within factory specifications[cite: 3].
**Key Insight:** By filtering out "phantom" detections along the X-axis, this module provides the most accurate wire count in the final stage[cite: 3].

## Technologies Used
* Python: Core programming language.
* OpenCV: Image processing and dynamic UI visualization.
* Ultralytics YOLO: Deep learning-based object detection.

## Privacy & Data Policy
This repository strictly contains no video or image assets. Source video files are excluded from version control to ensure data security. Users must provide local test data to run the inference scripts.