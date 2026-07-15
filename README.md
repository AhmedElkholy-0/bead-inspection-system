# Bead Inspection System

A modular computer vision solution for automated industrial bead inspection. This system leverages **YOLO-based object detection** to perform precise, real-time measurements across different camera perspectives (Front, Head, and Tail), ensuring high-quality control in manufacturing pipelines.

## Project Architecture
The system is built with a **Modular Design** pattern, separating logic for different inspection points to ensure scalability and ease of maintenance.

```text
/bead-inspection-system
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
├── .gitignore              # Privacy & file exclusion configuration
├── /camera_front
│   ├── inference.py        # Front camera inspection logic
│   └── /models             # Front model weights (.pt)
├── /camera_head
│   ├── inference.py        # Head camera inspection logic
│   └── /models             # Head model weights (.pt)
├── /camera_tail
│   ├── inference.py        # Tail camera inspection logic
│   └── /models             # Tail model weights (.pt)

Privacy & Data Policy
This repository does not contain any video or image assets.
To ensure data security and privacy, all source video files are strictly excluded from version control. Users must provide their own local test data to run the inference scripts.

Installation
Clone the repository:
git clone [repository-url]
Install dependencies:
pip install -r requirements.txt
Configure Models:
Place your trained YOLO weights (.pt files) into the respective models/ folders within each camera module directory.
Technologies Used
Python: Core programming language.

OpenCV: Image processing and visualization.

Ultralytics YOLO: Deep learning object detection.