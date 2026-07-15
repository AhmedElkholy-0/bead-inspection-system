# Model Card - Bead Inspection System

## Overview
Three specialized YOLO11 models for multi-view industrial bead quality inspection (Front, Head, and Tail cameras).

## Models Summary

| Camera | Focus Area                        | Classes | mAP50   | mAP50-95 | Status         |
|--------|-----------------------------------|---------|---------|----------|----------------|
| Front  | Geometric + Angular Analysis      | 5       | ~0.99   | ~0.88    | Excellent      |
| Head   | Vertical Alignment + Regions      | 5       | ~1.00   | ~0.92    | Outstanding    |
| Tail   | Wire Termination + Counting       | 3       | ~0.99   | ~0.82    | Very Strong    |

---

### 1. Front Camera Model
**Classes**: `bead`, `clamp`, `head`, `tail`, 

**Performance**:
- mAP50: ~0.99
- mAP50-95: ~0.88
- Precision: ~0.97
- Recall: ~0.98

**Dominant Class**: Clamp (405 instances)

---

### 2. Head Camera Model
**Classes**: `Region 1`, `Region 2`, `Region 3`, `head wires`

**Performance**:
- mAP50: ~1.00
- mAP50-95: ~0.92
- Precision: ~0.98
- Recall: ~0.99

**Dominant Class**: head wires (264 instances)

---

### 3. Tail Camera Model
**Classes**: `Tail_wire`, `clamp`

**Performance**:
- mAP50: ~0.99
- mAP50-95: ~0.82
- Precision: High
- Recall: High

**Dominant Class**: Tail_wire (61 instances)

---

## Training Highlights
- All models trained on real factory data.
- Stable convergence and high final metrics.
- Strong performance on dominant industrial classes (clamps & wires).
- Minor confusion mainly between visually similar classes (expected and acceptable).

## Usage
- Models are located in their respective folders: `camera_*/models/best.pt`
- See `README.md` for inference instructions.

**Last Updated**: July 16, 2026  
**License**: MIT
