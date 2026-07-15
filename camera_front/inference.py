import cv2
import numpy as np
from ultralytics import YOLO

# =====================================================================
# 1. INITIALIZATION & MODEL SETUP
# =====================================================================
# Define model path and input video source
model_path = 'models/best.pt'
model = YOLO(model_path)
video_test_path = '/content/cut_video.mp4'  # Input video path for Front Camera

# Define Class IDs based on the training dataset
CLASS_BEAD = 0
CLASS_CLAMP = 1
CLASS_HEAD = 2
CLASS_TAIL = 3

# Define color palette for visual representation
COLORS = {
    "clamp": (128, 0, 128),         # Purple for general clamps
    "head": (0, 0, 255),            # Red for Head
    "tail": (0, 255, 255),          # Yellow for Tail
    "overlap_clamp": (0, 255, 0)    # Green for valid overlap clamps
}

# Utility function: Draw text with a black background for high contrast/readability
def draw_clear_text(img, text, position, color, font_scale=0.6, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    # Draw solid rectangle behind text
    cv2.rectangle(img, (x, y - text_h - 4), (x + text_w + 4, y + baseline), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 2), font, font_scale, color, thickness, cv2.LINE_AA)

# =====================================================================
# 2. VIDEO STREAM PROPERTIES & WRITER GENERATION
# =====================================================================
cap = cv2.VideoCapture(video_test_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter to save processed footage
output_path = 'front_camera_measurement1_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("Processing Video with Strict 4-Clamp Maximum Overlap Constraint...")

# =====================================================================
# 3. REAL-TIME INFERENCE FRAME-BY-FRAME STREAM
# =====================================================================
results = model.predict(
    source=video_test_path,
    conf=0.35, # Confidence threshold for detection
    stream=True
)

for r in results:
    frame = r.orig_img.copy()

    # Lists to store detected objects for the current frame
    beads = []
    clamps = []
    heads = []
    tails = []

    # 3a. BOUNDING BOX EXTRACTION & CLASS SEGREGATION
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_score = float(box.conf[0])
        cls_id = int(box.cls[0])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        obj_data = {"center": (cx, cy), "box": (x1, y1, x2, y2), "conf": conf_score}

        if cls_id == CLASS_BEAD:
            beads.append(obj_data)
        elif cls_id == CLASS_CLAMP:
            clamps.append(obj_data)
        elif cls_id == CLASS_HEAD:
            heads.append(obj_data)
        elif cls_id == CLASS_TAIL:
            tails.append(obj_data)

    head_count = len(heads)
    tail_count = len(tails)

    # UI display variables initialization
    bead_center_text = "N/A"
    bead_radius_text = "N/A"
    arc_length_head_tail_text = "0 px"
    overlap_clamps_count_text = "0"
    arc_c1_c2 = "0 px"
    arc_c2_c3 = "0 px"
    arc_c3_c4 = "0 px"

    filtered_clamps = []

    # =====================================================================
    # 4. GEOMETRIC CALCULATIONS & INDEPENDENT RENDERING
    # =====================================================================
    if len(beads) > 0:
        # Select the bead with the highest confidence
        best_bead = max(beads, key=lambda b: b["conf"])
        bx1, by1, bx2, by2 = best_bead["box"]

        bead_cx = int((bx1 + bx2) / 2)
        bead_cy = int((by1 + by2) / 2)
        bead_radius = int(((bx2 - bx1) + (by2 - by1)) / 4)

        bead_center_text = f"({bead_cx}, {bead_cy})"
        bead_radius_text = f"{bead_radius} px"

        # -----------------------------------------------------------------
        # Logic: Smart Clamp Filtering (Angular and Proximity thresholding)
        # Prevents double detection or noise in clamp identification
        # -----------------------------------------------------------------
        if len(clamps) > 0:
            for c in clamps:
                c["theta"] = np.arctan2(c["center"][1] - bead_cy, c["center"][0] - bead_cx)

            # Sort clamps by angle to facilitate sequential filtering
            clamps_sorted = sorted(clamps, key=lambda c: c["theta"])
            for c in clamps_sorted:
                if not filtered_clamps:
                    filtered_clamps.append(c)
                else:
                    angle_diff = np.abs(c["theta"] - filtered_clamps[-1]["theta"])
                    if angle_diff > np.pi: angle_diff = 2 * np.pi - angle_diff

                    c1_x, c1_y = c["center"]
                    c2_x, c2_y = filtered_clamps[-1]["center"]
                    pixel_dist = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)

                    # Merge redundant detections if close in proximity and angle
                    if angle_diff < 0.03 and pixel_dist < 15:
                        if c["conf"] > filtered_clamps[-1]["conf"]: filtered_clamps[-1] = c
                    else:
                        filtered_clamps.append(c)

        total_clamps_count = len(filtered_clamps)

        # -----------------------------------------------------------------
        # Head and Tail Detection and Visualization
        # -----------------------------------------------------------------
        theta_head, theta_tail = None, None
        hx, hy, tx, ty = None, None, None, None

        if len(heads) > 0:
            head_obj = max(heads, key=lambda h: h["conf"])
            hx, hy = head_obj["center"]
            hx1, hy1, _, _ = head_obj["box"]
            cv2.circle(frame, (hx, hy), 8, COLORS["head"], -1)
            draw_clear_text(frame, f"Head: {head_obj['conf']:.2f}", (hx1, hy1 - 6), COLORS["head"])
            theta_head = np.arctan2(hy - bead_cy, hx - bead_cx)

        if len(tails) > 0:
            tail_obj = max(tails, key=lambda t: t["conf"])
            tx, ty = tail_obj["center"]
            tx1, ty1, _, _ = tail_obj["box"]
            cv2.circle(frame, (tx, ty), 8, COLORS["tail"], -1)
            draw_clear_text(frame, f"Tail: {tail_obj['conf']:.2f}", (tx1, ty1 - 6), COLORS["tail"])
            theta_tail = np.arctan2(ty - bead_cy, tx - bead_cx)

        # -----------------------------------------------------------------
        # Angular Boundary check to isolate overlapping area
        # -----------------------------------------------------------------
        overlap_clamps = []
        if theta_head is not None and theta_tail is not None:
            t_min = min(theta_head, theta_tail)
            t_max = max(theta_head, theta_tail)
            is_cross_boundary = (t_max - t_min) > np.pi

            for clamp in filtered_clamps:
                if not is_cross_boundary:
                    in_overlap = (t_min <= clamp["theta"] <= t_max)
                else:
                    in_overlap = (clamp["theta"] >= t_max or clamp["theta"] <= t_min)

                if in_overlap:
                    overlap_clamps.append(clamp)

            overlap_clamps_count_text = str(len(overlap_clamps))

        # -----------------------------------------------------------------
        # Safety Logic: Enforce strict 4-clamp constraint in overlap zone
        # Only proceed with calculations if clamp count is exactly 4
        # -----------------------------------------------------------------
        if theta_head is not None and theta_tail is not None and len(overlap_clamps) == 4:
            # 1. Calculate arc length between Head and Tail
            delta_theta = np.abs(theta_head - theta_tail)
            if delta_theta > np.pi:
                delta_theta = 2 * np.pi - delta_theta
            arc_length_head_tail = bead_radius * delta_theta
            arc_length_head_tail_text = f"{int(arc_length_head_tail)} px"

            # 2. Draw guidance lines from bead center to Head and Tail
            cv2.line(frame, (bead_cx, bead_cy), (hx, hy), COLORS["head"], 2, cv2.LINE_AA)
            cv2.line(frame, (bead_cx, bead_cy), (tx, ty), COLORS["tail"], 2, cv2.LINE_AA)

            # 3. Calculate spacing between consecutive clamps
            if not is_cross_boundary:
                overlap_clamps_sorted = sorted(overlap_clamps, key=lambda c: c["theta"])
            else:
                overlap_clamps_sorted = sorted(overlap_clamps, key=lambda c: (c["theta"] if c["theta"] >= 0 else c["theta"] + 2*np.pi))

            arcs = []
            for i in range(len(overlap_clamps_sorted) - 1):
                c1_theta = overlap_clamps_sorted[i]["theta"]
                c2_theta = overlap_clamps_sorted[i+1]["theta"]

                d_theta = np.abs(c2_theta - c1_theta)
                if d_theta > np.pi:
                    d_theta = 2 * np.pi - d_theta

                arc_dist = bead_radius * d_theta
                arcs.append(arc_dist)

                # Draw connecting lines between clamps
                ccx1, ccy1 = overlap_clamps_sorted[i]["center"]
                ccx2, ccy2 = overlap_clamps_sorted[i+1]["center"]
                cv2.line(frame, (ccx1, ccy1), (ccx2, ccy2), COLORS["overlap_clamp"], 2, cv2.LINE_AA)

            if len(arcs) >= 1: arc_c1_c2 = f"{int(arcs[0])} px"
            if len(arcs) >= 2: arc_c2_c3 = f"{int(arcs[1])} px"
            if len(arcs) >= 3: arc_c3_c4 = f"{int(arcs[2])} px"

            # 4. Highlight valid overlap clamps
            for oc in overlap_clamps_sorted:
                x1, y1, x2, y2 = oc["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["overlap_clamp"], 3)
                draw_clear_text(frame, f"C: {oc['conf']:.2f}", (x1, y1 - 6), COLORS["overlap_clamp"])
        
        elif theta_head is not None and theta_tail is not None and len(overlap_clamps) > 4:
            # Handle outlier condition
            arc_length_head_tail_text = "Invalid (Clamps > 4)"

    # Draw remaining clamps (General purpose visualization)
    for clamp in filtered_clamps:
        x1, y1, x2, y2 = clamp["box"]
        if frame[y1+5, x1+5][1] != 255: 
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["clamp"], 2)
            draw_clear_text(frame, f"C: {clamp['conf']:.2f}", (x1, y1 - 6), COLORS["clamp"])

    # =====================================================================
    # 5. DYNAMIC TELEMETRY DASHBOARD OVERLAY UI (FRONT)
    # =====================================================================
    box_x, box_y = width - 360, 20
    box_w, box_h = 340, 480

    # Draw semi-transparent overlay box
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_y = box_y + 30

    # Render Telemetry Data
    cv2.putText(frame, "Front Camera Analysis", (box_x + 15, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (box_x + 15, text_y + 8), (box_x + 325, text_y + 8), (100, 100, 100), 1)
    text_y += 40

    cv2.putText(frame, f"Total Clamps: {total_clamps_count}", (box_x + 20, text_y), font, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    text_y += 30
    cv2.putText(frame, f"Head Count: {head_count}  |  Tail Count: {tail_count}", (box_x + 20, text_y), font, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    text_y += 35

    cv2.putText(frame, "Bead Geometry (Hidden)", (box_x + 15, text_y), font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (box_x + 15, text_y + 5), (box_x + 325, text_y + 5), (70, 70, 70), 1)
    text_y += 30
    cv2.putText(frame, f"Center Coords: {bead_center_text}", (box_x + 20, text_y), font, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    text_y += 25
    cv2.putText(frame, f"Bead Radius: {bead_radius_text}", (box_x + 20, text_y), font, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    text_y += 40

    cv2.putText(frame, "Overlap & Arc Analysis", (box_x + 15, text_y), font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (box_x + 15, text_y + 5), (box_x + 325, text_y + 5), (70, 70, 70), 1)
    text_y += 30
    cv2.putText(frame, f"Head-to-Tail Arc: {arc_length_head_tail_text}", (box_x + 20, text_y), font, 0.55, (255, 182, 193), 1, cv2.LINE_AA)
    text_y += 30
    cv2.putText(frame, f"Clamps in Overlap: {overlap_clamps_count_text} / 4", (box_x + 20, text_y), font, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    text_y += 35

    cv2.putText(frame, f"Arc C1 -> C2: {arc_c1_c2}", (box_x + 20, text_y), font, 0.5, (200, 240, 200), 1, cv2.LINE_AA)
    text_y += 25
    cv2.putText(frame, f"Arc C2 -> C3: {arc_c2_c3}", (box_x + 20, text_y), font, 0.5, (200, 240, 200), 1, cv2.LINE_AA)
    text_y += 25
    cv2.putText(frame, f"Arc C3 -> C4: {arc_c3_c4}", (box_x + 20, text_y), font, 0.5, (200, 240, 200), 1, cv2.LINE_AA)

    out.write(frame)

cap.release()
out.release()
print("\nDone! The system will now discard any overlap calculations if the clamp count between Head and Tail exceeds 4.")