import cv2
import numpy as np
from ultralytics import YOLO

# =====================================================================
# 1. INITIALIZATION & MODEL SETUP
# =====================================================================
# Define model path and input video source
model_path = 'models/best.pt'
model = YOLO(model_path)
video_test_path = '/content/cut_video.mp4'

# Define label legend and color coding
colors_legend = {
    "Region 1": (255, 0, 0),    # Blue for Clamp
    "Wire":     (0, 255, 0)     # Green for Tail_wire
}

# =====================================================================
# 2. VIDEO STREAM PROPERTIES & WRITER GENERATION
# =====================================================================
cap = cv2.VideoCapture(video_test_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter to save processed footage
output_path = 'tail_wires_measurement_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("Processing video with TAIL WIRE Double-Detection Filter Logic...")

# =====================================================================
# 3. REAL-TIME INFERENCE FRAME-BY-FRAME STREAM
# =====================================================================
results = model.predict(
    source=video_test_path,
    conf=0.49, # Confidence threshold for detection
    stream=True
)

for r in results:
    frame = r.orig_img.copy()
    
    clamps = []           # List to store Clamps
    raw_wires = []        # List to store raw Tail Wires detections
    
    # Initialization of telemetry strings
    dist_clamp_to_shortest_text = "0 px"
    dist_shortest_to_longest_text = "0 px"
    
    # -----------------------------------------------------------------
    # 3a. BOUNDING BOX EXTRACTION & CLASS SEGREGATION
    # -----------------------------------------------------------------
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_score = float(box.conf[0])
        cls_id = int(box.cls[0])
        
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        # Classify detections based on ID
        if cls_id == 1: 
            clamps.append({"center": (cx, cy), "box": (x1, y1, x2, y2), "conf": conf_score})
        elif cls_id == 0: 
            raw_wires.append({"center": (cx, cy), "box": (x1, y1, x2, y2), "conf": conf_score})

    # =====================================================================
    # 4. ANTI DOUBLE-DETECTION FILTER (X-AXIS NMS)
    # =====================================================================
    wires = []
    if len(raw_wires) > 0:
        # Sort wires horizontally to suppress duplicate detections
        raw_wires_sorted_x = sorted(raw_wires, key=lambda w: w["center"][0])
        
        MIN_X_GAP = 20  # Minimum pixel gap between distinct wires
        
        for w in raw_wires_sorted_x:
            if not wires:
                wires.append(w)
            else:
                # Check for overlap/duplicate in X-axis
                if abs(w["center"][0] - wires[-1]["center"][0]) > MIN_X_GAP:
                    wires.append(w)
                else:
                    # Keep detection with higher confidence
                    if w["conf"] > wires[-1]["conf"]:
                        wires[-1] = w

    wire_count = len(wires)

    # -----------------------------------------------------------------
    # 3b. DRAW VISUAL BOUNDING BOXES FOR FILTERED OBJECTS
    # -----------------------------------------------------------------
    # Draw Clamps (Region 1)
    for clamp in clamps:
        x1, y1, x2, y2 = clamp["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(frame, f"{clamp['conf']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
    # Draw valid Filtered Wires
    for wire in wires:
        x1, y1, x2, y2 = wire["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"{wire['conf']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # =====================================================================
    # 5. TAIL MEASUREMENT LOGIC (TRIGGERED IF WIRES > 5)
    # =====================================================================
    if wire_count == 7:
        # 1. Sort wires vertically by Y-coordinate
        wire_distances = sorted(wires, key=lambda w: w["center"][1])
        
        highest_wire = wire_distances[0]   # Topmost wire
        lowest_wire = wire_distances[-1]   # Bottom-most wire
        
        hw_cx, hw_cy = highest_wire["center"]
        lw_cx, lw_cy = lowest_wire["center"]
        
        # Calculate bundle height
        dist_shortest_to_longest = int(abs(hw_cy - lw_cy))
        dist_shortest_to_longest_text = f"{dist_shortest_to_longest} px"
        
        # Draw bundle measurement line
        mid_x = int((hw_cx + lw_cx) / 2)
        cv2.line(frame, (mid_x, hw_cy), (mid_x, lw_cy), (255, 255, 0), 3)
        cv2.circle(frame, (mid_x, lw_cy), 7, (255, 255, 0), -1)
        cv2.putText(frame, dist_shortest_to_longest_text, (mid_x + 10, int((hw_cy + lw_cy) / 2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 2. Calculate vertical distance from Clamp to bottom wire
        if len(clamps) > 0:
            clamp_distances = []
            MAX_ALLOWED_X_GAP = 300 
            
            for clamp in clamps:
                ccx, ccy = clamp["center"]
                x_gap = abs(ccx - lw_cx)
                
                # Check proximity in X-axis to identify relevant clamp
                if x_gap <= MAX_ALLOWED_X_GAP:
                    perp_dist_y = abs(ccy - lw_cy)
                    clamp_distances.append((perp_dist_y, clamp))
            
            if len(clamp_distances) > 0:
                # Find nearest clamp
                clamp_distances.sort(key=lambda x: x[0])
                nearest_clamp_dist, nearest_clamp = clamp_distances[0]
                nc_cx, nc_cy = nearest_clamp["center"]
                
                dist_clamp_to_shortest_text = f"{int(nearest_clamp_dist)} px"
                
                # Draw measurement line from Clamp to Wire
                cv2.line(frame, (nc_cx, nc_cy), (nc_cx, lw_cy), (0, 165, 255), 3)
                cv2.circle(frame, (nc_cx, lw_cy), 7, (0, 165, 255), -1)
                cv2.putText(frame, dist_clamp_to_shortest_text, (nc_cx + 10, int((nc_cy + lw_cy) / 2)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                dist_clamp_to_shortest_text = "N/A (Out of Range)"

    # =====================================================================
    # 6. DYNAMIC DASHBOARD OVERLAY UI
    # =====================================================================
    box_x, box_y = width - 340, 20
    box_w, box_h = 320, 360  
    
    # Render semi-transparent UI dashboard
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_y = box_y + 35
    
    cv2.putText(frame, "Labels Legend", (box_x + 15, text_y), font, 0.75, (255, 255, 255), 2)
    cv2.line(frame, (box_x + 15, text_y + 8), (box_x + 305, text_y + 8), (100, 100, 100), 1)
    text_y += 45
    
    # Populate Legend
    for name, color in colors_legend.items():
        cv2.rectangle(frame, (box_x + 20, text_y - 18), (box_x + 45, text_y + 7), color, -1)
        cv2.putText(frame, name, (box_x + 60, text_y), font, 0.7, (240, 240, 240), 2)
        text_y += 40
        
    text_y += 10
    cv2.putText(frame, "Live Telemetry", (box_x + 15, text_y), font, 0.75, (0, 255, 255), 2)
    cv2.line(frame, (box_x + 15, text_y + 8), (box_x + 305, text_y + 8), (100, 100, 100), 1)
    text_y += 45
    
    # Display Metrics
    cv2.rectangle(frame, (box_x + 20, text_y - 15), (box_x + 35, text_y), (0, 165, 255), -1)
    cv2.putText(frame, f"R1-to-Short: {dist_clamp_to_shortest_text}", (box_x + 50, text_y), font, 0.65, (240, 240, 240), 2)
    text_y += 40
    
    cv2.rectangle(frame, (box_x + 20, text_y - 15), (box_x + 35, text_y), (255, 255, 0), -1)
    cv2.putText(frame, f"Short-to-Long: {dist_shortest_to_longest_text}", (box_x + 50, text_y), font, 0.65, (240, 240, 240), 2)
    text_y += 40

    cv2.rectangle(frame, (box_x + 20, text_y - 15), (box_x + 35, text_y), (0, 255, 0), -1)
    cv2.putText(frame, f"Wires Count: {wire_count}", (box_x + 50, text_y), font, 0.65, (240, 240, 240), 2)

    out.write(frame)

# =====================================================================
# 7. RESOURCE CLEANUP
# =====================================================================
cap.release()
out.release()
print("\nSuccess! Code updated with X-Axis Suppression to handle Double Detection.")