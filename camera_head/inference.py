import cv2
import numpy as np
from ultralytics import YOLO

# =====================================================================
# 1. INITIALIZATION & MODEL SETUP
# =====================================================================
model_path = 'models/best.pt'
model = YOLO(model_path)
video_test_path = '/content/cut_video.mp4'

colors_legend = {
    "Region 1": (255, 0, 0),    # Blue
    "Region 2": (0, 255, 255),  # Yellow / Cyan
    "Region 3": (0, 0, 255),    # Red
    "Wire":     (0, 255, 0)     # Green
}

# =====================================================================
# 2. VIDEO STREAM PROPERTIES & WRITER GENERATION
# =====================================================================
cap = cv2.VideoCapture(video_test_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = 'wires_measurement_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("Processing video with VERTICAL Perpendicular Distance Logic...")

# =====================================================================
# 3. REAL-TIME INFERENCE FRAME-BY-FRAME STREAM
# =====================================================================
results = model.predict(
    source=video_test_path,
    conf=0.6,
    stream=True
)

for r in results:
    frame = r.orig_img.copy()
    
    all_regions = []      # Global collection for safety guard scoring
    wires = []            # Head wires tracking list
    
    dist_clamp_to_shortest_text = "0 px"
    dist_shortest_to_longest_text = "0 px"
    region3_height_text = "0 px"  
    
    # -----------------------------------------------------------------
    # 3a. BOUNDING BOX EXTRACTION & CLASS SEGREGATION
    # -----------------------------------------------------------------
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_score = float(box.conf[0])
        cls_id = int(box.cls[0])
        
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        if cls_id == 0: color = (255, 0, 0)      # Blue (Region 1)
        elif cls_id == 1: color = (0, 255, 255)  # Yellow (Region 2)
        elif cls_id == 2: color = (0, 0, 255)    # Red (Region 3)
        elif cls_id == 3: color = (0, 255, 0)    # Green (Wires)
        else: color = (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        conf_text = f"{conf_score:.2f}"
        cv2.putText(frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if cls_id in [0, 1, 2]:
            all_regions.append({"center": (cx, cy), "cls_id": cls_id})
            
            if cls_id == 2:
                r3_height = y2 - y1
                region3_height_text = f"{r3_height} px"
                
        elif cls_id == 3:
            wires.append({"center": (cx, cy)})

    wire_count = len(wires)

    # =====================================================================
    # 4. ADVANCED LOGIC: VERTICAL PERPENDICULAR MEASUREMENTS
    # =====================================================================
    if wire_count >= 7 and len(all_regions) > 0:
        
        wire_centers = [w["center"] for w in wires]
        avg_wire_x = int(np.mean([pt[0] for pt in wire_centers]))
        avg_wire_y = int(np.mean([pt[1] for pt in wire_centers]))
        
        # حارس الأمن يستمر في استخدام فيثاغورس كمعيار قطري عام لمعرفة القرب الإجمالي
        regions_with_dist = []
        for reg in all_regions:
            rcx, rcy = reg["center"]
            d = np.sqrt((rcx - avg_wire_x)**2 + (rcy - avg_wire_y)**2)
            regions_with_dist.append((d, reg))
            
        regions_with_dist.sort(key=lambda x: x[0])
        nearest_distance = regions_with_dist[0][0]
        
        MAX_ALLOWED_CLAMP_DISTANCE = 300
        
        if nearest_distance <= MAX_ALLOWED_CLAMP_DISTANCE:
            filtered_regions = [reg for d, reg in regions_with_dist if d - nearest_distance < 150]
            active_r1_elements = [reg for reg in filtered_regions if reg["cls_id"] == 0]
            
            if len(active_r1_elements) > 0:
                r1_centers = [r["center"] for r in active_r1_elements]
                r1_cx = int(np.mean([pt[0] for pt in r1_centers]))
                r1_cy = int(np.mean([pt[1] for pt in r1_centers]))
                
                # ترتيب الأسلاك بناءً على البعد الرأسي العمودي (Y) عن الـ Region 1
                wire_distances = []
                for wire in wires:
                    wcx, wcy = wire["center"]
                    # حساب المسافة العمودية على المحور Y
                    perp_dist_y = abs(r1_cy - wcy)
                    wire_distances.append((perp_dist_y, wire))
                    
                wire_distances.sort(key=lambda x: x[0])
                
                shortest_wire = wire_distances[0][1]
                longest_wire = wire_distances[-1][1]
                
                sw_cx, sw_cy = shortest_wire["center"]
                lw_cx, lw_cy = longest_wire["center"]
                
                # --- الحساب الرأسي العمودي الجديد (فرق إحداثيات Y) ---
                dist_r1_to_shortest = int(wire_distances[0][0])     # المسافة الرأسية لأول سلك
                dist_shortest_to_longest = int(abs(sw_cy - lw_cy))  # الارتفاع الرأسي بين طرفي الحزمة
                
                dist_clamp_to_shortest_text = f"{dist_r1_to_shortest} px"
                dist_shortest_to_longest_text = f"{dist_shortest_to_longest} px"
                
                # رسم الخط العمودي الرأسي (البرتقالي) بتثبيت محور X ليكون رأسي تماماً
                cv2.line(frame, (r1_cx, r1_cy), (r1_cx, sw_cy), (0, 165, 255), 3)
                cv2.circle(frame, (r1_cx, sw_cy), 7, (0, 165, 255), -1)
                cv2.putText(frame, dist_clamp_to_shortest_text, (r1_cx + 10, int((r1_cy + sw_cy) / 2)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                # في حال غياب ريجون 1، ترتيب الأسلاك رأسياً لحساب المدى الرأسي للحزمة
                wire_y_sorted = sorted(wire_centers, key=lambda p: p[1])
                sw_cx, sw_cy = wire_y_sorted[0]
                lw_cx, lw_cy = wire_y_sorted[-1]
                dist_shortest_to_longest = int(abs(sw_cy - lw_cy))
                dist_shortest_to_longest_text = f"{dist_shortest_to_longest} px"
        else:
            wire_y_sorted = sorted(wire_centers, key=lambda p: p[1])
            sw_cx, sw_cy = wire_y_sorted[0]
            lw_cx, lw_cy = wire_y_sorted[-1]
            dist_shortest_to_longest = int(abs(sw_cy - lw_cy))
            dist_shortest_to_longest_text = f"{dist_shortest_to_longest} px"

        # رسم الخط العمودي الرأسي (البنفسجي) لعرض حزمة الأسلاك رأسياً
        if len(wires) >= 2:
            # استخدام إحداثي X موحد (منتصف المسافة أفقياً) ليخرج الخط رأسياً مستقيماً تماماً
            mid_x = int((sw_cx + lw_cx) / 2)
            cv2.line(frame, (mid_x, sw_cy), (mid_x, lw_cy), (255, 0, 255), 3)
            cv2.circle(frame, (mid_x, lw_cy), 7, (255, 0, 255), -1)
            cv2.putText(frame, dist_shortest_to_longest_text, (mid_x + 10, int((sw_cy + lw_cy) / 2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # =====================================================================
    # 5. DYNAMIC DASHBOARD OVERLAY UI
    # =====================================================================
    box_x, box_y = width - 340, 20
    box_w, box_h = 320, 480
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_y = box_y + 35
    
    cv2.putText(frame, "Labels Legend", (box_x + 15, text_y), font, 0.75, (255, 255, 255), 2)
    cv2.line(frame, (box_x + 15, text_y + 8), (box_x + 305, text_y + 8), (100, 100, 100), 1)
    text_y += 45
    
    for name, color in colors_legend.items():
        cv2.rectangle(frame, (box_x + 20, text_y - 18), (box_x + 45, text_y + 7), color, -1)
        cv2.putText(frame, name, (box_x + 60, text_y), font, 0.7, (240, 240, 240), 2)
        text_y += 40
        
    text_y += 10
    cv2.putText(frame, "Live Telemetry", (box_x + 15, text_y), font, 0.75, (0, 255, 255), 2)
    cv2.line(frame, (box_x + 15, text_y + 8), (box_x + 305, text_y + 8), (100, 100, 100), 1)
    text_y += 45
    
    cv2.rectangle(frame, (box_x + 20, text_y - 15), (box_x + 35, text_y), (0, 165, 255), -1)
    cv2.putText(frame, f"R1-to-Short: {dist_clamp_to_shortest_text}", (box_x + 50, text_y), font, 0.65, (240, 240, 240), 2)
    text_y += 40
    
    cv2.rectangle(frame, (box_x + 20, text_y - 15), (box_x + 35, text_y), (255, 0, 255), -1)
    cv2.putText(frame, f"Short-to-Long: {dist_shortest_to_longest_text}", (box_x + 50, text_y), font, 0.65, (240, 240, 240), 2)
    text_y += 40

    cv2.rectangle(frame, (box_x + 20, text_y - 15), (box_x + 35, text_y), (0, 0, 255), -1)
    cv2.putText(frame, f"R3 Height: {region3_height_text}", (box_x + 50, text_y), font, 0.65, (240, 240, 240), 2)
    text_y += 40

    cv2.rectangle(frame, (box_x + 20, text_y - 15), (box_x + 35, text_y), (0, 255, 0), -1)
    cv2.putText(frame, f"Wires Count: {wire_count}", (box_x + 50, text_y), font, 0.65, (240, 240, 240), 2)

    out.write(frame)

# =====================================================================
# 6. RESOURCE CLEANUP
# =====================================================================
cap.release()
out.release()
print("\nSuccess! Code updated to calculate and overlay absolute VERTICAL perpendicular distances.")