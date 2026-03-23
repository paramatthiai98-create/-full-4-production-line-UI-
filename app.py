import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="SmartSafe Factory 4-Line Dashboard", layout="wide")

DEFAULT_PERSON_MODEL = "yolov8n.pt"
DEFAULT_HELMET_MODEL = "best.pt"
DEFAULT_IMAGE_PATH = ""

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #08111f 0%, #0b1728 100%); color: #f8fafc; }
.block-container { max-width: 96rem; padding-top: 0.9rem; padding-bottom: 1rem; }
.main-title { font-size: 2.15rem; font-weight: 800; color: #ffffff; margin-bottom: 0.2rem; }
.sub-title { color: #9ca3af; margin-bottom: 1rem; }
.panel { background: rgba(10, 19, 34, 0.96); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 16px 18px; margin-bottom: 16px; }
.card { background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(9,18,32,0.98)); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 16px; min-height: 132px; }
.grid-card { background: linear-gradient(180deg, rgba(13,20,37,0.96), rgba(10,16,28,0.98)); border: 1px solid rgba(255,255,255,0.07); border-radius: 18px; padding: 16px; min-height: 170px; }
.section-title { font-size: 1.04rem; font-weight: 700; color: #e5e7eb; margin-bottom: 0.7rem; }
.metric-value { font-size: 1.9rem; font-weight: 800; color: #ffffff; line-height: 1.1; }
.small-item { font-size: 0.95rem; color: #e5e7eb; margin: 0.25rem 0; }
.badge { display: inline-block; padding: 5px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; margin-top: 0.4rem; }
.badge-safe { background: rgba(22,163,74,0.16); color: #86efac; border: 1px solid rgba(22,163,74,0.35); }
.badge-warn { background: rgba(245,158,11,0.16); color: #fcd34d; border: 1px solid rgba(245,158,11,0.35); }
.badge-risk { background: rgba(239,68,68,0.16); color: #fca5a5; border: 1px solid rgba(239,68,68,0.35); }
.alert-box { border-radius: 16px; padding: 14px 16px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.08); }
.alert-safe { background: rgba(22,163,74,0.10); color: #bbf7d0; }
.alert-warn { background: rgba(245,158,11,0.10); color: #fde68a; }
.alert-risk { background: rgba(239,68,68,0.10); color: #fecaca; }
.info-box { border-radius: 14px; padding: 12px 14px; background: rgba(59,130,246,0.10); border: 1px solid rgba(59,130,246,0.30); color: #dbeafe; margin-bottom: 14px; }
</style>
""", unsafe_allow_html=True)


def file_exists(path_str: str) -> bool:
    return bool(path_str) and Path(path_str).exists()


@st.cache_resource(show_spinner=False)
def load_model_safe(model_path: str):
    try:
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


def safe_model_names(model) -> dict:
    if model is None:
        return {}
    names = getattr(model, "names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return names


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix if uploaded_file and uploaded_file.name else ".jpg"
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded_file.read())
    tfile.flush()
    return tfile.name


def calc_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter_area / float(area_a + area_b - inter_area)


def center_distance(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def classify_helmet_label(label: str):
    label = str(label).lower().strip()
    helmet_words = ["helmet", "hardhat", "hard-hat", "hat"]
    no_helmet_words = ["no_helmet", "no helmet", "without helmet", "barehead", "bare_head", "head"]
    if any(w in label for w in no_helmet_words):
        return "no_helmet"
    if any(w in label for w in helmet_words):
        return "helmet"
    return None


def associate_person_with_head_detection(person_box, helmet_dets, no_helmet_dets):
    px1, py1, px2, py2 = person_box
    person_h = max(1, py2 - py1)
    head_roi = (px1, py1, px2, int(py1 + person_h * 0.42))

    best_type = "unknown"
    best_score = 0.0
    best_box = None

    for det_type, dets in (("helmet", helmet_dets), ("no_helmet", no_helmet_dets)):
        for det in dets:
            iou = calc_iou(head_roi, det)
            if iou <= 0:
                dist = center_distance(head_roi, det)
                diag = ((head_roi[2] - head_roi[0]) ** 2 + (head_roi[3] - head_roi[1]) ** 2) ** 0.5
                proximity_score = max(0.0, 1.0 - dist / max(1.0, diag * 1.3))
                score = proximity_score * 0.35
            else:
                score = iou + 0.25

            if score > best_score:
                best_score = score
                best_type = det_type
                best_box = det

    if best_score < 0.12:
        return "unknown", None
    return best_type, best_box


def get_line_zone(box, width, height):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if cx < width / 2 and cy < height / 2:
        return "Line 1"
    if cx >= width / 2 and cy < height / 2:
        return "Line 2"
    if cx < width / 2 and cy >= height / 2:
        return "Line 3"
    return "Line 4"


def estimate_distance_cm(person_box, width, height):
    zone = get_line_zone(person_box, width, height)
    anchors = {
        "Line 1": int(width * 0.43),
        "Line 2": int(width * 0.93),
        "Line 3": int(width * 0.43),
        "Line 4": int(width * 0.93),
    }
    x1, _, x2, _ = person_box
    person_center_x = (x1 + x2) / 2
    person_width = max(1, x2 - x1)
    pixel_distance = abs(anchors[zone] - person_center_x)
    return int(min(150, max(5, pixel_distance / person_width * 35)))


def calculate_risk(helmet_status: str, distance_cm: int):
    risk = 0
    reasons = []

    if helmet_status == "no_helmet":
        risk += 50
        reasons.append("No helmet detected")
    elif helmet_status == "unknown":
        risk += 18
        reasons.append("Helmet status uncertain")

    if distance_cm < 30:
        risk += 35
        reasons.append("Worker too close to machine")
    elif distance_cm < 45:
        risk += 20
        reasons.append("Worker near machine")

    return min(100, risk), reasons


def decision_logic(risk: int):
    if risk >= 75:
        return "RED", "STOP LINE"
    if risk >= 45:
        return "YELLOW", "CHECK LINE"
    return "GREEN", "NORMAL"


def render_status_badge(status: str):
    if status == "GREEN":
        return '<span class="badge badge-safe">GREEN</span>'
    if status == "YELLOW":
        return '<span class="badge badge-warn">YELLOW</span>'
    return '<span class="badge badge-risk">RED</span>'


def render_alert_box(status: str, text: str):
    css = "alert-box alert-safe" if status == "GREEN" else ("alert-box alert-warn" if status == "YELLOW" else "alert-box alert-risk")
    return f'<div class="{css}">{text}</div>'


def build_line_data():
    return {
        "Line 1": {"helmet": 0, "no_helmet": 0, "unknown": 0, "risk_total": 0, "people": 0, "reasons": []},
        "Line 2": {"helmet": 0, "no_helmet": 0, "unknown": 0, "risk_total": 0, "people": 0, "reasons": []},
        "Line 3": {"helmet": 0, "no_helmet": 0, "unknown": 0, "risk_total": 0, "people": 0, "reasons": []},
        "Line 4": {"helmet": 0, "no_helmet": 0, "unknown": 0, "risk_total": 0, "people": 0, "reasons": []},
    }


def add_line_overlay(annotated, w, h):
    cv2.line(annotated, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
    cv2.line(annotated, (0, h // 2), (w, h // 2), (255, 255, 255), 2)

    cv2.putText(annotated, "LINE 1", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(annotated, "LINE 2", (w // 2 + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(annotated, "LINE 3", (20, h // 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(annotated, "LINE 4", (w // 2 + 20, h // 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    machine_boxes = [
        ("L1 MACHINE", (int(w * 0.33), int(h * 0.10)), (int(w * 0.46), int(h * 0.42))),
        ("L2 MACHINE", (int(w * 0.83), int(h * 0.10)), (int(w * 0.96), int(h * 0.42))),
        ("L3 MACHINE", (int(w * 0.33), int(h * 0.58)), (int(w * 0.46), int(h * 0.92))),
        ("L4 MACHINE", (int(w * 0.83), int(h * 0.58)), (int(w * 0.96), int(h * 0.92))),
    ]
    for label, p1, p2 in machine_boxes:
        cv2.rectangle(annotated, p1, p2, (255, 170, 0), 2)
        cv2.putText(annotated, label, (p1[0], max(25, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 170, 0), 2)


def build_alerts(line_data):
    alerts = []
    for line, data in line_data.items():
        avg_risk = int(data["risk_total"] / data["people"]) if data["people"] else 0
        status, action = decision_logic(avg_risk)
        reasons = list(dict.fromkeys(data["reasons"]))

        if status == "RED":
            msg = f"🚨 {line}: RED — {action}. " + ("Reason: " + ", ".join(reasons[:2]) if reasons else "High risk detected.")
            alerts.append((status, msg))
        elif status == "YELLOW":
            msg = f"⚠️ {line}: YELLOW — {action}. " + ("Reason: " + ", ".join(reasons[:2]) if reasons else "Please inspect this line.")
            alerts.append((status, msg))

    if not alerts:
        alerts.append(("GREEN", "✅ All 4 production lines are in normal condition."))
    return alerts


st.sidebar.title("⚙️ Factory Control Panel")
image_source = st.sidebar.radio("Image Source", ["Upload image", "Use sample image"], index=0)
uploaded_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
sample_image_path = st.sidebar.text_input("Sample image path", value=DEFAULT_IMAGE_PATH)
helmet_model_path = st.sidebar.text_input("Helmet model path", value=DEFAULT_HELMET_MODEL)
person_conf = st.sidebar.slider("Person confidence", 0.10, 0.95, 0.25, 0.05)
helmet_conf = st.sidebar.slider("Helmet confidence", 0.10, 0.95, 0.25, 0.05)
enable_line_notify = st.sidebar.toggle("Enable LINE Notify message preview", value=True)
notify_token = st.sidebar.text_input("LINE Notify token / Messaging token", value="", type="password")
analyze_btn = st.sidebar.button("🔍 Analyze Factory", use_container_width=True)

st.markdown('<div class="main-title">SmartSafe Factory 4-Line Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Fallback mode: ถ้ายังไม่มี best.pt แอปก็ยังเปิดได้ และใช้ detect คนก่อน</div>', unsafe_allow_html=True)

person_model, person_model_error = load_model_safe(DEFAULT_PERSON_MODEL)
helmet_model, helmet_model_error = load_model_safe(helmet_model_path) if file_exists(helmet_model_path) else (None, "Helmet model file not found")

if person_model_error:
    st.error(f"โหลด person model ไม่ได้: {person_model_error}")

if helmet_model_error:
    st.markdown(
        f'<div class="info-box">ตอนนี้ยังใช้ <b>helmet fallback mode</b> อยู่ เพราะโหลดโมเดลหมวกไม่ได้<br><br>สาเหตุ: {helmet_model_error}</div>',
        unsafe_allow_html=True
    )

image_path = None
if image_source == "Upload image":
    if uploaded_image is not None:
        image_path = save_uploaded_file(uploaded_image)
    else:
        st.info("กรุณาอัปโหลดรูปภาพสำหรับโรงงาน")
else:
    if file_exists(sample_image_path):
        image_path = sample_image_path
    else:
        st.info("กรุณาใส่ sample image path ให้ถูกต้อง")

if analyze_btn:
    if person_model is None:
        st.error("ยังไม่สามารถโหลด person model ได้")
    elif not image_path:
        st.error("ยังไม่มีรูปภาพสำหรับวิเคราะห์")
    else:
        frame = cv2.imread(image_path)
        if frame is None:
            st.error("ไม่สามารถเปิดรูปภาพได้")
        else:
            h, w = frame.shape[:2]
            person_res = person_model.predict(frame, conf=person_conf, verbose=False)[0]
            person_names = safe_model_names(person_model)

            person_boxes = []
            for box in person_res.boxes:
                cls_id = int(box.cls[0])
                label = person_names.get(cls_id, str(cls_id)).lower()
                if label == "person":
                    person_boxes.append(tuple(map(int, box.xyxy[0].tolist())))

            helmet_boxes = []
            no_helmet_boxes = []

            if helmet_model is not None:
                helmet_res = helmet_model.predict(frame, conf=helmet_conf, verbose=False)[0]
                helmet_names = safe_model_names(helmet_model)

                for box in helmet_res.boxes:
                    cls_id = int(box.cls[0])
                    label = helmet_names.get(cls_id, str(cls_id))
                    kind = classify_helmet_label(label)
                    coords = tuple(map(int, box.xyxy[0].tolist()))
                    if kind == "helmet":
                        helmet_boxes.append(coords)
                    elif kind == "no_helmet":
                        no_helmet_boxes.append(coords)

            annotated = frame.copy()
            add_line_overlay(annotated, w, h)

            line_data = build_line_data()
            total_people = 0
            total_helmet = 0
            total_no_helmet = 0
            total_unknown = 0
            total_risk = 0

            for person_box in person_boxes:
                x1, y1, x2, y2 = person_box
                zone = get_line_zone(person_box, w, h)
                distance_cm = estimate_distance_cm(person_box, w, h)

                if helmet_model is not None:
                    helmet_status, matched_box = associate_person_with_head_detection(person_box, helmet_boxes, no_helmet_boxes)
                else:
                    helmet_status, matched_box = "unknown", None

                risk, reasons = calculate_risk(helmet_status, distance_cm)

                line_data[zone]["people"] += 1
                line_data[zone]["risk_total"] += risk
                line_data[zone]["reasons"].extend(reasons)

                total_people += 1
                total_risk += risk

                if helmet_status == "helmet":
                    line_data[zone]["helmet"] += 1
                    total_helmet += 1
                    color = (30, 200, 80)
                    label_text = f"{zone} | Helmet | {distance_cm} cm"
                elif helmet_status == "no_helmet":
                    line_data[zone]["no_helmet"] += 1
                    total_no_helmet += 1
                    color = (0, 0, 255)
                    label_text = f"{zone} | No Helmet | {distance_cm} cm"
                else:
                    line_data[zone]["unknown"] += 1
                    total_unknown += 1
                    color = (0, 200, 255)
                    label_text = f"{zone} | Person | {distance_cm} cm"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label_text, (x1, max(22, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                head_y2 = int(y1 + (y2 - y1) * 0.42)
                cv2.rectangle(annotated, (x1, y1), (x2, head_y2), (255, 255, 255), 1)

                if matched_box is not None:
                    mx1, my1, mx2, my2 = matched_box
                    cv2.rectangle(annotated, (mx1, my1), (mx2, my2), color, 2)

            overall_risk = int(total_risk / total_people) if total_people else 0
            overall_status, overall_action = decision_logic(overall_risk)
            alerts = build_alerts(line_data)

            left, right = st.columns([1.55, 1.05], gap="large")

            with left:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Factory Grid View</div>', unsafe_allow_html=True)
                st.image(annotated, channels="BGR", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with right:
                top1, top2, top3 = st.columns(3, gap="small")

                with top1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Factory Status</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{total_people}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="small-item">People detected</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="small-item">Helmet: <b>{total_helmet}</b></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="small-item">No helmet: <b>{total_no_helmet}</b></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="small-item">Unknown: <b>{total_unknown}</b></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with top2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Overall Risk</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{overall_risk}</div>', unsafe_allow_html=True)
                    st.markdown(render_status_badge(overall_status), unsafe_allow_html=True)
                    st.progress(overall_risk / 100)
                    st.markdown(f'<div class="small-item">Action: <b>{overall_action}</b></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with top3:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Model Info</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="small-item">Person model: <b>{DEFAULT_PERSON_MODEL}</b></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="small-item">Helmet model: <b>{"Loaded" if helmet_model is not None else "Fallback mode"}</b></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="small-item">Person conf: <b>{person_conf}</b></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">AI Alerts by Line</div>', unsafe_allow_html=True)
                for status, msg in alerts:
                    st.markdown(render_alert_box(status, msg), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">4 Production Lines</div>', unsafe_allow_html=True)

            cols = st.columns(4, gap="medium")
            for idx, line in enumerate(["Line 1", "Line 2", "Line 3", "Line 4"]):
                data = line_data[line]
                avg_risk = int(data["risk_total"] / data["people"]) if data["people"] else 0
                status, action = decision_logic(avg_risk)
                reasons = list(dict.fromkeys(data["reasons"]))

                with cols[idx]:
                    st.markdown('<div class="grid-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="section-title">{line}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="small-item">People: <b>{data["people"]}</b></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="small-item">Helmet: <b>{data["helmet"]}</b></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="small-item">No helmet: <b>{data["no_helmet"]}</b></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="small-item">Unknown: <b>{data["unknown"]}</b></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="small-item">Risk: <b>{avg_risk}</b></div>', unsafe_allow_html=True)
                    st.markdown(render_status_badge(status), unsafe_allow_html=True)
                    st.progress(avg_risk / 100)
                    st.markdown(f'<div class="small-item">Action: <b>{action}</b></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="small-item">Top reason: {reasons[0] if reasons else "No active risk detected"}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            compare1, compare2 = st.columns(2, gap="large")

            with compare1:
                compare_df = pd.DataFrame({
                    "Line": ["Line 1", "Line 2", "Line 3", "Line 4"],
                    "People": [line_data["Line 1"]["people"], line_data["Line 2"]["people"], line_data["Line 3"]["people"], line_data["Line 4"]["people"]],
                    "Unknown": [line_data["Line 1"]["unknown"], line_data["Line 2"]["unknown"], line_data["Line 3"]["unknown"], line_data["Line 4"]["unknown"]],
                }).set_index("Line")

                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Compare People Across 4 Lines</div>', unsafe_allow_html=True)
                st.bar_chart(compare_df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with compare2:
                risk_df = pd.DataFrame({
                    "Line": ["Line 1", "Line 2", "Line 3", "Line 4"],
                    "Risk Score": [
                        int(line_data["Line 1"]["risk_total"] / line_data["Line 1"]["people"]) if line_data["Line 1"]["people"] else 0,
                        int(line_data["Line 2"]["risk_total"] / line_data["Line 2"]["people"]) if line_data["Line 2"]["people"] else 0,
                        int(line_data["Line 3"]["risk_total"] / line_data["Line 3"]["people"]) if line_data["Line 3"]["people"] else 0,
                        int(line_data["Line 4"]["risk_total"] / line_data["Line 4"]["people"]) if line_data["Line 4"]["people"] else 0,
                    ]
                }).set_index("Line")

                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Compare Risk Across 4 Lines</div>', unsafe_allow_html=True)
                st.line_chart(risk_df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">LINE Alert Preview</div>', unsafe_allow_html=True)

            red_lines = []
            yellow_lines = []

            for line in ["Line 1", "Line 2", "Line 3", "Line 4"]:
                avg_risk = int(line_data[line]["risk_total"] / line_data[line]["people"]) if line_data[line]["people"] else 0
                status, action = decision_logic(avg_risk)
                if status == "RED":
                    red_lines.append((line, avg_risk, action))
                elif status == "YELLOW":
                    yellow_lines.append((line, avg_risk, action))

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if red_lines:
                preview = "🚨 SmartSafe Alert\n"
                preview += f"Time: {timestamp}\n"
                for line, risk, action in red_lines:
                    preview += f"- {line}: RED | Risk {risk} | {action}\n"
            elif yellow_lines:
                preview = "⚠️ SmartSafe Warning\n"
                preview += f"Time: {timestamp}\n"
                for line, risk, action in yellow_lines:
                    preview += f"- {line}: YELLOW | Risk {risk} | {action}\n"
            else:
                preview = f"✅ SmartSafe Update\nTime: {timestamp}\nAll 4 lines normal."

            if enable_line_notify:
                st.code(preview)
                if notify_token:
                    st.success("มี token แล้ว — พร้อมต่อยอดไปส่งจริงผ่าน LINE API")
                else:
                    st.info("ตอนนี้เป็น preview message ก่อน หากต้องการส่งจริง ให้ใส่ LINE token / Messaging API token")
            else:
                st.write("ปิด LINE preview อยู่")

            st.markdown('</div>', unsafe_allow_html=True)

            summary_df = pd.DataFrame({
                "Line": ["Line 1", "Line 2", "Line 3", "Line 4"],
                "People": [line_data["Line 1"]["people"], line_data["Line 2"]["people"], line_data["Line 3"]["people"], line_data["Line 4"]["people"]],
                "Helmet": [line_data["Line 1"]["helmet"], line_data["Line 2"]["helmet"], line_data["Line 3"]["helmet"], line_data["Line 4"]["helmet"]],
                "No Helmet": [line_data["Line 1"]["no_helmet"], line_data["Line 2"]["no_helmet"], line_data["Line 3"]["no_helmet"], line_data["Line 4"]["no_helmet"]],
                "Unknown": [line_data["Line 1"]["unknown"], line_data["Line 2"]["unknown"], line_data["Line 3"]["unknown"], line_data["Line 4"]["unknown"]],
                "Risk Score": [
                    int(line_data["Line 1"]["risk_total"] / line_data["Line 1"]["people"]) if line_data["Line 1"]["people"] else 0,
                    int(line_data["Line 2"]["risk_total"] / line_data["Line 2"]["people"]) if line_data["Line 2"]["people"] else 0,
                    int(line_data["Line 3"]["risk_total"] / line_data["Line 3"]["people"]) if line_data["Line 3"]["people"] else 0,
                    int(line_data["Line 4"]["risk_total"] / line_data["Line 4"]["people"]) if line_data["Line 4"]["people"] else 0,
                ]
            })

            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Production Summary Table</div>', unsafe_allow_html=True)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
### วิธีรัน
```bash
pip install -r requirements.txt
streamlit run app.py
