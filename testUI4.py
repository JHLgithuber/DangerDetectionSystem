"""
PyQt5‑based NVR GUI for multi‑camera monitoring with per‑camera AI (YOLOX) and polygonal area setting.
Unified control buttons are at the right‑hand side panel:
  • [영역 설정] – Toggle area‑input mode for the currently selected camera.
  • [되돌리기] – Remove the last point of the current polygon.
  • [설정 완료] – Close and fix the polygon (≥3 pts).
  • [영역 삭제] – Clear polygon for selected camera.

Update 2025‑07‑16 (#4):
  • Added robust **camera selection logic** so only one camera is selected at a time and the
    selection highlight does not accumulate.
  • `NVRGUI.select_camera()` maintains a single `selected_cam_id` and refreshes
    border styles of all `CameraWidget`s.
  • `CameraWidget` gets a `set_selected()` helper and emits selection on mouse press.
  • Polygon drawing is now re‑rendered fresh each frame to avoid thicker lines.
  • Added safeguards against clicks when no camera is selected.
"""

import math
import sys
import time
import traceback
from multiprocessing import Manager, Process, freeze_support
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QGridLayout,
    QCheckBox,
    QScrollArea,
    QSizePolicy,
    QMessageBox,
)

import fall_detection_adapt_layer
from yolo_run6 import YOLOWorldTRT, run_yolo_and_track


def camera_process(source, cam_id, viewer_queue, flags, io_queue):
    # yolox_model = YOLOX_TRT(engine_path="E:/YOLOX/yolox_custom.engine")
    yolo_model = YOLOWorldTRT(engine_path="yolov8x-worldv2.engine")

    cap = cv2.VideoCapture(source)
    prev_behavior = False
    prev_equipment = False

    # tracked_objects 변수를 루프 시작 전에 초기화
    tracked = []
    try:
        while True:
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                viewer_queue.put((cam_id, None, f"[카메라 {cam_id}] 프레임 수신 실패"))
                time.sleep(0.1)
                continue

            # 프레임 복사 (필요 시 AI용으로 별도 복사)
            processed_frame = frame.copy()

            # 낙상감지용 프레임 복사
            behavior_frame = frame.copy()

            current_behavior = flags[cam_id].get("behavior", False)
            current_equipment = flags[cam_id].get("equipment", False)

            # 상태 변경 감지: 메시지 1회만 출력
            if current_behavior != prev_behavior:
                if current_behavior:
                    viewer_queue.put((cam_id, None, f"[카메라 {cam_id}] 넘어짐 감지 실행"))
                else:
                    viewer_queue.put((cam_id, None, f"[카메라 {cam_id}] 넘어짐 감지 종료"))
                prev_behavior = current_behavior

            if current_equipment != prev_equipment:
                if current_equipment:
                    viewer_queue.put((cam_id, None, f"[카메라 {cam_id}] 안전장비 탐지 실행"))
                else:
                    viewer_queue.put((cam_id, None, f"[카메라 {cam_id}] 안전장비 탐지 종료"))
                prev_equipment = current_equipment

            # AI 처리
            if current_equipment:
                processed_frame = run_yolo_and_track(
                    processed_frame,
                    yolo_model,
                    conf=0.5,
                )

            # 낙상 감지
            processed_frame = fall_detection_adapt_layer.simple_detect(
                io_queue=io_queue,
                frame=behavior_frame,
                pre_processed_frame=processed_frame,
                need_detect=current_behavior
            )

            # 처리된 프레임 전달
            viewer_queue.put((cam_id, processed_frame, None))

            # 부하 방지

            proc_time = time.perf_counter() - start_time
            sleep_time = max(1 / 30 - proc_time, 0.000001)
            fps = 1 / (sleep_time + proc_time)
            print(f"{cam_id}\tFPS: {fps:.02f} ProcTime: {proc_time:.02f}  SleepTime: {sleep_time:.02f}")
            time.sleep(sleep_time)  # 30 FPS
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


# -------------------------------
# CameraWidget with selection & polygon handling
# -------------------------------
class CameraWidget(QWidget):
    def __init__(self, cam_id: int, shared_flags):
        super().__init__()
        self.cam_id = cam_id
        self.shared_flags = shared_flags
        self._points: list[QPoint] = []  # polygon points for this camera
        self._polygon_fixed = False
        self._selected = False
        self._init_ui()

    # --- UI & helper ---
    def _init_ui(self):
        self.label = QLabel(f"Camera {self.cam_id}")
        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # noinspection PyUnresolvedReferences
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.installEventFilter(self)  # capture mouse clicks

        self.toggle_behavior = QCheckBox("넘어짐 탐지 기능")
        self.toggle_equipment = QCheckBox("안전장비 착용유무 감지 기능")
        self.toggle_behavior.stateChanged.connect(self._update_flags)
        self.toggle_equipment.stateChanged.connect(self._update_flags)

        lay = QVBoxLayout()
        lay.addWidget(self.label)
        lay.addWidget(self.video_label)
        lay.addWidget(self.toggle_behavior)
        lay.addWidget(self.toggle_equipment)
        self.setLayout(lay)
        self._refresh_border()

    def set_selected(self, flag: bool):
        self._selected = flag
        self._refresh_border()

    def _refresh_border(self):
        if self._selected:
            self.setStyleSheet("border: 3px solid #00FF00;")
        else:
            self.setStyleSheet("")

    # --- Flags to child process ---
    def _update_flags(self):
        self.shared_flags[self.cam_id] = {
            "behavior": self.toggle_behavior.isChecked(),
            "equipment": self.toggle_equipment.isChecked(),
        }

    # --- Event filter for video label clicks ---
    def eventFilter(self, obj, event):
        if obj is self.video_label and event.type() == event.MouseButtonPress:
            window = self.window()
            if isinstance(window, NVRGUI):
                window.select_camera(self.cam_id)  # always select on click
                if window.is_area_mode:
                    # noinspection PyUnresolvedReferences
                    if event.button() == Qt.LeftButton:
                        if not self._polygon_fixed:
                            self._points.append(event.pos())
                    return True
        return super().eventFilter(obj, event)

    # --- Drawing & frame update ---
    # noinspection PyTypeChecker
    def update_frame_direct(self, frame: np.ndarray):
        # draw polygon if exists
        if self._points:
            pts_np = np.array([[p.x(), p.y()] for p in self._points], np.int32)
            cv2.polylines(frame, [pts_np], self._polygon_fixed, (0, 255, 0), 2)
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    # --- Polygon helpers ---
    def undo_last_point(self):
        if self._points and not self._polygon_fixed:
            self._points.pop()

    def finalize_polygon(self):
        if len(self._points) >= 3:
            self._polygon_fixed = True
        else:
            QMessageBox.warning(self, "오류", "3개 이상의 점이 필요합니다")

    def clear_polygon(self):
        self._points.clear()
        self._polygon_fixed = False


# -------------------------------
# NVRGUI with unified control & selection
# -------------------------------
class NVRGUI(QWidget):
    def __init__(self, camera_sources, queue, shared_flags):
        super().__init__()
        self.setWindowTitle("AI 기반 NVR 시스템")
        self.camera_sources = camera_sources
        self.queue = queue
        self.shared_flags = shared_flags
        self.cameras: list[CameraWidget] = []
        self.selected_cam_id: Optional[int] = None
        self.is_area_mode = False
        self._init_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._read_from_queue)
        self.timer.start(33)

    # --- UI
    def _init_ui(self):
        self.cam_container = QWidget()
        self.grid = QGridLayout(self.cam_container)
        self.grid.setSpacing(10)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.cam_container)

        for i in range(len(self.camera_sources)):
            cw = CameraWidget(i, self.shared_flags)
            self.cameras.append(cw)

        self._update_cam_grid()

        # noinspection PyArgumentList
        self.alerts = QTextEdit(readOnly=True)
        self.alerts.setPlaceholderText("위험행동 / 안전위험 알림...")

        # unified buttons
        self.btn_area = QPushButton("영역 설정")
        self.btn_undo = QPushButton("되돌리기")
        self.btn_done = QPushButton("설정 완료")
        self.btn_clear = QPushButton("영역 삭제")
        self.btn_area.clicked.connect(self._toggle_area_mode)
        self.btn_undo.clicked.connect(self._undo_point)
        self.btn_done.clicked.connect(self._fix_poly)
        self.btn_clear.clicked.connect(self._clear_poly)

        side_btn_lay = QHBoxLayout()
        side_btn_lay.addWidget(self.btn_area)
        side_btn_lay.addWidget(self.btn_undo)
        side_btn_lay.addWidget(self.btn_done)
        side_btn_lay.addWidget(self.btn_clear)

        side = QVBoxLayout()
        side.addWidget(QLabel("[시스템 알림]"))
        side.addWidget(self.alerts)
        side.addLayout(side_btn_lay)

        main = QHBoxLayout(self)
        main.addWidget(self.scroll_area, 3)
        main.addLayout(side, 1)

    def _update_cam_grid(self):
        # clear and re‑add widgets
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().setParent(None)
        cols = math.ceil(math.sqrt(len(self.cameras)))
        for idx, cam in enumerate(self.cameras):
            r, c = divmod(idx, cols)
            self.grid.addWidget(cam, r, c)

    # --- Camera selection
    def select_camera(self, cam_id: int):
        if self.selected_cam_id == cam_id:
            return  # already selected
        self.selected_cam_id = cam_id
        for idx, cam in enumerate(self.cameras):
            cam.set_selected(idx == cam_id)
        self.alerts.append(f"[INFO] 카메라 {cam_id} 선택")

    # --- Unified button callbacks
    def _toggle_area_mode(self):
        if self.selected_cam_id is None:
            QMessageBox.warning(self, "선택 필요", "먼저 카메라 프레임을 클릭해 선택하세요")
            return
        self.is_area_mode = not self.is_area_mode
        state = "ON" if self.is_area_mode else "OFF"
        self.alerts.append(f"[영역 모드] {state} (카메라 {self.selected_cam_id})")

    def _undo_point(self):
        cam = self._current_cam()
        if cam:
            cam.undo_last_point()

    def _fix_poly(self):
        cam = self._current_cam()
        if cam:
            cam.finalize_polygon()

    def _clear_poly(self):
        cam = self._current_cam()
        if cam:
            cam.clear_polygon()

    def _current_cam(self) -> Optional[CameraWidget]:
        if self.selected_cam_id is None:
            QMessageBox.warning(self, "선택 필요", "먼저 카메라 프레임을 클릭해 선택하세요")
            return None
        return self.cameras[self.selected_cam_id]

    # --- Queue reader ---
    def _read_from_queue(self):
        while not self.queue.empty():
            cam_id, frame, msg = self.queue.get()
            if frame is not None:
                self.cameras[cam_id].update_frame_direct(frame)
            if msg is not None:
                self.alerts.append(msg)


# -------------------------------
# Main launcher (unchanged AI workers)
# -------------------------------

def run():
    manager = Manager()
    shared_flags = manager.dict()
    queue = manager.Queue()
    sources = [0, 1]  # adjust to your cameras or RTSP streams
    for i in range(len(sources)):
        shared_flags[i] = {"behavior": False, "equipment": False}
    procs = []

    io_queues, frame_smm_mgr, shm_objs_dict, processes_dict = fall_detection_adapt_layer.fall_detect_init(sources)

    demo_process = None
    pose_processes = None
    fall_checker = None
    stream_instance_dict = None
    manager_process = None
    try:
        for i, src in enumerate(sources):
            p = Process(target=camera_process, args=(src, i, queue, shared_flags, io_queues[str(src)]))
            p.daemon = True
            p.start()
            procs.append(p)

        app = QApplication(sys.argv)
        gui = NVRGUI(sources, queue, shared_flags)
        gui.resize(1200, 800)
        gui.show()

        demo_process = processes_dict["demo_process"]
        pose_processes = processes_dict["pose_processes"]
        fall_checker = processes_dict["fall_checker"]
        stream_instance_dict = processes_dict["stream_instance_dict"]
        manager_process = processes_dict["manager_process"]
        while True:
            app.processEvents()

            # Watch Dog
            if not demo_process.is_alive():
                raise RuntimeError("[MAIN PROC WD ERROR] demo thread is dead")
            if not all(proc.is_alive() for proc in pose_processes):
                raise RuntimeError("[MAIN PROC WD ERROR] pose porc is dead")
            if not manager_process.is_alive():
                raise RuntimeError("[MAIN PROC WD ERROR] manager process is dead")
            if not fall_checker.is_alive():
                raise RuntimeError("[MAIN PROC WD ERROR] fall checker is dead")

            time.sleep(0.0001)

    except KeyboardInterrupt:
        print("NVR turnoff...")
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()

    finally:  # 리소스 정리
        exit_code = 0
        try:
            if stream_instance_dict:  # 입력스트림 종료
                for name, instance in stream_instance_dict.items():
                    thread = instance.kill_stream()
                    thread.join(timeout=5.0)
                    print(f"name: {name}, instance.is_alive: {thread.is_alive()}")

            if pose_processes:
                for proc in pose_processes:
                    if proc.is_alive():
                        proc.terminate()
                for proc in pose_processes:
                    proc.join(timeout=5.0)

            if manager_process and manager_process.is_alive():
                manager_process.terminate()
                manager_process.join(timeout=5.0)

            if demo_process and demo_process.is_alive():
                demo_process.terminate()
                demo_process.join(timeout=5.0)

            if fall_checker:
                fall_checker.terminate()
                fall_checker.join(timeout=5.0)

            for cam_proc in procs:
                if cam_proc.is_alive():
                    cam_proc.terminate()
                    cam_proc.join(timeout=5.0)

            if frame_smm_mgr:  # 공유메모리 정리
                frame_smm_mgr.shutdown()
                del frame_smm_mgr

            print("See you later!")

        except Exception as e:
            print(f"정리 중 오류 발생: {e}")
            traceback.print_exc()
            exit_code = 1

        finally:
            print(f"main END...{exit_code}")
    return exit_code


if __name__ == "__main__":
    freeze_support()
    run()
