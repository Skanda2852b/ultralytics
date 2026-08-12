# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Behavior Detection using YOLO Pose + MediaPipe Face Mesh.

This module combines YOLOv8 pose estimation for body keypoints with MediaPipe Face Mesh
for detailed facial landmark analysis to detect behaviors including:
- Head pose (pitch, yaw, roll)
- Gaze direction
- Facial expressions (smile, frown, surprise, etc.)
- Attention state (looking at screen, phone, away)
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from ultralytics import YOLO
from ultralytics.solutions import AIGym
from ultralytics.utils.plotting import Annotator, colors


class BehaviorDetector:
    """
    Behavior detector combining YOLO pose estimation with MediaPipe face mesh.

    Attributes:
        pose_model (YOLO): YOLO pose estimation model.
        face_mesh (mp.solutions.face_mesh.FaceMesh): MediaPipe face mesh detector.
        gym (AIGym): Workout monitoring solution.
        behavior_history (dict): Tracks behavior history per person for smoothing.
    """

    # MediaPipe Face Mesh landmark indices
    FACE_LANDMARKS = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_inner": 133,
        "left_eye_outer": 33,
        "right_eye_inner": 362,
        "right_eye_outer": 263,
        "left_eye_top": 159,
        "left_eye_bottom": 145,
        "right_eye_top": 386,
        "right_eye_bottom": 374,
        "left_iris": 468,
        "right_iris": 473,
        "mouth_left": 61,
        "mouth_right": 291,
        "mouth_top": 13,
        "mouth_bottom": 14,
        "left_eyebrow_inner": 70,
        "left_eyebrow_outer": 105,
        "right_eyebrow_inner": 300,
        "right_eyebrow_outer": 334,
        "forehead": 10,
    }

    # 3D model points for head pose estimation (generic face model)
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ], dtype=np.float64)

    def __init__(
        self,
        pose_model: str = "yolo26n-pose.pt",
        face_mesh_confidence: float = 0.5,
        max_faces: int = 5,
        smooth_frames: int = 5,
        attention_cap: float = 50.0,
        attention_window: int = 150,
        recover_frames: int = 15,
        device: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Initialize BehaviorDetector.

        Args:
            pose_model: Path to YOLO pose model.
            face_mesh_confidence: Minimum confidence for face detection.
            max_faces: Maximum faces to detect.
            smooth_frames: Number of frames to smooth behavior predictions.
            attention_cap: Non-attentive percentage threshold that flags a person as inattentive.
            attention_window: Number of frames used to compute the non-attentive percentage.
            recover_frames: Consecutive attentive frames needed to clear the inattentive flag.
            device: Device to run inference on ('cpu', 'cuda', 'mps', or '' for auto).
            **kwargs: Additional arguments passed to AIGym.
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is required for behavior detection. "
                "Install with: pip install mediapipe"
            )

        # Handle device selection
        if device == "":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pose_model = YOLO(pose_model).to(device)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=face_mesh_confidence,
            min_tracking_confidence=face_mesh_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.gym = AIGym(model=pose_model, **kwargs)
        self.smooth_frames = smooth_frames
        self.behavior_history: Dict[int, deque] = {}
        self.track_id_map: Dict[int, int] = {}  # Map face index to track ID
        self.attention_cap = attention_cap
        self.attention_window = attention_window
        self.recover_frames = recover_frames
        self.attention_history: Dict[int, deque] = {}
        self.reported_state: Dict[int, str] = {}
        self.attentive_count: Dict[int, int] = {}
        self.pose_history: Dict[int, deque] = {}
        self.head_deviation_threshold = 25.0
        self.non_attentive_states = ("looking_away", "looking_side", "looking_down", "on_phone")

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process frame for behavior detection.

        Args:
            frame: Input BGR image.

        Returns:
            Annotated frame and list of behavior dictionaries per tracked person.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO pose tracking
        pose_results = self.pose_model.track(frame, persist=True, classes=[0], verbose=False)

        # MediaPipe face mesh
        face_results = self.mp_face_mesh.process(rgb)

        # AIGym processing for workout monitoring
        gym_results = self.gym.process(frame)

        # Extract behaviors
        behaviors = self._extract_behaviors(frame, pose_results, face_results)

        # Annotate frame
        annotated_frame = self._annotate_frame(gym_results.plot_im, pose_results, face_results, behaviors)

        return annotated_frame, behaviors

    def _extract_behaviors(
        self,
        frame: np.ndarray,
        pose_results,
        face_results,
    ) -> List[Dict]:
        """Extract behavior data from pose and face landmarks."""
        h, w = frame.shape[:2]
        behaviors = []

        if not pose_results or not pose_results[0].boxes.is_track:
            return behaviors

        boxes = pose_results[0].boxes.xyxy.cpu().numpy()
        track_ids = pose_results[0].boxes.id.cpu().numpy() if pose_results[0].boxes.id is not None else []
        keypoints = pose_results[0].keypoints.data.cpu().numpy() if pose_results[0].keypoints is not None else []

        face_landmarks_list = face_results.multi_face_landmarks if face_results.multi_face_landmarks else []

        for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)

            # Get face landmarks for this person (match by spatial overlap)
            face_landmarks = self._match_face_to_person(face_landmarks_list, box, w, h)

            behavior = {
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                "gaze": {"direction": "center", "left_eye": (0.5, 0.5), "right_eye": (0.5, 0.5)},
                "expression": "neutral",
                "expression_confidence": 0.0,
                "attention": "unknown",
                "workout": {
                    "count": self.gym.states.get(track_id, {}).get("count", 0),
                    "stage": self.gym.states.get(track_id, {}).get("stage", "-"),
                    "angle": self.gym.states.get(track_id, {}).get("angle", 0.0),
                },
            }

            if face_landmarks:
                behavior["head_pose"] = self._estimate_head_pose(face_landmarks, w, h)
                behavior["gaze"] = self._estimate_gaze(face_landmarks, w, h)
                behavior["expression"], behavior["expression_confidence"] = self._analyze_expression(face_landmarks)
                hp = behavior["head_pose"]
                self.pose_history.setdefault(track_id, deque(maxlen=self.attention_window)).append((hp["pitch"], hp["yaw"]))
                behavior["attention"] = self._determine_attention(
                    hp, behavior["gaze"], keypoints[i] if i < len(keypoints) else None, list(self.pose_history[track_id])
                )

            # Smooth behavior over time
            behavior = self._smooth_behavior(track_id, behavior)

            # Track attention percentage over a sliding window and flag behavior changes
            behavior = self._track_attention(track_id, behavior)

            behaviors.append(behavior)

        return behaviors

    def _track_attention(self, track_id: int, behavior: Dict) -> Dict:
        """
        Track the percentage of non-attentive frames over a sliding window.

        Flags the person as inattentive once the non-attentive percentage exceeds
        `attention_cap`, and records behavior changes. Once the person looks back
        at the camera for `recover_frames` consecutive frames the flag clears so
        the status stays live until the camera is stopped.
        """
        history = self.attention_history.setdefault(track_id, deque(maxlen=self.attention_window))
        history.append(behavior["attention"])
        window = list(history)

        non_attentive = [state for state in window if state in self.non_attentive_states]
        pct = len(non_attentive) / len(window) * 100 if window else 0.0
        dominant = max(set(non_attentive), key=non_attentive.count) if non_attentive else "attentive"

        min_frames = min(self.attention_window, 30)
        inattentive = len(window) >= min_frames and pct >= self.attention_cap

        # Live recovery: a run of attentive frames clears the inattentive flag promptly
        if behavior["attention"] == "attentive":
            self.attentive_count[track_id] = self.attentive_count.get(track_id, 0) + 1
        else:
            self.attentive_count[track_id] = 0
        if self.attentive_count[track_id] >= self.recover_frames:
            inattentive = False

        change = ""
        if not inattentive and self.reported_state.get(track_id, "attentive") not in (None, "attentive"):
            prev = self.reported_state.get(track_id)
            self.reported_state[track_id] = "attentive"
            change = f"Behavior change: {prev} -> attentive"
        elif inattentive and self.reported_state.get(track_id) != dominant:
            prev = self.reported_state.get(track_id, "attentive")
            self.reported_state[track_id] = dominant
            change = f"Behavior change: {prev} -> {dominant}"

        behavior["attention_stats"] = {
            "non_attentive_pct": pct,
            "dominant": dominant,
            "inattentive": inattentive,
            "change": change,
        }
        return behavior

    def _match_face_to_person(
        self,
        face_landmarks_list,
        person_box: np.ndarray,
        w: int,
        h: int,
    ) -> Optional[Any]:
        """Match face landmarks to person bounding box by spatial overlap."""
        if not face_landmarks_list:
            return None

        px1, py1, px2, py2 = person_box
        person_center_x = (px1 + px2) / 2
        person_center_y = (py1 + py2) / 2

        best_match = None
        min_dist = float('inf')

        for face_landmarks in face_landmarks_list:
            # Get face center from nose tip
            nose = face_landmarks.landmark[self.FACE_LANDMARKS["nose_tip"]]
            face_x, face_y = nose.x * w, nose.y * h

            dist = np.sqrt((face_x - person_center_x) ** 2 + (face_y - person_center_y) ** 2)
            if dist < min_dist and dist < max(px2 - px1, py2 - py1) * 0.5:
                min_dist = dist
                best_match = face_landmarks

        return best_match

    def _estimate_head_pose(self, face_landmarks, w: int, h: int) -> Dict[str, float]:
        """Estimate head pose (pitch, yaw, roll) using PnP algorithm."""
        # Extract 2D image points from face landmarks
        image_points = np.array([
            (face_landmarks.landmark[self.FACE_LANDMARKS["nose_tip"]].x * w,
             face_landmarks.landmark[self.FACE_LANDMARKS["nose_tip"]].y * h),
            (face_landmarks.landmark[self.FACE_LANDMARKS["chin"]].x * w,
             face_landmarks.landmark[self.FACE_LANDMARKS["chin"]].y * h),
            (face_landmarks.landmark[self.FACE_LANDMARKS["left_eye_outer"]].x * w,
             face_landmarks.landmark[self.FACE_LANDMARKS["left_eye_outer"]].y * h),
            (face_landmarks.landmark[self.FACE_LANDMARKS["right_eye_outer"]].x * w,
             face_landmarks.landmark[self.FACE_LANDMARKS["right_eye_outer"]].y * h),
            (face_landmarks.landmark[self.FACE_LANDMARKS["mouth_left"]].x * w,
             face_landmarks.landmark[self.FACE_LANDMARKS["mouth_left"]].y * h),
            (face_landmarks.landmark[self.FACE_LANDMARKS["mouth_right"]].x * w,
             face_landmarks.landmark[self.FACE_LANDMARKS["mouth_right"]].y * h),
        ], dtype=np.float64)

        # Camera matrix (assuming simple pinhole camera)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

        # Convert rotation vector to Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0

        # Convert to degrees
        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)

        return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}

    def _estimate_gaze(self, face_landmarks, w: int, h: int) -> Dict:
        """Estimate gaze direction from iris positions."""
        # Get iris landmarks
        left_iris = face_landmarks.landmark[self.FACE_LANDMARKS["left_iris"]]
        right_iris = face_landmarks.landmark[self.FACE_LANDMARKS["right_iris"]]

        # Get eye corner landmarks for normalization
        left_eye_inner = face_landmarks.landmark[self.FACE_LANDMARKS["left_eye_inner"]]
        left_eye_outer = face_landmarks.landmark[self.FACE_LANDMARKS["left_eye_outer"]]
        right_eye_inner = face_landmarks.landmark[self.FACE_LANDMARKS["right_eye_inner"]]
        right_eye_outer = face_landmarks.landmark[self.FACE_LANDMARKS["right_eye_outer"]]

        # Normalize iris position within eye (0 to 1)
        def normalize_iris(iris, inner, outer):
            eye_width = outer.x - inner.x
            if eye_width == 0:
                return 0.5
            return (iris.x - inner.x) / eye_width

        left_gaze_x = normalize_iris(left_iris, left_eye_inner, left_eye_outer)
        right_gaze_x = normalize_iris(right_iris, right_eye_inner, right_eye_outer)

        # Vertical gaze (using top/bottom eyelids)
        left_eye_top = face_landmarks.landmark[self.FACE_LANDMARKS["left_eye_top"]]
        left_eye_bottom = face_landmarks.landmark[self.FACE_LANDMARKS["left_eye_bottom"]]
        right_eye_top = face_landmarks.landmark[self.FACE_LANDMARKS["right_eye_top"]]
        right_eye_bottom = face_landmarks.landmark[self.FACE_LANDMARKS["right_eye_bottom"]]

        def normalize_iris_y(iris, top, bottom):
            eye_height = bottom.y - top.y
            if eye_height == 0:
                return 0.5
            return (iris.y - top.y) / eye_height

        left_gaze_y = normalize_iris_y(left_iris, left_eye_top, left_eye_bottom)
        right_gaze_y = normalize_iris_y(right_iris, right_eye_top, right_eye_bottom)

        avg_gaze_x = (left_gaze_x + right_gaze_x) / 2
        avg_gaze_y = (left_gaze_y + right_gaze_y) / 2

        # Determine gaze direction
        if avg_gaze_x < 0.35:
            direction = "left"
        elif avg_gaze_x > 0.65:
            direction = "right"
        else:
            direction = "center"

        if avg_gaze_y < 0.35:
            direction += "_up"
        elif avg_gaze_y > 0.65:
            direction += "_down"

        return {
            "direction": direction,
            "left_eye": (float(left_gaze_x), float(left_gaze_y)),
            "right_eye": (float(right_gaze_x), float(right_gaze_y)),
        }

    def _analyze_expression(self, face_landmarks) -> Tuple[str, float]:
        """Analyze facial expression from landmarks."""
        # Mouth openness
        mouth_top = face_landmarks.landmark[self.FACE_LANDMARKS["mouth_top"]]
        mouth_bottom = face_landmarks.landmark[self.FACE_LANDMARKS["mouth_bottom"]]
        mouth_left = face_landmarks.landmark[self.FACE_LANDMARKS["mouth_left"]]
        mouth_right = face_landmarks.landmark[self.FACE_LANDMARKS["mouth_right"]]

        mouth_height = abs(mouth_bottom.y - mouth_top.y)
        mouth_width = abs(mouth_right.x - mouth_left.x)
        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

        # Eyebrow position (for surprise/anger)
        left_eyebrow_inner = face_landmarks.landmark[self.FACE_LANDMARKS["left_eyebrow_inner"]]
        left_eye_top = face_landmarks.landmark[self.FACE_LANDMARKS["left_eye_top"]]
        right_eyebrow_inner = face_landmarks.landmark[self.FACE_LANDMARKS["right_eyebrow_inner"]]
        right_eye_top = face_landmarks.landmark[self.FACE_LANDMARKS["right_eye_top"]]

        left_brow_height = left_eye_top.y - left_eyebrow_inner.y
        right_brow_height = right_eye_top.y - right_eyebrow_inner.y
        avg_brow_height = (left_brow_height + right_brow_height) / 2

        # Mouth corner position (smile/frown)
        mouth_corner_y = (mouth_left.y + mouth_right.y) / 2
        mouth_center_y = (mouth_top.y + mouth_bottom.y) / 2
        smile_score = mouth_center_y - mouth_corner_y  # Positive = smiling

        # Classify expression
        expressions = {}
        expressions["smile"] = max(0, smile_score * 10)
        expressions["surprise"] = max(0, avg_brow_height * 20 + mouth_ratio * 5)
        expressions["frown"] = max(0, -smile_score * 10 + (0.02 - avg_brow_height) * 20)
        expressions["neutral"] = 1.0 / (1.0 + sum(expressions.values()))

        best_expression = max(expressions, key=expressions.get)
        confidence = expressions[best_expression]

        return best_expression, float(confidence)

    def _determine_attention(
        self,
        head_pose: Dict,
        gaze: Dict,
        keypoints: Optional[np.ndarray],
        pose_window: Optional[List] = None,
    ) -> str:
        """
        Determine attention state from head pose, gaze, and body pose.

        The head pose baseline is estimated from a rolling median of the person's
        own history, so deviations are measured relative to their forward-facing
        direction rather than absolute angles (which vary with camera position).
        """
        # Normalize head pose to [-90, 90] so frontal faces (near +/-180 wrap) are not read as away
        pitch = head_pose["pitch"] % 360
        if pitch > 180:
            pitch -= 360
        yaw = head_pose["yaw"] % 360
        if yaw > 180:
            yaw -= 360
        gaze_dir = gaze["direction"]

        # Deviation from the person's own forward-facing baseline (circular, to avoid wraps)
        if pose_window and len(pose_window) >= 20:
            norm = lambda a: (a % 360) - 360 if (a % 360) > 180 else a % 360
            pitches = np.radians([norm(p) for p, _ in pose_window])
            yaws = np.radians([norm(y) for _, y in pose_window])
            base_pitch = np.degrees(np.arctan2(np.sin(pitches).sum(), np.cos(pitches).sum()))
            base_yaw = np.degrees(np.arctan2(np.sin(yaws).sum(), np.cos(yaws).sum()))
            dev_pitch = (pitch - base_pitch + 180) % 360 - 180
            dev_yaw = (yaw - base_yaw + 180) % 360 - 180
            if abs(dev_yaw) > self.head_deviation_threshold:
                return "looking_side"
            if dev_pitch > self.head_deviation_threshold:
                return "looking_down"
            if dev_pitch < -self.head_deviation_threshold:
                return "looking_up"
        elif abs(yaw) > 30 or abs(pitch) > 25:
            return "looking_away"

        # Gaze direction
        if "left" in gaze_dir or "right" in gaze_dir:
            return "looking_side"
        if "up" in gaze_dir:
            return "looking_up"
        if "down" in gaze_dir:
            return "looking_down"

        # Check if looking at phone (head down, hands near face)
        if keypoints is not None and pitch > 15:
            # Check wrist positions relative to face
            left_wrist = keypoints[9] if len(keypoints) > 9 else None
            right_wrist = keypoints[10] if len(keypoints) > 10 else None
            nose = keypoints[0] if len(keypoints) > 0 else None

            if nose is not None and nose[2] > 0.5:  # Confidence threshold
                nose_y = nose[1]
                for wrist in [left_wrist, right_wrist]:
                    if wrist is not None and wrist[2] > 0.5:
                        if abs(wrist[1] - nose_y) < 100:  # Wrist near face level
                            return "on_phone"

        return "attentive"

    def _smooth_behavior(self, track_id: int, behavior: Dict) -> Dict:
        """Smooth behavior predictions over time."""
        if track_id not in self.behavior_history:
            self.behavior_history[track_id] = deque(maxlen=self.smooth_frames)

        self.behavior_history[track_id].append(behavior)

        if len(self.behavior_history[track_id]) < 2:
            return behavior

        # Average numerical values
        smoothed = behavior.copy()
        history = list(self.behavior_history[track_id])

        # Smooth head pose
        smoothed["head_pose"] = {
            "pitch": np.mean([b["head_pose"]["pitch"] for b in history]),
            "yaw": np.mean([b["head_pose"]["yaw"] for b in history]),
            "roll": np.mean([b["head_pose"]["roll"] for b in history]),
        }

        # Smooth gaze (average coordinates)
        left_eyes = [b["gaze"]["left_eye"] for b in history]
        right_eyes = [b["gaze"]["right_eye"] for b in history]
        smoothed["gaze"]["left_eye"] = (np.mean([e[0] for e in left_eyes]), np.mean([e[1] for e in left_eyes]))
        smoothed["gaze"]["right_eye"] = (np.mean([e[0] for e in right_eyes]), np.mean([e[1] for e in right_eyes]))

        # Most common expression and attention
        expressions = [b["expression"] for b in history]
        attentions = [b["attention"] for b in history]
        smoothed["expression"] = max(set(expressions), key=expressions.count)
        smoothed["attention"] = max(set(attentions), key=attentions.count)

        return smoothed

    def _annotate_frame(
        self,
        frame: np.ndarray,
        pose_results,
        face_results,
        behaviors: List[Dict],
    ) -> np.ndarray:
        """Annotate frame with behavior information."""
        annotator = Annotator(frame, line_width=2, font_size=10, pil=False)

        # Draw face mesh
        if face_results and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

        # Draw behavior info for each person
        for behavior in behaviors:
            track_id = behavior["track_id"]
            x1, y1, x2, y2 = behavior["bbox"]
            color = colors(track_id, True)

            # Head pose
            hp = behavior["head_pose"]
            head_text = f"Head: P:{hp['pitch']:.1f} Y:{hp['yaw']:.1f} R:{hp['roll']:.1f}"

            # Gaze
            gaze_dir = behavior["gaze"]["direction"]
            gaze_text = f"Gaze: {gaze_dir}"

            # Expression
            expr = behavior["expression"]
            expr_conf = behavior["expression_confidence"]
            expr_text = f"Expr: {expr} ({expr_conf:.2f})"

            # Attention
            att_text = f"Att: {behavior['attention']}"

            # Workout
            wo = behavior["workout"]
            wo_text = f"Reps: {wo['count']} | Stage: {wo['stage']} | Angle: {wo['angle']:.1f}"

            # Draw all text
            texts = [f"ID:{track_id}", head_text, gaze_text, expr_text, att_text, wo_text]
            for j, text in enumerate(texts):
                annotator.text([x1, y1 - 20 - j * 20], text, txt_color=color)

            # Attention cap alert (live: red when inattentive, green when attentive)
            stats = behavior.get("attention_stats", {})
            if stats and "non_attentive_pct" in stats:
                if stats["inattentive"]:
                    annotator.box_label([x1, y1, x2, y2], f"INATTENTIVE {stats['non_attentive_pct']:.0f}%", color=(0, 0, 255))
                    annotator.text([x1, y2 + 20], f"{stats['dominant']} for {stats['non_attentive_pct']:.0f}% of window", txt_color=(0, 0, 255))
                    if stats["change"]:
                        annotator.text([x1, y2 + 40], stats["change"], txt_color=(0, 0, 255))
                else:
                    annotator.box_label([x1, y1, x2, y2], "ATTENTIVE", color=(0, 255, 0))
                    if stats["change"]:
                        annotator.text([x1, y2 + 20], stats["change"], txt_color=(0, 255, 0))

        return frame


def run_behavior_detection(
    source: str = "0",
    pose_model: str = "yolo26n-pose.pt",
    output_path: Optional[str] = None,
    device: str = "",
    show: bool = True,
    **kwargs,
) -> None:
    """
    Run behavior detection on video source.

    Args:
        source: Video source (camera index, video file, or YouTube URL).
        pose_model: YOLO pose model path.
        output_path: Optional output video path.
        device: Inference device.
        show: Whether to display output.
        **kwargs: Additional arguments for BehaviorDetector.
    """
    detector = BehaviorDetector(pose_model=pose_model, device=device, **kwargs)

    # Handle YouTube URLs
    if source.startswith("http"):
        from ultralytics.data.loaders import get_best_youtube_url
        source = get_best_youtube_url(source)

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source}")

    # Video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Behavior detection started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, behaviors = detector.process(frame)

        # Print behaviors to console
        for b in behaviors:
            print(f"Person {b['track_id']}: "
                  f"Attention={b['attention']}, "
                  f"Expression={b['expression']}, "
                  f"Gaze={b['gaze']['direction']}, "
                  f"Head(Y:{b['head_pose']['yaw']:.1f}, P:{b['head_pose']['pitch']:.1f}), "
                  f"Reps={b['workout']['count']}")
            stats = b.get("attention_stats", {})
            if stats and stats["inattentive"]:
                print(f"  -> {stats['change'] or 'Inattentive'}: {stats['dominant']} {stats['non_attentive_pct']:.0f}% "
                      f"over last {detector.attention_window} frames (cap={detector.attention_cap}%)")

        if writer:
            writer.write(annotated_frame)

        if show:
            cv2.imshow("Behavior Detection (YOLO Pose + MediaPipe Face Mesh)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    detector.mp_face_mesh.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Behavior Detection with YOLO Pose + MediaPipe Face Mesh")
    parser.add_argument("--source", type=str, default="0", help="Video source (camera index, file, or YouTube URL)")
    parser.add_argument("--pose-model", type=str, default="yolo26n-pose.pt", help="YOLO pose model path")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--device", type=str, default="", help="Device (cpu, cuda, mps)")
    parser.add_argument("--no-show", action="store_true", help="Disable display")
    parser.add_argument("--face-conf", type=float, default=0.5, help="Face mesh confidence threshold")
    parser.add_argument("--max-faces", type=int, default=5, help="Maximum faces to detect")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing frames")
    parser.add_argument("--attention-cap", type=float, default=50.0, help="Non-attentive percentage cap (0-100) that flags a person inattentive")
    parser.add_argument("--attention-window", type=int, default=150, help="Number of frames used to compute the non-attentive percentage")
    parser.add_argument("--recover-frames", type=int, default=15, help="Consecutive attentive frames needed to clear the inattentive flag")

    args = parser.parse_args()

    run_behavior_detection(
        source=args.source,
        pose_model=args.pose_model,
        output_path=args.output,
        device=args.device,
        show=not args.no_show,
        face_mesh_confidence=args.face_conf,
        max_faces=args.max_faces,
        smooth_frames=args.smooth,
        attention_cap=args.attention_cap,
        attention_window=args.attention_window,
        recover_frames=args.recover_frames,
    )