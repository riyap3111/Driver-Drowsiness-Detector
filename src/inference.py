from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.model import build_model, get_gradcam_target_layer
from src.utils import get_device


MIN_DECISION_THRESHOLD = 0.75


def overlay_heatmap_on_image(image_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 0.6, colored, 0.4, 0)


def enhance_frame(frame_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(enhanced, 5, 50, 50)


def compute_frame_quality(frame_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    low_light = brightness < 70
    blurry = blur_score < 75
    return {
        "brightness": brightness,
        "blur_score": blur_score,
        "low_light": low_light,
        "blurry": blurry,
        "quality_flag": "check_camera" if (low_light or blurry) else "usable",
    }


def stabilize_bbox(
    previous_bbox: tuple[int, int, int, int] | None,
    current_bbox: tuple[int, int, int, int] | None,
    alpha: float = 0.7,
) -> tuple[int, int, int, int] | None:
    if current_bbox is None:
        return previous_bbox
    if previous_bbox is None:
        return current_bbox
    return tuple(int(alpha * prev + (1 - alpha) * curr) for prev, curr in zip(previous_bbox, current_bbox))


class DrowsinessPredictor:
    def __init__(self, model_path: Path):
        checkpoint = torch.load(model_path, map_location="cpu")
        self.class_names = checkpoint["class_names"]
        self.image_size = checkpoint["image_size"]
        self.threshold = max(float(checkpoint.get("threshold", MIN_DECISION_THRESHOLD)), MIN_DECISION_THRESHOLD)
        self.device = get_device()
        self.model_name = checkpoint["model_name"]
        self.model = build_model(self.model_name, len(self.class_names), freeze_backbone=False).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.target_layer = get_gradcam_target_layer(self.model_name, self.model)
        self.previous_bbox: tuple[int, int, int, int] | None = None

    def detect_face(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            stabilized = stabilize_bbox(self.previous_bbox, None)
            if stabilized is None:
                return frame, None
            x, y, w, h = stabilized
            x = max(0, x)
            y = max(0, y)
            return frame[y : y + h, x : x + w], stabilized
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        stabilized = stabilize_bbox(self.previous_bbox, (x, y, w, h))
        self.previous_bbox = stabilized
        x, y, w, h = stabilized
        return frame[y : y + h, x : x + w], (x, y, w, h)

    def eye_count(self, frame: np.ndarray) -> int:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
        return len(eyes)

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[torch.Tensor, np.ndarray, np.ndarray, tuple[int, int, int, int] | None]:
        enhanced_frame = enhance_frame(frame_bgr)
        cropped, bbox = self.detect_face(enhanced_frame)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor, cropped, rgb, bbox

    def _gradcam(self, tensor: torch.Tensor, target_class: int) -> np.ndarray:
        activations = []
        gradients = []

        def forward_hook(_, __, output):
            activations.append(output.detach())

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            gradients.append(grad_output[0].detach())

        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        logits[:, target_class].backward()

        activation = activations[0]
        gradient = gradients[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        forward_handle.remove()
        backward_handle.remove()
        self.model.eval()
        return cam

    def predict(self, frame_bgr: np.ndarray, with_explanation: bool = False) -> dict:
        tensor, cropped_bgr, cropped_rgb, bbox = self._preprocess(frame_bgr)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        class_idx = int(np.argmax(probabilities))
        drowsy_idx = self.class_names.index("sleepy") if "sleepy" in self.class_names else min(1, len(self.class_names) - 1)
        drowsy_score = float(probabilities[drowsy_idx])
        predicted_label = self.class_names[drowsy_idx] if drowsy_score >= self.threshold else self.class_names[1 - drowsy_idx]

        result = {
            "label": predicted_label,
            "top_class": self.class_names[class_idx],
            "confidence": float(probabilities[class_idx]),
            "drowsy_score": drowsy_score,
            "threshold": self.threshold,
            "scores": {name: float(prob) for name, prob in zip(self.class_names, probabilities)},
            "eye_count": self.eye_count(cropped_bgr),
            "face_bbox": bbox,
            "face_crop_rgb": cropped_rgb,
            "quality": compute_frame_quality(frame_bgr),
            "risk_level": "high" if drowsy_score >= self.threshold else "moderate" if drowsy_score >= 0.5 else "low",
        }

        if with_explanation:
            cam = self._gradcam(tensor, drowsy_idx)
            result["attention_heatmap"] = overlay_heatmap_on_image(
                cv2.resize(cropped_rgb, (self.image_size, self.image_size)),
                cam,
            )
        return result

    def analyze_video(
        self,
        video_path: Path,
        sample_every_n_frames: int = 5,
        smoothing_window: int = 12,
        output_video_path: Path | None = None,
    ) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        writer = None
        if output_video_path is not None:
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        score_window: deque[float] = deque(maxlen=smoothing_window)
        timeline = []
        frame_index = 0
        drowsy_frames = 0
        max_score = 0.0
        last_result = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % sample_every_n_frames == 0:
                last_result = self.predict(frame, with_explanation=False)
                score_window.append(last_result["drowsy_score"])
                smoothed_score = float(np.mean(score_window))
                timestamp = frame_index / fps
                is_drowsy = smoothed_score >= self.threshold
                drowsy_frames += int(is_drowsy)
                max_score = max(max_score, smoothed_score)
                timeline.append(
                    {
                        "frame": frame_index,
                        "timestamp_sec": round(timestamp, 2),
                        "drowsy_score": round(last_result["drowsy_score"], 4),
                        "smoothed_drowsy_score": round(smoothed_score, 4),
                        "label": "sleepy" if is_drowsy else "awake",
                    }
                )

            if writer is not None and last_result is not None:
                smoothed_for_overlay = timeline[-1]["smoothed_drowsy_score"] if timeline else last_result["drowsy_score"]
                overlay_label = "DROWSY" if smoothed_for_overlay >= self.threshold else "ALERT"
                color = (0, 0, 255) if overlay_label == "DROWSY" else (0, 180, 0)
                annotated = frame.copy()
                cv2.putText(annotated, f"{overlay_label} score={smoothed_for_overlay:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(annotated, f"eyes={last_result['eye_count']}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                writer.write(annotated)

            frame_index += 1

        cap.release()
        if writer is not None:
            writer.release()

        processed_frames = max(len(timeline), 1)
        return {
            "video_path": str(video_path),
            "fps": fps,
            "total_frames": frame_count,
            "processed_frames": len(timeline),
            "max_drowsy_score": round(max_score, 4),
            "mean_drowsy_score": round(float(np.mean([item["smoothed_drowsy_score"] for item in timeline])) if timeline else 0.0, 4),
            "drowsy_ratio": round(drowsy_frames / processed_frames, 4),
            "threshold": self.threshold,
            "timeline": timeline,
            "output_video_path": str(output_video_path) if output_video_path else None,
        }


def run_webcam(model_path: Path):
    predictor = DrowsinessPredictor(model_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    score_window: deque[float] = deque(maxlen=10)
    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = predictor.predict(frame)
        score_window.append(result["drowsy_score"])
        smoothed_score = float(np.mean(score_window))
        label = "sleepy" if smoothed_score >= predictor.threshold else "awake"
        eye_count = result["eye_count"]
        color = (0, 255, 0) if label == "awake" else (0, 0, 255)

        cv2.putText(frame, f"{label} risk {smoothed_score:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"eyes_detected: {eye_count}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Run image, video, or webcam inference.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--output-video-path", type=Path)
    parser.add_argument("--sample-every-n-frames", type=int, default=5)
    parser.add_argument("--webcam", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    predictor = DrowsinessPredictor(args.model_path)

    if args.webcam:
        run_webcam(args.model_path)
        return

    if args.video_path is not None:
        result = predictor.analyze_video(
            args.video_path,
            sample_every_n_frames=args.sample_every_n_frames,
            output_video_path=args.output_video_path,
        )
        print(result)
        return

    if args.image_path is None:
        raise ValueError("Provide either --image-path, --video-path, or --webcam.")

    image = cv2.imread(str(args.image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image_path}")

    result = predictor.predict(image, with_explanation=True)
    print({key: value for key, value in result.items() if key not in {"attention_heatmap", "face_crop_rgb"}})


if __name__ == "__main__":
    main()
