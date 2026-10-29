
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "face_landmarker.task"


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access webcam.")
        return

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    frame_count = 0

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            print("Face and iris tracking started. Press Q to quit.")

            while True:
                success, frame = camera.read()

                if not success:
                    print("Unable to read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                frame_count += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms = frame_count * 33
                results = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                if results.face_landmarks:
                    for face in results.face_landmarks:
                        # Draw facial landmarks
                        for point in face:
                            x = int(point.x * width)
                            y = int(point.y * height)
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                        # MediaPipe Face Landmarker iris landmark indices
                        iris_groups = {
                            "Right iris": range(468, 473),
                            "Left iris": range(473, 478)
                        }

                        for label, indices in iris_groups.items():
                            points = [
                                face[i] for i in indices
                                if i < len(face)
                            ]

                            if not points:
                                continue

                            coords = [
                                (int(p.x * width), int(p.y * height))
                                for p in points
                            ]

                            for x, y in coords:
                                cv2.circle(
                                    frame, (x, y), 3, (0, 0, 255), -1
                                )

                            center_x = sum(x for x, y in coords) // len(coords)
                            center_y = sum(y for x, y in coords) // len(coords)

                            cv2.circle(
                                frame, (center_x, center_y),
                                5, (255, 0, 0), 2
                            )

                            row_y = 30 if label == "Right iris" else 60
                            cv2.putText(
                                frame,
                                f"{label}: ({center_x}, {center_y})",
                                (10, row_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 255, 255), 2
                            )
                else:
                    cv2.putText(
                        frame, "No face detected", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2
                    )

                cv2.imshow("Day 23 - Facial and Iris Tracking", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except Exception as error:
        print(f"Tracking error: {error}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam and OpenCV windows closed.")


if __name__ == "__main__":
    main()
