import cv2
import sys
from ultralytics import YOLO

VIDEO_SOURCE = 0  # number for webcam or path to video file
WINDOW_NAME = "Dice Recognition"
PRIMARY_COLOR = (0, 255, 0)
SECONDARY_COLOR = (0, 0, 255)

model = YOLO("runs/detect/train/weights/best.pt")
video_capture = cv2.VideoCapture(VIDEO_SOURCE)


def main():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        inference_loop()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if video_capture:
            video_capture.release()

        cv2.destroyAllWindows()
        sys.exit(0)


def inference_loop():
    while True:
        frame_captured, frame = video_capture.read()
        if not frame_captured:
            break

        detected_dice = model(frame)[0].boxes
        dice_count = len(detected_dice)
        total_dice_sum = sum(int(die_box.cls[0]) + 1 for die_box in detected_dice)

        for die_box in detected_dice:
            x1, y1, x2, y2 = map(int, die_box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), SECONDARY_COLOR, 2)
            cv2.putText(
                frame,
                str(int(die_box.cls[0]) + 1),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                SECONDARY_COLOR,
                2,
            )

        cv2.putText(
            frame,
            f"Dice Count: {dice_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            PRIMARY_COLOR,
            2,
        )
        cv2.putText(
            frame,
            f"Total Sum: {total_dice_sum}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            PRIMARY_COLOR,
            2,
        )

        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break


if __name__ == "__main__":
    main()
