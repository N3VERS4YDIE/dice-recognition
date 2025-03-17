from ultralytics import YOLO, settings

if __name__ == "__main__":
    model = YOLO("yolo12n.pt")
    model.train(
        data=f"{settings['datasets_dir']}/dices/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        half=True,
    )
