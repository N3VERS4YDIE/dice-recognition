# Dice Recognition 🎲 with YOLO

This project uses Ultralytics YOLO to detect dice, count them, and sum their values in real-time using a webcam or screen capture.

## Installation

### 1. Create a Virtual Environment
```sh
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

## Running the Model

### Option 1: Use Pretrained Model
If you want to directly run the model without training, download my pretrained model:
1. Download the pretrained model from [Releases](https://github.com/N3VERS4YDIE/dice-recognition/releases/pretrained-model)
2. Place it inside the `runs/detect/` path of the project (if the folders doesn't exist, create them).

Run the application:
```sh
python main.py
```

### Option 2: Train Your Own Model

#### Configure Ultralytics Dataset Directory
Ultralytics uses a global dataset directory for storing datasets. You need to configure it before training:
```sh
ultralytics settings
```
Set your dataset directory in the `datasets_dir` field. This is where you will store all datasets for YOLO training.

#### Download Dataset
Download the dataset from [Releases](https://github.com/N3VERS4YDIE/dice-recognition/releases/dataset), decompress and place it inside your configured dataset directory.

#### Train the Model
Run the following command to start training:
```sh
python train.py
```

Once training is complete, the best-trained model will be stored at:
```
runs/detect/train/weights/best.pt
```

You can now use this model by modifying `main.py`:
```python
model = YOLO("runs/detect/train/weights/best.pt")
```

> [!Note]
> If you train multiple times, new training folders (`train<n>`) will be created. You need to specify the exact folder containing `best.pt` in `main.py` before running.

## Usage
Run the application:
```sh
python main.py
```
The script will detect dice, count them, and display the total sum in real-time.

## Project Structure
```
.
├── main.py                # Runs the dice recognition model
├── train.py               # Trains the model
├── requirements.txt       # Project dependencies
├── yolo12n.pt             # Base YOLO model for training
├── runs/                  # Training outputs
│   └── detect/
│       ├── train/         # First training session
│       │   ├── weights/
│       │   │   ├── best.pt  # Best-trained model
│       │   │   ├── last.pt  # Last epoch model
```

## Notes
- If using screen capture mode, modify `IS_SCREEN_CAPTURE = True` in `main.py`.
- If you train the model multiple times, select the correct `train<n>` folder when using `best.pt` in `main.py`.

## License
This project is licensed under the MIT License.

