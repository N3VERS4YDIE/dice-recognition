# Dice Recognition 🎲🤖

Computer Vision project made with Python, YOLO and OpenCV to recognize dice, count them, and sum their values in real time using a webcam or video as input.

https://github.com/user-attachments/assets/6add3780-97fc-426d-94da-2fd304d312ad

## Installation

### 1. Create a Virtual Environment

```sh
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 2. Install Dependencies

> [!Note]
> Before installing the dependencies, if you want to **use CUDA for better performance**, you should install the appropriate CUDA versions of torch and torchvision:
>
> ```sh
> pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> ```
>
> The example above uses **cu126** (CUDA 12.6). However, you must ensure that:
>
> - Your system has a **compatible NVIDIA GPU**.
> - You have the **correct CUDA drivers installed**.
>
> If you're using an older GPU or have a lower CUDA version installed (e.g., CUDA 11.8), use the matching packages:
>
> ```sh
> pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
>
> For more options and compatibility information, check the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

```sh
pip3 install -r requirements.txt
```

## Running the Model

### Option 1: Use Pretrained Model

The pretrained model is already included in this repository at:

```text
runs/detect/train/weights/best.pt
```

You can run the application directly:

```sh
python3 main.py
```

### Option 2: Train Your Own Model

The dataset is already included in this repository at:

```text
datasets/dices
```

To train the model you can use the YOLO CLI:

```sh
yolo task=detect mode=train model=yolov8n.pt data=datasets/dices/data.yaml epochs=50 plots=True
```

> [!Note]
> You can customize the training process by for example modifying these opions:
>
> - **model**: YOLO model to use (e.g., `yolov8n.pt`, `yolov8s.pt`, etc.).
> - **epochs**: Max training cycles.
> - **patience**: Stop early if no improvement after this many epochs.

Once training is complete, the best-trained model will be stored at:

```text
runs/detect/train2/weights/best.pt
```

You can use this model by modifying `main.py`:

```python
model = YOLO("runs/detect/train2/weights/best.pt")
```

> [!Note]
> If you train multiple times, new training folders (e.g., `train2`, `train3`, etc.) will be created, so you can choose the best model from any of them by modifying the path in `main.py`.

## Usage

Simply run:

```sh
python3 main.py
```

The script will detect dice, count them, and display the total sum in real-time.

## Project Structure

```sh
.
├── datasets/                # Contains the datasets for training
│   └── dices/
│       ├── data.yaml        # Dataset configuration file
│       ├── test/            # Test dataset
│       ├── train/           # Training dataset
│       └── valid/           # Validation dataset
├── main.py                  # Runs the dice recognition model
├── requirements.txt         # Project dependencies
├── runs/                    # Training outputs
│   └── detect/
│       ├── train/           # First training session
│       │   ├── weights/
│       │   │   ├── best.pt  # Best-trained model
└── yolov8n.pt               # Base YOLO model for training
```

## License

This project is licensed under the MIT License.
