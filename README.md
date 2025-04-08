# Dice Recognition 🎲🤖

Computer Vision project made with Python, Ultralytics YOLO and OpenCV to recognize dice, count them, and sum their values in real time using a webcam or video as input.

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
pip install -r requirements.txt
```

## Running the Model

### Option 1: Use Pretrained Model

If you want to directly run the model without training, download my pretrained model:

1. Download the pretrained model from [Releases](https://github.com/N3VERS4YDIE/dice-recognition/releases/pretrained-model)
2. Place it inside the `runs/detect/` path of the project (if the folders doesn't exist, create them).

Run the application:

```sh
python3 main.py
```

### Option 2: Train Your Own Model

#### Configure Ultralytics Dataset Directory

Ultralytics uses a global dataset directory for storing datasets. You need to configure it before training:

```sh
yolo settings datasets_dir=/path/to/your/datasets
```

> [!Note]
> This is where you will store all datasets for YOLO training.

#### Download Dataset

Download the dataset from [Releases](https://github.com/N3VERS4YDIE/dice-recognition/releases/dataset), decompress and place it inside your configured dataset directory.

#### Train the Model

Run the following command to start training:

```sh
python3 train.py
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
> The `runs/detect/train/weights/best.pt` file is the best-trained model after training. If you train multiple times, new training folders (e.g., `train2`, `train3`, etc.) will be created, so you can choose the best model from any of them by modifying the path in `main.py`.

## Usage

Run the application:

```sh
python3 main.py
```

The script will detect dice, count them, and display the total sum in real-time.

## Project Structure

```sh
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

## License

This project is licensed under the MIT License.
