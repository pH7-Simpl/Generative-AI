import numpy as np

def load_mnist_images(path: str) -> np.ndarray[tuple[int, int, int]]:
    with open(path, 'rb') as f:
        data = f.read()

    magic = int.from_bytes(data[0:4], 'big')

    if magic != 2051:
        raise ValueError(f"Invalid MNIST image file: magic={magic}")
    
    num_images = int.from_bytes(data[4:8], 'big')
    rows = int.from_bytes(data[8:12], 'big')
    cols = int.from_bytes(data[12:16], 'big')

    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    images = images.reshape(num_images, rows, cols)

    return images

def load_mnist_labels(path):
    with open(path, 'rb') as f:
        data = f.read()

    magic = int.from_bytes(data[0:4], 'big')

    if magic != 2049:
        raise ValueError(f"Invalid MNIST image file: magic={magic}")

    labels = np.frombuffer(data, dtype=np.uint8, offset=8)

    return labels
