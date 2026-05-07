import numpy as np

def preprocess_mnist(
    images: np.ndarray,
    labels: np.ndarray,
    normalize: bool = True,
    flatten: bool = True,
    one_hot: bool = False,
    num_classes: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    
    # uint8 -> float32
    images = images.astype(np.float32)

    # normalisasi agar range menjadi dari 0 - 255 menjadi 0 - 1 dengan membagi semuanya dengan 255
    if normalize:
        images = images / 255.0

    # flatten images agar dari shape 2D (28 x 28) menjadi shape 1D (784)
    if flatten:
        images = images.reshape(images.shape[0], -1)

    # one-hot encoding jika nantinya diperlukan, mengubah dari nilai 0 - 9 menjadi 1D dengan 9 elemen ter one-hot
    # contoh 1-hot encoding dari angka 3 berarti menjadi 1D array [0,0,0,1,0,0,0,0,0,0]
    if one_hot:
        encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
        encoded[np.arange(labels.shape[0]), labels] = 1.0
        labels = encoded

    return images, labels