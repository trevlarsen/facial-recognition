import os
import numpy as np
from imageio.v2 import imread
from matplotlib import pyplot as plt
from scipy import linalg as la

class FacialRec:
    """A class for managing a facial recognition database using the Eigenfaces method.

    Attributes:
        original_faces ((mn,k) ndarray): Flattened images of the dataset.
        mean_face ((mn,) ndarray): Mean of all flattened images.
        shifted_faces ((mn,k) ndarray): Images centered by subtracting the mean.
        u ((mn,k) ndarray): Eigenfaces from the compact SVD of shifted_faces.
    """

    def __init__(self, path='./faces94'):
        """Initialize the dataset and compute the mean face, shifted faces, and eigenfaces."""
        faces = self.load_faces(path)
        self.original_faces = faces
        self.mean_face = np.mean(faces, axis=1)
        self.shifted_faces = faces - self.mean_face[:, None]
        self.u, _, _ = np.linalg.svd(self.shifted_faces, full_matrices=False)

    def load_faces(self, path='./faces94'):
        """Load one image per subdirectory, flatten, and convert to grayscale.

        Args:
            path (str): Directory containing the dataset of images.

        Returns:
            ndarray: Flattened images in columns.
        """
        faces = []
        for dirpath, _, filenames in os.walk(path):
            for fname in filenames:
                if fname.endswith(".jpg"):
                    faces.append(np.ravel(imread(os.path.join(dirpath, fname), pilmode="F")))
                    break
        if not faces:
            raise ValueError(f"No valid images found in directory: {path}")
        return np.transpose(faces)

    def show(self, image, m=200, n=180, display=False):
        """Display a grayscale image given its flattened representation.

        Args:
            image (ndarray): Flattened image.
            m (int): Number of rows in the image.
            n (int): Number of columns in the image.
            display (bool): Whether to show the plot immediately.
        """
        plt.imshow(image.reshape(m, n), cmap='gray')
        plt.axis('off')
        if display:
            plt.show()

    def project(self, A, s):
        """Project a face onto the subspace spanned by the first s eigenfaces.

        Args:
            A (ndarray): Face vector(s) to project.
            s (int): Number of eigenfaces to use.

        Returns:
            ndarray: Projected representation.
        """
        return self.u[:, :s].T @ A

    def find_nearest(self, g, s=38):
        """Find the closest match in the dataset to the given face.

        Args:
            g (ndarray): Flattened face image.
            s (int): Number of eigenfaces to use.

        Returns:
            int: Index of the closest matching face.
        """
        fhat = self.project(self.shifted_faces, s)
        ghat = self.project(g - self.mean_face, s)
        return np.argmin(la.norm(fhat.T - ghat.T, axis=1))

    def match(self, image, s=38, m=200, n=180):
        """Display an image alongside its closest match from the dataset.

        Args:
            image (ndarray): Flattened face image.
            s (int): Number of eigenfaces to use.
            m (int): Number of rows in the image.
            n (int): Number of columns in the image.
        """
        best = self.find_nearest(image, s)
        match = self.original_faces[:, best]
        plt.subplot(121)
        plt.title("Input Image")
        self.show(image, m, n, display=False)
        plt.subplot(122)
        plt.title("Closest Match")
        self.show(match, m, n, display=False)
        plt.show()

def sample_faces(k, path="./faces94"):
    """Generate k sample images from the given path.

    Parameters:
        n (int): The number of sample images to obtain. 
        path(str): The directory containing the dataset of images.  
    
    Yields:
        ((mn,) ndarray): An flattend mn-array representing a single
        image. k images are yielded in total.
    """
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for fname in filenames:
            if fname[-3:]=="jpg":
                files.append(dirpath+"/"+fname)

    test_files = np.random.choice(files, k, replace=False)
    for fname in test_files:
        yield np.ravel(imread(fname, mode='F'))