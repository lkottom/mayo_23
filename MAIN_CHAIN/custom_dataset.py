from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    """
    A custom PyTorch dataset for loading images from a specified directory.

    Parameters:
    data_path (str): Path to the directory containing image files.
    transform (callable, optional): A function/transform to apply to the images.

    Returns:
    torch.utils.data.Dataset: A custom dataset for loading images.
    """

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        # Get a list of image file paths in the specified directory
        self.image_paths = self.get_image_paths()

    def get_image_paths(self):
        """
        Fetch the list of image file paths from the 'data_path'.

        Returns:
        list: List of image file paths.
        """
        # Implement a function to fetch the list of image file paths from 'self.data_path'
        # For example, if your image files are in '.png' format, you can use glob as follows:
        import glob
        image_paths = glob.glob(self.data_path + "/*.png")
        return image_paths

    def __len__(self):
        """
        Get the total number of images in the dataset.

        Returns:
        int: Total number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get the image and its corresponding path at the specified index.

        Parameters:
        idx (int): Index of the image to retrieve.

        Returns:
        tuple: A tuple containing the image and its path.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_path  # Return both the image and its path

