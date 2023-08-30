from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        # Get a list of image file paths in the specified directory
        self.image_paths = self.get_image_paths()

    def get_image_paths(self):
        # Implement a function to fetch the list of image file paths from 'self.data_path'
        # For example, if your image files are in '.png' format, you can use glob as follows:
        import glob
        image_paths = glob.glob(self.data_path + "/*.png")
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_path  # Return both the image and its path
