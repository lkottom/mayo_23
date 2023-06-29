import os
import numpy as np

from constants import OPENSLIDE_DLLS_PATH

# Importing OpenSlide
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_DLLS_PATH):
        import openslide
else:
    import openslide

class OpenSlideHandler:
    '''This class helps with handling OpenSlide objects easier'''
    def __init__(self, wsi_path):
        '''Initiates an object of the class OpenSlideHandler

        Parameters:
        wsi_path (str): the path to the WSI svs file

        Returns:
        OpenSlideHandler: The created object

        '''
        # Creating native OpenSlide object
        self.openslide_obj = openslide.OpenSlide(wsi_path)
        
        # Creating the pyramid of magnifications and levels
        # List of all dimensions in different levels
        self.level_dimensions = self.openslide_obj.level_dimensions
        # List of downsampling ratio from the highest magnification 
        self.level_downsamples = self.openslide_obj.level_downsamples
        # Number of levels in the SVS file
        self.level_count = len(self.level_dimensions)
        
        # Trying to read the highest magnification
        # Trying to find aperio.AppMag or objective-power in properties
        if 'aperio.AppMag' in self.openslide_obj.properties.keys():
            self.max_magnification = float(self.openslide_obj.properties['aperio.AppMag'])
        elif 'objective-power' in self.openslide_obj.properties.keys():
            # Trying to find objective-power in properties, if aperio.AppMag not found
            self.max_magnification = float(self.openslide_obj.properties['objective-power'])
        else:
            raise ValueError('Maximum magnification not set')
        
        # Creating a list of level magnifications
        self.level_magnifications = [self.max_magnification / down_sample for down_sample in self.level_downsamples]
        # Largest WSI dimension
        self.wsi_dimensions = self.level_dimensions[0]

    def read_region(self, location, magnification, size):
        '''Returns a PIL object of the requested region

        Parameters:
        location (Tuple(int, int)): the left upermost coordinates of the rewired patch IN LEVEL 0 (HIGHEST MAGNIFICATION)
        magnification (int): the magnification needed
        size (Tuple(int, int)): the size of the patch IN THE REQUIRED MAGNIFICATION, NOT IN LEVEL 0

        Returns:
        PIL Image: The region image

        '''
        # Making sure that the required magnification is in the right range
        assert self.max_magnification >= magnification, 'Requested magnification is more than maximum magnification, Upsampling not recommended'
        assert magnification > 0, 'Negative magnification not feasible!'
        
        # Finding the level and down sampling needed
        level, down_sampling = self._find_level(magnification)
        
        # Selecting the patch size to load in loading level
        load_size = (int(size[0] * down_sampling), int(size[1] * down_sampling))
        
        # Finding the projected size of the patch on highest magnification
        path_size_on_level_0 = (
            int(size[0] * (self.max_magnification / magnification)),
            int(size[1] * (self.max_magnification / magnification))
            )
        
        # Making sure the projection of the patch on level 0 is within WSI boundary
        assert (
            location[0] + path_size_on_level_0[0] < self.wsi_dimensions[0]
            and location[1] + path_size_on_level_0[1] < self.wsi_dimensions[1]
            ), (
                f'Patch boundary out of WSI - WSI Dimensions: {self.wsi_dimensions}, ' +
                f'Patch Location on Level 0: {location}, Patch Size: {size}, ' +
                f'Projected Patch Size on Level 0: {path_size_on_level_0}'
            )

        # Checking if downsampling changed size, if changed -> resize else not
        if size == load_size:
            return self.openslide_obj.read_region(
                location=location,
                level=level,
                size=load_size
            ).convert('RGB')
        else:
            return self.openslide_obj.read_region(
                location=location,
                level=level,
                size=load_size
            ).convert('RGB').resize(size)
    
    def get_thumbnail(self, max_dim):
        '''Returns a thumbnail of the WSI

        Parameters:
        max_dim (Tuple(int, int)): The maximum dimension for the thumbnail in height and width

        Returns:
        PIL Image: The WSI thumbnail

        '''
        return self.openslide_obj.get_thumbnail(max_dim).convert('RGB')
    
    def _find_level(self, magnification):
        '''Finds the closest level with higher magnification of svs file to the given magnification

        Parameters:
        magnification (int): the needed magnification

        Returns:
        int: The level to load
        float: the downsampling required to get the needed magnification

        '''
        # Selecting the level to laod
        level = sum(np.array(self.level_magnifications) >= magnification) - 1
        
        assert level < self.level_count, 'Error finding the right level'
        
        # Determinign the downsampling needed
        down_sampling = self.level_magnifications[level] / magnification # Always bigger than 1
        
        return level, down_sampling    