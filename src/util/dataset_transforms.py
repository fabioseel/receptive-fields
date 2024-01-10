import torch
import PIL
import random
from torchvision.transforms import functional as F

class ComposeImage(torch.nn.Module):
    """Put the image at a random position in the RL Background
    """

    def __init__(self, background_image, fix_position=False, seed = None): # TODO: implement fix position to center
        super().__init__()
        if isinstance(background_image, PIL.Image.Image):
            self.bg_width = background_image.width
            self.bg_height = background_image.height
        else:
            self.bg_width = background_image.shape[-1]
            self.bg_height = background_image.shape[-2]

        self.background_image = background_image
        self.fix_position = fix_position
        self.seed = seed
        self.rng = random.Random(self.seed)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be put on background.

        Returns:
            PIL Image or Tensor: Image on background.
        """
        if isinstance(img, PIL.Image.Image):
            width = img.width
            height = img.height
        else:
            width = img.shape[-1]
            height = img.shape[-2]

        assert width <= self.bg_width
        assert height <= self.bg_height

        w_start = self.rng.randint(0, self.bg_width-width)
        h_start = self.rng.randint(0, self.bg_height-height)

        if isinstance(img, PIL.Image.Image):
            if not isinstance(self.background_image, PIL.Image.Image):
                composite = F.to_pil_image(self.background_image)
            else:
                composite = self.background_image.copy()
            composite.paste(img, (w_start,h_start))
        else:
            if not isinstance(self.background_image, PIL.Image.Image):
                composite = torch.clone(self.background_image)
            else:
                composite = F.to_tensor(self.background_image)
            composite[...,h_start:h_start+height, w_start:w_start+width] = img
        return composite

    def __repr__(self) -> str:
        detail = f"(fix_position={self.fix_position}, background={self.background_image.size()}, seed = {self.seed})"
        return f"{self.__class__.__name__}{detail}"
    

class RandomResize(torch.nn.Module):
    """Resizes the image to a random size in the given bounds. If bounds are defined for only one of width and height, the other will not be considered.
    """

    def __init__(self, factor_range, seed=None):
        super().__init__()
        self.factor_range = factor_range
        self.seed = seed
        self.rng = random.Random(self.seed)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be put on background.

        Returns:
            PIL Image or Tensor: Image on background.
        """
        if isinstance(img, PIL.Image.Image):
            width = img.width
            height = img.height
        else:
            width = img.shape[-1]
            height = img.shape[-2]
        
        scale = self.rng.random() * (self.factor_range[1] - self.factor_range[0]) + self.factor_range[0]

        return F.resize(img, (int(round(height*scale)), int(round(width*scale))))

    def __repr__(self) -> str:
        detail = f"(factor_range={self.factor_range}, seed = {self.seed})"
        return f"{self.__class__.__name__}{detail}"