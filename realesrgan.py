import torch
from PIL import Image
import numpy as np

from rrdbnet_arch import RRDBNet
from utils_sr import (
    pad_reflect,
    split_image_into_overlapping_patches,
    stich_together,
    unpad_image,
)


class RealESRGAN:
    def __init__(self, scale: int = 4, model_path: str = "weights/x4.pth") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = scale
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )
        self.load_weights(model_path)

    def load_weights(self, model_path: str) -> None:
        try:
            loadnet = torch.load(model_path)
        except FileNotFoundError:
            loadnet = torch.load("Real-ESRGAN/weights/x4.pth")
        if "params" in loadnet:
            self.model.load_state_dict(loadnet["params"], strict=True)
        elif "params_ema" in loadnet:
            self.model.load_state_dict(loadnet["params_ema"], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)
        self.model.eval()
        self.model.to(self.device)

    @torch.cuda.amp.autocast()  # type: ignore
    def predict(
        self,
        lr_image: Image,
        batch_size: int = 4,
        patches_size: int = 192,
        padding: int = 24,
        pad_size: int = 15,
    ) -> Image:
        scale = self.scale
        device = self.device
        lr_image = np.array(lr_image)
        lr_image = pad_reflect(lr_image, pad_size)

        patches, p_shape = split_image_into_overlapping_patches(
            lr_image, patch_size=patches_size, padding_size=padding
        )
        img = torch.FloatTensor(patches / 255).permute((0, 3, 1, 2)).to(device).detach()

        with torch.no_grad():
            # this part could be async maybe
            res = self.model(img[0:batch_size])
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat((res, self.model(img[i : i + batch_size])), 0)

        sr_image = res.permute((0, 2, 3, 1)).clamp_(0, 1).cpu()
        np_sr_image = sr_image.numpy()

        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        np_sr_image = stich_together(
            np_sr_image,
            padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape,
            padding_size=padding * scale,
        )
        sr_img = (np_sr_image * 255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size * scale)
        sr_img = Image.fromarray(sr_img)

        return sr_img

    def generate(self, input_path: str, result_image_path: str) -> None:
        if not result_image_path:
            result_image_path = input_path.replace("inputs/", "results/")
        image = Image.open(input_path).convert("RGB")
        sr_image = self.predict(np.array(image))
        sr_image.save(result_image_path)
