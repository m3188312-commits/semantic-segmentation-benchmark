import torch
import segmentation_models_pytorch as smp


def build_pretrained_unet(
    in_channels: int = 3,
    num_classes: int = 8,
    encoder_name: str = 'resnet34',  # Simple encoder
    encoder_weights: str = 'imagenet',
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    Constructs a pretrained U-Net model using segmentation_models.pytorch.

    :param in_channels: Number of input channels (e.g., 3 for RGB images or 4 for multi-spectral).
    :param num_classes: Number of target segmentation classes.
    :param encoder_name: Name of the encoder backbone (e.g., 'resnet34', 'resnet50').
    :param encoder_weights: Pretrained weights for the encoder. Use 'imagenet' or None.
    :param device: Compute device ('cuda' or 'cpu').
    :return: U-Net model moved to the specified device.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    return model.to(device)


if __name__ == '__main__':
    # Quick sanity check
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_pretrained_unet(device=device)
    x = torch.randn(1, model.encoder.in_channels, 512, 512).to(device)
    y = model(x)
    print(f"Model: {model.__class__.__name__} with encoder '{model.encoder.__class__.__name__}'")
    print('Output shape:', y.shape)
