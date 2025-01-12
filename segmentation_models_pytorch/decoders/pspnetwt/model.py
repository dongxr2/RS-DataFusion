from typing import Optional, Union

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead
)
from .decoder import (
    PSPDecoder,
    TransformerBlock
)

class PSPNetWT(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        encoder_depth: int = 3,
        psp_out_channels: int = 512,
        psp_use_batchnorm: bool = True,
        psp_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        upsampling: int = 8,
        aux_params: Optional[dict] = None,
        transformer_embed_size: int = 512,
        transformer_num_heads: int = 8,
        transformer_num_layers: int = 8,
        transformer_forward_expansion: int = 4,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = PSPDecoder(
            encoder_channels=self.encoder.out_channels,
            use_batchnorm=psp_use_batchnorm,
            out_channels=psp_out_channels,
            dropout=psp_dropout,
            transformer_embed_size=transformer_embed_size,
            transformer_num_heads=transformer_num_heads,
            transformer_num_layers=transformer_num_layers,
            transformer_forward_expansion=transformer_forward_expansion,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=psp_out_channels,
            out_channels=classes,
            kernel_size=3,
            activation=activation,
            upsampling=upsampling,
        )

        if aux_params:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "psp-{}".format(encoder_name)
        self.initialize()
