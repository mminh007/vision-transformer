import torch
import torch.nn as nn
from vit.embedding import PatchEmbedding
from vit.encoder import TransformerEncoder
from torchvision import transforms


class ViT(nn.Module):
    def __init__(self, 
                 depth=12,
                 num_heads=12,
                 embed_dim=768,
                 mlp_dim=3072,
                 num_classes=10,
                 patch_size=16,
                 image_size=224,
                 in_chans = 3,
                 dropout=0.1,
                 norm_eps=1e-12):
        
        super().__init__()

        # self.data_augmentation = nn.Sequential([
        #     transforms.Normalize(),
        #     transforms.Resize(image_size, image_size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(),
        # ])

        self.embedding = PatchEmbedding(patch_size=patch_size, image_size=image_size, embed_dim=embed_dim, in_chans=in_chans)

        self.encoder = TransformerEncoder(
            num_heads=num_heads,
            depth=depth,
            embed_dim=embed_dim,
            dropout=dropout,
            norm_eps=norm_eps
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim, eps=norm_eps),
            nn.Linear(in_features= embed_dim, out_features=mlp_dim),
            nn.Dropout(dropout),
            nn.Linear(in_features=mlp_dim, out_features=num_classes)
        )

        self.layer_norm = nn.LayerNorm(normalized_shape=num_classes, eps=norm_eps)


    def forward(self, inputs):
        
        # augmentation shape (..., image_size, image_size, channels)
        # augmented = self.data_augmentation(inputs)

        # embedding patches
        embedded = self.embedding(inputs)

        # Encoder block
        encoded = self.encoder(embedded)

        # embedded class token
        encoded_cls = encoded[:, 0]

        y = self.layer_norm(encoded_cls)

        # head mlp
        outputs = self.mlp_head(y)

        return outputs


class ViTBase(ViT):
    def __init__(self, num_classes=10, image_size=224, dropout=0.1, norm_eps=1e-12, in_chans = 3):
        super().__init__(depth=12,
                         num_heads=12,
                         embed_dim=768,
                         mlp_dim=3072,
                         patch_size=16,
                         num_classes=num_classes,
                         image_size=image_size,
                         in_chans = in_chans,
                         dropout=dropout,
                         norm_eps=norm_eps)
        

class ViTLarge(ViT):
    def __init__(self, num_classes=10, image_size=224, dropout=0.1, norm_eps=1e-12, in_chans = 3):
        super().__init__(depth=24,
                         num_heads=12,
                         embed_dim=1024,
                         mlp_dim=4096,
                         patch_size=16,
                         num_classes=num_classes,
                         image_size=image_size,
                         in_chans = in_chans,
                         dropout=dropout,
                         norm_eps=norm_eps)


class ViTHuge(ViT):
    def __init__(self, num_classes=10, image_size=224, dropout=0.1, norm_eps=1e-12, in_chans = 3):
        super().__init__(depth=32,
                         num_heads=16,
                         embed_dim=1280,
                         mlp_dim=5120,
                         patch_size=16,
                         num_classes=num_classes,
                         image_size=image_size,
                         in_chans = in_chans,
                         dropout=dropout,
                         norm_eps=norm_eps)


