import torch
import torch.nn as nn

class Patches(nn.Module):
    def __init__(self, 
                 kernel_size = 16, 
                 stride = 16, 
                 padding = 0, 
                 in_chans = 3, 
                 embed_dim = 768):
        
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size = kernel_size, stride = stride, padding = padding
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) 
        
        x = x.permute(0,2,3,1) # B C H W -> B H W C
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim, image_size):
        """ PatchEmbedding
            Parameters:
            -----------
            patch_size: int
                size of a patch (P)
            image_size: int
                size of a image (H or W)
        
        """
        super().__init__()
        
        self.num_patches = (image_size // patch_size) ** 2
        
        self.patches = Patches(kernel_size=patch_size, stride=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        # class token (1, 1, 768)
        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim), requires_grad=True)

        # learnable position embedding
        self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, embed_dim), requires_grad=True)


    def forward(self, images):
        """ Pass image to embed position infomation
            Parameters:
            -----------
            images: tensor
                image from dataset
                shape (..., W, H, C). Example (64, 32, 32, 3)
            
            Returns:
            ---------
            encoded_patches: tensor
            embed patchs with position infomation and concat with class token
            shape (..., S + 1, D) with S = (HW) / (P^2). Example (64, 65, 768)
        
        """

        x = self.patches(images)

        batch_size = x.shape[0]
        hidden_size = x.shape[-1] # hidden_size is embed_dim

        cls_broadcast = torch.broadcast_to(self.cls_token, (batch_size, 1, hidden_size)) # (1, 1, 768) -> (batch_size, 1, hidden_size)

        x_concat = torch.concat((cls_broadcast, x), dim = 1)

        encoded_patches = x_concat + self.position_embedding

        return encoded_patches