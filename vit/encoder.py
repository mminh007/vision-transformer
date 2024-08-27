import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 mlp_dim,
                 drop_out = 0.1,
                 norm_eps=1e-12):
        
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=mlp_dim),
            nn.GELU(),
            nn.Dropout(p=drop_out),
            nn.Linear(in_features=mlp_dim, out_features=embed_dim),
            #nn.Dropout(p=drop_out)
        )

        self.norm = nn.LayerNorm(normalized_shape=embed_dim, eps=norm_eps)
    
    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.mlp(x)

        return x
    

class TransformerBlock(nn.Module):
    def __init__(self,
                 num_heads,
                 embed_dim,
                 mlp_dim,
                 dropout = 0,
                 norm_eps= 1e-12):
        
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                               num_heads=num_heads,
                                               dropout=dropout,
                                               batch_first=True)
        
        self.norm_attention = nn.LayerNorm(normalized_shape=embed_dim, eps=norm_eps)

        self.mlp = MLPBlock(embed_dim=embed_dim, mlp_dim=mlp_dim)
        
        self.norm_mlp = nn.LayerNorm(normalized_shape=embed_dim, eps = norm_eps)

        
    def forward(self, inputs: torch.Tensor):
        """
            Parameters:
            -----------
            inputs: tensor,
                Embedded Patches
                shape (..., num_patches + 1, embed_dim). Example: (64, 65, 768)
            
            Returns:
            --------
            outputs: tensor
                attention + mlp outputs
                shape (..., num_patches + 1, embed_dim). Example: (64, 65, 768)
        """

        norm_attn = self.norm_attention(inputs)

        attention, _ = self.attention(query = norm_attn,
                                   key = norm_attn,
                                   value = norm_attn,
                                   need_weights = False)
        
        # skip connection
        attention += inputs

        # Feed MLP
        mlp_output = self.mlp(self.norm_mlp(attention))

        # skip connection
        outputs = mlp_output + attention

        return outputs
    

class TransformerEncoder(nn.Module):
    def __init__(self,
                 depth = 12,
                 num_heads = 12,
                 embed_dim = 768,
                 mlp_dim = 3072,
                 dropout=0.1,
                 norm_eps=1e-12):
        """
            Transformer Encoder which comprises several transformer layers
            Paramerters:
            ------------
            num_layers: int
                number of transformer layers. Exp: 12
            num_heads: int
                number of heads of multi-head attention layer. Exp: 12
            embed_dim: int
                size of each attention head for value
            ml_dim: int
                mlp size or dimension of hidden layer of mlp block
            dropout: float
                dropout rate of mlp block
            norm_eps: float
                eps of layer norm
        
        """
        super().__init__()
        self.encoder = nn.ModuleList([
            TransformerBlock(num_heads=num_heads,
                             embed_dim=embed_dim,
                             mlp_dim=mlp_dim,
                             dropout=dropout,
                             norm_eps=norm_eps)
            for _ in range(depth)
        ])


        def forward(self, inputs, *args, **kwargs):
            """
                Parameters:
                -----------
                inputs: tensor
                    Embedded Patches
                    shape (..., num_patches + 1, embed_dim). Example: (64, 65, 768)
                
                Returns:
                --------
                outputs: tensor
                    attention + mlp outputs
                    shape (..., num_patches + 1, embed_dim). Example: (64, 65, 768)

            """
            outputs = self.encoder(inputs, *args, **kwargs)

            return outputs



