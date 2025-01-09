# Vision Transformer

*Knowledge provided by* [AI VIET NAM](https://aivietnam.edu.vn/) 

Reimplementation of Vision Transformer model *based on* [The official implementation for Vision-Transformer](https://github.com/google-research/vision_transformer)

<div style="text-align: center;">
    <img src="../images/VIT.png" alt="Vision-Transformers">
</div>

Vision Transformers (ViT) is an architecture that uses self-attention mechanisms to process images. The Vision Transformer Architecture consists of a series of transformer blocks. Each transformer block consists of two sub-layers: a multi-head self-attention layer and a feed-forward layer.


---

## Overview
Self-attention-based architectures, in particular Transformers (Vaswani et al., 2017), have become the model of choice in natural language processing (NLP). The dominant approach is to pre-train on a large text corpus and then fine-tune on a smaller task-specific dataset (Devlin et al., 2019).

In computer vision, however, convolutional architectures remain dominant (LeCun et al., 1989; Krizhevsky et al., 2012; He et al., 2016). Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention (Wang et al., 2018; Carion et al., 2020), some replacing the convolutions entirely (Ramachandran et al., 2019; Wang et al., 2020a). The latter models, while theoretically efficient, have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns. Therefore, in large-scale image recognition, classic ResNetlike architectures are still state of the art.

Inspired by the Transformer scaling successes in NLP, authors experiment with applying a standard Transformer directly to images, with the fewest possible modifications. To do so, **authors split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application**.

---

## Architecture
The standard Transformer receives as input a 1D sequence of token embeddings. To handle 2D images, authors reshape the image $ \bold{x} \in \mathbb{R}^{H \times W \times C}$ into a sequence of flattened 2D patches $\bold{x}_p$ $\in \mathbb{R}^{N \times (P^2.C)}$, where $(H,W)$ is the resolution of the original image, $C$ is the number of channels, $(P,P)$ is the resolution of each image path, and $N = HW/P^2$ is the resulting number of patches, which also serves as the efficitive input length for the Transformer.

**The Transformer uses constant laten vector size $D$ through all of its layers**, so authors flatten the patches and map to $D$ dimensions with a trainable linear projection (Eq.1). Authors refer to the output of this projection as the patch embeddings.  

Similar to BERT's $[\mathnormal{class}]$ token, author prepend a learnable embedding to the sequence of embedded patches ($\bold{z}^0_0 =$ $\bold{x}_{class}$), whose state at the output of the Transformer encoder ($\bold{z}^0_L$) serves as the image representation $\bold{y}$ (Eq.4). Both during pre-training and fine-tuning, a classification head is attached to $\bold{z}^0_L$. The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time.

**Position embeddings** are added to the patch embeddings to retain positional information. Authors use standard learnable 1D position embeddings, since authors have not observed significant performance gains from using more advanced 2D-aware position embeddings (Appendix D.4). The resulting sequence of embedding vectors serves as input to the encoder.

The Transformer encoder consists of:
-   **Multi-Head Self Attention Layer (MSP):** This layer concatenates all the attention outputs linearly to the right dimensions. The many attention heads help train local and global dependencies in an image.
-   **Multi-Layer Perceptrons (MLP) Layer:** This layer contains a two-layer with Gaussian Error Linear Unit (GELU).
-   **Layer Norm (LN):** This is added before each block, as it does not include any new dependencies between the training images. This thereby helps improve the training time and overall performance. Moreover, residual connections are included after each block as they allow the components to flow through the network directly without passing through non-linear activations.

$$
\bold{z}_0  = [\bold{x}_{class}; \ \bold{x}^{1}_{p} \bold{E}; \ \bold{x}^{2}_{p} \bold{E};...; \ \bold{x}^{N}_{p} \bold{E}] \qquad \bold{E} \in \mathbb{R}^{(P^2.C) \times D}, \ \bold{E}_{pos} \in \mathbb{R}^{(N+1) \times D} \hspace{1cm} (1)
$$

$$
\bold{z'}_{l} = MSA(LN(\bold{z}_{l-1})) + \bold{z}_{l-1} \hspace{2.5cm} l = 1...L \hspace{3.3cm} (2)
$$

$$
\bold{z'}_{l} = MSA(LN(\bold{z}_{l})) + \bold{z}_{l}
\hspace{3.5cm} l = 1...L \hspace{3cm} (3)
$$

$$
\bold{y} = LN(\bold{z}^0_L) \hspace{9.7cm} (4)
$$


We can find several proposals for vision transformer models in the literature. The overall structure of the vision transformer architecture consists of the following steps:
1.  Split an image into patches (fixed sizes)
2.  Flatten the image patches
3.  Create lower-dimensional linear embeddings from these flattend image patches
4.  Include positional embeddings
5.  Feed the sequence as an input to a SOTA transformer encoder
6.  Pre-train the VIT model with image labels, then fully supervised on a bid dataset
7.  Fine-tun the downstream dataset for image classification
---

## How does a VIT work?
The performance of a vision transformer model depends on decisions such as that of the optimizer, network depth, and dataset-specific hyperparameters. **Compared to ViT, CNNs are easier to optimize.**

The disparity on a pure transformer is to marry a transformer to a CNN front end. The usual ViT stem leverages a $16 \times 16$ convolution with a $16$ stride. In comparison, a $3 \times 3$ convolution with stride $2$ increases the stability and improves precision.

CNN turns basic pixels into a feature map. Later, a tokenizer translates the feature map into a sequence of tokens and inputs them into the transformer. The transformer then applies the attention technique to create a sequence of output tokens.

Eventually, a projector reconnects the output tokens to the feature map. The latter allows the examination to navigate potentially crucial pixel-level details. This thereby lowers the number of tokens that need to be studied, lowering costs significantly.

Particularly, if the ViT model is trained on huge datasets that are over 14M images, it can outperform the CNNs. If not, the best option is to stick to ResNet or EfficientNet. The vision transformer model is trained on a huge dataset even before the process of fine-tuning. The only change is to disregard the MLP layer and add a new D times KD*K layer. This is where K is the number of classes of the small dataset.

To fine-tune in better resolutions, we perform the 2D representation of the pre-trained position embeddings. This is because the trainable liner layers model the positional embeddings.

---

## Limitted
The challenges of vision transformers are many, and they include issues related to architecture design, generalization, robustness, interpretability, and efficiency.

When trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNets of comparable size. This seemingly discouraging outcome may be expected: **Transformer lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amount of data.**

Additionally, it remains a challenge to fully understand why transformers work well on visual tasks. Furthermore, developing efficient transformer models for computer vision deployable on resource-limited devices is a challenging issue.

However, **the picture changes if the models are trained on larger datasets (14M-300M images)**. Authors find that large scale training trumps inductive bias. Our Vision Transformer (ViT) attains excellent results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints. 

