# Conditional UNet Architecture (DDPM)

This diagram visualizes exactly how the noise prediction network takes an image $x_t$, the current timestep $t$, and the class label, and processes them through symmetric Encoder and Decoder paths equipped with ResBlocks and Skip Connections.

```mermaid
flowchart TD
    %% Inputs
    X["Noisy Image x_t (3x64x64)"] --> InitC["Init Conv2d (64 channels)"]
    T["Timestep t"] --> TE["Sinusoidal Time Embedding"]
    L["Class Label (0 or 1)"] --> CE["nn.Embedding Class Embedding"]
    
    TE --> AddEmb["+"]
    CE --> AddEmb
    AddEmb --> Emb["Combined Embedding (emb)"]
    
    %% Downsampling
    subgraph Encoder ["Encoder (Downsampling Path)"]
        direction TB
        InitC --> D1["ResBlocks x2 (64 channels)"]
        D1 --> Down1["Conv2d Downsample"]
        Down1 --> D2["ResBlocks x2 (128 channels)"]
        D2 --> Down2["Conv2d Downsample"]
        Down2 --> D3["ResBlocks x2 (256 channels)"]
        D3 --> Down3["Conv2d Downsample"]
        Down3 --> D4["ResBlocks x2 (512 channels)"]
    end
    
    %% Bottleneck
    subgraph Bottleneck ["Bottleneck (Latent features)"]
        direction TB
        D4 --> Mid["ResBlock x2 (512 channels)"]
    end
    
    %% Upsampling
    subgraph Decoder ["Decoder (Upsampling Path)"]
        direction TB
        Mid --> U4["Concat Skips + ResBlocks x3 (512 channels)"]
        U4 --> Up1["Upsample + Conv2d"]
        Up1 --> U3["Concat Skips + ResBlocks x3 (256 channels)"]
        U3 --> Up2["Upsample + Conv2d"]
        Up2 --> U2["Concat Skips + ResBlocks x3 (128 channels)"]
        U2 --> Up3["Upsample + Conv2d"]
        Up3 --> U1["Concat Skips + ResBlocks x3 (64 channels)"]
    end
    
    %% Skips
    D1 -. "Skip Connection 1" .-> U1
    D2 -. "Skip Connection 2" .-> U2
    D3 -. "Skip Connection 3" .-> U3
    D4 -. "Skip Connection 4" .-> U4
    
    %% Output
    U1 --> Final["GroupNorm + SiLU + Final Conv2d"]
    Final --> Out["Predicted Noise ε (3x64x64)"]
    
    %% Embedding Injection
    Emb -. "Injected into ALL ResBlocks" .-> Encoder
    Emb -. "Injected into ALL ResBlocks" .-> Bottleneck
    Emb -. "Injected into ALL ResBlocks" .-> Decoder
```

### **Key Viva Takeaways from Architecture**
*   **The "U" Shape**: Notice how the image gets squeezed from `64x64` to smaller grids (with increasingly huge channel depths like 512) in the Encoder, then scales back up in the Decoder.
*   **Skip Connections (The Dotted Lines)**: If you compress a face all the way down to a tiny feature map, you lose the high-frequency pixel details (like eyelashes or hair strands). By copying the feature maps from the Encoder and directly concatenating them to the Decoder at the exact same size layer, the Decoder can rebuild the image flawlessly without guessing what the tiny details originally looked like.
*   **The Embedding Injection**: In standard networks, you input data at the top and wait. In a Diffusion UNet, the combined Timestep + Class Label embedding is broadcast and mathematically added directly into the middle of *every single ResBlock* throughout the entire network. This guarantees that whether it is analyzing tiny low-level pixels (Encoder start) or broad high-level features (Bottleneck), the network never forgets what timestep it is predicting noise for.
