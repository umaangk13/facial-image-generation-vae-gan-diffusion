# Conditional VAE Architecture (CVAE)

This diagram visualizes the macro-level architecture of the Conditional Variational Autoencoder, showing how the image and label are processed to create a latent space, and how that latent space is decoded back into an image.

```mermaid
flowchart TD
    %% Inputs
    X["Original Image (3x64x64)"] --> Enc_Concat
    L_Enc["Class Label (0 or 1)"] --> Emb_Enc["nn.Embedding (Label -> 64x64 map)"]
    Emb_Enc --> Enc_Concat["Concat (4x64x64)"]
    
    %% Encoder
    subgraph Encoder ["Encoder Network"]
        direction TB
        Enc_Concat --> Conv1["ResBlock (Downsample)"]
        Conv1 --> Conv2["ResBlock (Downsample)"]
        Conv2 --> Conv3["ResBlock (Downsample)"]
        Conv3 --> Conv4["ResBlock (Downsample)"]
        Conv4 --> Flatten["Flatten"]
        
        Flatten --> Mu["Linear (μ)"]
        Flatten --> LogVar["Linear (log σ²)"]
    end
    
    %% Reparameterization
    subgraph Reparam["Reparameterization Trick"]
        Mu --> Z_Calc["z = μ + ε * σ"]
        LogVar -.-> |"Convert to σ"| Z_Calc
        Epsilon["Random Noise ε ~ N(0,1)"] --> Z_Calc
    end
    
    Z_Calc --> Z["Latent Vector z (128-dim)"]
    
    %% Decoder Inputs
    Z --> Dec_Concat
    L_Dec["Class Label (0 or 1)"] --> Emb_Dec["nn.Embedding (Label -> 32-dim)"]
    Emb_Dec --> Dec_Concat["Concat (160-dim)"]
    
    %% Decoder
    subgraph Decoder ["Decoder Network"]
        direction TB
        Dec_Concat --> FC["Linear -> Reshape (512x4x4)"]
        FC --> Deconv1["ResBlockTranspose (Upsample)"]
        Deconv1 --> Deconv2["ResBlockTranspose (Upsample)"]
        Deconv2 --> Deconv3["ResBlockTranspose (Upsample)"]
        Deconv3 --> FinalOut["ConvTranspose2d + Tanh"]
    end
    
    FinalOut --> Out["Reconstructed Image (3x64x64)"]
    
    %% Losses
    Out -.-> |"MSE Loss (vs Original Image)"| Loss["Total VAE Loss"]
    Mu -.-> |"KL Divergence (vs N(0,1))"| Loss
    LogVar -.-> |"KL Divergence (vs N(0,1))"| Loss

    %% Styling
    classDef input fill:#f9f,stroke:#333,stroke-width:2px;
    classDef output fill:#ccf,stroke:#333,stroke-width:2px;
    classDef loss fill:#fcc,stroke:#333,stroke-width:2px;
    
    class X,L_Enc,L_Dec,Epsilon input;
    class Out output;
    class Loss loss;
```

### **Key Viva Takeaways from Architecture**
*   **The Bottleneck**: The entire image is compressed into just two vectors: Mean (μ) and Log-Variance (log(σ²)). This is the core of the Variational Autoencoder.
*   **Reparameterization**: The random noise ε is essential. It allows the network to sample a diverse point `z` from the learned distribution while still allowing backpropagation to push gradients back through μ and log(σ²).
*   **Conditioning (Labels)**: Notice how the class label is injected *twice*. It's given to the Encoder (so it learns separate distributions for glasses vs. no glasses) and the Decoder (so it knows which distribution to generate from).
*   **The Loss Tug-of-War**: The total loss explicitly tries to balance making the generated image match the original (MSE) while forcing $\mu$ and $\log\sigma^2$ to look like a standard normal distribution (KL).
