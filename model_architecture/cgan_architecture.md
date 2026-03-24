# Conditional GAN Architecture (cGAN)

This diagram visualizes the macro-level architecture of the Conditional Generative Adversarial Network, showing the adversarial minimax game between the Generator and the Discriminator, and how labels are used to condition both networks.

```mermaid
flowchart TD
    %% Generator Inputs
    Z["Random Noise z (128-dim)"] --> G_Concat
    L_G["Class Label (0 or 1)"] --> Emb_G["nn.Embedding (Label -> 32-dim)"]
    Emb_G --> G_Concat["Concat (160-dim)"]
    
    %% Generator
    subgraph Generator ["Generator Network (The Counterfeiter)"]
        direction TB
        G_Concat --> G_FC["Linear -> Reshape (512x4x4)"]
        G_FC --> G_Deconv1["ConvTranspose2d (Upsample)"]
        G_Deconv1 --> G_Deconv2["ConvTranspose2d (Upsample)"]
        G_Deconv2 --> G_Deconv3["ConvTranspose2d (Upsample)"]
        G_Deconv3 --> G_Final["ConvTranspose2d + Tanh"]
    end
    
    G_Final --> FakeImg["Fake Image (3x64x64)"]
    
    %% Discriminator Inputs
    RealImg["Real Image (3x64x64)"] --> D_Input
    FakeImg --> D_Input["Image Input to D"]
    
    L_D["Class Label (0 or 1)"] --> Emb_D["nn.Embedding (Label -> 64x64 map)"]
    Emb_D -.-> |"Stacked as 4th Channel"| D_Input
    
    D_Concat["Concat (4x64x64)"]
    D_Input --> D_Concat
    
    %% Discriminator
    subgraph Discriminator ["Discriminator Network (The Detective)"]
        direction TB
        D_Concat --> D_Conv1["Conv2d (Downsample)"]
        D_Conv1 --> D_Conv2["Conv2d (Downsample)"]
        D_Conv2 --> D_Conv3["Conv2d (Downsample)"]
        D_Conv3 --> D_Conv4["Conv2d (Downsample)"]
        D_Conv4 --> D_Flatten["Flatten"]
        D_Flatten --> D_FC["Linear + Sigmoid"]
    end
    
    D_FC --> Out["Probability (Real=1, Fake=0)"]
    
    %% Losses and Gradients
    Out -.-> |"BCE Loss (Target: 0 for Fake, 1-smooth for Real)"| D_Update["Update D Weights"]
    Out -.-> |"BCE Loss (Target: 1 for Fake)"| G_Update["Update G Weights"]

    %% Styling
    classDef input fill:#f9f,stroke:#333,stroke-width:2px;
    classDef output fill:#ccf,stroke:#333,stroke-width:2px;
    classDef fake fill:#ffc,stroke:#333,stroke-width:2px;
    classDef update fill:#fcc,stroke:#333,stroke-width:2px;
    
    class Z,L_G,RealImg,L_D input;
    class Out output;
    class FakeImg fake;
    class D_Update,G_Update update;
```

### **Key Viva Takeaways from Architecture**
*   **The Minimax Game**: The Generator is trying to maximize the Discriminator's error (make the probability closer to 1 for fakes), while the Discriminator is trying to minimize it (make the probability 0 for fakes and ~1 for reals).
*   **Conditioning differences**: 
    *   In the **Generator**, the class label is simply concatenated to the flat 1D noise vector `z`.
    *   In the **Discriminator**, the class label is embedded into a massive 2D spatial map (`64x64`) and glued underneath the spatial image as a 4th channel. This forces the Discriminator to look at the pixels *in context* of the requested label.
*   **Label Smoothing**: Notice the target for Real images is "1-smooth" (e.g., 0.9). This prevents the Discriminator from becoming overconfident and killing the gradients.
