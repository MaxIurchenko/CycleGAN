CycleGAN: Implementation and Analysis
Introduction
CycleGAN is a type of Generative Adversarial Network (GAN) that enables image-to-image translation without requiring paired examples. It employs two generators and two discriminators to map images between two domains X and Y, ensuring cycle-consistency by enforcing that a transformation from X→Y→X or Y→X→Y returns the original image.
<img width="468" alt="image" src="https://github.com/user-attachments/assets/19ed5b03-0fe3-45e8-af79-28e15350a2a1" />

This project aimed to:
Implement CycleGAN architecture.
Train it on Cityscape and art datasets.
Evaluate its ability to translate cityscape images to artistic styles.
Network Architecture
  <img width="468" alt="image" src="https://github.com/user-attachments/assets/d8c2e087-20c9-44c6-8435-9bae4e87da6e" />

Generator
The generators are based on ResNet architectures with the following characteristics:
Input: 256×256 RGB images.
Downsampling Layers: 2 convolutional layers with instance normalization and ReLU activation. Residual Blocks: 9 ResNet blocks with instance normalization and reflection padding to reduce edge artifacts.
Upsampling Layers: 2 transposed convolutional layers to restore the original image resolution. Output: A 3-channel RGB image using 7×7 convolution followed by Tanh activation.
<img width="468" alt="image" src="https://github.com/user-attachments/assets/3e3662d3-b926-473b-90c7-dc11249f572b" />

Discriminator
The discriminators follow a PatchGAN approach, distinguishing 70×70 image patches as real or fake: Layers: 4 convolutional layers with instance normalization and leaky ReLU activation.
Output: A single-channel output that predicts real/fake status for each patch.
 
Loss Functions
The training process uses the following loss functions:
Adversarial Loss: Encourages generators to produce images indistinguishable from the target domain. Cycle Consistency Loss: Ensures G(F(x))≈x and F(G(y))≈y, maintaining content.
Identity Loss: Preserves color and structure when translating images already in the target domain.
The final generator loss is a weighted sum of these components is Lg=Ladv+λcycleLcycle+λidentityLidentity
Dataset
We used the Cityscapes dataset for real-world images and art dataset of paintings. The datasets were
preprocessed with:
Resizing images to 256×256.
Normalizing pixel values to [−1,1].
Data augmentation: random cropping and horizontal flipping.
Training Setup
Hyperparameters
Learning Rate: 0.0002 with a decay after 100 epochs. Batch Size: 1 (due to computational constraints). Optimizers: Adam optimizers (β1=0.5,β2=0.999). Epochs: Trained for 20 epochs (target: 200 epochs).
Training Procedure
The training loop involved:
Updating generators G and F to minimize adversarial, cycle-consistency, and identity losses. Updating discriminators Dx and Dy to maximize adversarial accuracy.
Results
After 20 epochs, the model's performance was suboptimal:
Translated images lacked detail and often exhibited artifacts.
Cycle consistency was weak, failing to preserve the content of real-world images during transformation.
Intermediate Outputs
Sample outputs (epoch 20) revealed:
Generated Images: Lacked realism and clarity. Cycle-Consistency Images: Significant deviations from input.
<img width="128" alt="image" src="https://github.com/user-attachments/assets/1fefe2d1-8f3e-46e3-912c-9f70680181f6" />

   First tests
    ![image](https://github.com/user-attachments/assets/e40b2dc8-e215-46d0-8eb6-c0d838ffe770)

Epoch 0:
Generator Loss: 6.429175633533745 Discriminator X Loss: 0.43364873708813056 Discriminator Y Loss: 0.48220846662001243
<img width="128" alt="image" src="https://github.com/user-attachments/assets/49125b73-9461-4bc0-875d-b84d047b094c" />

Epoch 1:
Generator Loss: 6.007449833450804 Discriminator X Loss: 0.29539287854341945 Discriminator Y Loss: 0.42873355525599166
<img width="128" alt="image" src="https://github.com/user-attachments/assets/580d9669-ed1e-4a0a-9684-d95bca773411" />

Epoch 2:
Generator Loss: 5.618030848017164 Discriminator X Loss: 0.24251476284946985 Discriminator Y Loss: 0.39511890364158303
         
Training going pretty well!!!
Challenges and Issues
Insufficient Training Time: CycleGAN requires extensive training (200+ epochs) for optimal results.
Dataset Size and Diversity: Limited variation in the training dataset may hinder generalization.
Hyperparameter Tuning: The weights for cycle and identity losses (λcycle, λidentity) need fine-tuning.
Mode Collapse: Discriminators may overpower generators, leading to repetitive outputs.
Proposed Improvements
Increase Training Duration: Extend training to 200 epochs with learning rate decay after 100 epochs.
Enhance Dataset: Include a larger, more diverse dataset with more artistic styles.
Data Augmentation: Apply stronger augmentations like rotation and color jitter.
Loss Balance: Experiment with loss weights to prioritize cycle-consistency.
Gradient Penalty: Incorporate a gradient penalty to stabilize training and reduce artifacts. Pre-trained Models: Use pre-trained CycleGAN weights to accelerate convergence.
Conclusion
The current CycleGAN implementation demonstrates the fundamental capability for image-to-image translation but requires further training and optimization to produce high-quality results. Future work will focus on addressing the challenges identified and implementing the proposed improvements.
References
1. Zhu, J., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.
2. Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Perceptual Losses for Real-Time Style Transfer and Super-Resolution.
3. https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master
4. https://medium.com/@chilldenaya/cyclegan-introduction-pytorch-implementation-
5b53913741ca
   
