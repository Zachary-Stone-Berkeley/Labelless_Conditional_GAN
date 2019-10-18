# Labelless Conditional GAN

Typically, conditional image synthesis with a generative adversarial network (GAN) is achieved by providing the generator a class label. This code implements a novel (to my knowledge) GAN loss function that allows for image synthesis conditioned on a raw input image. In detail, the GAN is provided both sampled noise and an image to condition on. The noise is up-sampled as usual while the image is down-sampled and the resulting hidden representations are concatenated. But for any standard GAN loss, the optimal generator would act as an identity function on the provided image. The solution I propose is the following loss function:

## Prerequisites

python (3.6.1)  
tf-nightly-gpu (1.13.0.dev20190221)  
numpy (1.12.1)  
scikit-learn (0.18.1)  

Note: Newer versions of scikit-learn will raise an error as it is no longer possible to pass a 1D array to one-hot encoder.

## Running the Code

To run the code, enter

```
python SemiConditionalGAN.py --args
```

where possible args are

```
--noise_dim: The dimensionality of the sampled noise provided to the generator.
--noise_type: The type of noise. Choose between uniform, normal, or truncated.
--loss_type: The loss type. Choose between KL, wasserstein, or hinge.
--batch_size: The batch size.
--num_steps: The total number of updates.
--show_images: Bool. If True, images on test data will be shown every 2500 updates.
--save: Bool. If true, the model will be preiodically saved.
```
## Credits

The arch_ops and losses files are modified from github.com/google/compare_gan.  
The make_montage file used to display a collection of images as a montage follows Parag Mital's concise implementation (see github.com/pkmital/CADL). 
