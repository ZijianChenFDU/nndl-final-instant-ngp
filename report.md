# 3D Generation using Instant-NGP

## Github repository

- Instant-NGP (source): [https://github.com/NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp) 
- Our GitHub repo: [https://github.com/ZijianChenFDU/nndl-final-instant-ngp](https://github.com/ZijianChenFDU/nndl-final-instant-ngp)
- Baidu Cloud:
  - [Instant-NGP (our own data sets)](https://pan.baidu.com/s/1EnPhPK0G0YvNiNfH8OCGLA?pwd=x8ew)
  - [Requirements](https://pan.baidu.com/s/1QoyVv9NJitTBoE42pR-jBw?pwd=mq26)



Detailed steps for installation, training, and testing are provided in our [README](https://github.com/ZijianChenFDU/nndl-final-instant-ngp/blob/main/README.md) file on the GitHub repository.


## Paper Summary

### Notes on NeRF

NeRF is a 3D rendering technique that utilizes neural networks to generate three-dimensional objects/scenes from a sparse set of two-dimensional images. From a graphics perspective, it is essentially a differentiable rendering technique, meaning that its rendering process is a differentiable mapping. The neural network it uses is actually just a regular Multi-Layer Perceptron (MLP), but it produces impressive rendering results, which has led to widespread attention. It works by taking input images representing a scene and interpolating between them to render one complete scene. 

NeRF employs the classical volume rendering technique. Volume rendering is a method of generating images by accumulating or integrating light rays and was originally used for rendering non-rigid objects such as clouds. In NeRF, the MLP models the points along the rays in the three-dimensional object as volumetric representations with color and density values, which enables the generation of the shape of the object from unseen viewpoints based on images from a sparse set of given perspectives.


By representing the scene as a continuous 5D radiance field in a neural network, NeRF can generate scene images from input camera poses. As in Figure 1, based on the given position $\mathbf{x} = (x, y, z)$ and viewing direction $(\theta, \varphi)$ of a point, the neural network $\mathbf{F}_\theta$ outputs the emissive color $\mathbf{c}=(r,g,b)$ in that direction and the voxel density $\sigma$ of the point. Then, the volume rendering equation is used to render the color of the pixels, and the network is trained by calculating the error between the rendered color and the ground truth color.


<center>
<img src="demo\report_images\nerf.png"></img>
<p>Figure 1 The Structure of NeRF</p>
</center>

The neural network structure of NeRF is not complex. the MLP $\mathbf{F}_\theta$ first processes the input 3D coordinate $\mathbf{x}$ with 8 fully-connected layers (using ReLU activations and 256 channels per layer), and outputs $\sigma$ and a 256-dimensional feature vector. This feature vector is then concatenated with the camera ray’s viewing direction and passed to one additional fully-connected layer (using a ReLU activation and 128 channels) that output the view-dependent RGB color.

In the original NeRF paper, $\mathbf{F}_\theta=\mathbf{F}_\theta' \circ \gamma$ is a composite function consisting of the MLP  $\mathbf{F}_\theta'$ and a positional encoding 

$$
\gamma(p)=\left(\sin \left(2^0 \pi p\right), \cos \left(2^0 \pi p\right), \cdots, \sin \left(2^{L-1} \pi p\right), \cos \left(2^{L-1} \pi p\right)\right).
$$

This function $\gamma(\cdot)$ is applied separately to each of the three coordinate values in $\mathbf{x}$ (which are normalized to lie in $[-1,1]$ ) and to the three components of the Cartesian viewing direction unit vector $\mathbf{d}$ (which by construction lie in $[-1,1]$ ). In our experiments, we set $L=10$ for $\gamma(\mathbf{x})$ and $L=4$ for $\gamma(\mathbf{d})$.

for each viewing direction $\mathbf{d}=(\theta, \varphi)$ and each distance coefficient $t$, a camera ray is defined by $\mathbf{r}=\mathbf{o}+t\mathbf{d}$. The volume rendering algorithm combines the $\sigma$ output and the RGB output together using the following formula:

$$
\hat{C}(\mathbf{r})=\sum_{i=1}^N T_i\left(1-\exp \left(-\sigma_i \delta_i\right)\right) \mathbf{c}_i, \text { where } T_i=\exp \left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)
$$

where $\delta_i=t_{i+1}-t_i$ is the distance between adjacent samples. The whole pipline of NeRF is demonstrated in Figure 2.

<center>
<img src="demo\report_images\nerf_pipeline.png"></img>
<p>Figure 2 The Whole Pipeline of NeRF</p>
</center>

### Main Contributions of Instant-NGP

The MLP used in NeRF suffers from the issue of a large number of parameters, leading to longer training times. Instant Neural Graphics Primitives (Instant-NGP) addresses this problem by replacing the original input encoding method with multiresolution hash encoding. This new input encoding allows for the use of a smaller MLP without compromising the quality of the generated images. With significantly reduced network layers and parameter counts, both the computational time and memory consumption for floating-point operations are reduced. Specifically, the rendering time for a 3D scene is shortened to around 20 seconds.


It is important to note that Instant-NGP is not just an improvement for NeRF. The proposed multiresolution hash encoding can be utilized for multiple 3D vision tasks, including:


1. Gigapixel image: the MLP learns the mapping from 2D coordinates to RGB colors of a high-resolution image.
2. Neural signed distance functions (SDF): the MLP learns the mapping from 3D coordinates to the distance to a surface.
3. Neural radiance caching (NRC): the MLP learns the 5D light field of a given scene from a Monte Carlo path tracer.
4. Neural radiance and density fields (NeRF): the MLP learns the 3D density and 5D light field of a given scene from image observations and corresponding perspective transforms.

### Methods of Input Encoding

Low-dimension-to-high-dimension input encoding is an important field in the machine learning literature. Early examples include the one-hot encoding and the kernel method. Recall that NeRF encodes the feature vectors through a explicit formula:

$$
\gamma(p)=\left(\sin \left(2^0 \pi p\right), \cos \left(2^0 \pi p\right), \cdots, \sin \left(2^{L-1} \pi p\right), \cos \left(2^{L-1} \pi p\right)\right).
$$

This method is referred as a type of **frequency encodings**.


Recently, state-of-the-art results have been achieved by **parametric encodings** which blur the line between classical data structures and neural approaches. The idea is to arrange additional trainable parameters (beyond weights and biases) in an auxiliary data structure, such as a grid or a tree, and to look-up and (optionally) interpolate these parameters depending on the input vector $x \in \mathbb{R}^d$. This arrangement trades a larger memory footprint for a smaller computational cost: whereas for each gradient propagated backwards through the network, every weight in the fully connected MLP network must be updated, for the trainable input encoding parameters (“feature vectors”), only a very small number are affected. For example, with a trilinearly interpolated 3D grid of feature vectors, only 8 such grid points need to be updated for each sample back-propagated to the encoding. In this way, although the total number of parameters is much higher for a parametric encoding than a fixed input encoding, the number of FLOPs and memory accesses required for the update during training is not increased significantly. By reducing the size of the MLP, such parametric models can typically be trained to convergence much faster without sacrificing approximation quality.


The method proposed in Instant-NGP combines both ideas to reduce waste. The trainable feature vectors stored in a compact spatial hash table, whose size is a hyper-parameter $T$ which can be tuned to trade the number of parameters for reconstruction quality. Instant-NGP uses multiple separate hash tables indexed at different resolutions, whose interpolated outputs are concatenated before being passed through the MLP.

Instant-NGP does not explicitly handle collisions of the hash functions. Instead, it relies on the neural network to learn to disambiguate hash collisions itself, avoiding control flow divergence, reducing implementation complexity and improving performance.


### Multiresolution Hash Encoding

Given an MLP $m(\mathbf{y} ; \Phi)$, we are interested in an encoding of its inputs $\mathbf{y}=\mathrm{enc}(\mathbf{x} ; \theta)$. Our neural network not only has trainable weight parameters $\Phi$, but also trainable encoding parameters $\theta$. These are arranged into $L$ levels, each containing up to $T$ feature vectors with dimensionality $F$. Typical values for these hyperparameters are shown in Table 1. Figure 3 illustrates the steps performed in our multiresolution hash encoding. Each level (two of which are shown as red and blue in the figure) is independent and conceptually stores feature vectors at the vertices of a grid, the resolution of which is chosen to be a geometric progression between the coarsest and finest resolutions $\left[N_{\min }, N_{\max }\right]$ :
$$
\begin{aligned}
N_l & :=\left\lfloor N_{\min } \cdot b^l\right\rfloor, \\
b & :=\exp \left(\frac{\ln N_{\max }-\ln N_{\min }}{L-1}\right) .
\end{aligned}
$$
$N_{\max }$ is chosen to match the finest detail in the training data. Due to the large number of levels $L$, the growth factor is usually small. Our use cases have $b \in[1.26,2]$.


Table 1 Hash encoding parameters and their ranges
| Parameter | Symbol | Value |
| :--- | :--- | ---: |
| Number of levels | $L$ | $16$ |
| Max. entries per level (hash table size) | $T$ | $2^{14}$ to $2^{24}$ |
| Number of feature dimensions per entry | $F$ | $2$ |
| Coarsest resolution | $N_{\min}$ | $16$ |
| Finest resolution | $N_{\max}$ | $512$ to $524288$ |

Consider a single level $l$. The input coordinate $\mathbf{x} \in \mathbb{R}^d$ is scaled by that level's grid resolution before rounding down and up $\left\lfloor\mathbf{x}_l\right\rfloor:=$ $\left\lfloor\mathbf{x} \cdot N_l\right\rfloor,\left\lceil\mathbf{x}_l\right\rceil:=\left\lceil\mathbf{x} \cdot N_l\right\rceil$.


$\left\lfloor\mathbf{x}_l\right\rfloor$ and $\left\lceil\mathbf{x}_l\right\rceil$ span a voxel with $2^d$ integer vertices in $\mathbb{Z}^d$. We map each corner to an entry in the level's respective feature vector array, which has fixed size of at most $T$. For coarse levels where a dense grid requires fewer than $T$ parameters, i.e. $\left(N_l+1\right)^d \leq T$, this mapping is $1: 1$. At finer levels, we use a hash function $h: \mathbb{Z}^d \rightarrow \mathbb{Z}_T$ to index into the array, effectively treating it as a hash table, although there is no explicit collision handling. We rely instead on the gradient-based optimization to store appropriate sparse detail in the array, and the subsequent neural network $m(\mathbf{y} ; \Phi)$ for collision resolution. The number of trainable encoding parameters $\theta$ is therefore $O(T)$ and bounded by $T \cdot L \cdot F$ which in our case is always $T \cdot 16 \cdot 2$ (Table 1). We use a spatial hash function of the form
$$
h(\mathbf{x})=\left(\bigoplus_{i=1}^d x_i \pi_i\right) \bmod T,
$$
where $\oplus$ denotes the bit-wise XOR operation and $\pi_i$ are unique,large prime numbers. Notably, to achieve (pseudo-)independence, only $d-1$ of the $d$ dimensions must be permuted, so we choose $\pi_1:=1$ for better cache coherence, $\pi_2=2654435761$, and $\pi_3=805459861$.


Lastly, the feature vectors at each corner are $d$-linearly interpolated according to the relative position of $\mathbf{x}$ within its hypercube, i.e. the interpolation weight is $\mathbf{w}_l:=\mathbf{x}_l-\left\lfloor\mathbf{x}_l\right\rfloor$.

Recall that this process takes place independently for each of the $L$ levels. The interpolated feature vectors of each level, as well as auxiliary inputs $\xi \in \mathbb{R}^E$ (such as the encoded view direction and textures in neural radiance caching), are concatenated to produce $\mathbf{y} \in \mathbb{R}^{L F+E}$, which is the encoded input enc $(\mathbf{x} ; \theta)$ to the MLP $m(\mathbf{y} ; \Phi)$.


Figure 3 illustrates of the multiresolution hash encoding in 2D. (1) for a given input coordinate $\mathbf{x}$, we find the surrounding voxels at $L$ resolution levels and assign indices to their corners by hashing their integer coordinates. (2) for all resulting corner indices, we look up the corresponding $F$-dimensional feature vectors from the hash tables $\theta_l$ and (3) linearly interpolate them according to the relative position of $\mathbf{x}$ within the respective $l$-th voxel. (4) we concatenate the result of each level, as well as auxiliary inputs $\xi \in \mathbb{R}^E$, producing the encoded MLP input $y \in \mathbb{R}^{L F+E}$, which (5) is evaluated last. To train the encoding, loss gradients are backpropagated through the MLP (5), the concatenation (4), the linear interpolation (3), and then accumulated in the looked-up feature vectors.

<center>
<img src="demo\report_images\hashing.png"></img>
<p>Figure 3 Illustration of the multiresolution hash encoding in 2D</p>
</center>


The authors argues that the presence of hash collisions will not influence the quality of reconstructed scenes with a series of reasons. In practice, when training samples collide in this way, their gradients average. More details on hash collisions are discussed in the original paper.


### The Network Architecture of Instant-NGP

 For this part, we only focus on the NeRF application of Instant-NGP. Similar to the vanilla NeRF model, the Instant-NGP NeRF model consists of two concatenated MLPs: a density MLP followed by a color MLP. The density MLP maps the hash encoded position $\mathbf{y}=\mathrm{enc}(\mathbf{x} ; \theta)$ to 16 output values, the first of which we treat as log-space density. The color MLP adds view-dependent color variation. Its input is the concatenation of
- the 16 output values of the density MLP, and
- the view direction projected onto the first 16 coefficients of the spherical harmonics basis (i.e. up to degree 4). This is a natural frequency encoding over unit vectors.

Its output is an RGB color triplet, for which we use either a sigmoid activation when the training data has low dynamic-range (sRGB) or an exponential activation when it has high dynamic range (linear HDR). For either case, the results were generated with a 1-hidden-layer density MLP and a 2-hidden-layer color MLP, both 64 neurons wide, which is far less complicated than the vanilla NeRF MLP. The Instant-NGP NeRF pipeline is depicted in Figure 4.

<center>
<img src="demo\report_images\ngp_pipeline.png"></img>
<p>Figure 4 The Pipeline of Instant-NGP NeRF</p>
</center>



## Experiment Procedure

This section utilizes FFmpeg+COLMAP+Instant-NGP to create 3D models of selected statues in the campus of Fudan University. A detailed introduction of the implementation of the experiments in Chinese is presented on [the GitHub repository](https://github.com/ZijianChenFDU/nndl-final-instant-ngp).

### The Dataset

Our experiment is based on a unique data set on the statues in Fudan University, including the statue of Su Buqing(苏步青), the statue of Chen Wangdao(陈望道), the “The Books and the Clock” in front of the 4th Teaching Building, and the “The poet pondering while seated on a donkey”(驴背诗思) on the corner of Guanghua Building. The original video was captured using a smartphone. Extracted frames are shown in Figure 5.

<center>
<img src="demo\report_images\original.png"></img>
<p>Figure 5 Extracted frames from the videos of the statues</p>
</center>

### The Environment

The environment in this experiment is quite special. The source code in the Instant-NGP repository were written in CUDA and compiled using CMAKE. We performed these operations on Windows, so the obtained `instant-ngp.exe` can only be run on Windows. The environment used for our project, hosted on the server provided by [matgo.cn](https://matgo.cn), includes:

- Windows Server 2019
- Python 3.8
- Pytorch 1.12
- CUDA 11.3
- cuDNN 8.2.0

#### Installation of Requirements

First, you should make sure that these softwares are installed on your computer.

- [COLMAP](https://demuc.de/colmap/#download) (necessary)
- [FFmpeg](https://ffmpeg.org/download.html) (necessary)
- [VLC Media Player](https://www.videolan.org/vlc/) (suggested)
- [Infranview](https://www.irfanview.com/) (suggested)

You can also just copy the installed programs from our  [Baidu Cloud Link](https://pan.baidu.com/s/1QoyVv9NJitTBoE42pR-jBw?pwd=mq26) by directly downloading and unzip the `Downloads.7z` file and you do not need to install them. You **must** add the directories of `colmap.exe` and `ffmpeg.exe` to the PATH of your system. Also, for the COLMAP  folder, it is necessary to copy **all files** in `lib` (**not the subfolder**) to `bin`.


The `./Instant-NGP-for-RTX-2000/scripts/colmap2nerf.py` script in the folder will install COLMAP and FFmpeg automatically from the Internet. But the speed can be slow due to unstability of the network connection. Also, the script cannot automatically add COLMAP and FFmpeg to the system PATH. Therefore, you are suggested to install them by yourself before your inplementation.

Instant-NGP can be downloaded from its own [repository](https://github.com/NVlabs/instant-ngp) or directly from our [Baidu Cloud Link](https://pan.baidu.com/s/1EnPhPK0G0YvNiNfH8OCGLA?pwd=x8ew) where our data sets are already included.

#### Required Python Packages
- `commentjson`
- `imageio`
- `numpy`
- `opencv-python-headless`
- `pybind11`
- `pyquaternion`
- `scipy`
- `tqdm`

You can install them through `requirements.txt`
```
cd [your-own-path]\Instant-NGP-for-RTX-2000
pip install -r requirements.txt
```

### Training and Testing Guide

#### Exploring the trained example

All the following codes can be processed by the command line in Windows.
```
cd [your-own-path]\Instant-NGP-for-RTX-2000
instant-ngp subuqing
```
In the above commands, `subuqing` can be replaced by `chenwangdao`, `sijiao`, or `lvbeishisi`.

#### Training a new case

The statue of Maxiangbo(马相伯) in the folder is an untrained example. You can render your own scene with the following steps.

- **Step 1:** Use FFmpeg to split an input video in MP4 format into frames at a rate of two frames per second and save them as images.

```
cd [your-own-path]\Instant-NGP-for-RTX-2000
python scripts\colmap2nerf.py --video_in maxiangbo\maxiangbo.mp4 --video_fps 2 --run_colmap --aabb_scale 32 --overwrite
```
- **Step 2:** The generated images are placed in the `images` subfolder under the folder where the corresponding video resides. According to an official guide, you are advised to delete any blurry frames or frames that are irrelevant to the objects that need to be modeled before proceeding to the next step.


- **Step 3:** Use COLMAP to generate a dataset and determine the mapping relationship between images and 3D scene viewpoints. COLMAP will automatically calculate the center coordinates of the scene and the maximum coordinates along the positive and negative directions of the x, y, and z axes.

```
cd maxiangbo
python [your-own-path]\Instant-NGP-for-RTX-2000\scripts\colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 32 --overwrite
```
- **Step 4:** Check your result.
```
cd ..
instant-ngp maxiangbo
```

## Experiment Results

This section shows the results of the four statues (Figure 6 to Figure 9). The original results of Instant-NGP is a volume dataset. Users can generate MP4 videos of given view points within the Instant-NGP program. The results generated by Instant-NGP show the entire scene captured by the camera. In the program interface, you can set the spatial coordinate range to display only the objects of interest on the screen. The following figures show the extracted frames of the output videos.



<center>
<img src="demo\report_images\subuqing_video.png"></img>
<p>Figure 6 Constructed 3D Model of the Statue of Buqing Su</p>
</center>

<center>
<img src="demo\report_images\chenwangdao_video.png"></img>
<p>Figure 7 Constructed 3D Model of the Statue of Wangdao Chen</p>
</center>

<center>
<img src="mat_screen_images\sijiao_video.png"></img>
<p>Figure 8 Constructed 3D Model of “The Books and the Clock”</p>
</center>

<center>
<img src="demo\report_images\lvbeishisi_video.png"></img>
<p>Figure 9 Constructed 3D Model of “The poet pondering while seated on a donkey”</p>
</center>

## Discussions

In this section, the strong and weak points of the rendering quality of Instant-NGP NeRF are discussed. We find that Instant-NGP has very fast rendering speed and an excellent reconstruction of local details. However, its result may be not as good when inplementing a “out-of-sample” viewing direction. Also, the rendering of the backgrounds is not robust.

### High Speed of Rendering

Instant-NGP exhibits a very fast convergence speed, taking only around 30 seconds to generate a recognizable 3D scene. Generally, the quality of the rendering no longer improves after rendering for approximately 2-3 minutes. The display results below are all scenes obtained from the final rendering. Figure 10 exhibits the rendering process of a scene from initialization to convergence.

<center>
<img src="demo\report_images\rendering.png"></img>
<p>Figure 10 The Rendering Process of “The Books and the Clock”</p>
</center>

### High-quality portrayal of details

The results generated by Instant-NGP have a very high quality. Figure 11 showcases the details of the flowers and Chinese characters in the generated scene. Both are portrayed very close to reality.


<center>
<img src="demo\report_images\details.png"></img>
<p>Figure 11 The Details of the Flowers (a) and Chinese Characters (b) Rendered in the 3D Scene</p>
</center>

### Distortions in out-of-sample viewing directions


When observing a scene from an out-of-sample view, there may be some distortions. In our experiments, when moving the coordination to an “highly” out-of-sample direction (For example, most of our original videos lacks the perspective of a bird's-eye view (Figure 12) or a worm's-eye view) may display blurry or distorted details, and there may be geometric shape distortions. This is a common issue that can be encountered in many existing rendering engines, which also plagues Instant-NGP.

As depicted in Figure 12, when flipping the object vertically, it is noticed that the rendering quality of both the object and the background is excellent from the frontal view perspective (the second line). However, in the bird's-eye (the last two lines) and worm's-eye views (the first line), the background becomes very blurry, and the quality of the central object's rendering also significantly decreases.

<center>
<img src="demo\report_images\vertical.png"></img>
<p>Figure 12 Comparison of the Rendering Quality of Different Viewing Directions</p>
</center>

### Shortcomings in Background Rendering

Compared to the near-perfect portrayal of the object of interest in terms of details, Instant-NGP has many flaws in rendering the background. Sometimes, it generates inexplicable "clouds" in mid-air (Figure 13). Additionally, the quality of the background can undergo significant changes with a change in perspective. When zooming in, it becomes apparent that Instant-NGP actually treats the background as colorful clouds floating in mid-air. Only under specific distances and angles do these "clouds" come together to form a background scene that approaches the actual situation.

<center>
<img src="demo\report_images\clouds.png"></img>
<p>Figure 13 "Clouds" in Mid-air around the Object</p>
</center>

<center>
<img src="demo\report_images\background.png"></img>
<p>Figure 14 The Background is Treated as Colorful Clouds Floating in Mid-air</p>
</center>
