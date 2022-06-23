<html>
<h1 align="center">Paper Review on <b>StyleSDF</b></h1>
<div style="display: flex; flex-direction: column;">
<p>
This Blog article is the review of paper <a href="https://arxiv.org/pdf/2112.11427.pdf" targer="_blank">StyleSDF: High-Resolution 3D-Consistent Image and Geometry Generation</a> published in CVPR 2022 <a href="#1">[1]</a>. First we will look at a brief introduction to the topic, followed by relevant concepts and some related works. Then we will discuss the implementation details in depth, followed by evaluations and results. Finally, we conclude this review by looking at the limitations & future work.
</p>

<h2 class="title" align="center">Introduction</h2>

<p>
2D Image Generation, the task of generating new images, is becoming increasingly popular now a days. Extending this generation task to another dimension brings us to the concept of 3D image generation. Techniques such as GRAF <a href="#2">[2]</a>, HoloGAN <a href="#3">[3]</a>, PiGAN <a href="#4">[4]</a> have made a great amount of contribution in this area. Although these techniques have their pros and cons, overall 3D-image generation poses two main challenges, high resolution view-consistent generation of RGB images and detailed 3D shape generation. <b>StyleSDF</b> <a href="#1">[1]</a> attempts to achieve these challenges. StyleSDF is a technique that generates high resolution 3D-consistent RGB images and detailed 3D shapes, with nover views that are globally aligned, while having stylistic awareness that enables image editing. StyleSDF is trained on single-view RGB data only and Fig.01 summarises the goal of StyleSDF in a concise manner.
</p>

<br/>
<div align="center">
<img width="400px" class="image" src="./images/fig_01.png"/>
</div>
<p align="center" class="figure">Fig.01 The goal of StyleSDF</p>
<br/>

<p>
Fig.02 summarizes how StyleSDF is implemented to achieve it's goals. As we can see that StyleSDF first generates a view-consistent 3D shape, from which it then extracts a 64x64 RGB image and its corresponding feature vector. Later, it combines both the low resolution RGB-image and feature vector to generate a high resolution 1024x1024 RGB-image that is 3D-consistent.
</p>

<br/>
<div align="center">
<img width="400px" class="image" src="./images/fig_02.png"/>
</div>
<p align="center" class="figure">Fig.02 High level view of StyleSDF algorithm</p>
<br/>

<p>Before discussing about StyleSDF more detailly, Let us look at some relavant topics and related works that better equip us to understand the technique in a much efficient manner.</p>

<h2 class="title" align="center">Relevant Topics</h2>

<h2>Signed Distance Field</h2>
<p>
Signed Distance Field (SDF) is a 3D Volumetric representation in which each 3D spatial coordinate will have a value, called Distance value (Fig.03). This distance value can be either positive, 0 or negative and the value tells us how farther away spatially are we from the nearest point on the surface. A zero distance value at a particular location indicates that there exists a surface at that spatial point. A positive values indicates the distance away from surface in the direction of surface normal and a negative value indicates the distance away from the surface in the opposite direction of surface normal.
</p>

<div class="img-container">
<img width="400px" class="image" src="./images/fig_03.png"/>
<span class="figure">Fig.03 Signed Distance Field</span>
</div>

<h2>Neural Rendering</h2>
<p>
The concept of neural rendering combines ideas from classical computer graphics and machine learning to create algorithms for synthesizing images from real-world observations <a href="#5">[5]</a>. Neural rendering is 3D consistent by design and it enables applications such as novel viewpoint synthesis of a captured scene.
</p>

<div class="img-container">
<img width="400px" class="image" src="./images/fig_04.png"/>
<span class="figure">Fig.04 Neural Rendering</span>
</div>

<h2>Neural Radiance Field - NeRF</h2>
<p>
Neural Radiance Field is a functional representation that jointly models geometry and appearance, and is able to model view-dependent effects <a href="#6">[6]</a>. Fig.05 shows the Radiance field function. The function takes 5D cordinates (3 spacial coordinates + 2 viewing directions) as input and produces Radiance field (view-dependent emmitted radiance + Volume density) as output.
</p>

<div class="img-container">
<img width="400px" class="image" src="./images/fig_05.png"/>
<span class="figure">Fig.05 Radiance Field Function</span>
</div>

<p>
NeRF is a technique that introduced the use of volume rendering for reconstructing a 3D scene using Radiance Field to synthesize novel views (Fig.06). 
</p>

<div class="img-container">
<img width="400px" class="image" src="./images/fig_06.png"/>
<span class="figure">Fig.06 NeRF</span>
</div>

<h2>Generative Adversarial Networks - GANs</h2>
<p>
Generative Adversarial Networks, or GANs for short, are an approach to generative modeling using deep learning methods, such as convolutional neural networks (Fig.07). GANs can synthesize high-resolution RGB images that are practically indistinguishable from real images. You can learn more about GANs <a href="https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/" target="_blank">here</a>.
</p>

<div class="img-container">
<img width="400px" class="image" src="./images/fig_07.png"/>
<span class="figure">Fig.07 GAN Architecture</span>
</div>

<h2>StyleGAN & StyleGAN2</h2>
<p>
<b>StyleGAN</b> <a href="#7">[7]</a> is the current state-of-the-art method for high-resolution image synthesis (Fig.08 a) and <b>StyleGAN2</b> <a href="#8">[8]</a> extends the work of StyleGAN by focusing on fixing StyleGAN’s characteristic artifacts & improving the result quality further (Fig.08 b). We can observe from Fig.08 a that the output of StyleGAN contains some artifacts at bottom right corner, which are eliminated in the output of StyleGAN2 as we can see at Fig.08 b.

</p>

<div class="img-container">
<img width="400px" class="image" src="./images/fig_08.png"/>
<span class="figure">Fig.08 a) StyleGAN b) StyleGAN2</span>
</div>

<h2 class="title" align="center">Related Works</h2>
<h2>Single-View Supervised 3D-Aware GANs</h2>

<p>
In the previous section, we had a brief introduction to what GANs are. Now let us look at 3D-Aware GANs. A GAN that generates images that are 3D-consistent are known as 3D-aware GANs. Extending on that concept, Single-View Supervised 3D-Aware GANs are those GANs, that are trained only on single-view RGB data. In contrast, NeRF is Multi-view supervised GAN, meaning that NeRF takes multiple views of same scene to train and generate 3D aware images. Some of the popular Single-View Supervised 3D-Aware GANs are:
</p>

* GRAF
* Pi-GAN
* HoloGAN
* StyleNeRF

<h2>Pi-GAN</h2>

<p>
Pi-GAN is one of the most advanced Single-View Supervised 3D-Aware GAN that achieves the same goal as StyleSDF. That is why Pi-GAN is a strong baseline for evaluating the results of StyleSDF. Pi-GAN, as NeRF, generates 3D shapes using radiance fields. Fig.09 describes Pi-GAN in a concise manner. 
</p>

<div class="img-container">
<img width="400px" class="image" src="./images/fig_09.png"/>
<span class="figure">Fig.09 Pi-GAN</span>
</div>

<h2 class="title" align="center">How StyleSDF works?</h2>

<p>
Back to the original topic, now let us discuss the algorithm of StyleSDF detailly in this section. Fig.10 shows the overall architecture of StyleSDF. The entire architecture can be mainly divided into 2 components: Mapping Networks, Volume renderer, and 2D Generator.
</p>

<div class="img-container">
<img width="400px" class="image" src="./images/fig_10.png"/>
<span class="figure">Fig.10 StyleSDF architecture</span>
</div>

<h2>Mapping Networks</h2>
<p>
The components Volume Renderer and 2D generator has their own corresponding mapping networks which map the input latent vector into modulation signals for each layer. For simplicity, we ignore the mapping networks and concentrate on the other components in this discussion.
</p>

<h2>Mapping Networks</h2>
<p>
The components Volume Renderer and 2D generator has their own corresponding mapping networks which map the input latent vector into modulation signals for each layer. For simplicity, we ignore the mapping networks and concentrate on the other components in this discussion.
</p>

</div>

</html>
