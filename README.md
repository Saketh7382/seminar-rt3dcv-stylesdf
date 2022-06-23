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
<img width=500px" class="image" src="./images/fig_01.png"/>
</div>
<p align="center" class="figure">Fig.01 The goal of StyleSDF</p>
<br/>

<p>
Fig.02 summarizes how StyleSDF is implemented to achieve it's goals. As we can see that StyleSDF first generates a view-consistent 3D shape, from which it then extracts a 64x64 RGB image and its corresponding feature vector. Later, it combines both the low resolution RGB-image and feature vector to generate a high resolution 1024x1024 RGB-image that is 3D-consistent.
</p>

<br/>
<div align="center">
<img width=500px" class="image" src="./images/fig_02.png"/>
</div>
<p align="center" class="figure">Fig.02 High level view of StyleSDF algorithm</p>
<br/>

<p>Before discussing about StyleSDF more detailly, Let us look at some relavant topics and related works that better equip us to understand the technique in a much efficient manner.</p>

<h2 class="title" align="center">Relevant Topics</h2>

<h2>Signed Distance Field</h2>
<p>
Signed Distance Field (SDF) is a 3D Volumetric representation in which each 3D spatial coordinate will have a value, called Distance value (Fig.03). This distance value can be either positive, 0 or negative and the value tells us how farther away spatially are we from the nearest point on the surface. A zero distance value at a particular location indicates that there exists a surface at that spatial point. A positive values indicates the distance away from surface in the direction of surface normal and a negative value indicates the distance away from the surface in the opposite direction of surface normal.
</p>

<br/>
<div align="center">
<img width=500px" class="image" src="./images/fig_03.png"/>
</div>
<p align="center" class="figure">Fig.03 Signed Distance Field</p>
<br/>

<h2>Neural Rendering</h2>
<p>
The concept of neural rendering combines ideas from classical computer graphics and machine learning to create algorithms for synthesizing images from real-world observations <a href="#5">[5]</a>. Neural rendering is 3D consistent by design and it enables applications such as novel viewpoint synthesis of a captured scene.
</p>

<br/>
<div align="center">
<img width=500px" class="image" src="./images/fig_04.png"/>
</div>
<p align="center" class="figure">Fig.04 Neural Rendering</p>
<br/>

<h2>Neural Radiance Field - NeRF</h2>
<p>
Neural Radiance Field is a functional representation that jointly models geometry and appearance, and is able to model view-dependent effects <a href="#6">[6]</a>. Fig.05 shows the Radiance field function. The function takes 5D cordinates (3 spacial coordinates + 2 viewing directions) as input and produces Radiance field (view-dependent emmitted radiance + Volume density) as output.
</p>

<br/>
<div align="center">
<img width=500px" class="image" src="./images/fig_05.png"/>
</div>
<p align="center" class="figure">Fig.05 Radiance Field Function</p>
<br/>

<p>
NeRF is a technique that introduced the use of volume rendering for reconstructing a 3D scene using Radiance Field to synthesize novel views (Fig.06). 
</p>

<br/>
<div align="center">
<img width=500px" class="image" src="./images/fig_06.png"/>
</div>
<p align="center" class="figure">Fig.06 NeRF</p>
<br/>

<h2>Generative Adversarial Networks - GANs</h2>
<p>
Generative Adversarial Networks, or GANs for short, are an approach to generative modeling using deep learning methods, such as convolutional neural networks (Fig.07). GANs can synthesize high-resolution RGB images that are practically indistinguishable from real images. You can learn more about GANs <a href="https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/" target="_blank">here</a>.
</p>

<br/>
<div align="center">
<img width=500px" class="image" src="./images/fig_07.png"/>
</div>
<p align="center" class="figure">Fig.07 GAN Architecture</p>
<br/>

<h2>StyleGAN & StyleGAN2</h2>
<p>
<b>StyleGAN</b> <a href="#7">[7]</a> is the current state-of-the-art method for high-resolution image synthesis (Fig.08 a) and <b>StyleGAN2</b> <a href="#8">[8]</a> extends the work of StyleGAN by focusing on fixing StyleGAN’s characteristic artifacts & improving the result quality further (Fig.08 b). We can observe from Fig.08 a that the output of StyleGAN contains some artifacts at bottom right corner, which are eliminated in the output of StyleGAN2 as we can see at Fig.08 b.

</p>

<br/>
<div align="center">
<img width=500px" class="image" src="./images/fig_08.png"/>
</div>
<p align="center" class="figure">Fig.08 a) StyleGAN b) StyleGAN2</p>
<br/>

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

<br/>
<div align="center">
<img width=500px" class="image" src="./images/fig_09.png"/>
</div>
<p align="center" class="figure">Fig.09 Pi-GAN</p>
<br/>

<h2 class="title" align="center">How StyleSDF works?</h2>

<p>
Back to the original topic, now let us discuss the algorithm of StyleSDF detailly in this section. Fig.10 shows the overall architecture of StyleSDF. The entire architecture can be mainly divided into 2 components: Mapping Networks, Volume renderer, and 2D Generator.
</p>

<br/>
<div align="center">
<img width=500px" class="image" src="./images/fig_10.png"/>
</div>
<p align="center" class="figure">Fig.10 StyleSDF architecture</p>
<br/>

<h2>Mapping Networks</h2>
<p>
The components Volume Renderer and 2D generator has their own corresponding mapping networks which map the input latent vector into modulation signals for each layer. For simplicity, we ignore the mapping networks and concentrate on the other components in this discussion.
</p>

<h2>Volume Renderer</h2>
<p>
This component takes 5D cordinates (3 spacial coordinates: x + 2 viewing directions: v) as input and outputs SDF value at spatial location x, view-dependent color value at x with view v, and feature vector, represented by d(x), c(x,v) and f(x,v) respectively (Fig.11).
</p>

<br/>
<div align="center">
<img width=500px" class="image" src="./images/fig_11.png"/>
</div>
<p align="center" class="figure">Fig.11 Volume Renderer</p>
<br/>

<p>
Now let us take a step further deep and look at the architecture of the volume renderer (Fig.12). With notations mentioned above, we can see from the figure that volume renderer has 3 FC layers. The first FC layer outputs d(x), SDF value at poisition x. We can use an algorithhm called marching cubes (<a href="https://en.wikipedia.org/wiki/Marching_cubes">learn more</a>) to visualize the 3D shape represented by d(x). Further, the second layer outputs c(x,v) and third FC layer outputs f(x,v), which are view-dependent color value at x with view v, and its corresponding feature vector respectively. If we observe there are two more components in the architecure that require additional description. They are Density function (K-alpha) and Volume aggregation. 
</p>

<br/>
<div align="center">
<img width=500px" class="image" src="./images/fig_12.png"/>
</div>
<p align="center" class="figure">Fig.12 Volume Renderer Architecture</p>
<br/>

<h3><li>Density function</li></h3>
This function controls the tightness of the density around the surface boundary. It takes d(x) as input an produces the density at spacial location x. As we can see from the formula for density function (Fig.13) the output depends on two terms, one being the input d(x) and the other is alpha, which is learned by the network during training. So volume renderer learns the value of alpha in such a way that it controls the tightness of the density around the surface boundary based on the SDF value at that location.
</div>

<br/>
<div align="center">
<img width=500px" class="image" src="./images/fig_13.png"/>
</div>
<p align="center" class="figure">Fig.13 Density function</p>
<br/>

<h3><li>Volume Aggregation</li></h3>
This Component is responsible to generate low resolution view-dependent 64x64 RGB image and its corresponding feature vector by taking the output of density function, c(x,v) and f(x,v) as inputs.
</div>

<br/>
<div align="center">
<img width=500px" class="image" src="./images/fig_14.png"/>
</div>
<p align="center" class="figure">Fig.14 Volume aggreagation formulas</p>
<br/>

<p>
Formulas Description:
</p>
<p>
<li>r(t) = Camera direction</li>
<li>C(r), F(r) = Expected color/feature of camera ray r(t)</li>
<li>T(t) = the probability that the ray travels from tn to t without hitting any other particle.</li>
</p>

<h2>2D Generator</h2>
<p>
This component Aim is to generate High resolution Image at viewpoint v, given feature vector (Fig.15). We have already learned that state-of-the-art techniue to generate High resolution images is StyleGAN2, so StyleSDF uses StyleGAN2 as 2D generator, which takes the 64x64 RGB image and its corresponding feature vector as inputs and produces a High-resolution 1024x1024 RGB image as output.
</p>


<br/>
<div align="center">
<img width=500px" class="image" src="./images/fig_15.png"/>
</div>
<p align="center" class="figure">Fig.15 2D Generator</p>
<br/>

<h2 class="title" align="center">StyleSDF Training</h2>

<p>
In this section, let us briefly discuss about various loss functions implemented to train both volume rendered and 2D generator.
</p>

</html>
