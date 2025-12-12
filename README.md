# Emergent Image

Emergent Image allows you to create a video in which an image emerges from the frames of the video. It does this using Stable Diffusions img2img method and with a ControlNet to apply the outline of the emerging image. 

The INPUT is a video or folder of images which form the bakground of the video.

The CONTROL is an image, video, or folder of images in which the outline of the forms emerge from the background frames provided in INPUT. 

Example of combining two still images:

<div style="display:flex; flex-direction:column; gap:18px; align-items:center;">
  <div style="display:flex; justify-content:center; gap:18px;">
    <figure style="margin:0; text-align:center;">
      <img src="assets/background.png" alt="Background" width="220" />
      <figcaption>Background</figcaption>
    </figure>
  </div>
  <div style="display:flex; justify-content:center; gap:18px;">
    <figure style="margin:0; text-align:center;">
      <img src="assets/gingko.png" alt="Control drawing" width="220" />
      <figcaption>Control drawing</figcaption>
    </figure>
    <figure style="margin:0; text-align:center;">
      <img src="assets/gingko_canny.jpg" alt="Canny (ControlNet input)" width="220" />
      <figcaption>Canny (ControlNet input)</figcaption>
    </figure>
  </div>
  <div style="display:flex; justify-content:center; gap:18px;">
    <figure style="margin:0; text-align:center;">
      <img src="assets/output.png" alt="Result" width="220" />
      <figcaption>Result</figcaption>
    </figure>
  </div>
</div>

We can use this to create videos, using the same control image but with changing denoising strength. As we increase the denoising amount, the control image emerges from the background.

<div align="center" style="margin:16px 0;">
  <video width="640" controls muted>
    <source src="assets/gingko-video.mp4" type="video/mp4" />
    Your browser does not support the video tag.
  </video>
</div>

The CONTROL can also be a video and we can use it to create an emergent video:

<!-- <div align="center" style="margin:16px 0;">
  <video width="640" controls muted>
    <source src="media/inputs/circle.mp4" type="video/mp4" />
    Your browser does not support the video tag.
  </video>
</div> -->



<div align="center" style="margin:16px 0;">
  <video width="640" controls muted>
    <source src="assets/circle-video.mp4" type="video/mp4" />
    Your browser does not support the video tag.
  </video>
</div>



