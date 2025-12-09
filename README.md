# Emergent Image

Emergent Image allows you to create a video in which an image emerges from the frames of the video. It does this using Stable Diffusions img2img method and with a ControlNet to apply the outline of the emerging image. 

The INPUT is a video or folder of images which form the bakground of the video.

The CONTROL is an image, video, or folder of images in which the outline of the forms emerge from the background frames provided in INPUT. 

Example of combining two still images:

<div align="center" style="display:flex; flex-wrap:wrap; gap:18px; justify-content:center;">
  <figure style="margin:0;">
    <img src="media/videos/run14/inputs/resized_input_download-6.png" alt="Background" width="220" />
    <figcaption>Background</figcaption>
  </figure>
  <figure style="margin:0;">
    <img src="media/inputs/gingko.jpg" alt="Control drawing" width="220" />
    <figcaption>Control drawing</figcaption>
  </figure>
  <figure style="margin:0;">
    <img src="media/videos/run14/inputs/raw_control_gingko.png" alt="Canny (ControlNet input)" width="220" />
    <figcaption>Canny (ControlNet input)</figcaption>
  </figure>
  <figure style="margin:0;">
    <img src="media/videos/run14/frames/frame_00039.png" alt="Result" width="220" />
    <figcaption>Result</figcaption>
  </figure>
</div>

We can use this to create videos, using the same control image but with changing denoising strength. As we increase the denoising amount, the control image emerges from the background.

<div align="center" style="margin:16px 0;">
  <video width="640" controls muted>
    <source src="media/videos/run24/video.mp4" type="video/mp4" />
    Your browser does not support the video tag.
  </video>
</div>

The CONTROL can also be a video:

<div align="center" style="margin:16px 0;">
  <video width="640" controls muted>
    <source src="media/inputs/circle.mp4" type="video/mp4" />
    Your browser does not support the video tag.
  </video>
</div>

And we can use it to create an emergent video:

<div align="center" style="margin:16px 0;">
  <video width="640" controls muted>
    <source src="media/videos/run23/video.mp4" type="video/mp4" />
    Your browser does not support the video tag.
  </video>
</div>



