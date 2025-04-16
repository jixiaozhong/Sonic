# Sonic
Sonic: Shifting Focus to Global Audio Perception in Portrait Animation, CVPR 2025.


<a href='https://jixiaozhong.github.io/Sonic/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
  <a href='https://replicate.com/zsxkib/sonic'><img src='https://replicate.com/zsxkib/sonic/badge'></a>
<a href="http://demo.sonic.jixiaozhong.online/" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
<a href='https://arxiv.org/pdf/2411.16331'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
  <a href="https://huggingface.co/spaces/xiaozhongji/Sonic" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Space-ZeroGPU-orange?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
    </a>
  <a href="https://raw.githubusercontent.com/jixiaozhong/Sonic/refs/heads/main/LICENSE" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>

<p align="center">
    👋 Join our <a href="examples/image/QQ.png" target="_blank">QQ Chat Group</a> 
</p>
<p align="center">


## 🔥🔥🔥 NEWS

**`2025/03/14`**: Super stoked to share that our Sonic is accpted by the CVPR 2025! See you Nashville!!

**`2025/02/08`**: Many thanks to the open-source community contributors for making the ComfyUI version of Sonic a reality. Your efforts are truly appreciated! [**ComfyUI version of Sonic**](https://github.com/smthemex/ComfyUI_Sonic)

**`2025/02/06`**: Commercialization: Note that our license is **non-commercial**. If commercialization is required, please use Tencent Cloud Video Creation Large Model: [**Introduction**](https://cloud.tencent.com/product/vclm) / [**API documentation**](https://cloud.tencent.com/document/api/1616/109378)

**`2025/01/17`**: Our [**Online huggingface Demo**](https://huggingface.co/spaces/xiaozhongji/Sonic/) is released.

**`2025/01/17`**: Thank you to NewGenAI for promoting our Sonic and creating a Windows-based tutorial on [**YouTube**](https://www.youtube.com/watch?v=KiDDtcvQyS0).

**`2024/12/16`**: Our [**Online Demo**](http://demo.sonic.jixiaozhong.online/) is released.


## 🎥 Demo
| Input                | Output                | Input                | Output                |
|----------------------|-----------------------|----------------------|-----------------------|
|<img src="examples/image/anime1.png" width="360">|<video src="https://github.com/user-attachments/assets/636c3ff5-210e-44b8-b901-acf828071133" width="360"> </video>|<img src="examples/image/female_diaosu.png" width="360">|<video src="https://github.com/user-attachments/assets/e8207300-2569-47d1-9ad4-4b4c9b0f0bd4" width="360"> </video>|
|<img src="examples/image/hair.png" width="360">|<video src="https://github.com/user-attachments/assets/dcb755c1-de01-4afe-8b4f-0e0b2c2439c1" width="360"> </video>|<img src="examples/image/leonnado.jpg" width="360">|<video src="https://github.com/user-attachments/assets/b50e61bb-62d4-469d-b402-b37cda3fbd27" width="360"> </video>|


For more visual demos, please visit our [**Page**](https://jixiaozhong.github.io/Sonic/).

## 🧩 Community Contributions
If you develop/use Sonic in your projects, welcome to let us know.

- ComfyUI version of Sonic: [**ComfyUI_Sonic**](https://github.com/smthemex/ComfyUI_Sonic)


## 📑 Updates
**`2025/01/14`**: Our inference code and weights are released. Stay tuned, we will continue to polish the model.


## 📜 Requirements
* An NVIDIA GPU with CUDA support is required. 
  * The model is tested on a single 32G GPU.
* Tested operating system: Linux

## 🔑 Inference

### Installtion

- install pytorch
```shell
  pip3 install -r requirements.txt
```
- All models are stored in `checkpoints` by default, and the file structure is as follows
```shell
Sonic
  ├──checkpoints
  │  ├──Sonic
  │  │  ├──audio2bucket.pth
  │  │  ├──audio2token.pth
  │  │  ├──unet.pth
  │  ├──stable-video-diffusion-img2vid-xt
  │  │  ├──...
  │  ├──whisper-tiny
  │  │  ├──...
  │  ├──RIFE
  │  │  ├──flownet.pkl
  │  ├──yoloface_v5m.pt
  ├──...
```
Download by `huggingface-cli` follow
```shell
  python3 -m pip install "huggingface_hub[cli]"
  huggingface-cli download LeonJoe13/Sonic --local-dir  checkpoints
  huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir  checkpoints/stable-video-diffusion-img2vid-xt
  huggingface-cli download openai/whisper-tiny --local-dir checkpoints/whisper-tiny
```

or manully download [pretrain model](https://drive.google.com/drive/folders/1oe8VTPUy0-MHHW2a_NJ1F8xL-0VN5G7W?usp=drive_link), [svd-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and [whisper-tiny](https://huggingface.co/openai/whisper-tiny) to checkpoints/ 


### Run demo
```shell
  python3 demo.py \
  '/path/to/input_image' \
  '/path/to/input_audio' \
  '/path/to/output_video'
```



 
## 🔗 Citation

If you find our work helpful for your research, please consider citing our work.   

```bibtex
@article{ji2024sonic,
  title={Sonic: Shifting Focus to Global Audio Perception in Portrait Animation},
  author={Ji, Xiaozhong and Hu, Xiaobin and Xu, Zhihong and Zhu, Junwei and Lin, Chuming and He, Qingdong and Zhang, Jiangning and Luo, Donghao and Chen, Yi and Lin, Qin and others},
  journal={arXiv preprint arXiv:2411.16331},
  year={2024}
}

@article{ji2024realtalk,
  title={Realtalk: Real-time and realistic audio-driven face generation with 3d facial prior-guided identity alignment network},
  author={Ji, Xiaozhong and Lin, Chuming and Ding, Zhonggan and Tai, Ying and Zhu, Junwei and Hu, Xiaobin and Luo, Donghao and Ge, Yanhao and Wang, Chengjie},
  journal={arXiv preprint arXiv:2406.18284},
  year={2024}
}
```

## 📜 Related Works

Explore our related researches:
-  **[Super-fast talk：real-time and less GPU computation]** [Realtalk: Real-time and realistic audio-driven face generation with 3d facial prior-guided identity alignment network](https://arxiv.org/pdf/2406.18284)

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jixiaozhong/Sonic&type=Date)](https://star-history.com/#jixiaozhong/Sonic&Date)
