import os
import torch
import torch.utils.checkpoint
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import cv2

from diffusers import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import WhisperModel, CLIPVisionModelWithProjection, AutoFeatureExtractor

from src.utils.util import save_videos_grid, seed_everything
from src.dataset.test_preprocess import process_bbox, image_audio_to_tensor
from src.models.base.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel, add_ip_adapters
from src.pipelines.pipeline_sonic import SonicPipeline
from src.models.audio_adapter.audio_proj import AudioProjModel
from src.models.audio_adapter.audio_to_bucket import Audio2bucketModel
from src.utils.RIFE.RIFE_HDv3 import RIFEModel
from src.dataset.face_align.align import AlignImage


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def test(
    pipe,
    config,
    wav_enc,
    audio_pe,
    audio2bucket,
    image_encoder,
    width,
    height,
    batch
):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(pipe.device).float()
    ref_img = batch['ref_img']
    clip_img = batch['clip_images']
    face_mask = batch['face_mask']
    image_embeds = image_encoder(
        clip_img
            ).image_embeds

    audio_feature = batch['audio_feature']
    audio_len = batch['audio_len']
    step = int(config.step)

    window = 3000
    audio_prompts = []
    last_audio_prompts = []
    for i in range(0, audio_feature.shape[-1], window):
        audio_prompt = wav_enc.encoder(audio_feature[:,:,i:i+window], output_hidden_states=True).hidden_states
        last_audio_prompt = wav_enc.encoder(audio_feature[:,:,i:i+window]).last_hidden_state
        last_audio_prompt = last_audio_prompt.unsqueeze(-2)
        audio_prompt = torch.stack(audio_prompt, dim=2)
        audio_prompts.append(audio_prompt)
        last_audio_prompts.append(last_audio_prompt)

    audio_prompts = torch.cat(audio_prompts, dim=1)
    audio_prompts = audio_prompts[:,:audio_len*2]
    audio_prompts = torch.cat([torch.zeros_like(audio_prompts[:,:4]), audio_prompts, torch.zeros_like(audio_prompts[:,:6])], 1)

    last_audio_prompts = torch.cat(last_audio_prompts, dim=1)
    last_audio_prompts = last_audio_prompts[:,:audio_len*2]
    last_audio_prompts = torch.cat([torch.zeros_like(last_audio_prompts[:,:24]), last_audio_prompts, torch.zeros_like(last_audio_prompts[:,:26])], 1)


    ref_tensor_list = []
    audio_tensor_list = []
    uncond_audio_tensor_list = []
    motion_buckets = []
    for i in tqdm(range(audio_len//step)):


        audio_clip = audio_prompts[:,i*2*step:i*2*step+10].unsqueeze(0)
        audio_clip_for_bucket = last_audio_prompts[:,i*2*step:i*2*step+50].unsqueeze(0)
        motion_bucket = audio2bucket(audio_clip_for_bucket, image_embeds)
        motion_bucket = motion_bucket * 16 + 16
        motion_buckets.append(motion_bucket[0])

        cond_audio_clip = audio_pe(audio_clip).squeeze(0)
        uncond_audio_clip = audio_pe(torch.zeros_like(audio_clip)).squeeze(0)

        ref_tensor_list.append(ref_img[0])
        audio_tensor_list.append(cond_audio_clip[0])
        uncond_audio_tensor_list.append(uncond_audio_clip[0])

    video = pipe(
        ref_img,
        clip_img,
        face_mask,
        audio_tensor_list,
        uncond_audio_tensor_list,
        motion_buckets,
        height=height,
        width=width,
        num_frames=len(audio_tensor_list),
        decode_chunk_size=config.decode_chunk_size,
        motion_bucket_scale=config.motion_bucket_scale,
        fps=config.fps,
        noise_aug_strength=config.noise_aug_strength,
        min_guidance_scale1=config.min_appearance_guidance_scale, # 1.0,
        max_guidance_scale1=config.max_appearance_guidance_scale,
        min_guidance_scale2=config.audio_guidance_scale, # 1.0,
        max_guidance_scale2=config.audio_guidance_scale,
        overlap=config.overlap,
        shift_offset=config.shift_offset,
        frames_per_batch=config.n_sample_frames,
        num_inference_steps=config.num_inference_steps,
        i2i_noise_strength=config.i2i_noise_strength
    ).frames


    # Concat it with pose tensor
    # pose_tensor = torch.stack(pose_tensor_list,1).unsqueeze(0)
    video = (video*0.5 + 0.5).clamp(0, 1)
    video = torch.cat([video.to(pipe.device)], dim=0).cpu()

    return video


class Sonic():
    config_file = os.path.join(BASE_DIR, 'config/inference/sonic.yaml')
    config = OmegaConf.load(config_file)

    def __init__(self, 
                 device_id=0,
                 enable_interpolate_frame=True,
                 ):
        
        config = self.config
        config.use_interframe = enable_interpolate_frame

        device = 'cuda:{}'.format(device_id) if device_id > -1 else 'cpu'

        config.pretrained_model_name_or_path = os.path.join(BASE_DIR, config.pretrained_model_name_or_path)

        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            config.pretrained_model_name_or_path, 
            subfolder="vae",
            variant="fp16")
        
        val_noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            config.pretrained_model_name_or_path, 
            subfolder="scheduler")
        
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            config.pretrained_model_name_or_path, 
            subfolder="image_encoder",
            variant="fp16")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="unet",
            variant="fp16")
        add_ip_adapters(unet, [32], [config.ip_audio_scale])
        
        audio2token = AudioProjModel(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=1024, context_tokens=32).to(device)
        audio2bucket = Audio2bucketModel(seq_len=50, blocks=1, channels=384, clip_channels=1024, intermediate_dim=1024, output_dim=1, context_tokens=2).to(device)

        unet_checkpoint_path = os.path.join(BASE_DIR, config.unet_checkpoint_path)
        audio2token_checkpoint_path = os.path.join(BASE_DIR, config.audio2token_checkpoint_path)
        audio2bucket_checkpoint_path = os.path.join(BASE_DIR, config.audio2bucket_checkpoint_path)

        unet.load_state_dict(
            torch.load(unet_checkpoint_path, map_location="cpu"),
            strict=True,
        )
        
        audio2token.load_state_dict(
            torch.load(audio2token_checkpoint_path, map_location="cpu"),
            strict=True,
        )

        audio2bucket.load_state_dict(
            torch.load(audio2bucket_checkpoint_path, map_location="cpu"),
            strict=True,
        )
        

        if config.weight_dtype == "fp16":
            weight_dtype = torch.float16
        elif config.weight_dtype == "fp32":
            weight_dtype = torch.float32
        elif config.weight_dtype == "bf16":
            weight_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"Do not support weight dtype: {config.weight_dtype} during training"
            )

        whisper = WhisperModel.from_pretrained(os.path.join(BASE_DIR, 'checkpoints/whisper-tiny/')).to(device).eval()
        
        whisper.requires_grad_(False)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(os.path.join(BASE_DIR, 'checkpoints/whisper-tiny/'))

        det_path = os.path.join(BASE_DIR, os.path.join(BASE_DIR, 'checkpoints/yoloface_v5m.pt'))
        self.face_det = AlignImage(device, det_path=det_path)
        if config.use_interframe:
            rife = RIFEModel(device=device)
            rife.load_model(os.path.join(BASE_DIR, 'checkpoints', 'RIFE/'))
            self.rife = rife


        image_encoder.to(weight_dtype)
        vae.to(weight_dtype)
        unet.to(weight_dtype)

        pipe = SonicPipeline(
            unet=unet,
            image_encoder=image_encoder,
            vae=vae,
            scheduler=val_noise_scheduler,
        )
        pipe = pipe.to(device=device, dtype=weight_dtype)


        self.pipe = pipe
        self.whisper = whisper
        self.audio2token = audio2token
        self.audio2bucket = audio2bucket
        self.image_encoder = image_encoder
        self.device = device

        print('init done')


    def preprocess(self,
              image_path, expand_ratio=1.0):
        face_image = cv2.imread(image_path)
        h, w = face_image.shape[:2]
        _, _, bboxes = self.face_det(face_image, maxface=True)
        face_num = len(bboxes)
        bbox = []
        bbox_s = None  # Initialize bbox_s to None
        
        if face_num > 0:
            x1, y1, ww, hh = bboxes[0]
            x2, y2 = x1 + ww, y1 + hh
            bbox = x1, y1, x2, y2
            bbox_s = process_bbox(bbox, expand_radio=expand_ratio, height=h, width=w)
        else:
            # If no face is detected, use the entire image as the bounding box
            bbox_s = [0, 0, w, h]

        return {
            'face_num': face_num,
            'crop_bbox': bbox_s,
        }
    
    def crop_image(self,
                   input_image_path,
                   output_image_path,
                   crop_bbox):
        face_image = cv2.imread(input_image_path)
        crop_image = face_image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
        cv2.imwrite(output_image_path, crop_image)

    @torch.no_grad()
    def process(self,
                image_path,
                audio_path,
                output_path,
                min_resolution=512,
                inference_steps=25,
                dynamic_scale=1.0,
                keep_resolution=False,
                seed=None):
        
        config = self.config
        device = self.device
        pipe = self.pipe
        whisper = self.whisper
        audio2token = self.audio2token
        audio2bucket = self.audio2bucket
        image_encoder = self.image_encoder

        # specific parameters
        if seed:
            config.seed = seed

        config.num_inference_steps = inference_steps

        config.motion_bucket_scale = dynamic_scale

        seed_everything(config.seed)

        video_path = output_path.replace('.mp4', '_noaudio.mp4')
        audio_video_path = output_path

        imSrc_ = Image.open(image_path).convert('RGB')
        raw_w, raw_h = imSrc_.size

        test_data = image_audio_to_tensor(self.face_det, self.feature_extractor, image_path, audio_path, limit=config.frame_num, image_size=min_resolution, area=config.area)
        if test_data is None:
            return -1
        height, width = test_data['ref_img'].shape[-2:]
        if keep_resolution:
            resolution = f'{raw_w//2*2}x{raw_h//2*2}'
        else:
            resolution = f'{width}x{height}'

        video = test(
            pipe,
            config,
            wav_enc=whisper,
            audio_pe=audio2token,
            audio2bucket=audio2bucket,
            image_encoder=image_encoder,
            width=width,
            height=height,
            batch=test_data,
            )

        if config.use_interframe:
            rife = self.rife
            out = video.to(device)
            results = []
            video_len = out.shape[2]
            for idx in tqdm(range(video_len-1), ncols=0):
                I1 = out[:, :, idx]
                I2 = out[:, :, idx+1]
                middle = rife.inference(I1, I2).clamp(0, 1).detach()
                results.append(out[:, :, idx])
                results.append(middle)
            results.append(out[:, :, video_len-1])
            video = torch.stack(results, 2).cpu()
        
        save_videos_grid(video, video_path, n_rows=video.shape[0], fps=config.fps * 2 if config.use_interframe else config.fps)
        ffmpeg_command = f'ffmpeg -i "{video_path}" -i "{audio_path}" -s {resolution} -vcodec libx264 -acodec aac -crf 18 -shortest -y "{audio_video_path}"'
        os.system(ffmpeg_command)
        os.remove(video_path)  # Use os.remove instead of rm for Windows compatibility
        
        return 0
        