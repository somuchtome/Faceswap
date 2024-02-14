import glob
import os
import random
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader
from diffusers import ControlNetModel, DDIMScheduler
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint_face import StableDiffusionControlNetFaceInpaintPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from lora_diffusion import patch_pipe, tune_lora_scale
from generate_annotation import generate_annotation
from train_lora_dreambooth import train_lora
from parse import parse_args
from mask import Mask
from utils import cosin_metric, show_editied_masked_image, ImageAugmentations
from models.gaze_estimation.gaze_estimator import Gaze_estimator
from eye_crop import get_eye_coords
from accelerate.utils import set_seed

def make_canny_condition(image):
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size=512):
        super().__init__()
        self.files = glob.glob(path + '/**/*.jpg', recursive=True)
        self.files.extend(glob.glob(path + '/**/*.png', recursive=True))
        self.files.sort()
        self.img_size = img_size
        self.transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),])

    def __getitem__(self, index):
        file = self.files[index]
        image = Image.open(file).convert('RGB')
        x = self.transform(image)
        return x, file

    def __len__(self):
        return len(self.files)

def id_distance(src, targ, netArc, device):
    src = TF.to_tensor(src).unsqueeze(0).to(device)
    src = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(src)
    src = F.interpolate(src, (112, 112))
    src_id = netArc(src)

    targ = TF.to_tensor(targ).unsqueeze(0).to(device)
    targ = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(targ)
    targ = F.interpolate(targ, (112, 112))
    targ_id = netArc(targ)

    id_loss = cosin_metric(src_id, targ_id)
    print('cosin_metric_id: {}'.format(id_loss.item()))
    return id_loss.item()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    src_dataset  = Dataset(path=args.src_dir,  img_size=args.resolution)
    targ_dataset = Dataset(path=args.targ_dir, img_size=args.resolution)
    src_loader   = DataLoader(src_dataset,  num_workers=1, shuffle=False, batch_size=1)
    targ_loader  = DataLoader(targ_dataset, num_workers=1, shuffle=False, batch_size=1)
    src_iter     = iter(src_loader)
    targ_iter    = iter(targ_loader)
    mask_processor = Mask()
    length = len(src_loader)
    print('Number of Test Data: ', length)
    device = "cuda"
    netArc_checkpoint = 'checkpoints/arcface_checkpoint.tar'
    netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
    netArc = netArc_checkpoint
    netArc = netArc.to(device).eval()
    netGaze = Gaze_estimator().to(device)
    image_augmentations = ImageAugmentations(112, args.aug_num)

    controlnet_canny = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32
    )
    controlnet_face = ControlNetModel.from_pretrained("CrucibleAI/ControlNetMediaPipeFace", subfolder="diffusion_sd15", torch_dtype=torch.float32)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", 
        subfolder="models/image_encoder",
        torch_dtype=torch.float32,
    ).to("cuda")
    pipe = StableDiffusionControlNetFaceInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=[controlnet_canny,controlnet_face], image_encoder=image_encoder, torch_dtype=torch.float32, safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    for step in range(length):
            try:
                src_tensor, instance_path = next(src_iter)
                targ_tensor, targ_instance_path = next(targ_iter)
            except StopIteration:
                src_iter        = iter(src_loader)
                targ_iter       = iter(targ_loader)
                src_tensor , instance_path = next(src_iter)
                targ_tensor, targ_instance_path =  next(targ_iter)
            
            lora_path = os.path.join(args.output_dir, "lora")
            os.makedirs(lora_path, exist_ok=True)
            # train_lora
            train_lora(args, instance_path[0])
            # calculate Canny image
            targ_img_name = targ_instance_path[0].split("/")[-1].split(".")[0]
            src_img_name = instance_path[0].split("/")[-1].split(".")[0]
            targ_img = Image.open(targ_instance_path[0]).convert("RGB").resize((args.resolution,args.resolution))
            src_img = Image.open(instance_path[0]).convert("RGB").resize((args.resolution,args.resolution))
            control_canny = make_canny_condition(targ_img)
            # calculate face annotation
            control_face = generate_annotation(targ_img, 1)
            #get face mask
            mask_image = mask_processor(targ_tensor.to(device).float())
            mask_image = mask_image.resize((args.resolution, args.resolution))

            #sets the default attention implementation. 
            pipe.unet.set_default_attn_processor()
            #load lora
            lora_path = os.path.join(args.output_dir, f"lora/{src_img_name}_lora_weight.safetensors")
            patch_pipe(
                        pipe,
                        lora_path,
                        patch_text=True,
                        patch_ti=False,
                        patch_unet=True,
            )
            tune_lora_scale(pipe.unet, 1.0)
            tune_lora_scale(pipe.text_encoder, 1.0)
            #load ip-adapter
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin", torch_dtype=torch.float32)
            pipe.set_ip_adapter_scale(0.5)
          
            # generate image
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            image = pipe(
                prompt="a photo of sks person",
                num_inference_steps=50,
                generator=generator,
                image=targ_img,
                mask_image=mask_image,
                control_image=[control_canny, control_face],
                conditioning_scale=[0.5,1.0],
                ip_adapter_image=src_img,
                netArc_model=netArc,
                netGaze_model=netGaze,
                netSeg_model=mask_processor.netSeg,
                spNorm=mask_processor.spNorm,
                aug=image_augmentations,
                get_eye_coords=get_eye_coords,
                src_image=src_img,
                lora_path=os.path.join(args.output_dir, f"lora/{src_img_name}_lora_weight.pt"),
                args_seed=args.seed
            ).images[0]
            #swap face results
            output_path = os.path.join(args.output_dir, "results")
            os.makedirs(output_path, exist_ok=True)
            image.save(f"{output_path}/{src_img_name}_{targ_img_name}.png")
            #vis src and target
            vis_path = os.path.join(args.output_dir, "vis")
            os.makedirs(vis_path, exist_ok=True)
            final_distance = id_distance(image, src_img, netArc, device)
            formatted_distance = f"{final_distance:.4f}"
            show_editied_masked_image(
                                title='Results',
                                source_image=src_img,
                                target_image=targ_img,
                                edited_image=image,
                                mask=mask_image,
                                path=f"{vis_path}/{src_img_name}_{targ_img_name}.png",
                                distance=formatted_distance,
                            )