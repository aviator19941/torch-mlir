# Textual-inversion fine-tuning for Stable Diffusion using diffusers
# This script shows how to "teach" Stable Diffusion a new concept via
# textual-inversion using 🤗 Hugging Face [🧨 Diffusers library](https://github.com/huggingface/diffusers).
# By using just 3-5 images you can teach new concepts to Stable Diffusion
# and personalize the model on your own images.

#pip install diffusers["training"]==0.9.0 transformers ftfy opencv-python spacy
#pip install opencv-python diffusers ftfy spacy accelerate

import argparse
import itertools
import math
import os
import random
import cv2

# import torchdynamo
# import torch._dynamo as dynamo
import torch
import torch_mlir
# from shark.sharkdynamo.utils import make_shark_compiler

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint
from torch.utils.data import Dataset
from typing import List
from contextlib import redirect_stdout

import PIL
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

# from shark.shark_inference import SharkInference
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
import torch_mlir

YOUR_TOKEN = "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"

p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
p.add_argument(
    "--input_dir",
    type=str,
    default="input_images/",
    help="the directory contains the images used for fine tuning",
)
p.add_argument(
    "--output_dir",
    type=str,
    default="sd_result",
    help="the directory contains the images used for fine tuning",
)
p.add_argument(
    "--training_steps",
    type=int,
    default=3000,
    help="the maximum number of training steps",
)
p.add_argument("--seed", type=int, default=42, help="the random seed")
p.add_argument(
    "--what_to_teach",
    type=str,
    choices=["object", "style"],
    default="object",
    help="what is it that you are teaching?",
)
p.add_argument(
    "--placeholder_token",
    type=str,
    default="<cat-toy>",
    help="It is the token you are going to use to represent your new concept",
)
p.add_argument(
    "--initializer_token",
    type=str,
    default="toy",
    help="It is a word that can summarise what is your new concept",
)
p.add_argument(
    "--inference_steps",
    type=int,
    default=50,
    help="the number of steps for inference",
)
p.add_argument(
    "--num_inference_samples",
    type=int,
    default=4,
    help="the number of samples for inference",
)
p.add_argument(
    "--prompt",
    type=str,
    default="a grafitti in a wall with a *s on it",
    help="the text prompt to use",
)
p.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="The device to use",
)
args = p.parse_args()

if "*s" not in args.prompt:
    raise ValueError(
        f'The prompt should have a "*s" which will be replaced by a placeholder token.'
    )

prompt1, prompt2 = args.prompt.split("*s")
args.prompt = prompt1 + args.placeholder_token + prompt2


def compile_through_fx(model, inputs):

    fx_g = make_fx(
        model,
        decomposition_table=get_decompositions(
            [
                torch.ops.aten.embedding_dense_backward,
                torch.ops.aten.native_layer_norm_backward,
                torch.ops.aten.slice_backward,
                torch.ops.aten.select_backward,
                torch.ops.aten.norm.ScalarOpt_dim,
                torch.ops.aten.native_group_norm,
                torch.ops.aten.upsample_bilinear2d.vec,
                torch.ops.aten.split.Tensor,
                torch.ops.aten.split_with_sizes,
                torch.ops.aten.mse_loss_backward,
                torch.ops.aten.native_group_norm_backward,
                torch.ops.aten.sigmoid_backward,
            ]
        ),
    )(*inputs)

    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()

    def strip_overloads(gm):
        """
        Modifies the target of graph nodes in :attr:`gm` to strip overloads.
        Args:
            gm(fx.GraphModule): The input Fx graph module to be modified
        """
        for node in gm.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload):
                node.target = node.target.overloadpacket
        gm.recompile()

    strip_overloads(fx_g)
    print("******Fx graph generated*****")
    # print(fx_g.graph)

    # with open('sd_backward_fx_graph_21_11.txt', 'w') as f:
    #     with redirect_stdout(f):
    #         print(fx_g.graph)

    ts_g = torch.jit.script(fx_g)
    print("******Torchscript generated*****")
    # print(ts_g.graph)

    # with open('sd_backward_torchscript_graph_21_11.txt', 'w') as f:
    #     with redirect_stdout(f):
    #         print(ts_g.graph)

    module = torch_mlir.compile(
        ts_g,
        inputs,
        torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=False,
        verbose=False,
    )

    print("******Torch-MLIR compilation done*****")

    with open('sd_backward_linalg_ir_12_12.mlir', 'w') as f:
        with redirect_stdout(f):
            print(module.operation.get_asm())
    
    # print("******IR writing in file successful*****")

    # mlir_model = module
    # func_name = "forward"

    # shark_module = SharkInference(
    #     mlir_model, func_name, device=args.device, mlir_dialect="tm_tensor"
    # )
    # shark_module.compile()
    # print("******SHARK compilation done*****")

    return module


pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

# Load input images.
images = []
for filename in os.listdir(args.input_dir):
    img = cv2.imread(os.path.join(args.input_dir, filename))
    if img is not None:
        images.append(img)

# Setup the prompt templates for training
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

# Setup the dataset
class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [
            os.path.join(self.data_root, file_path)
            for file_path in os.listdir(self.data_root)
        ]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.templates = (
            imagenet_style_templates_small
            if learnable_property == "style"
            else imagenet_templates_small
        )
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[
                (h - crop) // 2 : (h + crop) // 2,
                (w - crop) // 2 : (w + crop) // 2,
            ]

        image = Image.fromarray(img)
        image = image.resize(
            (self.size, self.size), resample=self.interpolation
        )

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example
    
# Setting up the model
# Load the tokenizer and add the placeholder token as a additional special token.
# Please read and if you agree accept the LICENSE
# [here](https://huggingface.co/CompVis/stable-diffusion-v1-4) if you see an error
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
    use_auth_token=YOUR_TOKEN,
)

# Add the placeholder token in tokenizer
num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
if num_added_tokens == 0:
    raise ValueError(
        f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
        " `placeholder_token` that is not already in the tokenizer."
    )

# Get token ids for our placeholder and initializer token.
# This code block will complain if initializer string is not a single token
# Convert the initializer_token, placeholder_token to ids
token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
# Check if initializer_token is a single token or a sequence of tokens
if len(token_ids) > 1:
    raise ValueError("The initializer token must be a single token.")

initializer_token_id = token_ids[0]
placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)

# Load models and create wrapper for stable diffusion
# pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
# del pipeline
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae"
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
)

text_encoder.resize_token_embeddings(len(tokenizer))

token_embeds = text_encoder.get_input_embeddings().weight.data
token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

def freeze_params(params):
    for param in params:
        param.requires_grad = False

# Freeze vae and unet
freeze_params(vae.parameters())
freeze_params(unet.parameters())
# Freeze all parameters except for the token embeddings in text encoder
params_to_freeze = itertools.chain(
    text_encoder.text_model.encoder.parameters(),
    text_encoder.text_model.final_layer_norm.parameters(),
    text_encoder.text_model.embeddings.position_embedding.parameters(),
)
freeze_params(params_to_freeze)

train_dataset = TextualInversionDataset(
    data_root=args.input_dir,
    tokenizer=tokenizer,
    size=512,
    placeholder_token=args.placeholder_token,
    repeats=100,
    learnable_property=args.what_to_teach,  # Option selected above between object and style
    center_crop=False,
    set="train",
)

def create_dataloader(train_batch_size=1):
    return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

# Create noise_scheduler for training.
noise_scheduler = DDPMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    # tensor_format="pt",
)

# Define hyperparameters for our training
hyperparameters = {
    "learning_rate": 5e-04,
    "scale_lr": True,
    "max_train_steps": args.training_steps,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "seed": args.seed,
    "output_dir": "sd-concept-output",
}

gradient_accumulation_steps = hyperparameters[
    "gradient_accumulation_steps"
]

learning_rate = hyperparameters["learning_rate"]

# Initialize the optimizer
optimizer = torch.optim.AdamW(
    text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
    lr=learning_rate,
)

from torch_mlir.dynamo import make_simple_dynamo_backend
import torch._dynamo as dynamo
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
torch._dynamo.config.verbose=True
@make_simple_dynamo_backend
def refbackend_torchdynamo_backend(fx_graph: torch.fx.GraphModule,
                                   example_inputs: List[torch.Tensor]):
    mlir_module = torch_mlir.compile(
        fx_graph, example_inputs, output_type="linalg-on-tensors")
    with open('out.txt', 'w') as f:
        print(mlir_module, file=f)
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(mlir_module)
    loaded = backend.load(compiled)

    def compiled_callable(*inputs):
        inputs = [x.numpy() for x in inputs]
        result = loaded.forward(*inputs)
        if not isinstance(result, tuple):
            result = torch.from_numpy(result)
        else:
            result = tuple(torch.from_numpy(x) for x in result)
        return result
    return compiled_callable

def train_func(batch_input_ids, batch_pixel_values, params):
    # Convert images to latent space
    latents = vae.encode(batch_pixel_values).latent_dist.sample().detach()
    latents = latents * 0.18215

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(batch_input_ids)[0]

    # Predict the noise residual
    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    loss = (
        F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
    )
    # normalize loss to account for batch accumulation
    loss = loss / gradient_accumulation_steps
    loss.backward()

    # grads = text_encoder.get_input_embeddings().weight.grad
    grads = params.grad
    # Get the index for tokens that we want to zero the grads for
    if grads is not None:
        index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
        grads.data[index_grads_to_zero, :] = grads.data[
            index_grads_to_zero, :
        ].fill_(0)

    optimizer.step()
    optimizer.zero_grad()
    return params


def predictions(torch_func, jit_func, batchA, batchB, params):
    golden_prediction = torch_func(batchA, batchB, params)
    print("TORCHDYNAMO UPDATED PARAMS")
    print(golden_prediction)
    prediction = torch.from_numpy(jit_func(batchA.numpy(), batchB.numpy(), params.numpy()))
    print("TORCH-MLIR UPDATED PARAMS")
    print(prediction)


def training_function(text_encoder, vae, unet):
    train_batch_size = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    learning_rate = hyperparameters["learning_rate"]
    max_train_steps = hyperparameters["max_train_steps"]
    output_dir = hyperparameters["output_dir"]

    train_dataloader = create_dataloader(train_batch_size)

    if hyperparameters["scale_lr"]:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size
        )

    # Move vae and unet to device
    vae.to(args.device)
    unet.to(args.device)

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.train()

    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Instantaneous batch size per device = {train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    global_step = 0

    for epoch in range(num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            '''# Convert images to latent space
            latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # Predict the noise residual
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = (
                F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
            )
            # normalize loss to account for batch accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # grads = text_encoder.get_input_embeddings().weight.grad
            params = [i for i in text_encoder.get_input_embeddings().parameters()]
            print("PyTorch PARAMS")
            print(params)
            grads = params[0].grad
            # Get the index for tokens that we want to zero the grads for
            if grads is not None:
                index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                grads.data[index_grads_to_zero, :] = grads.data[
                    index_grads_to_zero, :
                ].fill_(0)

            optimizer.step()
            optimizer.zero_grad()

            if global_step >= max_train_steps:
                break'''
            params = [i for i in text_encoder.get_input_embeddings().parameters()]
            print("PyTorch PARAMS")
            print(params)
            result = train_func(batch["input_ids"], batch["pixel_values"], params[0])
            print("TORCH UPDATED PARAMS")
            print(result)
            dynamo_callable = dynamo.optimize(refbackend_torchdynamo_backend)(train_func)
            predictions(train_func, lambda x, y, z: dynamo_callable(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)).detach().numpy(), batch["input_ids"], batch["pixel_values"], params[0].detach())
            print("AFTER PREDICTIONS")

            if global_step >= max_train_steps:
                break


    # Create the pipeline using using the trained modules and save it.
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        unet=unet,
    )
    pipeline.save_pretrained(output_dir)
    # Also save the newly trained embeddings
    save_path = os.path.join(output_dir, f"learned_embeds.bin")


training_function(text_encoder, vae, unet)

for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
  if param.grad is not None:
    del param.grad  # free some memory
  torch.cuda.empty_cache()