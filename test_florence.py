import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, PretrainedConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Monkey-patch 1: Fix for missing forced_bos_token_id
PretrainedConfig.forced_bos_token_id = None

# Monkey-patch 2: Fix for _supports_sdpa
_original_getattr = torch.nn.Module.__getattr__
def _patched_getattr(self, name):
    if name == "_supports_sdpa":
        return False
    return _original_getattr(self, name)
torch.nn.Module.__getattr__ = _patched_getattr

# Monkey-patch 3: Fix for additional_special_tokens in Tokenizer
_orig_tok_getattr = PreTrainedTokenizerBase.__getattr__
def _patched_tok_getattr(self, name):
    if name == "additional_special_tokens":
        return []
    return _orig_tok_getattr(self, name)
PreTrainedTokenizerBase.__getattr__ = _patched_tok_getattr


# 1. Load the model and processor
model_id = "microsoft/Florence-2-large"

# Automatically use GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading {model_id} on {device}...")
# Note: trust_remote_code=True is required for Florence-2
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, trust_remote_code=True).to(device)

# Fix 5: Manually tie weights - newer transformers doesn't handle this for Florence-2.
# Without this, the embed/lm_head weights are randomly initialized and output is garbage.
shared_embed = model.language_model.model.encoder.embed_tokens.weight
model.language_model.model.decoder.embed_tokens.weight = shared_embed
model.language_model.lm_head.weight = shared_embed

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 2. Helper function to run tasks
def run_example(task_prompt, image):
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
      use_cache=False,
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer

# 3. Load a test image
print("Downloading test image...")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# Fix 4: The model currently expects square feature maps, so we pad the image
def make_square(im):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), (0, 0, 0))
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

image = make_square(image)


# 4. Run some Florence-2 specific tasks
print("\n--- Running Tests ---")

# Task A: Detailed Captioning
print("\nTask: <MORE_DETAILED_CAPTION>")
result = run_example("<MORE_DETAILED_CAPTION>", image)
print(result)

# Task B: Object Detection
print("\nTask: <OD> (Object Detection)")
result = run_example("<OD>", image)
print(result)
