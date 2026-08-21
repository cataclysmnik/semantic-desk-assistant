import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
model_id = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)

prompt = "Where is my bag"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }
]
text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
print("TEXT PROMPT:")
print(text_prompt)

# Let's also test model generation with a simple black image
image = Image.new('RGB', (224, 224), color = 'black')
inputs = processor(
    text=[text_prompt],
    images=[image],
    return_tensors="pt",
)
print("INPUT IDS SHAPE:", inputs.input_ids.shape)
