import requests
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# 1. Load the model and processor
model_id = "Qwen/Qwen2-VL-2B-Instruct"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading {model_id} on {device}...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch_dtype,
).to(device)
processor = AutoProcessor.from_pretrained(model_id)


# 2. Helper function to run a visual question
def ask_about_image(image, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=256)

    # Trim the input tokens from the output
    output_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return output_text


# 3. Load a test image
print("Downloading test image...")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# 4. Run some tests
print("\n--- Running Tests ---\n")

print("Q: Describe this image in detail.")
answer = ask_about_image(image, "Describe this image in detail.")
print(f"A: {answer}\n")

print("Q: What objects can you see?")
answer = ask_about_image(image, "What objects can you see? List them.")
print(f"A: {answer}\n")

print("Q: What color is the car?")
answer = ask_about_image(image, "What color is the car?")
print(f"A: {answer}\n")
