import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

try:
    print("Loading model...")
    model_id = "./moondream2"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # Load real desk image
    image = Image.open("esp32_captures/raw/capture_20260818-183454.jpg").convert("RGB")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Encoding image...")
    enc_image = model.encode_image(image)
    
    print("Answering...")
    answer = model.answer_question(enc_image, "What objects are on this desk?", tokenizer=tokenizer)
    print("Answer:", answer)
except Exception as e:
    import traceback
    traceback.print_exc()
