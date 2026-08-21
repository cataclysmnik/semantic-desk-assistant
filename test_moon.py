import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

try:
    print("Loading model...")
    model_id = "./moondream2"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # Create dummy white image
    image = Image.new('RGB', (800, 600), color = 'white')
    
    print("Encoding image...")
    enc_image = model.encode_image(image)
    
    print("Answering...")
    answer = model.answer_question(enc_image, "What is this?")
    print("Answer:", answer)
except Exception as e:
    import traceback
    traceback.print_exc()
