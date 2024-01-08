from PIL import Image
import torch
import os
from transformers import AutoProcessor, LlavaForConditionalGeneration


model_id = "llava-hf/bakLlava-v1-hf"
saved_model_path = "LLava_Model_weights/bakLlava-v1-hf"

if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)
    
    model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    load_in_4bit=True,
    # attn_implementation="flash_attention_2",
    # force_download=True,
    # resume_download=False
    )
    processor = AutoProcessor.from_pretrained(model_id)

    model.save_pretrained(saved_model_path)
    processor.save_pretrained(saved_model_path)
else:
    model = LlavaForConditionalGeneration.from_pretrained(
        saved_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        # attn_implementation="flash_attention_2",
        # force_download=True,
        # resume_download=False
    )

    processor = AutoProcessor.from_pretrained(saved_model_path)


def run_inference(user_input):
    image_file = "static/frames/captured_frame.jpg"
    raw_image = Image.open(image_file)

    prompt = f"USER: <image>\n{user_input}\nASSISTANT:"
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float32)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return processor.decode(output[0][2:], skip_special_tokens=True)

