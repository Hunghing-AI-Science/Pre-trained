from transformers import AutoModel, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = "/home/admin/.cache/huggingface/hub/models--prithivMLmods--DeepSeek-OCR-Latest-BF16.I64/snapshots/244823a4cf4089862aab71ce1d4883253ac4b88f"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', torch_dtype=torch.float16, trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\n Could you give me the title of this image ?"
# prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = '/home/admin/Pre-trained/temp.png'
output_path = './'

res =  model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_file,
            output_path=output_path,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True,
            test_compress=True,
            eval_mode=True,
        )


print(res)
