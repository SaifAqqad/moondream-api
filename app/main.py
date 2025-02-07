from transformers import AutoModelForCausalLM
from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
from typing import Annotated
import io
import os

# vipsbin = r'C:\Users\saifw\OneDrive\Desktop\moondream\vips-dev-8.16\bin'
# add_dll_dir = getattr(os, 'add_dll_directory', None)
# if callable(add_dll_dir):
#     add_dll_dir(vipsbin)
# else:
#     os.environ['PATH'] = os.pathsep.join((vipsbin, os.environ['PATH']))

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    # Uncomment to run on GPU.
    # device_map="cuda",
)

app = FastAPI()


# "Give me a json array output of the items left on the backseats of the car"
@app.post("/query/")
async def query_moondream(file: UploadFile, prompt: Annotated[str, Form()]):
    # Read the contents of the uploaded file into a BytesIO object
    contents = await file.read()  # async read
    image = Image.open(io.BytesIO(contents))

    try:
        encoded_image = model.encode_image(image)
        result = model.query(
            encoded_image,
            prompt,
        )

        return {"result": result}
    except (IOError, SyntaxError):
        return {"message": "Invalid image file"}
