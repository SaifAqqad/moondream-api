import io
import os
from typing import Annotated

import scalar_fastapi.scalar_fastapi as scalar
from PIL import Image
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.security import APIKeyHeader
from starlette.responses import RedirectResponse
from transformers import AutoModelForCausalLM

from app.enums import CaptionLength

# Load vips binaries from env var
vips_path = os.getenv("VIPS_PATH", "")
if vips_path != "":
    add_dll_dir = getattr(os, 'add_dll_directory', None)
    if callable(add_dll_dir):
        add_dll_dir(vips_path)
    else:
        os.environ['PATH'] = os.pathsep.join((vips_path, os.environ['PATH']))

using_gpu = os.getenv("USE_GPU", "") == "true"
if using_gpu:
    print("Loading GPU model")
else:
    print("Loading CPU model")

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map=("cuda" if using_gpu else "cpu"),
)

app = FastAPI(title="Moondream API")
default_api_key = os.getenv("DEFAULT_API_KEY", "1234")
api_key_scheme = APIKeyHeader(name="X-Api-Key", scheme_name="API Key Header")


@app.post("/detect/")
async def detect_moondream(
        file: UploadFile,
        object_description: Annotated[str, Form()],
        api_key: str = Depends(api_key_scheme),
):
    if api_key != default_api_key:
        raise HTTPException(status_code=401, detail="Invalid Credentials")

    try:
        encoded_image = await _model_encoded_image(file)
        result = model.detect(
            encoded_image,
            object_description,
        )

        return {"result": result["objects"]}
    except (IOError, SyntaxError):
        return {"message": "Invalid image file"}


@app.post("/caption/")
async def caption_moondream(
        file: UploadFile,
        length: Annotated[CaptionLength, Form()] = CaptionLength.NORMAL,
        api_key: str = Depends(api_key_scheme),
):
    if api_key != default_api_key:
        raise HTTPException(status_code=401, detail="Invalid Credentials")
    try:
        encoded_image = await _model_encoded_image(file)
        result = model.caption(
            encoded_image,
            length=length.value if length is not None else CaptionLength.NORMAL.value
        )

        return {"result": result["caption"]}
    except (IOError, SyntaxError):
        return {"message": "Invalid image file"}


@app.post("/query/")
async def query_moondream(
        file: UploadFile,
        prompt: Annotated[str, Form()],
        api_key: str = Depends(api_key_scheme),
):
    if api_key != default_api_key:
        raise HTTPException(status_code=401, detail="Invalid Credentials")

    try:
        encoded_image = await _model_encoded_image(file)
        result = model.query(
            encoded_image,
            prompt,
        )

        return {"result": result}
    except (IOError, SyntaxError):
        return {"message": "Invalid image file"}


@app.get("/", include_in_schema=False)
def default_redirect():
    return RedirectResponse(url="/scalar")


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return scalar.get_scalar_api_reference(
        openapi_url=app.openapi_url, title=app.title, layout=scalar.Layout.CLASSIC
    )


async def _model_encoded_image(file: UploadFile):
    contents = await file.read()  # async read
    image = Image.open(io.BytesIO(contents))
    return model.encode_image(image)
