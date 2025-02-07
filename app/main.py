import io
import os
from typing import Annotated

import scalar_fastapi.scalar_fastapi as scalar
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import APIKeyHeader
from PIL import Image
from starlette.responses import RedirectResponse
from transformers import AutoModelForCausalLM

# On windows you'll need to set the path to vips
# vipsbin = r'<vips/bin/path>'
# add_dll_dir = getattr(os, 'add_dll_directory', None)
# if callable(add_dll_dir):
#     add_dll_dir(vipsbin)
# else:
#     os.environ['PATH'] = os.pathsep.join((vipsbin, os.environ['PATH']))

default_api_key = os.getenv("DEFAULT_API_KEY", "1234")

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
header_scheme = APIKeyHeader(name="X-Api-Key", scheme_name="API Key Header")


@app.post("/query/")
async def query_moondream(
    file: UploadFile,
    prompt: Annotated[str, Form()],
    api_key: str = Depends(header_scheme),
):
    if api_key != default_api_key:
        raise HTTPException(status_code=401, detail="Invalid Credentials")

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


@app.get("/", include_in_schema=False)
def default_redirect():
    return RedirectResponse(url="/scalar")


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return scalar.get_scalar_api_reference(
        openapi_url=app.openapi_url, title=app.title, layout=scalar.Layout.CLASSIC
    )
