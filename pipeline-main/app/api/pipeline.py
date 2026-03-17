from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from app.schemas.schema import Inference_Response
import requests
import os

from app.schemas.schema import (
    ScrapeRequest,
    article,
    coref_request,
    Coref_Article
)


router = APIRouter()

import os

# def build_url(env_key: str, path: str) -> str:
#     base = os.environ.get(env_key)
#     if not base:
#         raise RuntimeError(f"Missing environment variable: {env_key}")
#     return f"{base.rstrip('/')}/{path.lstrip('/')}"

# SCRAPE_API= 'http://localhost:8001'
# PREPROCESS_API = 'http://localhost:8002'
# COREF_API = 'http://attention-regardless-buy-twist.trycloudflare.com'
# BIAS_API='http://localhost:8005'

# SCRAPE_API = build_url("SCRAPE_API", "api/v1/scrape")
# PREPROCESS_API = build_url("PREPROCESS_API", "api/v1/preprocess")
# COREF_API = build_url("COREF_API", "api/v1/coref")
# BIAS_API = build_url("BIAS_API", "api/v1/inference")

##SCRAPE_API= "http://localhost:8001/api/v1/scrape"
##PREPROCESS_API="http://localhost:8002/api/v1/preprocess"
##BIAS_API="https://instruction-vpn-observations-bedrooms.trycloudflare.com/api/v1/bias/inference"
##COREF_API="https://sean-favourite-ranging-mysimon.trycloudflare.com/api/v1/coref"





def post_json(url: str, payload: dict):
    try:
        resp = requests.post(
            url,
            json=payload,
            timeout=(10, 280)  # ✅ 10 sec connect, 280 sec read
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ReadTimeout:
        raise HTTPException(
            status_code=504,
            detail=f"Timeout while calling {url}"
        )


@router.post("/pipeline", response_model=Inference_Response)
async def pipeline(data: ScrapeRequest):

    if not all([SCRAPE_API, PREPROCESS_API, COREF_API]):
        raise HTTPException(status_code=500, detail="API endpoints not configured")

    # 1️⃣ Scrape
    scraped_json = await run_in_threadpool(
        post_json,
        SCRAPE_API,
        {"url": str(data.url)}
    )
    scraped_article = article(**scraped_json)
    scraped_article.pipeline_status = ["scraped"]

    # 2️⃣ Preprocess
    preprocessed_json = await run_in_threadpool(
        post_json,
        PREPROCESS_API,
        scraped_article.model_dump()
    )
    preprocessed = article(**preprocessed_json)
    preprocessed.pipeline_status = scraped_article.pipeline_status + ["preprocessed"]

    # 3️⃣ Coreference
    coref_payload = coref_request(
        content=preprocessed.content,
        url=str(preprocessed.url)
    )
    

    coref_json = await run_in_threadpool(
        post_json,
        COREF_API,
        coref_payload.model_dump()
    )
    coref_result = Coref_Article(**coref_json)


    # Replace content with coref output
    preprocessed.content = coref_result.content
    preprocessed.pipeline_status.append("coref_resolved")   

    bias_payload = {
    "ner_list": [i.model_dump() for i in coref_result.ner_list]
}

    bias_json = await run_in_threadpool(
        post_json,
        BIAS_API,
        bias_payload
    )

    Bias_Score_result = Inference_Response(**bias_json)



    return Bias_Score_result