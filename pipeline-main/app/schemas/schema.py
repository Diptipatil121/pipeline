from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class ScrapeRequest(BaseModel):
    url: str

class article(BaseModel):

    title: str | None = None
    content: str | None = None
    url: str
    error_code: int | None = None
    error_message: str | None = None
    source: str | None = None
    pipeline_status: List[str] | None = None
    

class coref_request(BaseModel): # the schema for the input request for coreference api. change this according to orchestration method

    content: str | None = None
    url: str
    
class item(BaseModel):
    sent: str | None = None
    label: str | None = None 

class ItemScored(BaseModel):
    sent: Optional[str] = None
    label: Optional[str] = None
    score: Optional[float] = None

class Coref_Article(BaseModel): # the schema for the input request for preprocessing api. change this according to orchestration method

    content: str | None = None
    url: str
    chains: list | None = None 
    ner_list : List[item] | None = None

class Inference_Response(BaseModel):
    aggregate_score: Optional[float] = None
    aggregate_label:Optional[str] = None
    scored_list: Optional[List[ItemScored]] = None
    median_score: Optional[float] = None
    mode_value: Optional[str] = None
