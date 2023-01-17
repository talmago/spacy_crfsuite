import os
import spacy

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseSettings, BaseModel
from spacy.language import Language
from typing import Union, List

from spacy_crfsuite import CRFExtractor, CRFEntityExtractor


class Settings(BaseSettings):
    spacy_language_model: str = os.getenv("SPACY_MODEL", "en_core_web_md")
    crf_model: str = os.getenv("CRF_MODEL", "spacy_crfsuite_conll03_sm.bz2")


class Entity(BaseModel):
    start: int
    end: int
    value: str
    entity: str


class Sentence(BaseModel):
    text: str
    entities: List[Entity]


class Request(BaseModel):
    text: Union[str, List[str]]


class Response(BaseModel):
    data: List[Sentence] = []


def get_pipe(settings: Settings):
    """Build a spaCy pipeline for entity extraction based on ``settings``.

    Args:
        settings (Settings): global settings

    Returns:
        spacy.lang.Language
    """

    @Language.factory("ner_crf")
    def create_component(nlp, name):
        crf_extractor = CRFExtractor().from_disk(settings.crf_model)
        return CRFEntityExtractor(nlp, crf_extractor=crf_extractor)

    nlp = spacy.load(settings.spacy_language_model)
    nlp.add_pipe("ner_crf")
    return nlp


settings = Settings()
app = FastAPI()
nlp = get_pipe(settings)


@app.get("/status")
async def handle_status_request() -> JSONResponse:
    return JSONResponse(
        content={
            "status": "OK",
            "spacy_model": settings.spacy_language_model,
            "crf_model": os.path.basename(settings.crf_model),
        }
    )


@app.post("/parse")
async def handle_ner_request(request: Request) -> Response:
    it = [request.text] if isinstance(request.text, str) else request.text

    response_data = []
    for doc in nlp.pipe(it, disable="ner"):
        sentence: Sentence = Sentence(
            text=doc.text,
            entities=[
                {
                    "entity": ent.label_,
                    "value": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
                for ent in doc.ents
            ],
        )
        response_data.append(sentence)
    return Response(data=response_data)
