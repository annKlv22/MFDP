"""REST: справочник видов."""
from typing import List

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.schemas import SpeciesOut
from common.database import get_session
from common.models import Species

router = APIRouter()


@router.get("/", response_model=List[SpeciesOut], summary="Список поддерживаемых видов")
def list_species(session: Session = Depends(get_session)):
    rows = session.exec(select(Species).order_by(Species.order_index)).all()
    return [
        SpeciesOut(
            id=s.id, name=s.name, latin=s.latin,
            description=s.description, photo_url=s.photo_url,
        )
        for s in rows
    ]
