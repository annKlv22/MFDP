"""Сущность «Вид» — справочник из 20 региональных видов (СПб / Ленобласть).

Имя вида = метка класса модели = русское название из `classes.json`.
"""
from typing import Optional

from sqlmodel import Field, SQLModel


class Species(SQLModel, table=True):
    __tablename__ = "species"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)   # русское название = метка класса
    latin: str = ""                              # латинские названия (через запятую)
    description: str = ""                         # краткое описание для карточки результата
    photo_url: str = ""                           # /static/species/<latin>.jpg или внешний URL
    order_index: int = 0                          # порядок для списка видов в UI
