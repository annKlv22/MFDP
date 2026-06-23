"""Наполнение справочника видов (20 региональных видов СПб / Ленобласти).

Порядок и названия совпадают с метками классов модели (`classes.json`).
Фото подтягиваются из /static/species/<latin>.jpg; пока файла нет — в UI показывается заглушка.
"""
from sqlmodel import select

from common.database import session_scope
from common.logging_conf import get_logger
from common.models import Species

logger = get_logger("seed")

# name = метка класса (рус.), latin = биологические виды, photo = slug файла в static/species/
SPECIES_SEED = [
    {"name": "Белобровик", "latin": "Turdus iliacus", "photo": "turdus_iliacus",
     "description": "Некрупный дрозд с белой бровью и рыжими боками; звонкая флейтовая песня в сумерках."},
    {"name": "Большая синица", "latin": "Parus major", "photo": "parus_major",
     "description": "Самая крупная синица: жёлтая грудь с чёрным «галстуком», звонкое «ци-ци-фи»."},
    {"name": "Большой пёстрый дятел", "latin": "Dendrocopos major", "photo": "dendrocopos_major",
     "description": "Чёрно-белый дятел с красным подхвостьем; весной барабанит по сухим стволам."},
    {"name": "Воробей", "latin": "Passer domesticus, Passer montanus", "photo": "passer_domesticus",
     "description": "Домовый и полевой воробьи; шумные стайки и характерное «чик-чирик»."},
    {"name": "Зяблик", "latin": "Fringilla coelebs", "photo": "fringilla_coelebs",
     "description": "Один из самых обычных лесных певцов; бодрая песня с «росчерком» в конце."},
    {"name": "Крапивник", "latin": "Troglodytes troglodytes", "photo": "troglodytes_troglodytes",
     "description": "Крошечная птичка с задранным хвостом и неожиданно громкой звонкой трелью."},
    {"name": "Кряква", "latin": "Anas platyrhynchos", "photo": "anas_platyrhynchos",
     "description": "Самая известная дикая утка; громкое «кря-кря» у самок."},
    {"name": "Кукушка", "latin": "Cuculus canorus", "photo": "cuculus_canorus",
     "description": "Узнаётся по знаменитому «ку-ку»; гнездовой паразит."},
    {"name": "Лазоревка", "latin": "Cyanistes caeruleus", "photo": "cyanistes_caeruleus",
     "description": "Маленькая синица с голубой шапочкой; подвижная и звонкая."},
    {"name": "Мухоловка-пеструшка", "latin": "Ficedula hypoleuca", "photo": "ficedula_hypoleuca",
     "description": "Чёрно-белый самец; ловит насекомых на лету, песня — короткие коленца."},
    {"name": "Пищуха", "latin": "Certhia familiaris", "photo": "certhia_familiaris",
     "description": "Лазает по стволам снизу вверх, обследуя кору; тонкий высокий голос."},
    {"name": "Поползень", "latin": "Sitta europaea", "photo": "sitta_europaea",
     "description": "Единственная птица, бегающая по стволу вниз головой; громкое «тви-тви»."},
    {"name": "Серая ворона", "latin": "Corvus cornix", "photo": "corvus_cornix",
     "description": "Серо-чёрная ворона города и пойм; хриплое «карр»."},
    {"name": "Сизый голубь", "latin": "Columba livia", "photo": "columba_livia",
     "description": "Привычный городской голубь; глухое воркование «грру-грру»."},
    {"name": "Скворец", "latin": "Sturnus vulgaris", "photo": "sturnus_vulgaris",
     "description": "Блестящее оперение; искусный пересмешник, копирует чужие звуки."},
    {"name": "Снегирь", "latin": "Pyrrhula pyrrhula", "photo": "pyrrhula_pyrrhula",
     "description": "Самец с ярко-красной грудью; мягкое флейтовое «фью»."},
    {"name": "Соловей", "latin": "Luscinia luscinia", "photo": "luscinia_luscinia",
     "description": "Восточный соловей; мощная щёлкающая песня тёплыми ночами."},
    {"name": "Чайка", "latin": "Hydrocoloeus minutus, Chroicocephalus ridibundus, Larus argentatus",
     "photo": "chroicocephalus_ridibundus",
     "description": "Малая, озёрная и серебристая чайки; резкие крики у воды."},
    {"name": "Чиж", "latin": "Spinus spinus", "photo": "spinus_spinus",
     "description": "Мелкая жёлто-зелёная птичка; щебечущая песня с жужжащим «чжии»."},
    {"name": "Чёрный дрозд", "latin": "Turdus merula", "photo": "turdus_merula",
     "description": "Чёрный самец с оранжевым клювом; богатая флейтовая песня на заре."},
]


def seed_species() -> int:
    """Добавляет отсутствующие виды. Возвращает число добавленных. Идемпотентно."""
    created = 0
    with session_scope() as session:
        for i, item in enumerate(SPECIES_SEED):
            exists = session.exec(select(Species).where(Species.name == item["name"])).first()
            if exists:
                continue
            session.add(Species(
                name=item["name"],
                latin=item["latin"],
                description=item["description"],
                photo_url=f"/static/species/{item['photo']}.jpg",
                order_index=i,
            ))
            created += 1
    logger.info("Сидинг видов: добавлено %d (всего в справочнике %d)", created, len(SPECIES_SEED))
    return created
