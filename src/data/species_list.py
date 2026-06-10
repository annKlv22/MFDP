# -*- coding: utf-8 -*-
"""Целевые виды регионального определителя птиц (СПб / Ленинградская область).

Метка класса = русское название = имя папки в `data/audio/<Вид>/` и `data/processed/<Вид>/`.
Некоторые классы объединяют несколько биологических видов (по продуктовому решению):
  - «Воробей» = домовый + полевой
  - «Чайка»   = малая + озёрная + серебристая
«Чёрный дрозд» и «Белобровик» — отдельные классы.

Порядок совпадает с `sorted(...)` (как строится список классов в ноутбуке).
"""

# Класс (рус., = имя папки) -> биологические виды (Genus species), которые в него входят
SPECIES_MAP = {
    "Белобровик":            ["Turdus iliacus"],
    "Большая синица":        ["Parus major"],
    "Большой пёстрый дятел": ["Dendrocopos major"],
    "Воробей":               ["Passer domesticus", "Passer montanus"],
    "Зяблик":                ["Fringilla coelebs"],
    "Крапивник":             ["Troglodytes troglodytes"],
    "Кряква":                ["Anas platyrhynchos"],
    "Кукушка":               ["Cuculus canorus"],
    "Лазоревка":             ["Cyanistes caeruleus"],
    "Мухоловка-пеструшка":   ["Ficedula hypoleuca"],
    "Пищуха":                ["Certhia familiaris"],
    "Поползень":             ["Sitta europaea"],
    "Серая ворона":          ["Corvus cornix"],
    "Сизый голубь":          ["Columba livia"],
    "Скворец":               ["Sturnus vulgaris"],
    "Снегирь":               ["Pyrrhula pyrrhula"],
    "Соловей":               ["Luscinia luscinia"],
    "Чайка":                 ["Hydrocoloeus minutus", "Chroicocephalus ridibundus", "Larus argentatus"],
    "Чиж":                   ["Spinus spinus"],
    "Чёрный дрозд":          ["Turdus merula"],
}

# Список меток классов (= имена папок в data/audio/)
TARGET_SPECIES = list(SPECIES_MAP.keys())

# Обратное соответствие: биологический вид -> класс (для справки)
LATIN_TO_CLASS = {lat: cls for cls, lats in SPECIES_MAP.items() for lat in lats}

assert len(TARGET_SPECIES) == 20, f"ожидалось 20 классов, получено {len(TARGET_SPECIES)}"
