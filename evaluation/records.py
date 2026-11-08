# Copyright 2025-2026 FIZ Karlsruhe - Leibniz-Institut für Informationsinfrastruktur GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parse harvested OAI-PMH pages into plain dataset dicts."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, List

from config import HARVEST_DIR

OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element, name: str) -> List[ET.Element]:
    # Match on local names, so a new RADAR schema version (namespace) doesn't break parsing.
    return [child for child in element if _local(child.tag) == name]


def _text(element) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _split_keywords(raw: List[str]) -> List[str]:
    # Depositors often put several keywords into one <keyword>, separated by commas or semicolons.
    keywords = []
    for value in raw:
        for part in re.split(r"[;,]", value):
            part = part.strip()
            if part and part not in keywords:
                keywords.append(part)
    return keywords


def parse_dataset(record: ET.Element) -> Dict:
    header = record.find(f"{OAI_NS}header")
    identifier = _text(header.find(f"{OAI_NS}identifier"))
    metadata = record.find(f"{OAI_NS}metadata")
    dataset = next(iter(metadata), None) if metadata is not None else None
    if dataset is None:
        raise ValueError(f"record {identifier} has no metadata")

    titles = [_text(t) for t in _children(dataset, "title")]
    descriptions = []
    for container in _children(dataset, "descriptions"):
        for d in _children(container, "description"):
            descriptions.append({"type": d.get("descriptionType"), "text": _text(d)})
    abstracts = [d["text"] for d in descriptions if d["type"] == "Abstract" and d["text"]]

    keywords = []
    for container in _children(dataset, "keywords"):
        keywords.extend(_text(k) for k in _children(container, "keyword"))

    subject_areas = []
    for container in _children(dataset, "subjectAreas"):
        for area in _children(container, "subjectArea"):
            subject_areas.append(_text(area))

    return {
        "identifier": identifier,
        "datestamp": _text(header.find(f"{OAI_NS}datestamp")),
        "title": " ".join(titles),
        "abstract": "\n\n".join(abstracts),
        "author_keywords": _split_keywords(keywords),
        "subject_areas": [s for s in subject_areas if s],
    }


def load_datasets(oai_set: str) -> Iterator[Dict]:
    """Yield the datasets of a harvested set in OAI order, skipping deleted records."""
    directory = HARVEST_DIR / oai_set
    pages = sorted(directory.glob("page_*.xml"))
    if not pages:
        raise FileNotFoundError(f"No harvested pages in {directory} - run harvest.py first.")
    for page in pages:
        root = ET.parse(page).getroot()
        for record in root.iterfind(f"{OAI_NS}ListRecords/{OAI_NS}record"):
            header = record.find(f"{OAI_NS}header")
            if header is not None and header.get("status") == "deleted":
                continue
            yield parse_dataset(record)


def document_text(dataset: Dict) -> str:
    """The text sent to the keyword service: title and abstract."""
    return "\n\n".join(part for part in (dataset["title"], dataset["abstract"]) if part)
