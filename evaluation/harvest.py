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

"""Harvest RADAR metadata via OAI-PMH ListRecords to disk.

Each response page is stored verbatim as page_NNNN.xml under
HARVEST_DIR/<oai_set>/, so later analysis never has to hit the OAI endpoint
again. Re-running replaces a set's pages with a fresh harvest.

    python evaluation/harvest.py              # all configured domains
    python evaluation/harvest.py culture      # just one
"""

import argparse
import shutil
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from config import DOMAINS, HARVEST_DIR, OAI_BASE_URL, OAI_METADATA_PREFIX

OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"
USER_AGENT = "radar-keyword-eval/1.0"


def fetch(params: dict) -> bytes:
    url = f"{OAI_BASE_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except OSError as e:
            if attempt == 2:
                raise
            print(f"  retrying after error: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))


def harvest_set(oai_set: str) -> None:
    target = HARVEST_DIR / oai_set
    tmp = target.with_name(target.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    params = {"verb": "ListRecords", "metadataPrefix": OAI_METADATA_PREFIX, "set": oai_set}
    page, total = 0, 0
    while True:
        page += 1
        body = fetch(params)
        (tmp / f"page_{page:04d}.xml").write_bytes(body)

        root = ET.fromstring(body)
        error = root.find(f"{OAI_NS}error")
        if error is not None:
            raise RuntimeError(f"OAI error for set {oai_set}: {error.get('code')} {error.text}")
        records = root.findall(f"{OAI_NS}ListRecords/{OAI_NS}record")
        total += len(records)
        token = root.find(f"{OAI_NS}ListRecords/{OAI_NS}resumptionToken")
        size = token.get("completeListSize") if token is not None else None
        print(f"  {oai_set}: page {page}, {total}{f'/{size}' if size else ''} records")

        if token is None or not (token.text or "").strip():
            break
        params = {"verb": "ListRecords", "resumptionToken": token.text.strip()}

    (tmp / "harvest.txt").write_text(
        f"set={oai_set}\nurl={OAI_BASE_URL}\nharvested={datetime.now(timezone.utc).isoformat()}\n"
        f"pages={page}\nrecords={total}\n",
        encoding="utf-8",
    )
    # Swap in the complete harvest only once it succeeded.
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
    print(f"  {oai_set}: {total} records stored in {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("domains", nargs="*", metavar="domain", help=f"one of {', '.join(DOMAINS)} (default: all)")
    args = parser.parse_args()
    unknown = set(args.domains) - set(DOMAINS)
    if unknown:
        parser.error(f"unknown domain(s): {', '.join(sorted(unknown))}")
    for domain in args.domains or DOMAINS:
        harvest_set(DOMAINS[domain]["oai_set"])


if __name__ == "__main__":
    main()
