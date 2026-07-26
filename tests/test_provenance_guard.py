import hashlib
import re
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PROMPT_FINGERPRINTS = {
    (5, "6cf3557601e94be892a10775b6c487425e89a2a2b3a87647eea3da995c98f8e9"),
    (6, "565b8f2d4a0eec2e87be03c83d203bb207616dbabb77657288da36e894b47dcc"),
    (7, "96fdfd4aea6e05cd20f7edc5a13d31cad4c059940e0fdb15e5678706e19ccbd1"),
    (8, "50cbb9046722bd6a172585488860906c47a2d4f918df58f4560e770b5465a006"),
    (13, "1944ada03f971094e29827b4fa24915d3c8eb641488236763918764a8252b4e2"),
    (16, "e4d5387b1260eb7df664d82566dd6f09aa76073dd95921807c318326a3e51783"),
    (17, "a4f8412b07318cd608ba2855ed1cfd46f6105d309caea2a8654e2ecf1a563565"),
    (24, "fd14eb2912ce61d27631c9129951a9b5ba1942387992b76bf7c434c27640decc"),
}


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _fingerprint(tokens: List[str]) -> str:
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()


def test_legacy_public_prompts_are_absent_from_repository_text():
    matches = []
    lengths = {length for length, _ in LEGACY_PROMPT_FINGERPRINTS}

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            tokens = _tokens(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        for length in lengths:
            for start in range(len(tokens) - length + 1):
                digest = _fingerprint(tokens[start : start + length])
                if (length, digest) in LEGACY_PROMPT_FINGERPRINTS:
                    matches.append(str(path.relative_to(ROOT)))

    assert matches == []
