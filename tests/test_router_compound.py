from pathlib import Path

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.generation.engine import GenerationEngine
from app.generation.extractor import TaskExtractor
from tests.support_backends import DeterministicTestBackend


def test_compound_prompt_is_classified_without_runtime_template_route():
    spec = TaskExtractor().extract(
        prompt="Для wf.vars.json.IDOC.ZCDF_HEAD.ZCDF_PACKAGES собери новый массив объектов через _utils.array.new(). Для каждого пакета верни таблицу {number = pkg.number, itemsCount = N}, где N равно количеству элементов в pkg.items. Если pkg.items уже массив, используй его длину. Если pkg.items является одним объектом, считай itemsCount равным 1. Если pkg.items отсутствует, пакет пропускай. Если ZCDF_PACKAGES отсутствует или пустой, верни пустой массив.",
        context={
            "wf": {
                "vars": {
                    "json": {
                        "IDOC": {
                            "ZCDF_HEAD": {
                                "ZCDF_PACKAGES": [
                                    {"number": "PKG001", "items": [{"sku": "A"}, {"sku": "B"}]},
                                    {"number": "PKG002", "items": {"sku": "C"}},
                                ]
                            }
                        }
                    }
                }
            }
        },
    )

    assert spec.family is not None
    assert spec.safety_fallback is False
    assert spec.generation_hints
    assert spec.composition_score > 0.34


def test_engine_routes_compound_prompt_to_model_chain(tmp_path):
    trace_store = TraceStore(root=tmp_path / "traces")
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=trace_store,
        backend=DeterministicTestBackend(),
    )

    result = engine.generate(
        prompt="Возьми wf.vars.contacts и подготовь список таблиц с id и email в нижнем регистре только для активных записей.",
        context={"wf": {"vars": {"contacts": [{"id": 1, "email": "USER@EXAMPLE.COM", "active": True}]}}},
    )

    assert result.strategy == "ollama_chain"
    trace_files = list(Path(trace_store.root).glob("**/*.json"))
    assert trace_files


def test_email_normalization_matcher_does_not_capture_array_projection():
    spec = TaskExtractor().extract(
        prompt="Возьми wf.vars.contacts и подготовь список таблиц с id и email в нижнем регистре только для активных записей.",
        context={"wf": {"vars": {"contacts": [{"id": 1, "email": "USER@EXAMPLE.COM", "active": True}]}}},
    )

    assert spec.family != "normalize_email_string"
