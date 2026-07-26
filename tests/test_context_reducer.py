from app.generation.context_reducer import ContextReducer
from app.generation.extractor import TaskExtractor


def test_context_reducer_returns_original_context_when_small():
    context = {"wf": {"vars": {"email": "a@example.com"}}}
    task_spec = TaskExtractor().extract(
        prompt="Проверь email и верни boolean.",
        context=context,
    )

    reduced = ContextReducer().reduce(context, task_spec)

    assert reduced == context


def test_context_reducer_builds_summary_for_dense_context():
    context = {
        "wf": {
            "vars": {
                "orders": [{"id": i, "amount": i * 10, "status": "paid"} for i in range(20)],
                "customer": {
                    "profile": {
                        "name": "Test",
                        "email": "USER@EXAMPLE.COM",
                        "address": {"city": "Moscow", "zip": "101000"},
                    }
                },
                "metadata": {"source": "erp", "tenant": "demo"},
            }
        }
    }
    task_spec = TaskExtractor().extract(
        prompt="Из массива wf.vars.orders верни новый массив order_id только для заказов, где status равен paid и amount больше 1000.",
        context=context,
    )

    reduced = ContextReducer(max_serialized_chars=120, max_paths=6).reduce(context, task_spec)

    assert reduced["reduced"] is True
    assert "roots" in reduced
    assert "available_paths_sample" in reduced
    assert "shape_summary" in reduced
