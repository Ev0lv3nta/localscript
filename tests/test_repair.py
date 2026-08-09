from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.generation.engine import GenerationEngine
from app.generation.extractor import TaskExtractor
from app.generation.formatter import OutputFormatter
from app.repair.loop import RepairLoop
from app.validation.validators import ValidationPipeline


class JsonPathBackend:
    def generate(self, prompt, context=None):
        return "return $.wf.vars.value"


class UnsupportedRootBackend:
    def generate(self, prompt, context=None):
        return "local items = ctx.body.items\nreturn items[1]"


class MissingArrayGuardBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"family":"generic_lua","root":"wf.vars","source_paths":["wf.vars.orders"],'
                '"return_shape":"array","constraints":["Do not use JsonPath"],'
                '"assumptions":[],"clarification_needed":false,"clarification_question":"",'
                '"semantic_checks":[{"kind":"empty_array_on_missing_source","source_path":"wf.vars.orders"}]}'
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return '{"repairable":true,"issues":["generic_empty_array_behavior_mismatch"],"minimal_actions":["add_nil_safe_array_iteration","normalize_empty_array_return"]}'
        return "local result = _utils.array.new()\nfor _, item in ipairs(wf.vars.orders) do\n  table.insert(result, item.order_id)\nend\nreturn nil"

    def generate(self, prompt, context=None):
        return self.complete(prompt)


class StringTrimBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"family":"normalize_email_string","root":"wf.initVariables","source_paths":["wf.initVariables.userEmail"],'
                '"return_shape":"scalar","constraints":["Do not use JsonPath"],'
                '"assumptions":[],"clarification_needed":false,"clarification_question":"",'
                '"semantic_checks":["must return normalized lowercase string without whitespace"]}'
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return '{"repairable":true,"issues":["lua_runtime_error"],"minimal_actions":["rewrite_string_trim"]}'
        return (
            "local userEmail = wf.initVariables.userEmail\n"
            "local normalizedEmail = string.lower(string.trim(userEmail))\n"
            "return normalizedEmail"
        )

    def generate(self, prompt, context=None):
        return self.complete(prompt)


class MethodTrimBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"family":"normalize_email_string","root":"wf.initVariables","source_paths":["wf.initVariables.userEmail"],'
                '"return_shape":"scalar","constraints":["Do not use JsonPath"],'
                '"assumptions":[],"clarification_needed":false,"clarification_question":"",'
                '"semantic_checks":["must return normalized lowercase string without whitespace"]}'
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return '{"repairable":true,"issues":["lua_runtime_error"],"minimal_actions":["rewrite_string_trim"]}'
        return (
            "local userEmail = wf.initVariables.userEmail\n"
            "userEmail = userEmail:trim()\n"
            "userEmail = userEmail:lower()\n"
            "return userEmail"
        )

    def generate(self, prompt, context=None):
        return self.complete(prompt)


class ArrayContainsBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"family":"table_transform","root":"wf.vars","source_paths":["wf.vars.subscribers"],'
                '"return_shape":"array","constraints":[],"assumptions":[],'
                '"clarification_needed":false,"clarification_question":"",'
                '"semantic_checks":[{"kind":"array_equals","value":["example.com"]}]}'
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"repairable":true,"issues":["lua_runtime_error"],'
                '"minimal_actions":["replace unsupported contains method"]}'
            )
        return (
            "local result = _utils.array.new()\n"
            'local domain = "example.com"\n'
            "if not result:contains(domain) then\n"
            "  table.insert(result, domain)\n"
            "end\n"
            "return result"
        )

    def generate(self, prompt, context=None):
        return self.complete(prompt)


class WrappedScalarBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"family":"normalize_email_string","root":"wf.initVariables","source_paths":["wf.initVariables.userEmail"],'
                '"return_shape":"scalar","constraints":["Do not use JsonPath"],'
                '"assumptions":[],"clarification_needed":false,"clarification_question":"",'
                '"semantic_checks":["must return normalized lowercase string without whitespace"]}'
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return '{"repairable":true,"issues":["generic_return_shape_scalar_mismatch"],"minimal_actions":["normalize_scalar_return_shape"]}'
        return (
            "local userEmail = wf.initVariables.userEmail\n"
            "userEmail = string.gsub((userEmail or \"\"), \"^%s*(.-)%s*$\", \"%1\")\n"
            "userEmail = userEmail:lower()\n"
            "return _utils.array.new({userEmail})"
        )

    def generate(self, prompt, context=None):
        return self.complete(prompt)


class RecallTimeOsBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"family":"iso8601_to_epoch","root":"wf.initVariables","source_paths":["wf.initVariables.recallTime"],'
                '"return_shape":"scalar","constraints":["Do not use JsonPath"],'
                '"assumptions":[],"clarification_needed":false,"clarification_question":"",'
                '"semantic_checks":["must convert ISO 8601 string to epoch seconds"]}'
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"repairable":true,"issues":["dangerous_stdlib_os_forbidden"],'
                '"minimal_actions":["rewrite_iso8601_to_epoch"]}'
            )
        return (
            'local iso = wf.initVariables.recallTime\n'
            'local total_seconds = os.time{year=2023, month=10, day=15, hour=15, minute=30, second=0}\n'
            'return total_seconds'
        )

    def generate(self, prompt, context=None):
        return self.complete(prompt)


class DatumTimeBrokenBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"family":"datum_time_to_iso8601","root":"wf.vars","source_paths":["wf.vars.json.IDOC.ZCDF_HEAD.DATUM","wf.vars.json.IDOC.ZCDF_HEAD.TIME"],'
                '"return_shape":"scalar","constraints":["Do not use JsonPath"],'
                '"assumptions":[],"clarification_needed":false,"clarification_question":"",'
                '"semantic_checks":[]}'
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"repairable":true,"issues":["semantic_mismatch"],'
                '"minimal_actions":["rewrite_datum_time_to_iso8601"]}'
            )
        return (
            "local DATUM = wf.vars.json.IDOC.ZCDF_HEAD.DATUM\n"
            "local TIME = wf.vars.json.IDOC.ZCDF_HEAD.TIME\n"
            "local function safe_sub(str, start, finish)\n"
            "  local s = string.sub(str or \"\", start, math.min(finish, #(str or \"\")))\n"
            "  return s ~= \"\" and s or \"00\"\n"
            "end\n"
            "local year = safe_sub(DATUM, 1, 4)\n"
            "local month = safe_sub(DATUM, 5, 6)\n"
            "local day = safe_sub(DATUM, 7, 8)\n"
            "local hour = safe_sub(TIME, 1, 2)\n"
            "local minute = safe_sub(TIME, 3, 4)\n"
        )

    def generate(self, prompt, context=None):
        return self.complete(prompt)


class RestCleanupLiteralDeleteBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"family":"rest_cleanup","root":"wf.vars","source_paths":["wf.vars.RESTbody.result"],'
                '"return_shape":"array","constraints":["Do not use JsonPath"],'
                '"assumptions":[],"clarification_needed":false,"clarification_question":"",'
                '"semantic_checks":[]}'
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return '{"repairable":true,"issues":["rest_cleanup_excluded_key_reference::CALL"],"minimal_actions":["rewrite_rest_cleanup_keep_only"]}'
        return (
            "local result = wf.vars.RESTbody.result\n"
            "for _, entry in ipairs(result or {}) do\n"
            "  entry.CALL = nil\n"
            "  entry.OTHER = nil\n"
            "end\n"
            "return result"
        )

    def generate(self, prompt, context=None):
        return self.complete(prompt)


class AugmentAliasBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"family":"augment_existing_code","root":"unknown","source_paths":["wf.vars"],'
                '"return_shape":"json_envelope","constraints":["Do not use JsonPath"],'
                '"assumptions":[],"clarification_needed":false,"clarification_question":"",'
                '"semantic_checks":[]}'
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"repairable":true,"issues":["augment_existing_code_missing_key::num","augment_existing_code_missing_key::squared"],'
                '"minimal_actions":["add num","add squared"]}'
            )
        return '{\n  "sqr": "lua{local n = tonumber(\'5\')\\nreturn n * n}lua"\n}'

    def generate(self, prompt, context=None):
        return self.complete(prompt)


class AugmentLiteralSyntaxBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"family":"augment_existing_code","root":"unknown","source_paths":["wf.vars"],'
                '"return_shape":"json_envelope","constraints":["Do not use JsonPath"],'
                '"assumptions":[],"clarification_needed":false,"clarification_question":"",'
                '"semantic_checks":[]}'
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return '{"repairable":true,"issues":["lua_syntax_error"],"minimal_actions":["fix num envelope syntax"]}'
        return '{\n  "num": "lua{7}lua",\n  "squared": "lua{local n = 7\\nreturn n * n}lua"\n}'

    def generate(self, prompt, context=None):
        return self.complete(prompt)


class EnsureItemsRuntimeBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"family":"ensure_items_array","root":"wf.vars","source_paths":["wf.vars.json.IDOC.ZCDF_HEAD.ZCDF_PACKAGES"],'
                '"return_shape":"array","constraints":["Do not use JsonPath"],'
                '"assumptions":[],"clarification_needed":false,"clarification_question":"",'
                '"semantic_checks":[]}'
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return '{"repairable":true,"issues":["lua_runtime_error"],"minimal_actions":["rewrite ensure items array"]}'
        return (
            "local packages = wf.vars.json.IDOC.ZCDF_HEAD.ZCDF_PACKAGES or {}\n"
            "local result = _utils.array.new()\n"
            "for _, pkg in ipairs(packages) do\n"
            "  if type(pkg) == \"table\" and pkg.items ~= nil then\n"
            "    result = _utils.array.insert(result, {items = pkg.items})\n"
            "  end\n"
            "end\n"
            "return result"
        )

    def generate(self, prompt, context=None):
        return self.complete(prompt)


class EmailValidationSemanticBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"family":"email_validation","root":"wf.vars","source_paths":["wf.vars.profile.email"],'
                '"return_shape":"scalar","constraints":["Do not use JsonPath"],'
                '"assumptions":[],"clarification_needed":false,"clarification_question":"",'
                '"semantic_checks":[]}'
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return '{"repairable":true,"issues":["semantic_mismatch"],"minimal_actions":["fix email pattern"]}'
        return (
            "local email = wf.vars.profile.email\n"
            "if not email or email == \"\" then\n"
            "  return false\n"
            "end\n"
            "return string.match(email, \"^[A-Za-z0-9._%+%-]+@[A-Za-z0-9.%-]+%.([A-Za-z]{2,})$\") ~= nil"
        )

    def generate(self, prompt, context=None):
        return self.complete(prompt)


class ShadowedTableLocalBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"family":"generic_lua","root":"wf.vars","source_paths":["wf.vars.orders"],'
                '"return_shape":"array","constraints":["Do not use JsonPath"],'
                '"assumptions":[],"clarification_needed":false,"clarification_question":"",'
                '"semantic_checks":[]}'
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return '{"repairable":true,"issues":["shadowed_stdlib_local::table"],"minimal_actions":["rename shadowed table local"]}'
        return (
            "local result = _utils.array.new()\n"
            "for _, item in ipairs(wf.vars.orders or {}) do\n"
            "  if item.amount >= 100 then\n"
            "    local table = {}\n"
            "    table.id = item.order_id\n"
            "    table.status = item.status\n"
            "    table.insert(result, table)\n"
            "  end\n"
            "end\n"
            "return result"
        )

    def generate(self, prompt, context=None):
        return self.complete(prompt)


class ClarifiedRootMismatchBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return (
                '{"family":"generic_lua","root":"wf.vars","source_paths":["wf.vars.orders"],'
                '"return_shape":"scalar","constraints":["Do not use JsonPath"],'
                '"assumptions":[],"clarification_needed":false,"clarification_question":"",'
                '"semantic_checks":[]}'
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return '{"repairable":true,"issues":["unexpected_root_reference::wf.vars"],"minimal_actions":["rewrite root to clarified selection"]}'
        return (
            "local count = 0\n"
            "for _, item in ipairs(wf.vars.orders or {}) do\n"
            "  if item.status == \"paid\" then\n"
            "    count = count + 1\n"
            "  end\n"
            "end\n"
            "return count"
        )

    def generate(self, prompt, context=None):
        return self.complete(prompt)


def test_repair_loop_strips_markdown_fences():
    extractor = TaskExtractor()
    task_spec = extractor.extract(
        prompt="Верни последний адрес из массива wf.vars.emails; если массив пустой, верни nil.",
        context={"wf": {"vars": {"emails": ["a@example.com"]}}},
    )
    pipeline = ValidationPipeline()
    formatter = OutputFormatter()
    repair_loop = RepairLoop(
        validation_pipeline=pipeline,
        formatter=formatter,
    )
    initial_code = "```lua\nreturn wf.vars.emails[#wf.vars.emails]\n```"
    initial_report = pipeline.run(
        code=initial_code,
        task_spec=task_spec,
        profile=get_runtime_profile(),
    )

    result = repair_loop.run(
        code=initial_code,
        task_spec=task_spec,
        validation_report=initial_report,
        profile=get_runtime_profile(),
        max_rounds=2,
    )

    assert result.code == "return wf.vars.emails[#wf.vars.emails]"
    assert result.validation_report.has_errors is False
    assert result.rounds == 1


def test_engine_repairs_jsonpath_backend_output(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=JsonPathBackend(),
    )

    result = engine.generate(
        prompt="Сгенерируй код для значения value",
        context={"wf": {"vars": {"value": 7}}},
    )

    assert result.code == "return wf.vars.value"
    assert result.verification_errors == []
    assert result.repair_rounds == 1


def test_engine_repairs_unsupported_root_backend_output(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=UnsupportedRootBackend(),
    )

    result = engine.generate(
        prompt="Iterate over the incoming items collection and return the first value.",
        context={"wf": {"vars": {"items": [1, 2, 3]}}},
    )

    assert "ctx.body" not in result.code
    assert "wf.vars.items" in result.code
    assert result.verification_errors == []
    assert result.repair_rounds >= 1


def test_engine_repairs_missing_array_guard_and_empty_array_behavior(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=MissingArrayGuardBackend(),
    )

    result = engine.generate(
        prompt="Из массива wf.vars.orders верни новый массив order_id. Если orders пустой или nil, верни пустой массив.",
        context={"wf": {"vars": {"orders": [{"order_id": "A"}]}}},
    )

    assert "ipairs(wf.vars.orders or {})" in result.code
    assert result.code.endswith("return _utils.array.new()")
    assert result.repair_rounds >= 1


def test_engine_repairs_unsupported_string_trim_runtime_error(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=StringTrimBackend(),
    )

    result = engine.generate(
        prompt="Нормализуй launch variable wf.initVariables.userEmail: убери пробелы по краям, затем переведи строку в lower case и верни результат.",
        context={"wf": {"initVariables": {"userEmail": "  USER@Example.COM  "}}},
    )

    assert "string.trim" not in result.code
    assert 'string.gsub((userEmail or ""), "^%s*(.-)%s*$", "%1")' in result.code
    assert result.verification_errors == []
    assert result.repair_rounds >= 1


def test_engine_repairs_unsupported_method_trim_runtime_error(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=MethodTrimBackend(),
    )

    result = engine.generate(
        prompt="Нормализуй launch variable wf.initVariables.userEmail",
        context={"wf": {"initVariables": {"userEmail": "  USER@Example.COM  "}}},
    )

    assert ":trim()" not in result.code
    assert 'string.gsub((userEmail or ""), "^%s*(.-)%s*$", "%1")' in result.code
    assert result.verification_errors == []
    assert result.repair_rounds >= 1


def test_engine_repairs_unsupported_array_contains_runtime_error(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=ArrayContainsBackend(),
    )

    result = engine.generate(
        prompt="Верни уникальные домены подписчиков.",
        context={"wf": {"vars": {"subscribers": [{"email": "user@example.com"}]}}},
    )

    assert ":contains(" not in result.code
    assert "for _, candidate_value in ipairs(result or {})" in result.code
    assert result.verification_errors == []
    assert result.repair_rounds >= 1


def test_engine_repairs_wrapped_scalar_return_shape(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=WrappedScalarBackend(),
    )

    result = engine.generate(
        prompt="Нормализуй launch variable wf.initVariables.userEmail",
        context={"wf": {"initVariables": {"userEmail": "  USER@Example.COM  "}}},
    )

    assert "return _utils.array.new" not in result.code
    assert result.code.endswith("return userEmail")
    assert result.verification_errors == []
    assert result.repair_rounds >= 1


def test_engine_repairs_iso8601_epoch_without_os_namespace(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=RecallTimeOsBackend(),
    )

    result = engine.generate(
        prompt="Преобразуй ISO 8601 из wf.initVariables.recallTime в Unix timestamp.",
        context={"wf": {"initVariables": {"recallTime": "2023-10-15T15:30:00+00:00"}}},
    )

    assert "os." not in result.code
    assert "wf.initVariables.recallTime" in result.code
    assert result.verification_errors == []
    assert result.repair_rounds >= 1


def test_engine_repairs_datum_time_iso8601_truncation(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=DatumTimeBrokenBackend(),
    )

    result = engine.generate(
        prompt="Convert DATUM and short TIME to ISO 8601.",
        context={"wf": {"vars": {"json": {"IDOC": {"ZCDF_HEAD": {"DATUM": "20231201", "TIME": "945"}}}}}},
    )

    assert "wf.vars.json.IDOC.ZCDF_HEAD.DATUM" in result.code
    assert "wf.vars.json.IDOC.ZCDF_HEAD.TIME" in result.code
    assert 'string.gsub(TIME or "", "%D", "")' in result.code
    assert result.verification_errors == []
    assert result.repair_rounds >= 1


def test_engine_repairs_rest_cleanup_by_removing_excluded_key_references(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=RestCleanupLiteralDeleteBackend(),
    )

    result = engine.generate(
        prompt="Оставь в result только ID и ENTITY_ID, без CALL.",
        context={"wf": {"vars": {"RESTbody": {"result": [{"ID": 1, "ENTITY_ID": 2, "CALL": "x", "OTHER": "y"}]}}}},
    )

    assert '"CALL"' not in result.code
    assert ".CALL" not in result.code
    assert 'key ~= "ID"' in result.code
    assert 'key ~= "ENTITY_ID"' in result.code
    assert result.verification_errors == []
    assert result.repair_rounds >= 1


def test_engine_repairs_augment_existing_code_alias_to_canonical_envelope(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=AugmentAliasBackend(),
    )

    result = engine.generate(
        prompt="добавь sqr переменную",
        context=None,
    )

    assert '"num"' in result.code
    assert '"squared"' in result.code
    assert '"sqr"' not in result.code
    assert result.verification_errors == []
    assert result.repair_rounds >= 1


def test_engine_repairs_augment_existing_code_literal_syntax(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=AugmentLiteralSyntaxBackend(),
    )

    result = engine.generate(
        prompt="Сформируй JSON envelope: переменная num равна 7, а squared хранит её квадрат.",
        context=None,
    )

    assert "lua{7}lua" not in result.code
    assert "tonumber('7')" in result.code
    assert result.verification_errors == []
    assert result.repair_rounds >= 1


def test_engine_repairs_ensure_items_array_runtime_misuse(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=EnsureItemsRuntimeBackend(),
    )

    result = engine.generate(
        prompt="Make items in ZCDF_PACKAGES always arrays, direct access only.",
        context={"wf": {"vars": {"json": {"IDOC": {"ZCDF_HEAD": {"ZCDF_PACKAGES": [{"items": {"sku": "R"}}]}}}}}},
    )

    assert "_utils.array.insert" not in result.code
    assert "function ensureArray" in result.code
    assert "pkg.items = ensureArray(pkg.items)" in result.code
    assert result.verification_errors == []
    assert result.repair_rounds >= 1


def test_engine_repairs_email_validation_semantic_pattern(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=EmailValidationSemanticBackend(),
    )

    result = engine.generate(
        prompt="Проверь корректность profile.email и верни boolean.",
        context={"wf": {"vars": {"profile": {"email": "profile@example.com"}}}},
    )

    assert "%.[A-Za-z]+$" in result.code
    assert "{2,}" not in result.code
    assert result.verification_errors == []
    assert result.repair_rounds >= 1


def test_engine_repairs_shadowed_table_stdlib_local(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=ShadowedTableLocalBackend(),
    )

    result = engine.generate(
        prompt="На основе wf.vars.orders подготовь список кратких таблиц только для заказов amount >= 100. В каждой таблице создай поле id из order_id и поле status без изменений.",
        context={"wf": {"vars": {"orders": [{"order_id": "A", "status": "paid", "amount": 150}]}}},
    )

    assert "local table =" not in result.code
    assert "table.insert(result, entry)" in result.code
    assert "local entry = {}" in result.code
    assert result.verification_errors == []
    assert result.repair_rounds >= 1


def test_generate_rich_repairs_code_to_clarified_root(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=ClarifiedRootMismatchBackend(),
    )

    context = {
        "wf": {
            "vars": {"orders": [{"status": "paid"}, {"status": "paid"}]},
            "initVariables": {"orders": [{"status": "paid"}]},
        }
    }
    first = engine.generate_rich(
        prompt="Посчитай, сколько элементов в orders имеют status paid, и верни число.",
        context=context,
    )
    second = engine.generate_rich(
        session_id=first.session_id,
        clarification_answer="Use wf.initVariables for this task.",
    )

    assert first.status == "clarification_needed"
    assert "wf.vars or wf.initVariables" in (first.question or "")
    assert "wf.vars.orders" not in (second.code or "")
    assert "wf.initVariables.orders" in (second.code or "")
    assert second.verification_errors == []
    assert second.repair_rounds >= 1
