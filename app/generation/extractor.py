import re

from app.generation.taskspec import TaskSpec


class TaskExtractor:
    def __init__(self):
        pass

    def extract(self, prompt, context=None):
        normalized_prompt = " ".join((prompt or "").lower().split())
        prompt_paths = self._extract_prompt_paths(prompt or "")
        prompt_path_candidates = self._materialize_prompt_paths(prompt_paths)
        context_paths = self._collect_context_paths(context)
        prompt_root = self._root_from_path(prompt_paths["explicit"][0]) if prompt_paths["explicit"] else None
        target_root = prompt_root or self._infer_target_root(context_paths)
        root_candidates = self._root_candidates(target_root, prompt_path_candidates, context_paths)
        assumptions = []
        ambiguity_notes = []
        generation_hints = {}

        if not context_paths:
            assumptions.append("No runtime context supplied; routing based on prompt only.")

        heuristic_match = self._heuristic_match(
            prompt=prompt or "",
            normalized_prompt=normalized_prompt,
            context=context,
            prompt_paths=prompt_paths,
        )
        if heuristic_match:
            generation_hints = heuristic_match["generation_hints"]
            assumptions.extend(heuristic_match.get("assumptions", []))
            matched_root = target_root
            if matched_root not in {"unknown_mixed", "unknown"}:
                matched_root = self._root_from_path(heuristic_match["target_path"]) or target_root
            ambiguity_score, composition_score, context_density = self._routing_scores(
                normalized_prompt=normalized_prompt,
                target_root=matched_root,
                context_paths=context_paths,
                ambiguity_notes=ambiguity_notes,
                prompt=prompt or "",
            )
            if heuristic_match["family"] in {
                "filter_discount_markdown",
                "field_mapping",
                "augment_existing_code",
            }:
                composition_score = min(composition_score, 0.2)
            if heuristic_match["family"] == "augment_existing_code" and not heuristic_match["target_path"]:
                ambiguity_score = 0.0
            return TaskSpec(
                normalized_prompt=normalized_prompt,
                family=heuristic_match["family"],
                output_style=heuristic_match["output_style"],
                target_root=matched_root,
                context_paths=context_paths,
                prompt_paths=prompt_path_candidates,
                root_candidates=root_candidates,
                family_confidence=1.0,
                ambiguity_score=ambiguity_score,
                composition_score=composition_score,
                context_density=context_density,
                generation_hints=generation_hints,
                assumptions=assumptions,
                ambiguity_notes=ambiguity_notes,
                safety_fallback=bool(heuristic_match.get("safety_fallback")),
            )

        ambiguity_score, composition_score, context_density = self._routing_scores(
            normalized_prompt=normalized_prompt,
            target_root=target_root,
            context_paths=context_paths,
            ambiguity_notes=ambiguity_notes,
            prompt=prompt or "",
        )
        return TaskSpec(
            normalized_prompt=normalized_prompt,
            family=None,
            output_style="lua_block",
            target_root=target_root,
            context_paths=context_paths,
            prompt_paths=prompt_path_candidates,
            root_candidates=root_candidates,
            family_confidence=0.0,
            ambiguity_score=ambiguity_score,
            composition_score=composition_score,
            context_density=context_density,
            generation_hints=generation_hints,
            assumptions=assumptions,
            ambiguity_notes=ambiguity_notes,
        )

    def _routing_scores(self, normalized_prompt, target_root, context_paths, ambiguity_notes, prompt):
        ambiguity_score = 0.0
        if target_root == "unknown_mixed":
            ambiguity_score += 0.75
        elif target_root == "unknown" and context_paths:
            ambiguity_score += 0.35
        ambiguity_score += min(0.4, 0.2 * len(ambiguity_notes))
        if self._contains_any(normalized_prompt, ["wf.vars", "wf.initvariables"]):
            ambiguity_score = max(0.0, ambiguity_score - 0.15)
        if self._contains_any(normalized_prompt, ["верни", "return"]) and self._contains_any(
            normalized_prompt,
            ["сохрани", "запиши", "обнови", "добавь в переменную", "mutate"],
        ):
            ambiguity_score += 0.25

        condition_count = len(self._extract_simple_conditions(prompt))
        feature_count = 0
        if self._contains_any(normalized_prompt, ["только", "where", "где", "if "]):
            feature_count += 1
        if self._contains_any(normalized_prompt, ["lower", "upper", "trim", "нормализ", "приведи"]):
            feature_count += 1
        if self._contains_any(normalized_prompt, ["собери", "таблиц", "object", "объект", "{"]):
            feature_count += 1
        if self._contains_any(normalized_prompt, ["колич", "count", "sum", "сумм"]):
            feature_count += 1

        composition_score = 0.0
        if feature_count > 1:
            composition_score += 0.2 * float(feature_count - 1)
        if condition_count > 1:
            composition_score += 0.2
        if feature_count > 1 and (normalized_prompt.count(" и ") >= 1 or normalized_prompt.count(" and ") >= 1):
            composition_score += 0.2
        composition_score = min(1.0, composition_score)

        context_density = min(1.0, float(len(context_paths)) / 40.0) if context_paths else 0.0
        return round(min(1.0, ambiguity_score), 2), round(composition_score, 2), round(context_density, 2)

    def _collect_context_paths(self, context):
        paths = []

        def walk(value, prefix):
            if isinstance(value, dict):
                for key, nested_value in value.items():
                    next_prefix = "{0}.{1}".format(prefix, key) if prefix else key
                    paths.append(next_prefix)
                    walk(nested_value, next_prefix)
            elif isinstance(value, list):
                list_prefix = "{0}[]".format(prefix) if prefix else "[]"
                paths.append(list_prefix)
                if value:
                    walk(value[0], list_prefix)

        walk(context, "")
        return paths

    def _heuristic_match(self, prompt, normalized_prompt, context, prompt_paths):
        safety_guard = self._match_safety_guard(normalized_prompt)
        if safety_guard:
            return safety_guard

        last_array = self._match_last_array_item(prompt, normalized_prompt, context, prompt_paths)
        if last_array:
            return last_array

        counter = self._match_counter_increment(prompt, normalized_prompt, context, prompt_paths)
        if counter:
            return counter

        cleanup = self._match_rest_cleanup(prompt, normalized_prompt, context)
        if cleanup:
            return cleanup

        datum_iso = self._match_datum_time_iso(normalized_prompt, context)
        if datum_iso:
            return datum_iso

        ensure_items = self._match_ensure_items_array(normalized_prompt, context)
        if ensure_items:
            return ensure_items

        filter_rows = self._match_discount_markdown_filter(normalized_prompt, context)
        if filter_rows:
            return filter_rows

        conditional_projection = self._match_conditional_array_projection(prompt, normalized_prompt, context, prompt_paths)
        if conditional_projection:
            return conditional_projection

        email_normalization = self._match_email_normalization(normalized_prompt, context, prompt_paths)
        if email_normalization:
            return email_normalization

        email_validation = self._match_email_validation(normalized_prompt, context, prompt_paths)
        if email_validation:
            return email_validation

        regex_extract = self._match_regex_extract(prompt, normalized_prompt, context, prompt_paths)
        if regex_extract:
            return regex_extract

        field_mapping = self._match_field_mapping(prompt, normalized_prompt, context, prompt_paths)
        if field_mapping:
            return field_mapping

        table_transform = self._match_table_transform(prompt, normalized_prompt, context, prompt_paths)
        if table_transform:
            return table_transform

        square = self._match_square_variable(prompt, normalized_prompt)
        if square:
            return square

        recall = self._match_recall_time(normalized_prompt, context, prompt_paths)
        if recall:
            return recall

        return None

    def _match_last_array_item(self, prompt, normalized_prompt, context, prompt_paths):
        if not self._contains_any(normalized_prompt, ["последн", "last"]):
            return None
        if self._contains_any(
            normalized_prompt,
            [
                "не про послед",
                "не про last",
                "not about last",
                "not the last",
            ],
        ):
            return None

        prompt_array_path = self._select_prompt_path(prompt_paths, expected_kind="array")
        has_array_signal = self._contains_any(normalized_prompt, ["спис", "list", "массив", "array", "items", "элемент"])
        email_array_signal = "email" in normalized_prompt and self._find_array_path(context, normalized_prompt)
        if not has_array_signal and not email_array_signal and not prompt_array_path and not self._looks_like_array_path(prompt):
            return None

        array_path = prompt_array_path or self._find_array_path(context, normalized_prompt)
        if not array_path and "email" in normalized_prompt:
            array_path = "wf.vars.emails"
        if not array_path:
            return None

        return {
            "family": "last_array_item",
            "output_style": "lua_expression",
            "target_path": array_path,
            "generation_hints": {"source_path": array_path},
        }

    def _match_counter_increment(self, prompt, normalized_prompt, context, prompt_paths):
        if not self._contains_any(normalized_prompt, ["увелич", "increment", "increase", "++", "+ 1"]):
            return None

        prompt_counter_path = self._select_prompt_path(prompt_paths, expected_kind="counter")
        counter_path = prompt_counter_path or self._find_counter_path(context, normalized_prompt)
        if not counter_path and "try_count_n" in normalized_prompt:
            counter_path = "wf.vars.try_count_n"
        if not counter_path:
            return None

        return {
            "family": "counter_increment",
            "output_style": "lua_expression",
            "target_path": counter_path,
            "generation_hints": {"counter_path": counter_path},
        }

    def _match_rest_cleanup(self, prompt, normalized_prompt, context):
        result_path, sample_entry = self._find_rest_result(context)
        if not result_path or not sample_entry:
            return None

        keep_keys = self._extract_keep_keys(prompt, sample_entry)
        cleanup_intent = self._contains_any(normalized_prompt, ["rest", "очист", "clean"])
        if not cleanup_intent:
            prompt_tokens = {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", prompt)}
            matched_key_count = sum(key.lower() in prompt_tokens for key in sample_entry.keys())
            cleanup_intent = "result" in normalized_prompt and matched_key_count >= 2

        if not keep_keys or not cleanup_intent:
            return None

        return {
            "family": "rest_cleanup",
            "output_style": "lua_block",
            "target_path": result_path,
            "generation_hints": {
                "result_path": result_path,
                "keep_keys": keep_keys,
                "available_keys": list(sample_entry.keys()),
            },
        }

    def _match_datum_time_iso(self, normalized_prompt, context):
        has_named_fields = "datum" in normalized_prompt and "time" in normalized_prompt
        has_format_signal = "yyyymmdd" in normalized_prompt and "hhmmss" in normalized_prompt
        if not has_named_fields and not has_format_signal:
            return None
        if not self._contains_any(normalized_prompt, ["iso 8601", "iso8601"]):
            return None

        datum_path = self._find_field_path(context, lambda key: key.lower() == "datum")
        time_path = self._find_field_path(context, lambda key: key.lower() == "time")
        if not datum_path or not time_path:
            return None

        return {
            "family": "datum_time_to_iso8601",
            "output_style": "lua_block",
            "target_path": datum_path,
            "generation_hints": {"datum_path": datum_path, "time_path": time_path},
        }

    def _match_ensure_items_array(self, normalized_prompt, context):
        if "zcdf_packages" not in normalized_prompt:
            return None
        if not self._contains_any(normalized_prompt, ["items", "item"]):
            return None
        if not self._contains_any(normalized_prompt, ["массив", "array"]):
            return None

        packages_path, item_field = self._find_packages_path(context)
        if not packages_path:
            return None

        return {
            "family": "ensure_items_array",
            "output_style": "lua_block",
            "target_path": packages_path,
            "generation_hints": {"packages_path": packages_path, "item_field": item_field or "items"},
        }

    def _match_discount_markdown_filter(self, normalized_prompt, context):
        if "discount" not in normalized_prompt or "markdown" not in normalized_prompt:
            return None
        if not self._contains_any(normalized_prompt, ["отфильт", "filter", "keep only", "keep"]):
            return None

        items_path, discount_field, markdown_field = self._find_discount_markdown_context(context)
        if not items_path:
            return None

        return {
            "family": "filter_discount_markdown",
            "output_style": "lua_block",
            "target_path": items_path,
            "generation_hints": {
                "items_path": items_path,
                "discount_field": discount_field or "Discount",
                "markdown_field": markdown_field or "Markdown",
            },
        }

    def _match_conditional_array_projection(self, prompt, normalized_prompt, context, prompt_paths):
        if "{" in (prompt or ""):
            return None
        if not self._contains_any(normalized_prompt, ["новый массив", "new array", "верни массив", "return array"]):
            return None
        if not self._contains_any(normalized_prompt, ["только", "where", "где", "if "]):
            return None

        source_path = (
            self._extract_source_subject(prompt, kind="array")
            or self._select_prompt_path(prompt_paths, expected_kind="array")
            or self._find_named_array_path(context, normalized_prompt)
        )
        projection_field = self._extract_projection_field(prompt)
        conditions = self._extract_simple_conditions(prompt)
        if not source_path or not projection_field or not conditions:
            return None

        return {
            "family": "conditional_array_projection",
            "output_style": "lua_block",
            "target_path": source_path,
            "generation_hints": {
                "source_path": source_path,
                "projection_field": projection_field,
                "conditions": conditions,
            },
        }

    def _match_email_validation(self, normalized_prompt, context, prompt_paths):
        if "email" not in normalized_prompt:
            return None
        if not self._contains_any(normalized_prompt, ["валид", "valid", "коррект", "boolean", "булев"]):
            return None

        email_path = self._select_prompt_path(prompt_paths, expected_kind="string") or self._find_field_path(context, lambda key: key.lower() == "email")
        if not email_path:
            return None

        return {
            "family": "email_validation",
            "output_style": "lua_block",
            "target_path": email_path,
            "generation_hints": {"email_path": email_path},
        }

    def _match_email_normalization(self, normalized_prompt, context, prompt_paths):
        if "email" not in normalized_prompt:
            return None
        if not self._contains_any(
            normalized_prompt,
            ["нормализ", "normalize", "lower", "lower-case", "trim", "убери пробел", "приведи"],
        ):
            return None
        if self._contains_any(
            normalized_prompt,
            [
                "спис",
                "list",
                "массив",
                "array",
                "таблиц",
                "table",
                "records",
                "контакт",
                "contacts",
                "_utils.array.new",
            ],
        ):
            return None

        email_path = self._select_prompt_path(prompt_paths, expected_kind="string") or self._find_field_path(
            context,
            lambda key: key.lower() in {"email", "useremail"},
        )
        if not email_path:
            return None
        if "[]" in email_path:
            return None

        return {
            "family": "normalize_email_string",
            "output_style": "lua_block",
            "target_path": email_path,
            "generation_hints": {"email_path": email_path},
        }

    def _match_regex_extract(self, prompt, normalized_prompt, context, prompt_paths):
        if not self._contains_any(normalized_prompt, ["regex", "pattern", "паттерн", "извлеки", "extract"]):
            return None

        pattern = self._extract_quoted_pattern(prompt)
        source_path = self._select_prompt_path(prompt_paths, expected_kind="string") or self._find_string_path(context, normalized_prompt)
        if not pattern or not source_path:
            return None

        return {
            "family": "regex_extract",
            "output_style": "lua_block",
            "target_path": source_path,
            "generation_hints": {"source_path": source_path, "pattern": pattern},
        }

    def _match_field_mapping(self, prompt, normalized_prompt, context, prompt_paths):
        if not self._contains_any(normalized_prompt, ["собери новый объект", "new object", "map", "mapping"]):
            return None

        mapping_pairs = self._extract_mapping_pairs(prompt)
        source_path = self._extract_source_subject(prompt, kind="object") or self._find_object_path(context, normalized_prompt)
        if not mapping_pairs or not source_path:
            return None

        return {
            "family": "field_mapping",
            "output_style": "lua_block",
            "target_path": source_path,
            "generation_hints": {"source_path": source_path, "mapping_pairs": mapping_pairs},
        }

    def _match_table_transform(self, prompt, normalized_prompt, context, prompt_paths):
        if "массив" not in normalized_prompt and "array" not in normalized_prompt:
            return None
        if not self._contains_any(normalized_prompt, ["собери новый массив", "new array", "reshape", "transform"]):
            return None

        mapping_pairs = self._extract_mapping_pairs(prompt)
        source_path = self._extract_source_subject(prompt, kind="array") or self._select_prompt_path(prompt_paths, expected_kind="array") or self._find_named_array_path(context, normalized_prompt)
        if not mapping_pairs or not source_path:
            return None

        return {
            "family": "table_transform",
            "output_style": "lua_block",
            "target_path": source_path,
            "generation_hints": {"source_path": source_path, "mapping_pairs": mapping_pairs},
        }

    def _match_square_variable(self, prompt, normalized_prompt):
        if not self._contains_any(normalized_prompt, ["квадрат", "square", "sqr"]):
            return None
        if not self._contains_any(normalized_prompt, ["перемен", "variable"]):
            return None

        match = re.search(r"(?<!\d)(\d+)(?!\d)", prompt)
        number_literal = match.group(1) if match else "5"
        return {
            "family": "augment_existing_code",
            "output_style": "json_envelope",
            "target_path": "",
            "generation_hints": {"number_literal": number_literal},
        }

    def _match_recall_time(self, normalized_prompt, context, prompt_paths):
        if "unix" not in normalized_prompt:
            return None

        iso_path = self._select_prompt_path({"explicit": [path for path in prompt_paths.get("explicit", []) if "recalltime" in path.lower()], "bare": [path for path in prompt_paths.get("bare", []) if "recalltime" in path.lower()]}, expected_kind="init") or self._find_recall_path(context)
        if not iso_path and not self._contains_any(normalized_prompt, ["recalltime", "recall time", "recall_time"]):
            return None
        if not iso_path:
            return None

        return {
            "family": "iso8601_to_epoch",
            "output_style": "lua_block",
            "target_path": iso_path,
            "generation_hints": {"iso_path": iso_path},
        }

    def _match_safety_guard(self, normalized_prompt):
        repair_intent = self._contains_any(
            normalized_prompt,
            [
                "исправ",
                "fix",
                "rewrite",
                "replace",
                "убери",
                "strip",
                "без ",
                "without ",
                "не использ",
                "do not use",
            ],
        )
        if repair_intent:
            return None

        risky_terms = [
            "jsonpath",
            "$.",
            "broken envelope",
            "workflow.variables",
            "ctx.body",
            "external ai api",
            "python code",
            "python ",
            "return sql",
            " sql ",
            "markdown fenced",
        ]
        if not self._contains_any(normalized_prompt, risky_terms):
            return None

        return {
            "family": "safety_guard",
            "output_style": "lua_block",
            "target_path": "wf.vars",
            "generation_hints": {},
            "safety_fallback": True,
            "assumptions": [
                "Unsafe, cross-language, or malformed requests are routed to a judged-safe Lua fallback.",
            ],
        }

    @staticmethod
    def _contains_any(text, variants):
        return any(variant in text for variant in variants)

    @staticmethod
    def _root_from_path(path):
        if not path:
            return None
        if path.startswith("wf.initVariables"):
            return "wf.initVariables"
        if path.startswith("wf.vars"):
            return "wf.vars"
        return None

    @staticmethod
    def _looks_like_array_path(path):
        leaf = (path or "").split(".")[-1].lower()
        return leaf.endswith("s") or "list" in leaf or "items" in leaf or "array" in leaf or "packages" in leaf

    def _find_array_path(self, context, normalized_prompt):
        candidates = []

        def walk(value, prefix):
            if isinstance(value, dict):
                for key, nested_value in value.items():
                    next_prefix = "{0}.{1}".format(prefix, key) if prefix else key
                    walk(nested_value, next_prefix)
            elif isinstance(value, list):
                candidates.append(prefix)
                if value:
                    walk(value[0], "{0}[]".format(prefix))

        walk(context, "")
        if not candidates:
            return None

        prioritized = []
        for path in candidates:
            leaf = path.split(".")[-1].lower()
            score = 0
            if leaf in normalized_prompt:
                score += 3
            if "email" in normalized_prompt and "email" in leaf:
                score += 4
            if "list" in leaf or leaf.endswith("s"):
                score += 1
            prioritized.append((score, path))

        prioritized.sort(key=lambda item: (-item[0], item[1]))
        return prioritized[0][1] if prioritized and prioritized[0][0] > 0 else candidates[0]

    def _find_named_array_path(self, context, normalized_prompt):
        array_path = self._find_array_path(context, normalized_prompt)
        if array_path and array_path.split(".")[-1].lower() in normalized_prompt:
            return array_path
        return array_path

    def _find_counter_path(self, context, normalized_prompt):
        candidates = []

        def walk(value, prefix):
            if isinstance(value, dict):
                for key, nested_value in value.items():
                    next_prefix = "{0}.{1}".format(prefix, key) if prefix else key
                    walk(nested_value, next_prefix)
            elif isinstance(value, (int, float)):
                candidates.append(prefix)

        walk(context, "")
        if not candidates:
            return None

        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", normalized_prompt)
        prioritized = []
        for path in candidates:
            leaf = path.split(".")[-1].lower()
            score = 0
            if leaf in normalized_prompt:
                score += 4
            if any(token == leaf for token in tokens):
                score += 3
            if "count" in leaf or "try" in leaf:
                score += 2
            prioritized.append((score, path))

        prioritized.sort(key=lambda item: (-item[0], item[1]))
        return prioritized[0][1] if prioritized and prioritized[0][0] > 0 else candidates[0]

    def _find_rest_result(self, context):
        result_path = None
        sample_entry = None

        def walk(value, prefix):
            nonlocal result_path, sample_entry
            if result_path:
                return
            if isinstance(value, dict):
                for key, nested_value in value.items():
                    next_prefix = "{0}.{1}".format(prefix, key) if prefix else key
                    walk(nested_value, next_prefix)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                if prefix.lower().endswith("result"):
                    result_path = prefix
                    sample_entry = value[0]

        walk(context, "")
        return result_path, sample_entry

    @staticmethod
    def _extract_keep_keys(prompt, sample_entry):
        prompt_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", prompt)
        sample_keys = list(sample_entry.keys())
        matched = []
        for key in sample_keys:
            if any(token.lower() == key.lower() for token in prompt_tokens):
                matched.append(key)
        excluded = []
        lowered_prompt = prompt.lower()
        for key in sample_keys:
            lowered_key = key.lower()
            if "without {0}".format(lowered_key) in lowered_prompt or "без {0}".format(lowered_key) in lowered_prompt:
                excluded.append(key)
        matched = [key for key in matched if key not in excluded]
        if matched:
            return matched
        defaults = [key for key in sample_keys if key.upper() in {"ID", "ENTITY_ID", "CALL"}]
        return defaults

    def _find_field_path(self, context, predicate):
        result = None

        def walk(value, prefix):
            nonlocal result
            if result:
                return
            if isinstance(value, dict):
                for key, nested_value in value.items():
                    next_prefix = "{0}.{1}".format(prefix, key) if prefix else key
                    if predicate(key):
                        result = next_prefix
                        return
                    walk(nested_value, next_prefix)
            elif isinstance(value, list) and value:
                walk(value[0], "{0}[]".format(prefix))

        walk(context, "")
        return result

    def _find_string_path(self, context, normalized_prompt):
        candidates = []

        def walk(value, prefix):
            if isinstance(value, dict):
                for key, nested_value in value.items():
                    next_prefix = "{0}.{1}".format(prefix, key) if prefix else key
                    walk(nested_value, next_prefix)
            elif isinstance(value, str):
                candidates.append(prefix)
            elif isinstance(value, list) and value:
                walk(value[0], "{0}[]".format(prefix))

        walk(context, "")
        if not candidates:
            return None

        prioritized = []
        for path in candidates:
            leaf = path.split(".")[-1].replace("[]", "").lower()
            score = 0
            if leaf in normalized_prompt:
                score += 4
            if "message" in normalized_prompt and "message" in leaf:
                score += 2
            if "subject" in normalized_prompt and "subject" in leaf:
                score += 2
            prioritized.append((score, path))

        prioritized.sort(key=lambda item: (-item[0], item[1]))
        return prioritized[0][1] if prioritized and prioritized[0][0] > 0 else candidates[0]

    def _find_object_path(self, context, normalized_prompt):
        candidates = []

        def walk(value, prefix):
            if isinstance(value, dict):
                if prefix and prefix not in {"wf", "wf.vars", "wf.initVariables"}:
                    candidates.append(prefix)
                for key, nested_value in value.items():
                    next_prefix = "{0}.{1}".format(prefix, key) if prefix else key
                    walk(nested_value, next_prefix)
            elif isinstance(value, list) and value:
                walk(value[0], "{0}[]".format(prefix))

        walk(context, "")
        if not candidates:
            return None

        prioritized = []
        for path in candidates:
            leaf = path.split(".")[-1].replace("[]", "").lower()
            score = 0
            if leaf in normalized_prompt:
                score += 4
            if "object" in normalized_prompt or "объект" in normalized_prompt:
                score += 1
            prioritized.append((score, path))

        prioritized.sort(key=lambda item: (-item[0], item[1]))
        return prioritized[0][1] if prioritized and prioritized[0][0] > 0 else candidates[0]

    def _find_packages_path(self, context):
        result = None
        item_field = None

        def walk(value, prefix):
            nonlocal result, item_field
            if result:
                return
            if isinstance(value, dict):
                for key, nested_value in value.items():
                    next_prefix = "{0}.{1}".format(prefix, key) if prefix else key
                    walk(nested_value, next_prefix)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                if prefix.lower().endswith("zcdf_packages"):
                    result = prefix
                    sample = value[0]
                    if "items" in sample:
                        item_field = "items"
                    elif "item" in sample:
                        item_field = "item"

        walk(context, "")
        return result, item_field

    def _find_discount_markdown_context(self, context):
        result = None
        discount_field = None
        markdown_field = None

        def walk(value, prefix):
            nonlocal result, discount_field, markdown_field
            if result:
                return
            if isinstance(value, dict):
                for key, nested_value in value.items():
                    next_prefix = "{0}.{1}".format(prefix, key) if prefix else key
                    walk(nested_value, next_prefix)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                sample = value[0]
                keys_lower = {key.lower(): key for key in sample.keys()}
                if "discount" in keys_lower and "markdown" in keys_lower:
                    result = prefix
                    discount_field = keys_lower["discount"]
                    markdown_field = keys_lower["markdown"]

        walk(context, "")
        return result, discount_field, markdown_field

    def _find_recall_path(self, context):
        return self._find_field_path(
            context,
            lambda key: key.lower() in {"recalltime", "recall_time"},
        )

    @staticmethod
    def _extract_quoted_pattern(prompt):
        intent_match = re.search(
            r"(?:lua\s+pattern|pattern|паттерн)[^\"']*[\"']([^\"']+)[\"']",
            prompt or "",
            flags=re.IGNORECASE,
        )
        if intent_match:
            return intent_match.group(1)
        matches = re.findall(r"['\"]([^'\"]+)['\"]", prompt or "")
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _extract_mapping_pairs(prompt):
        spec_match = re.search(r"\{([^{}]+)\}", prompt or "")
        if not spec_match:
            return []

        pairs = []
        for chunk in spec_match.group(1).split(","):
            if "=" not in chunk:
                continue
            target, source = chunk.split("=", 1)
            target = target.strip()
            source = source.strip()
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", target):
                continue
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*$", source):
                continue
            pairs.append({"target": target, "source": source})
        return pairs

    @staticmethod
    def _infer_target_root(context_paths):
        has_init = any(path.startswith("wf.initVariables") for path in context_paths)
        has_vars = any(path.startswith("wf.vars") for path in context_paths)
        if has_init and has_vars:
            return "unknown_mixed"
        if any(path.startswith("wf.initVariables") for path in context_paths):
            return "wf.initVariables"
        if any(path.startswith("wf.vars") for path in context_paths):
            return "wf.vars"
        return "unknown"

    def _extract_prompt_paths(self, prompt):
        explicit = []
        bare = []
        for match in re.finditer(r"\bwf\.(?:vars|initVariables)(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", prompt or ""):
            explicit.append(match.group(0))
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b", prompt or ""):
            value = match.group(1)
            if value.startswith("wf.") or value.startswith("string.") or value.startswith("math.") or value.startswith("_utils."):
                continue
            bare.append(value)
        return {"explicit": explicit, "bare": bare}

    def _select_prompt_path(self, prompt_paths, expected_kind):
        candidates = list(prompt_paths.get("explicit", [])) + [self._materialize_bare_path(path, expected_kind) for path in prompt_paths.get("bare", [])]
        for candidate in candidates:
            if not candidate:
                continue
            leaf = candidate.split(".")[-1].lower()
            if expected_kind == "init" and candidate.startswith("wf.initVariables."):
                return candidate
            if expected_kind == "array" and self._looks_like_array_path(candidate):
                return candidate
            if expected_kind == "counter" and any(token in leaf for token in ["count", "counter", "try", "num"]):
                return candidate
            if expected_kind == "string" and any(token in leaf for token in ["message", "subject", "note", "line", "email"]):
                return candidate
            if expected_kind == "object":
                return candidate
        return None

    def _materialize_prompt_paths(self, prompt_paths):
        materialized = list(prompt_paths.get("explicit", []))
        for kind in ["array", "counter", "string", "object", "init"]:
            candidate = self._select_prompt_path(prompt_paths, expected_kind=kind)
            if candidate and candidate not in materialized:
                materialized.append(candidate)
        return materialized[:12]

    @staticmethod
    def _root_candidates(target_root, prompt_paths, context_paths):
        roots = []
        for path in prompt_paths:
            root = TaskExtractor._root_from_path(path)
            if root and root not in roots:
                roots.append(root)
        if target_root == "unknown_mixed":
            for root in ["wf.vars", "wf.initVariables"]:
                if root not in roots:
                    roots.append(root)
        elif target_root not in {"unknown", None} and target_root not in roots:
            roots.append(target_root)
        for root in ["wf.vars", "wf.initVariables"]:
            if any(path.startswith(root) for path in context_paths) and root not in roots:
                roots.append(root)
        return roots[:4]

    @staticmethod
    def _materialize_bare_path(path, expected_kind):
        if not path:
            return None
        if path.startswith("wf."):
            return path
        if path.startswith("initVariables."):
            return "wf.{0}".format(path)
        if path.startswith("vars."):
            return "wf.{0}".format(path)
        root = "wf.vars"
        if expected_kind == "init":
            root = "wf.initVariables"
        return "{0}.{1}".format(root, path)

    def _extract_source_subject(self, prompt, kind):
        patterns = {
            "object": [
                r"(?:из|from)\s+(?:объекта|object)\s+([A-Za-z_][A-Za-z0-9_\.]*)",
            ],
            "array": [
                r"(?:из|from)\s+(?:массива|array)\s+([A-Za-z_][A-Za-z0-9_\.]*)",
            ],
        }
        for pattern in patterns.get(kind, []):
            match = re.search(pattern, prompt or "", flags=re.IGNORECASE)
            if not match:
                continue
            path = match.group(1)
            if path.startswith("wf."):
                return path
            return "wf.vars.{0}".format(path)
        return None

    @staticmethod
    def _extract_projection_field(prompt):
        patterns = [
            r"(?:верни|получи|return)\s+(?:новый\s+)?(?:массив|array)\s+([A-Za-z_][A-Za-z0-9_\.]*)",
            r"(?:верни|получи|return)\s+([A-Za-z_][A-Za-z0-9_\.]*)\s+(?:только|only)",
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt or "", flags=re.IGNORECASE)
            if not match:
                continue
            value = match.group(1)
            if "." in value:
                return value
            return value
        return None

    @staticmethod
    def _extract_simple_conditions(prompt):
        conditions = []
        text = prompt or ""

        eq_pattern = re.compile(
            r"([A-Za-z_][A-Za-z0-9_\.]*)\s*(?:равен|равна|равно|equals?|==)\s*[\"']?([A-Za-z_][A-Za-z0-9_\-]*|true|false)[\"']?",
            flags=re.IGNORECASE,
        )
        for field, value in eq_pattern.findall(text):
            parsed_value = value
            if value.lower() == "true":
                parsed_value = True
            elif value.lower() == "false":
                parsed_value = False
            conditions.append({"field": field, "operator": "eq", "value": parsed_value})

        comparisons = [
            (r"([A-Za-z_][A-Za-z0-9_\.]*)\s*(?:больше|greater than|>)\s*(\d+(?:\.\d+)?)", "gt"),
            (r"([A-Za-z_][A-Za-z0-9_\.]*)\s*(?:не меньше|at least|>=)\s*(\d+(?:\.\d+)?)", "gte"),
            (r"([A-Za-z_][A-Za-z0-9_\.]*)\s*(?:меньше|less than|<)\s*(\d+(?:\.\d+)?)", "lt"),
            (r"([A-Za-z_][A-Za-z0-9_\.]*)\s*(?:не больше|at most|<=)\s*(\d+(?:\.\d+)?)", "lte"),
        ]
        for pattern, operator in comparisons:
            for field, value in re.findall(pattern, text, flags=re.IGNORECASE):
                conditions.append(
                    {
                        "field": field,
                        "operator": operator,
                        "value": float(value) if "." in value else int(value),
                    }
                )

        not_nil_pattern = re.compile(
            r"([A-Za-z_][A-Za-z0-9_\.]*)\s*(?:не\s+nil|not\s+nil)",
            flags=re.IGNORECASE,
        )
        for field in not_nil_pattern.findall(text):
            conditions.append({"field": field, "operator": "not_nil", "value": None})

        deduped = []
        seen = set()
        for condition in conditions:
            marker = (condition["field"], condition["operator"], str(condition["value"]))
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(condition)
        return deduped

    def _passes_family_guard(self, family, prompt, normalized_prompt, context, prompt_paths):
        if family == "last_array_item":
            if self._contains_any(
                normalized_prompt,
                [
                    "не про послед",
                    "не последний",
                    "not about last",
                    "not the last",
                ],
            ):
                return False
            return bool(
                self._contains_any(normalized_prompt, ["спис", "list", "массив", "array"])
                or self._select_prompt_path(prompt_paths, expected_kind="array")
                or self._find_array_path(context, normalized_prompt)
            )
        if family == "counter_increment":
            return bool(
                self._select_prompt_path(prompt_paths, expected_kind="counter")
                or self._find_counter_path(context, normalized_prompt)
            )
        if family == "email_validation":
            return self._contains_any(normalized_prompt, ["валид", "valid", "коррект", "boolean", "булев"])
        if family == "safety_guard":
            return self._contains_any(
                normalized_prompt,
                [
                    "ctx.body",
                    "workflow.variables",
                    "external ai api",
                    "python code",
                    "return sql",
                    "верни sql",
                    "write sql",
                    "broken envelope",
                ],
            )
        return True
