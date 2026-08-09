import json


def _dump_json(value):
    return json.dumps(value or {}, ensure_ascii=False, indent=2, sort_keys=True)


def _session_block(session_state):
    if not session_state:
        return "No session state."
    return _dump_json(
        {
            "original_task": session_state.get("original_task"),
            "clarification_history": session_state.get("clarification_history", []),
            "feedback_history": session_state.get("feedback_history", []),
            "previous_candidate_code": session_state.get("previous_candidate_code"),
            "previous_validation_report": session_state.get("previous_validation_report", {}),
        }
    )


def _family_writer_guidance(planner):
    family = (planner or {}).get("family")
    guidance = {
        "last_array_item": [
            "Preserve the original source path in the final indexed expression, for example source[#source].",
            "Guard nil/empty safely, but keep the final return as one scalar element from the original array path.",
            "Do not create wrapper arrays, loops, or workflow-state mutations for this family.",
        ],
        "counter_increment": [
            "Return one scalar expression equal to the current counter plus one.",
            "Do not mutate wf.vars or wf.initVariables for this family.",
        ],
        "datum_time_to_iso8601": [
            "Use DATUM and TIME fragments from context and format the final value as YYYY-MM-DDTHH:MM:SS.00000Z.",
            "Strip non-digits from TIME and normalize short fragments with HH -> HH0000 and HHMM -> HHMM00 before slicing hour/minute/second.",
            "Guard missing fragments with safe substring logic instead of crashing.",
        ],
        "ensure_items_array": [
            "Preserve the package objects and normalize only each `.items` field into an array.",
            "Do not flatten nested package contents into one output array.",
        ],
        "augment_existing_code": [
            "Return a valid JSON object whose values are lua{...}lua strings.",
            "Use the canonical keys `num` and `squared`; do not invent aliases such as `sqr` and do not add extra envelope keys.",
            "The `num` field must itself be executable Lua, for example `lua{return tonumber('7')}lua`, not a raw literal like `lua{7}lua`.",
            "If no workflow value is provided, use the extracted numeric literal from the task hints instead of inventing wf.vars paths.",
            "Do not emit raw Lua statements outside the JSON envelope.",
        ],
        "email_validation": [
            "Read one email string from the extracted source path and return one boolean scalar.",
            "Use string.match(... ) ~= nil so the final result is a boolean, not the matched string.",
        ],
        "normalize_email_string": [
            "Read one email string from the extracted source path, trim surrounding whitespace, lower-case it, and return one scalar string.",
            "Do not wrap the result into an object or array.",
        ],
        "regex_extract": [
            "Read one source string, store it in a local variable, and return string.match(value, pattern).",
            "Use the exact Lua pattern from the planner/source hints and return one scalar match value or nil.",
        ],
        "iso8601_to_epoch": [
            "Parse the ISO string with one Lua pattern match into year, month, day, hour, minute, second, sign, offset hour, and offset minute.",
            "Compute epoch seconds with pure arithmetic and leap-year logic from 1970-01-01.",
            "Subtract the parsed timezone offset in seconds.",
            "Never use os.*, io.*, debug.*, package.*, _utils.date.*, or external helpers for this family.",
        ],
    }
    lines = guidance.get(family, [])
    if not lines:
        return "No extra family-specific guidance."
    return "\n".join("- {0}".format(line) for line in lines)


def build_planner_prompt(prompt, context, task_spec, rules, session_state=None):
    rules_block = "\n".join("- {0}".format(rule) for rule in rules) or "- No extra rules."
    return """You are the planner for a LocalScript/Lua generation pipeline.
Return JSON only. No markdown, no explanations.

Allowed planner families:
- generic_lua
- last_array_item
- counter_increment
- rest_cleanup
- datum_time_to_iso8601
- ensure_items_array
- filter_discount_markdown
- augment_existing_code
- email_validation
- normalize_email_string
- iso8601_to_epoch
- conditional_array_projection
- field_mapping
- regex_extract
- table_transform

Contract:
{{
  "family": "generic_lua | last_array_item | counter_increment | rest_cleanup | datum_time_to_iso8601 | ensure_items_array | filter_discount_markdown | augment_existing_code | email_validation | normalize_email_string | iso8601_to_epoch | conditional_array_projection | field_mapping | regex_extract | table_transform",
  "root": "wf.vars | wf.initVariables | unknown_mixed | unknown",
  "source_paths": ["wf.vars.orders"],
  "return_shape": "scalar | object | array | json_envelope",
  "constraints": ["Do not use JsonPath"],
  "assumptions": [],
  "clarification_needed": false,
  "clarification_question": "",
  "semantic_checks": ["must ..."]
}}

Routing notes:
- If the task asks for the last element from an array, use last_array_item.
- If the task asks to increment a counter and return one value, use counter_increment.
- If the task asks to clean a REST result and keep only specific keys, use rest_cleanup.
- If the task asks to build ISO 8601 from DATUM and TIME, use datum_time_to_iso8601.
- If the task asks to normalize package items into arrays, use ensure_items_array.
- If the task asks to filter Discount/Markdown rows, use filter_discount_markdown.
- If the task asks to add or preserve named variables in a JSON envelope, use augment_existing_code.
- If the task asks to validate one email and return boolean, use email_validation.
- If the task asks to normalize one email-like string and return the normalized scalar, use normalize_email_string.
- If the task asks to parse an ISO 8601 timestamp into epoch seconds, use iso8601_to_epoch.
- If the task is clearly array filter+projection, prefer conditional_array_projection.
- If the task builds one object from one source object, prefer field_mapping.
- If the task builds an array of objects from an input array, prefer table_transform.
- If the task asks for a scalar result such as count / sum / first match / normalized string / boolean, use generic_lua.
- If the task is not confidently any supported family, use generic_lua.
- Never invent external APIs or unsupported namespaces.
- If the task is ambiguous in target root, return shape, mutate-vs-return behavior, or field selection, set `clarification_needed=true` and ask one short concrete question.

Global rules:
{rules_block}

Extractor hints:
{extractor_hints}

Session state JSON:
{session_state}

User prompt:
{user_prompt}

Context JSON:
{context_json}
""".format(
        rules_block=rules_block,
        extractor_hints=_dump_json(task_spec.model_dump()),
        session_state=_session_block(session_state),
        user_prompt=prompt or "",
        context_json=_dump_json(context),
    )


def build_writer_prompt(prompt, context, task_spec, planner, rules, examples, session_state=None):
    rules_block = "\n".join("- {0}".format(rule) for rule in rules) or "- No extra rules."
    examples_block = []
    for example in examples:
        examples_block.append(
            "Example {id} ({family})\nPrompt: {prompt}\nConstraints:\n{constraints}\nReference code:\n{reference_code}".format(
                id=example.get("id"),
                family=example.get("family"),
                prompt=example.get("prompt"),
                constraints="\n".join("- {0}".format(item) for item in example.get("key_constraints", [])) or "- none",
                reference_code=example.get("reference_code") or "-- no code reference",
            )
        )
    examples_text = "\n\n".join(examples_block) if examples_block else "No few-shot examples selected."
    return """You are the writer for a LocalScript/Lua generation pipeline.
Return code only. No markdown fences. No explanations.

Required style:
- Use wf.vars or wf.initVariables directly.
- Do not use JsonPath.
- When creating a new array, prefer:
  local result = _utils.array.new()
  for _, item in ipairs(source or {{}}) do
    ...
    table.insert(result, value)
  end
  return result
- Prefer explicit loops over callback-style _utils.array.new(function(...), source).
- If the task says empty/nil input should return an empty array, handle that explicitly and safely.
- Preserve field names from prompt/context.
- Do not mutate wf.vars or wf.initVariables unless the task explicitly asks to write updated values back into the workflow state.
- Never shadow protected runtime/global identifiers with local variables: do not declare locals named `table`, `string`, `math`, `utf8`, `_utils`, or `wf`.
- Respect planner return_shape exactly:
  - array => return an array, usually via _utils.array.new()
  - object => return one Lua table object, not an array
  - scalar => return one scalar value, not an array/table wrapper
  - json_envelope => return one valid JSON object with `lua{{...}}lua` string values
- Respect planner family exactly:
  - last_array_item => return the last element from the source array and do not build wrapper arrays
  - counter_increment => return `counter + 1` as a scalar and do not create loops or mutate workflow state
  - rest_cleanup => keep the original row structure but preserve only requested keys
  - rest_cleanup => iterate over entry keys and remove everything except the extracted keep_keys; do not name excluded keys explicitly unless they are preserved
  - datum_time_to_iso8601 => build `YYYY-MM-DDTHH:MM:SS.00000Z` from DATUM/TIME fragments
  - ensure_items_array => preserve package objects and normalize each `.items` field into an array in place
  - filter_discount_markdown => return a filtered array of the original rows
  - augment_existing_code => return a JSON envelope, not raw Lua statements
  - augment_existing_code => return exactly the canonical `num` and `squared` keys unless the task explicitly names a different contract
  - email_validation => return one boolean scalar and use string.match(... ) ~= nil
  - normalize_email_string => trim whitespace, lower-case the value, and return one scalar string
  - iso8601_to_epoch => parse the actual ISO string from context; never hardcode date parts, do not use os.*, and prefer the shortest correct arithmetic implementation
  - field_mapping => build one object from one source object
  - table_transform => build an array of objects
  - conditional_array_projection => filter an array and project values
  - generic_lua => solve the task directly without forcing array constructors

Planner JSON:
{planner_json}

Extractor hints:
{extractor_hints}

Global rules:
{rules_block}

Relevant local examples:
{examples_text}

If a reference example matches the same family, follow its structure closely unless the current context requires a small local adaptation.

Family-specific guidance:
{family_guidance}

Session state JSON:
{session_state}

User prompt:
{user_prompt}

Context JSON:
{context_json}
""".format(
        planner_json=_dump_json(planner),
        extractor_hints=_dump_json(task_spec.model_dump()),
        rules_block=rules_block,
        examples_text=examples_text,
        family_guidance=_family_writer_guidance(planner),
        session_state=_session_block(session_state),
        user_prompt=prompt or "",
        context_json=_dump_json(context),
    )


def build_critic_prompt(prompt, context, planner, code, validation_errors, critic_rules, session_state=None):
    critic_rules_block = []
    for rule in critic_rules:
        critic_rules_block.append(
            "- {id}: {repair}".format(
                id=rule.get("id", "rule"),
                repair=rule.get("repair", ""),
            )
        )
    rules_text = "\n".join(critic_rules_block) or "- No critic rules selected."
    return """You are the critic for a LocalScript/Lua generation pipeline.
Return JSON only. No markdown, no explanations.

Contract:
{{
  "repairable": true,
  "issues": ["issue_code"],
  "minimal_actions": ["minimal_local_fix"]
}}

Keep actions minimal and concrete. Mention only actual problems.
Do not invent array-constructor fixes for scalar or json_envelope tasks.

Planner JSON:
{planner_json}

Validation errors:
{validation_errors}

Critic rules:
{rules_text}

Session state JSON:
{session_state}

User prompt:
{user_prompt}

Context JSON:
{context_json}

Current code:
{code}
""".format(
        planner_json=_dump_json(planner),
        validation_errors=_dump_json({"errors": validation_errors}),
        rules_text=rules_text,
        session_state=_session_block(session_state),
        user_prompt=prompt or "",
        context_json=_dump_json(context),
        code=code or "",
    )


def build_fixer_prompt(prompt, context, task_spec, planner, code, critic, validation_errors, rules, examples, session_state=None):
    rules_block = "\n".join("- {0}".format(rule) for rule in rules) or "- No extra rules."
    examples_block = []
    for example in examples:
        examples_block.append(
            "Example {id} ({family})\nPrompt: {prompt}\nReference code:\n{reference_code}".format(
                id=example.get("id"),
                family=example.get("family"),
                prompt=example.get("prompt"),
                reference_code=example.get("reference_code") or "-- no code reference",
            )
        )
    examples_text = "\n\n".join(examples_block) if examples_block else "No few-shot examples selected."
    return """You are the fixer for a LocalScript/Lua generation pipeline.
Return corrected code only. No markdown fences. No explanations.

Hard requirements:
- Keep the fix minimal.
- Preserve correct parts of the code.
- Use only LocalScript/Lua.
- Do not use JsonPath.
- Prefer canonical explicit loops and _utils.array.new() for returned arrays.
- Never use dangerous stdlib namespaces such as os, io, debug, or package.

Planner JSON:
{planner_json}

Extractor hints:
{extractor_hints}

Critic JSON:
{critic_json}

Validation errors:
{validation_errors}

Global rules:
{rules_block}

Relevant local examples:
{examples_text}

Family-specific guidance:
{family_guidance}

Session state JSON:
{session_state}

User prompt:
{user_prompt}

Context JSON:
{context_json}

Current code:
{code}
""".format(
        planner_json=_dump_json(planner),
        extractor_hints=_dump_json(task_spec.model_dump()),
        critic_json=_dump_json(critic),
        validation_errors=_dump_json({"errors": validation_errors}),
        rules_block=rules_block,
        examples_text=examples_text,
        family_guidance=_family_writer_guidance(planner),
        session_state=_session_block(session_state),
        user_prompt=prompt or "",
        context_json=_dump_json(context),
        code=code or "",
    )
