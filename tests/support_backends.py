import json

from app.generation.backend_errors import BackendUnavailable


class FailIfCalledBackend:
    def generate(self, prompt, context=None):
        raise AssertionError("backend should not be called for this test")

    def complete(self, prompt, response_format=None, model=None):
        raise AssertionError("backend should not be called for this test")


class UnavailableBackend:
    def generate(self, prompt, context=None):
        raise BackendUnavailable(reason="test_backend_unavailable")

    def complete(self, prompt, response_format=None, model=None):
        raise BackendUnavailable(reason="test_backend_unavailable")


class DeterministicTestBackend:
    def __init__(self):
        self._codes = {
            "Нужен Lua для wf.vars.orders": (
                "local result = _utils.array.new()\n"
                "for _, item in ipairs(wf.vars.orders or {}) do\n"
                "  if item.status == \"paid\" and item.amount > 1000 then\n"
                "    table.insert(result, item.order_id)\n"
                "  end\n"
                "end\n"
                "return result"
            ),
            "Возьми wf.vars.contacts и подготовь список таблиц": (
                "local result = _utils.array.new()\n"
                "for _, item in ipairs(wf.vars.contacts or {}) do\n"
                "  if item.active == true and item.email ~= nil then\n"
                "    table.insert(result, {id = item.id, email = string.lower(item.email)})\n"
                "  end\n"
                "end\n"
                "return result"
            ),
            "Посчитай, сколько элементов в wf.vars.orders": (
                "local count = 0\n"
                "for _, item in ipairs(wf.vars.orders or {}) do\n"
                "  if item.status == \"paid\" then\n"
                "    count = count + 1\n"
                "  end\n"
                "end\n"
                "return count"
            ),
            "Для wf.vars.orders сложи amount": (
                "local total = 0\n"
                "for _, item in ipairs(wf.vars.orders or {}) do\n"
                "  if item.status == \"shipped\" then\n"
                "    total = total + (item.amount or 0)\n"
                "  end\n"
                "end\n"
                "return total"
            ),
            "Найди в wf.vars.contacts первый email": (
                "for _, item in ipairs(wf.vars.contacts or {}) do\n"
                "  if item.vip == true then\n"
                "    return item.email\n"
                "  end\n"
                "end\n"
                "return nil"
            ),
            "Из wf.vars.profile подготовь payload": (
                "local profile = wf.vars.profile or {}\n"
                "return {\n"
                "  name_upper = string.upper(profile.name or \"\"),\n"
                "  city = profile.address and profile.address.city or nil\n"
                "}"
            ),
            "По wf.vars.items верни список sku": (
                "local result = _utils.array.new()\n"
                "for _, item in ipairs(wf.vars.items or {}) do\n"
                "  if item.stock > 0 and item.archived == false then\n"
                "    table.insert(result, item.sku)\n"
                "  end\n"
                "end\n"
                "return result"
            ),
            "Нормализуй launch variable wf.initVariables.userEmail": (
                "local value = wf.initVariables.userEmail or \"\"\n"
                "value = string.gsub(value, \"^%s*(.-)%s*$\", \"%1\")\n"
                "return string.lower(value)"
            ),
            "Верни boolean: есть ли в wf.vars.contacts": (
                "for _, item in ipairs(wf.vars.contacts or {}) do\n"
                "  if item.channel == \"sms\" and item.enabled == true then\n"
                "    return true\n"
                "  end\n"
                "end\n"
                "return false"
            ),
            "По данным wf.vars.contacts собери список phone": (
                "local result = _utils.array.new()\n"
                "for _, item in ipairs(wf.vars.contacts or {}) do\n"
                "  if item.channel == \"whatsapp\" and item.phone ~= nil then\n"
                "    table.insert(result, item.phone)\n"
                "  end\n"
                "end\n"
                "return result"
            ),
            "На основе wf.vars.orders подготовь список кратких таблиц": (
                "local result = _utils.array.new()\n"
                "for _, item in ipairs(wf.vars.orders or {}) do\n"
                "  if item.amount >= 100 then\n"
                "    table.insert(result, {id = item.order_id, status = item.status})\n"
                "  end\n"
                "end\n"
                "return result"
            ),
            "Верни city из wf.vars.customer.address.city": (
                "local customer = wf.vars.customer\n"
                "if not customer or not customer.address then\n"
                "  return nil\n"
                "end\n"
                "return customer.address.city"
            ),
            "Нормализуй email и верни его в lower-case.": (
                "return string.lower(wf.vars.email)"
            ),
            "Посчитай количество элементов в выбранном массиве items.": (
                "local count = 0\n"
                "for _, _ in ipairs(wf.vars.items or {}) do\n"
                "  count = count + 1\n"
                "end\n"
                "return count"
            ),
        }
        self._clarification_prompts = {
            "Нормализуй email и верни его в lower-case.",
            "Посчитай количество элементов в выбранном массиве items.",
            "Посчитай количество элементов в массиве items и верни число.",
            "Верни первый элемент из массива items.",
            "Верни последний элемент из массива items.",
            "Посчитай, сколько элементов в orders имеют status paid, и верни число.",
            "Сложи amount только у записей orders со status shipped.",
            "Верни city из customer.address.city.",
            "Верни boolean: есть ли в contacts хотя бы одна запись, где channel равно sms и enabled равно true.",
            "Верни name_upper из profile.name в верхнем регистре.",
            "Найди первый phone в contacts, где channel равно whatsapp. Если такого нет, верни nil.",
        }

    @staticmethod
    def _chosen_root(prompt):
        if "Use wf.initVariables for email root." in prompt:
            return "wf.initVariables"
        if "Use wf.vars for email root." in prompt:
            return "wf.vars"
        if "Use wf.initVariables for this task." in prompt:
            return "wf.initVariables"
        if "Use wf.vars for this task." in prompt:
            return "wf.vars"
        return None

    def complete(self, prompt, response_format=None, model=None):
        user_prompt = prompt.rsplit("User prompt:\n", 1)[-1]
        if "\n\nContext JSON:" in user_prompt:
            user_prompt = user_prompt.split("\n\nContext JSON:", 1)[0]
        user_prompt = user_prompt.strip()
        lowered_user_prompt = user_prompt.lower()
        chosen_root = self._chosen_root(prompt)

        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            root = chosen_root or ("wf.initVariables" if "wf.initVariables" in user_prompt else "wf.vars")
            if (
                user_prompt in self._clarification_prompts
                and "wf.initVariables" not in user_prompt
                and "wf.vars" not in user_prompt
                and chosen_root is None
            ):
                question = "Use wf.vars or wf.initVariables for email root?"
                if "email" not in lowered_user_prompt:
                    question = "Use wf.vars or wf.initVariables for this task?"
                return json.dumps(
                    {
                        "family": "generic_lua",
                        "root": "unknown_mixed",
                        "source_paths": [],
                        "return_shape": "scalar",
                        "constraints": ["Do not use JsonPath"],
                        "assumptions": [],
                        "clarification_needed": True,
                        "clarification_question": question,
                        "semantic_checks": [],
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "family": "generic_lua",
                    "root": root,
                    "source_paths": [],
                    "return_shape": "scalar",
                    "constraints": ["Do not use JsonPath"],
                    "assumptions": [],
                    "clarification_needed": False,
                    "clarification_question": "",
                    "semantic_checks": [],
                },
                ensure_ascii=False,
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return json.dumps(
                {"repairable": False, "issues": [], "minimal_actions": []},
                ensure_ascii=False,
            )
        if "Нормализуй email и верни его в lower-case." in user_prompt:
            if chosen_root == "wf.initVariables":
                return "return string.lower(wf.initVariables.email)"
            return "return string.lower(wf.vars.email)"
        if "Посчитай количество элементов в массиве items и верни число." in user_prompt:
            root = chosen_root or "wf.vars"
            return (
                "local count = 0\n"
                "for _, _ in ipairs({0}.items or {{}}) do\n"
                "  count = count + 1\n"
                "end\n"
                "return count"
            ).format(root)
        if "Верни первый элемент из массива items." in user_prompt:
            root = chosen_root or "wf.vars"
            return (
                "local items = {0}.items or {{}}\n"
                "return items[1]"
            ).format(root)
        if "Посчитай, сколько элементов в orders имеют status paid, и верни число." in user_prompt:
            root = chosen_root or "wf.vars"
            return (
                "local count = 0\n"
                "for _, item in ipairs({0}.orders or {{}}) do\n"
                "  if item.status == \"paid\" then\n"
                "    count = count + 1\n"
                "  end\n"
                "end\n"
                "return count"
            ).format(root)
        if "Верни city из customer.address.city." in user_prompt:
            root = chosen_root or "wf.vars"
            return (
                "local customer = {0}.customer\n"
                "if not customer or not customer.address then\n"
                "  return nil\n"
                "end\n"
                "return customer.address.city"
            ).format(root)
        if "wf.initVariables.userEmail" in user_prompt and "нормализ" in lowered_user_prompt:
            return (
                "local value = wf.initVariables.userEmail or \"\"\n"
                "value = string.gsub(value, \"^%s*(.-)%s*$\", \"%1\")\n"
                "return string.lower(value)"
            )
        if "discount" in lowered_user_prompt and "markdown" in lowered_user_prompt:
            return (
                "local result = _utils.array.new()\n"
                "for _, item in ipairs(wf.vars.parsedCsv or {}) do\n"
                "  if item.Discount ~= nil or item.Markdown ~= nil then\n"
                "    table.insert(result, item)\n"
                "  end\n"
                "end\n"
                "return result"
            )
        if "restbody" in lowered_user_prompt and "entity_id" in lowered_user_prompt and "call" in lowered_user_prompt:
            return (
                "local result = wf.vars.RESTbody.result\n"
                "for _, filteredEntry in pairs(result) do\n"
                "  for key, value in pairs(filteredEntry) do\n"
                "    if key ~= \"ID\" and key ~= \"ENTITY_ID\" and key ~= \"CALL\" then\n"
                "      filteredEntry[key] = nil\n"
                "    end\n"
                "  end\n"
                "end\n"
                "return result"
            )
        if "try_count_n" in lowered_user_prompt:
            return "return wf.vars.try_count_n + 1"
        if "datum" in lowered_user_prompt and "time" in lowered_user_prompt and "iso" in lowered_user_prompt:
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
                "local second = safe_sub(TIME, 5, 6)\n"
                "return string.format('%s-%s-%sT%s:%s:%s.00000Z', year, month, day, hour, minute, second)"
            )
        for snippet, code in self._codes.items():
            if snippet in user_prompt:
                return code
        return "return nil"

    def generate(self, prompt, context=None):
        return self.complete(prompt)
