import json


class OutputFormatter:
    def format(self, code, output_style):
        rendered = (code or "").strip()
        if output_style == "json_envelope":
            try:
                payload = json.loads(rendered)
            except json.JSONDecodeError:
                return rendered
            return json.dumps(payload, ensure_ascii=False, indent=2)
        return rendered
