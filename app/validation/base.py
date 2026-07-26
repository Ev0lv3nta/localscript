from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationMessage:
    validator: str
    level: str
    code: str
    message: str


@dataclass
class ValidationReport:
    messages: List[ValidationMessage] = field(default_factory=list)

    def add(self, validator, level, code, message):
        self.messages.append(
            ValidationMessage(
                validator=validator,
                level=level,
                code=code,
                message=message,
            )
        )

    @property
    def has_errors(self):
        return any(message.level == "error" for message in self.messages)

    @property
    def has_warnings(self):
        return any(message.level == "warning" for message in self.messages)

    def error_codes(self):
        return [message.code for message in self.messages if message.level == "error"]

    def to_dict(self):
        return {
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "messages": [
                {
                    "validator": message.validator,
                    "level": message.level,
                    "code": message.code,
                    "message": message.message,
                }
                for message in self.messages
            ],
        }


@dataclass
class ValidatorContext:
    profile: object
    task_spec: object
    prompt: str = ""
    source_context: object = None
    planner_semantic_checks: object = None


class BaseValidator:
    name = "base"

    def validate(self, code, context):
        raise NotImplementedError
