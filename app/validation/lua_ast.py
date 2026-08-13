"""Static Lua policy built on the concrete syntax tree, not source text matching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import tree_sitter_lua
from tree_sitter import Language, Node, Parser

ALLOWED_GLOBALS: Final[frozenset[str]] = frozenset(
    {
        "wf",
        "_utils",
        "math",
        "string",
        "table",
        "tonumber",
        "tostring",
        "type",
        "pairs",
        "ipairs",
        "select",
        "assert",
        "error",
        "pcall",
        "xpcall",
        "utf8",
    }
)

FORBIDDEN_GLOBALS: Final[dict[str, tuple[str, str]]] = {
    "os": (
        "dangerous_stdlib_os_forbidden",
        "Access to the `os` namespace is forbidden.",
    ),
    "io": (
        "dangerous_stdlib_io_forbidden",
        "Access to the `io` namespace is forbidden.",
    ),
    "package": (
        "dangerous_stdlib_package_forbidden",
        "Access to the `package` namespace is forbidden.",
    ),
    "debug": (
        "dangerous_stdlib_debug_forbidden",
        "Access to the `debug` namespace is forbidden.",
    ),
    "require": ("dangerous_stdlib_require_forbidden", "`require` is forbidden."),
    "load": (
        "dangerous_stdlib_load_forbidden",
        "Dynamic code loading through `load` is forbidden.",
    ),
    "loadfile": ("dangerous_stdlib_loadfile_forbidden", "`loadfile` is forbidden."),
    "dofile": ("dangerous_stdlib_dofile_forbidden", "`dofile` is forbidden."),
    "collectgarbage": (
        "dangerous_stdlib_collectgarbage_forbidden",
        "`collectgarbage` is forbidden.",
    ),
}


@dataclass(frozen=True, slots=True)
class LuaPolicyFinding:
    code: str
    message: str
    line: int
    column: int
    chunk_index: int = 1


@dataclass(frozen=True, slots=True)
class LuaPolicyResult:
    findings: tuple[LuaPolicyFinding, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.findings


class _Scope:
    def __init__(self, parent: _Scope | None = None) -> None:
        self.parent = parent
        self.locals: dict[str, bool] = {}

    def declare(self, name: str, *, aliases_wf: bool = False) -> None:
        self.locals[name] = aliases_wf

    def owner(self, name: str) -> _Scope | None:
        if name in self.locals:
            return self
        if self.parent is None:
            return None
        return self.parent.owner(name)

    def aliases_wf(self, name: str) -> bool:
        owner = self.owner(name)
        return owner is not None and owner.locals[name]

    def assign_alias(self, name: str, aliases_wf: bool) -> None:
        owner = self.owner(name)
        if owner is not None:
            owner.locals[name] = aliases_wf


@lru_cache(maxsize=1)
def _parser() -> Parser:
    language = Language(tree_sitter_lua.language())
    return Parser(language)


def extract_lua_chunks(code: str, output_style: str) -> tuple[str, ...]:
    if output_style != "json_envelope":
        if not isinstance(code, str) or not code.strip():
            return ()
        return (_strip_lua_wrapper(code),)

    if not isinstance(code, str):
        return ()
    try:
        payload = json.loads(code)
    except (ValueError, TypeError, RecursionError):
        return ()
    if not isinstance(payload, dict) or not payload:
        return ()

    chunks: list[str] = []
    for value in payload.values():
        if not isinstance(value, str) or not value.startswith("lua{") or not value.endswith("}lua"):
            return ()
        chunks.append(_strip_lua_wrapper(value))
    return tuple(chunks)


def analyze_lua_output(code: str, output_style: str) -> LuaPolicyResult:
    findings: list[LuaPolicyFinding] = []
    for chunk_index, chunk in enumerate(extract_lua_chunks(code, output_style), start=1):
        result = analyze_lua_chunk(chunk, chunk_index=chunk_index)
        findings.extend(result.findings)
    return LuaPolicyResult(tuple(findings))


def analyze_lua_chunk(chunk: str, *, chunk_index: int = 1) -> LuaPolicyResult:
    source = chunk.encode("utf-8")
    tree = _parser().parse(source)
    analyzer = _Analyzer(source=source, chunk_index=chunk_index)
    if tree.root_node.has_error:
        analyzer.report_syntax_errors(tree.root_node)
    else:
        analyzer.visit_sequence(tree.root_node, _Scope())
    return LuaPolicyResult(tuple(analyzer.findings))


def _strip_lua_wrapper(code: str) -> str:
    stripped = code.strip()
    if stripped.startswith("lua{") and stripped.endswith("}lua"):
        return stripped[4:-4]
    return stripped


class _Analyzer:
    def __init__(self, *, source: bytes, chunk_index: int) -> None:
        self.source = source
        self.chunk_index = chunk_index
        self.findings: list[LuaPolicyFinding] = []

    def text(self, node: Node) -> str:
        return self.source[node.start_byte : node.end_byte].decode("utf-8")

    def add(self, node: Node, code: str, message: str) -> None:
        point = node.start_point
        finding = LuaPolicyFinding(
            code=code,
            message=message,
            line=point.row + 1,
            column=point.column + 1,
            chunk_index=self.chunk_index,
        )
        if finding not in self.findings:
            self.findings.append(finding)

    def report_syntax_errors(self, node: Node) -> None:
        if node.is_error or node.is_missing:
            self.add(
                node,
                "lua_ast_syntax_error",
                "Lua chunk contains an invalid or incomplete syntax node.",
            )
        for child in node.children:
            if child.has_error or child.is_error or child.is_missing:
                self.report_syntax_errors(child)

    def visit_sequence(self, node: Node, scope: _Scope) -> None:
        for child in node.named_children:
            self.visit(child, scope)

    def visit(self, node: Node, scope: _Scope) -> None:
        handler = getattr(self, f"visit_{node.type}", None)
        if handler is not None:
            handler(node, scope)
            return
        if node.type == "identifier":
            self.check_reference(node, scope)
            return
        if node.type == "block":
            self.visit_sequence(node, _Scope(scope))
            return
        for child in node.named_children:
            self.visit(child, scope)

    def visit_variable_declaration(self, node: Node, scope: _Scope) -> None:
        assignment = next(
            (child for child in node.named_children if child.type == "assignment_statement"),
            None,
        )
        if assignment is None:
            variable_list = next(
                (child for child in node.named_children if child.type == "variable_list"),
                None,
            )
            if variable_list is not None:
                for identifier in self.declared_identifiers(variable_list):
                    self.declare_local(identifier, scope)
            return
        variable_list = next(
            (child for child in assignment.named_children if child.type == "variable_list"),
            None,
        )
        expression_list = next(
            (child for child in assignment.named_children if child.type == "expression_list"),
            None,
        )
        expressions = list(expression_list.named_children) if expression_list is not None else []
        if expression_list is not None:
            self.visit(expression_list, scope)
        if variable_list is None:
            return
        for index, identifier in enumerate(self.declared_identifiers(variable_list)):
            aliases_wf = index < len(expressions) and self.is_wf_value(expressions[index], scope)
            self.declare_local(identifier, scope, aliases_wf=aliases_wf)

    def visit_assignment_statement(self, node: Node, scope: _Scope) -> None:
        variable_list = next(
            (child for child in node.named_children if child.type == "variable_list"),
            None,
        )
        expression_list = next(
            (child for child in node.named_children if child.type == "expression_list"),
            None,
        )
        expressions = list(expression_list.named_children) if expression_list is not None else []
        expression_aliases = [self.is_wf_value(expression, scope) for expression in expressions]
        if expression_list is not None:
            self.visit(expression_list, scope)
        if variable_list is None:
            return
        for index, target in enumerate(variable_list.named_children):
            self.visit_assignment_target(target, scope)
            if target.type == "identifier" and scope.owner(self.text(target)) is not None:
                aliases_wf = index < len(expression_aliases) and expression_aliases[index]
                scope.assign_alias(self.text(target), aliases_wf)

    def visit_assignment_target(self, target: Node, scope: _Scope) -> None:
        if target.type == "identifier":
            name = self.text(target)
            if scope.owner(name) is None:
                self.add(
                    target,
                    f"lua_global_assignment_forbidden::{name}",
                    f"Assignment to global identifier `{name}` is forbidden.",
                )
            return

        root = self.root_identifier(target)
        if root is not None:
            root_name = self.text(root)
            if root_name == "wf" or scope.aliases_wf(root_name):
                self.add(
                    target,
                    "lua_wf_mutation_forbidden",
                    "Workflow input `wf` is read-only.",
                )
            elif scope.owner(root_name) is None:
                self.add(
                    target,
                    f"lua_global_assignment_forbidden::{root_name}",
                    f"Mutation through global identifier `{root_name}` is forbidden.",
                )
        self.visit(target, scope)

    def visit_function_declaration(self, node: Node, scope: _Scope) -> None:
        name = node.child_by_field_name("name")
        is_local = any(child.type == "local" for child in node.children)
        if name is not None:
            if is_local and name.type == "identifier":
                self.declare_local(name, scope)
            elif not is_local:
                self.visit_assignment_target(name, scope)
        self.visit_function_body(
            node, scope, add_self=name is not None and name.type == "method_index_expression"
        )

    def visit_function_definition(self, node: Node, scope: _Scope) -> None:
        self.visit_function_body(node, scope, add_self=False)

    def visit_function_call(self, node: Node, scope: _Scope) -> None:
        name = node.child_by_field_name("name")
        arguments = node.child_by_field_name("arguments")
        if name is not None and arguments is not None and self.is_table_mutator(name, scope):
            first_argument = next(iter(arguments.named_children), None)
            if first_argument is not None and self.is_wf_value(first_argument, scope):
                self.add(
                    first_argument,
                    "lua_wf_mutation_forbidden",
                    "Workflow input `wf` is read-only.",
                )
        if name is not None:
            self.visit(name, scope)
        if arguments is not None:
            self.visit(arguments, scope)

    def visit_function_body(self, node: Node, scope: _Scope, *, add_self: bool) -> None:
        function_scope = _Scope(scope)
        if add_self:
            function_scope.declare("self")
        parameters = node.child_by_field_name("parameters")
        if parameters is not None:
            for child in parameters.named_children:
                if child.type == "identifier":
                    self.declare_local(child, function_scope)
        body = node.child_by_field_name("body")
        if body is not None:
            self.visit_sequence(body, function_scope)

    def visit_for_statement(self, node: Node, scope: _Scope) -> None:
        clause = node.child_by_field_name("clause")
        body = node.child_by_field_name("body")
        loop_scope = _Scope(scope)
        if clause is not None and clause.type == "for_numeric_clause":
            for field_name in ("start", "end", "step"):
                expression = clause.child_by_field_name(field_name)
                if expression is not None:
                    self.visit(expression, scope)
            identifier = clause.child_by_field_name("name")
            if identifier is not None:
                self.declare_local(identifier, loop_scope)
        elif clause is not None:
            variable_list = next(
                (child for child in clause.named_children if child.type == "variable_list"),
                None,
            )
            expression_list = next(
                (child for child in clause.named_children if child.type == "expression_list"),
                None,
            )
            if expression_list is not None:
                self.visit(expression_list, scope)
            if variable_list is not None:
                for identifier in self.declared_identifiers(variable_list):
                    self.declare_local(identifier, loop_scope)
        if body is not None:
            self.visit_sequence(body, loop_scope)

    def visit_repeat_statement(self, node: Node, scope: _Scope) -> None:
        repeat_scope = _Scope(scope)
        body = node.child_by_field_name("body")
        if body is not None:
            self.visit_sequence(body, repeat_scope)
        condition = node.child_by_field_name("condition")
        if condition is not None:
            self.visit(condition, repeat_scope)

    def visit_dot_index_expression(self, node: Node, scope: _Scope) -> None:
        table = node.child_by_field_name("table")
        if table is not None:
            self.visit(table, scope)

    def visit_method_index_expression(self, node: Node, scope: _Scope) -> None:
        table = node.child_by_field_name("table")
        if table is not None:
            self.visit(table, scope)

    def visit_bracket_index_expression(self, node: Node, scope: _Scope) -> None:
        table = node.child_by_field_name("table")
        field = node.child_by_field_name("field")
        if table is not None:
            self.visit(table, scope)
        if field is not None:
            self.visit(field, scope)

    def visit_field(self, node: Node, scope: _Scope) -> None:
        name = node.child_by_field_name("name")
        value = node.child_by_field_name("value")
        if name is not None and name.type != "identifier":
            self.visit(name, scope)
        if value is not None:
            self.visit(value, scope)

    def visit_parameters(self, node: Node, scope: _Scope) -> None:
        return

    def visit_variable_list(self, node: Node, scope: _Scope) -> None:
        return

    def visit_label_statement(self, node: Node, scope: _Scope) -> None:
        self.add(node, "lua_construct_forbidden::label", "Lua labels are not supported.")

    def visit_goto_statement(self, node: Node, scope: _Scope) -> None:
        self.add(node, "lua_construct_forbidden::goto", "Lua goto is not supported.")

    def check_reference(self, node: Node, scope: _Scope) -> None:
        name = self.text(node)
        if scope.owner(name) is not None or name in ALLOWED_GLOBALS:
            return
        code, message = FORBIDDEN_GLOBALS.get(
            name,
            (
                f"lua_global_not_allowed::{name}",
                f"Global identifier `{name}` is not available in the restricted runtime.",
            ),
        )
        self.add(node, code, message)

    def declare_local(self, node: Node, scope: _Scope, *, aliases_wf: bool = False) -> None:
        name = self.text(node)
        if name == "wf":
            self.add(
                node,
                "lua_reserved_identifier_shadowed::wf",
                "Local declaration cannot shadow the reserved runtime identifier `wf`.",
            )
        scope.declare(name, aliases_wf=aliases_wf)

    @staticmethod
    def declared_identifiers(variable_list: Node) -> tuple[Node, ...]:
        return tuple(child for child in variable_list.named_children if child.type == "identifier")

    def root_identifier(self, node: Node) -> Node | None:
        current = node
        while current.type in {
            "dot_index_expression",
            "bracket_index_expression",
            "method_index_expression",
        }:
            table = current.child_by_field_name("table")
            if table is None:
                return None
            current = table
        return current if current.type == "identifier" else None

    def is_wf_value(self, node: Node, scope: _Scope) -> bool:
        root = self.root_identifier(node)
        if root is not None:
            name = self.text(root)
            return name == "wf" or scope.aliases_wf(name)
        if node.type == "parenthesized_expression" and len(node.named_children) == 1:
            return self.is_wf_value(node.named_children[0], scope)
        return False

    def is_table_mutator(self, node: Node, scope: _Scope) -> bool:
        if node.type != "dot_index_expression":
            return False
        table = node.child_by_field_name("table")
        field = node.child_by_field_name("field")
        if table is None or field is None or self.text(field) not in {"insert", "remove", "sort"}:
            return False
        root = self.root_identifier(table)
        return root is not None and self.text(root) == "table" and scope.owner("table") is None
