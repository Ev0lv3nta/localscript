def verify_code(code):
    errors = []
    stripped = (code or "").strip()
    if not stripped:
        errors.append("empty_code")
        return errors

    if "$." in stripped or "$[" in stripped:
        errors.append("jsonpath_forbidden")

    if "```" in stripped:
        errors.append("markdown_fence_forbidden")

    if "ctx.body" in stripped:
        errors.append("ctx_body_forbidden")

    if "workflow.variables" in stripped:
        errors.append("workflow_variables_forbidden")

    return errors
