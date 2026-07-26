class BackendError(RuntimeError):
    """A safe-to-publish backend failure."""

    code = "backend_error"
    default_message = "Backend request failed."

    def __init__(self, message=None, *, reason=None, status_code=None):
        self.public_message = message or self.default_message
        self.reason = reason or self.code
        self.status_code = status_code
        super().__init__(self.public_message)


class BackendUnavailable(BackendError):
    code = "backend_unavailable"
    default_message = "Backend is unavailable."


class BackendTimeout(BackendError):
    code = "backend_timeout"
    default_message = "Backend request timed out."


class BackendProtocol(BackendError):
    code = "backend_protocol_error"
    default_message = "Backend returned an invalid response."


class BackendModel(BackendError):
    code = "backend_model_error"
    default_message = "Backend model is unavailable or inconsistent."


# Descriptive aliases for callers that prefer the conventional ``Error`` suffix.
BackendProtocolError = BackendProtocol
BackendModelError = BackendModel
