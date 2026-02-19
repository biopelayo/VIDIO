import json


class VidioException(Exception):
    def __init__(self, code, operation, message):
        self.code = code
        self.operation = operation
        self.message = message

    def __str__(self):
        return json.dumps({
            'code': self.code,
            'operation': self.operation,
            'message': self.message,
        })


class DBException(VidioException):
    def __init__(self, code, operation, message):
        super().__init__(code, operation, message)


class WSException(VidioException):
    def __init__(self, code, operation, message):
        super().__init__(code, operation, message)
