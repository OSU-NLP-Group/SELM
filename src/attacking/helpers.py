class InsistedError(Exception):
    condition: object
    message: object

    def __init__(self, condition: object, message: object):
        self.condition = condition
        self.message = message

    def __str__(self):
        return f"Internal consistency error: {self.message}"


def insist(condition: object, message: object) -> None:
    if not condition:
        raise InsistedError(condition, message)
