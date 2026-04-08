from typing import Any, Dict


class Client:
    def __init__(self):
        pass

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method will be overridden by inference.py during evaluation.
        """
        return {}