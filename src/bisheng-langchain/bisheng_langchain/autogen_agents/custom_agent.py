import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union
from autogen import ConversableAgent, Agent

logger = logging.getLogger(__name__)


class AutoGenCustomAgent(ConversableAgent):
    """Custom agent that can use langchain agent and chain."""

    def __init__(
        self,
        name: str,
        system_message: str,
        func: Callable[..., str],
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            human_input_mode='NEVER',
            **kwargs,
        )
        self.func = func
        self.register_reply(Agent, AutoGenCustomAgent.generate_custom_reply)

    def generate_custom_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Union[str, Dict, None]:
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]

        if "content" in message:
            query = message["content"]
            reply = self.func(query)
            if isinstance(reply, dict):
                reply = reply.values()[0]
            return True, reply

        return False, None