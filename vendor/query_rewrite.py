from vendor.base_agent import BaseAgent
import logging


class AgentQR(BaseAgent):

    def _set_messages(self, role, content):
        super()._set_messages(role, content)
        self.payload[role] = content
        logging.info(f"Added message to payload - {content}")

        logging.info(f"{content}")

    def _build_context(self, historical_payload, k):
        filtered_messages = [
            msg for msg in historical_payload.get("messages", [])
            if msg.get("role") in {"user"}
        ]

        k = min(k, len(filtered_messages))

        return filtered_messages[-k:]

    def get_response_AgentQR(self, last_msg, historical_payload, k=3):

        historical = self._build_context(historical_payload, k)
        self._set_messages("prompt",
                           f"""Você é um assistente útil que reescreve perguntas. Gere uma nova pergunta de pesquisa relacionada às seguintes entradas:
                                        Entrada anterior: {historical}
                                        Entrada atual: {last_msg}
                                        Nova pergunta:
                                    """)
        response_data = self._send_request()
        answer = response_data.get("response", "No content found.")

        return answer