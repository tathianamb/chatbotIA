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
                           f"Você é um assistente útil que reescreve perguntas. Gere uma nova pergunta de "
                           f"pesquisa relacionada às seguintes entradas:\nEntrada anterior: {historical}"
                           f"\nEntrada atual: {last_msg}\nNova pergunta:")
        response_data = self._send_request()
        answer = response_data.get("response") or logging.warning(
            f"Não foi possível reescrever a última mensagem adicionando contexto do histórico.") or last_msg
        return answer