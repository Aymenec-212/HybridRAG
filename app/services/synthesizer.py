from typing import List
import pandas as pd
from pydantic import BaseModel, Field
from app.services.llm_factory import LLMFactory


class SynthesizedResponse(BaseModel):
    thought_process: List[str] = Field(
        description="Liste des réflexions que l'assistant IA a eues en synthétisant la réponse"
    )
    answer: str = Field(description="La réponse synthétisée à la question de l'utilisateur")
    enough_context: bool = Field(
        description="Indique si l'assistant dispose de suffisamment de contexte pour répondre à la question"
    )


class Synthesizer:
    SYSTEM_PROMPT = """
    # Rôle et Objectif
    Vous êtes un assistant IA spécialisé dans l'analyse de documents réglementaires et juridiques marocains. 
    Votre tâche est de synthétiser une réponse cohérente, factuelle et précise en vous basant uniquement sur la question de l'utilisateur et les extraits de textes pertinents fournis en contexte.

    # Directives Strictes:
    1.  Exactitude et Précision : Fournir une réponse claire et précise à la question posée. La rigueur est primordiale.
    2.  Exclusivité du Contexte : Baser votre réponse **exclusivement** sur les informations contenues dans le contexte fourni ("informations extraites"). N'utilisez JAMAIS de connaissances externes ou d'informations préalables.
    3.  Ne Pas Fabuler : Ne jamais inventer, extrapoler ou inférer des informations qui ne sont pas explicitement présentes dans le contexte. Si une information n'est pas dans le texte, elle n'existe pas pour vous.
    4.  Gestion de l'Incertitude : Le contexte est extrait via une recherche sémantique et peut être partiel ou non pertinent. Si les informations fournies sont insuffisantes ou ne permettent pas de répondre à la question, déclarez-le clairement. Il est préférable de dire "L'information n'est pas disponible dans les documents fournis" plutôt que de donner une réponse incorrecte.
    5.  Citation des Sources : Si le contexte le permet (par exemple, en mentionnant un numéro d'article), citez la source de votre information pour garantir la traçabilité.
    6.  Ton Professionnel : Maintenir un ton formel, neutre et informatif, adapté à un contexte réglementaire. Évitez toute familiarité ou opinion.

    Examinez la question de l'utilisateur :
    """

    @staticmethod
    def generate_response(question: str, context: pd.DataFrame) -> SynthesizedResponse:
        """Generates a synthesized response based on the question and context.

        Args:
            question: The user's question.
            context: The relevant context retrieved from the knowledge base.

        Returns:
            A SynthesizedResponse containing thought process and answer.
        """
        context_str = Synthesizer.dataframe_to_json(
            context, columns_to_keep=["content"]
        )

        messages = [
            {"role": "system", "content": Synthesizer.SYSTEM_PROMPT},
            {"role": "user", "content": f"# Question de l'utilisateur :\n{question}"},
            {
                "role": "assistant",
                "content": f"# Informations extraites :\n{context_str}",
            },
        ]

        llm = LLMFactory("openai")
        return llm.create_completion(
            response_model=SynthesizedResponse,
            messages=messages,
        )

    @staticmethod
    def dataframe_to_json(
            context: pd.DataFrame,
            columns_to_keep: List[str],
    ) -> str:
        """
        Convert the context DataFrame to a JSON string.

        Args:
            context (pd.DataFrame): The context DataFrame.
            columns_to_keep (List[str]): The columns to include in the output.

        Returns:
            str: A JSON string representation of the selected columns.
        """
        return context[columns_to_keep].to_json(orient="records", indent=2)