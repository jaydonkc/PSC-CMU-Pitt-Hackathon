"""GraphReader Agent
-------------------
A simplified multi-step reasoning agent that interacts with a Neo4j knowledge
graph to answer biomedical queries. The implementation follows the outline of
GraphRAG as described in the project documentation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from neo4j import GraphDatabase
from pydantic import BaseModel

try:
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None  # type: ignore
    PromptTemplate = None  # type: ignore

try:  # Optional local inference backend
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover - vllm is optional
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OverallState(BaseModel):
    question: str
    rational_plan: str | None = None
    notebook: str = ""
    previous_actions: List[str] = []
    check_atomic_facts_queue: List[str] = []
    check_chunks_queue: List[str] = []
    neighbor_check_queue: List[str] = []
    chosen_action: str | None = None
    answer: str | None = None
    step: str = "rational_plan"


@dataclass
class GraphReaderAgent:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    llm_model: str = "gpt-4"
    llm_backend: str = "openai"  # "openai" or "vllm"
    vllm_model: str | None = None
    _driver: object = field(init=False, repr=False)
    _vllm_params: object = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self._driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )
        if self.llm_backend == "openai":
            if ChatOpenAI is None:
                raise ImportError(
                    "LangChain with ChatOpenAI is required for the agent"
                )
            self.llm = ChatOpenAI(model=self.llm_model, temperature=0)
        else:
            if LLM is None:
                raise ImportError(
                    "vllm is required for local inference. Install vllm and set llm_backend='vllm'"
                )
            model_name = self.vllm_model or self.llm_model
            self.llm = LLM(model=model_name)
            self._vllm_params = SamplingParams(temperature=0.0)

    def close(self):
        self._driver.close()

    def vector_search_nodes(self, query: str) -> List[str]:
        """Dummy vector search over KeyElement nodes by simple substring match."""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (k:KeyElement) WHERE toLower(k.text) CONTAINS toLower($q) RETURN k.text LIMIT 10",
                q=query,
            )
            return [record["k.text"] for record in result]

    def run_llm(self, prompt: str) -> str:
        if self.llm_backend == "openai":
            return self.llm.predict(prompt)
        outputs = self.llm.generate([prompt], self._vllm_params)
        return outputs[0].outputs[0].text.strip()

    def run(self, question: str) -> str:
        state = OverallState(question=question)
        while True:
            logger.info("Step: %s", state.step)
            if state.step == "rational_plan":
                plan_prompt = f"You are an expert biomedical researcher. Formulate a plan to answer: {state.question}"
                state.rational_plan = self.run_llm(plan_prompt)
                state.step = "initial_node_selection"
            elif state.step == "initial_node_selection":
                candidates = self.vector_search_nodes(state.question)
                state.check_atomic_facts_queue = candidates[:5]
                state.step = "answer_reasoning"
            elif state.step == "answer_reasoning":
                evidence = "; ".join(state.check_atomic_facts_queue)
                answer_prompt = f"Question: {state.question}\nEvidence: {evidence}\nProvide a concise answer."
                state.answer = self.run_llm(answer_prompt)
                break
            else:
                break
        return state.answer or ""
