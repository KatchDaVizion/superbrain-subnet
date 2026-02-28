# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Protocol — RAG Knowledge Query + Knowledge Sync Synapses

import typing
import pydantic
import bittensor as bt


class RAGSynapse(bt.Synapse):
    """
    Protocol for RAG knowledge queries between validators and miners.
    Validators send query + context_chunks + chunk_sources.
    Miners return response + citations + confidence_score.
    """
    # Sent by Validator
    query: str = ""
    context_chunks: typing.List[str] = pydantic.Field(default_factory=list)
    chunk_sources: typing.List[str] = pydantic.Field(default_factory=list)

    # Filled by Miner
    response: typing.Optional[str] = None
    citations: typing.Optional[typing.List[int]] = None
    confidence_score: typing.Optional[float] = None

    def deserialize(self) -> dict:
        return {
            "response": self.response,
            "citations": self.citations or [],
            "confidence_score": self.confidence_score,
        }


class KnowledgeSyncSynapse(bt.Synapse):
    """
    Protocol for knowledge chunk sync between validators and miners.
    Validators send hashes of chunks they already have.
    Miners return new chunks as a compressed SyncBatch (base64-encoded).
    """
    # Sent by Validator
    known_hashes: typing.List[str] = pydantic.Field(default_factory=list)
    max_chunks: int = 50
    node_id: str = ""

    # Filled by Miner
    batch_data: typing.Optional[str] = None
    chunk_count: typing.Optional[int] = None
    batch_id: typing.Optional[str] = None

    def deserialize(self) -> dict:
        return {
            "batch_data": self.batch_data,
            "chunk_count": self.chunk_count,
            "batch_id": self.batch_id,
        }
