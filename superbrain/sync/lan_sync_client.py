"""
SuperBrain LAN Sync Client
===========================

Production-quality asyncio WebSocket client for the SuperBrain decentralized
AI knowledge network. Connects to LANSyncServer peers and synchronizes
knowledge chunks over the local network.

Protocol: JSON messages over WebSocket
Discovery: Zeroconf mDNS (_superbrain._tcp.local.)
Verification: Ed25519 signatures on every chunk
Storage: Qdrant vector DB with Ollama embeddings
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import websockets
import websockets.exceptions
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from zeroconf import ServiceBrowser, ServiceStateChange, Zeroconf

# ---------------------------------------------------------------------------
# Try to reuse ChunkStore from the server module; fall back to a minimal
# local implementation so the client is self-contained.
# ---------------------------------------------------------------------------
try:
    from superbrain.sync.lan_sync_server import ChunkStore  # type: ignore[import-untyped]
except ImportError:

    @dataclass
    class ChunkStore:
        """Minimal in-memory chunk store used when lan_sync_server is unavailable."""

        chunks: dict[str, dict[str, Any]] = field(default_factory=dict)

        def add(self, chunk: dict[str, Any]) -> None:
            """Add a chunk keyed by its hash."""
            self.chunks[chunk["hash"]] = chunk

        def get(self, chunk_hash: str) -> Optional[dict[str, Any]]:
            """Return a chunk by hash, or ``None``."""
            return self.chunks.get(chunk_hash)

        def hashes(self) -> list[str]:
            """Return all stored chunk hashes."""
            return list(self.chunks.keys())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROTOCOL_VERSION = "1.0"
MDNS_SERVICE_TYPE = "_superbrain._tcp.local."
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
QDRANT_COLLECTION = "sb_docs_v1_ollama"
EMBEDDING_MODEL = "nomic-embed-text"

CONNECT_TIMEOUT_S = 10.0
HANDSHAKE_TIMEOUT_S = 10.0
SYNC_TIMEOUT_S = 60.0
RECV_TIMEOUT_S = 30.0
RECONNECT_DELAY_BASE_S = 1.0
RECONNECT_MAX_RETRIES = 5
MDNS_BROWSE_TIMEOUT_S = 5.0

logger = logging.getLogger("superbrain.sync.lan_client")


# ---------------------------------------------------------------------------
# mDNS peer collector
# ---------------------------------------------------------------------------
class _PeerCollector:
    """Zeroconf listener that accumulates discovered SuperBrain peers."""

    def __init__(self) -> None:
        self.peers: list[tuple[str, int]] = []

    def on_service_state_change(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        if state_change is not ServiceStateChange.Added:
            return
        info = zeroconf.get_service_info(service_type, name)
        if info is None:
            return

        addresses = info.parsed_addresses()
        port = info.port
        for addr in addresses:
            if port and addr:
                peer = (addr, port)
                if peer not in self.peers:
                    self.peers.append(peer)
                    logger.info("mDNS: discovered peer %s:%d", addr, port)


# ---------------------------------------------------------------------------
# LANSyncClient
# ---------------------------------------------------------------------------
class LANSyncClient:
    """Asyncio WebSocket client that synchronises knowledge chunks with
    SuperBrain LAN peers.

    Parameters
    ----------
    node_id:
        SHA-256 hex digest of this node's Ed25519 public key.
    private_key:
        This node's Ed25519 private key (``Ed25519PrivateKey`` instance).
        Used only to derive the public key for identity; chunk *verification*
        uses the contributor's public key embedded in each chunk.
    chunk_store:
        Local ``ChunkStore`` instance for caching chunks in memory.
    qdrant_url:
        Base URL of the Qdrant vector database.
    ollama_url:
        Base URL of the Ollama inference server.
    """

    def __init__(
        self,
        node_id: str,
        private_key: Any,
        chunk_store: ChunkStore,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
    ) -> None:
        self.node_id: str = node_id
        self.private_key = private_key
        self.chunk_store: ChunkStore = chunk_store
        self.qdrant_url: str = qdrant_url.rstrip("/")
        self.ollama_url: str = ollama_url.rstrip("/")

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected: bool = False
        self._http: httpx.AsyncClient = httpx.AsyncClient(timeout=30.0)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self, host: str, port: int) -> None:
        """Open a WebSocket connection to *host*:*port*.

        Retries with exponential back-off up to ``RECONNECT_MAX_RETRIES``
        times before raising.
        """
        uri = f"ws://{host}:{port}"
        delay = RECONNECT_DELAY_BASE_S

        for attempt in range(1, RECONNECT_MAX_RETRIES + 1):
            try:
                logger.info(
                    "Connecting to %s (attempt %d/%d) ...",
                    uri,
                    attempt,
                    RECONNECT_MAX_RETRIES,
                )
                self._ws = await asyncio.wait_for(
                    websockets.connect(uri),  # type: ignore[arg-type]
                    timeout=CONNECT_TIMEOUT_S,
                )
                self._connected = True
                logger.info("Connected to %s", uri)
                return
            except (
                OSError,
                websockets.exceptions.WebSocketException,
                asyncio.TimeoutError,
            ) as exc:
                logger.warning(
                    "Connection attempt %d failed: %s", attempt, exc
                )
                if attempt < RECONNECT_MAX_RETRIES:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)

        raise ConnectionError(
            f"Failed to connect to {uri} after {RECONNECT_MAX_RETRIES} attempts"
        )

    async def discover_and_connect(self) -> list[tuple[str, int]]:
        """Use Zeroconf mDNS to find SuperBrain peers on the LAN.

        Returns the list of ``(host, port)`` tuples discovered.  Does **not**
        open persistent connections — call :meth:`sync_with_peer` or
        :meth:`auto_sync` for that.
        """
        logger.info("Starting mDNS discovery for %s ...", MDNS_SERVICE_TYPE)
        collector = _PeerCollector()
        zc = Zeroconf()
        try:
            ServiceBrowser(zc, MDNS_SERVICE_TYPE, handlers=[collector.on_service_state_change])
            await asyncio.sleep(MDNS_BROWSE_TIMEOUT_S)
        finally:
            zc.close()

        if not collector.peers:
            logger.warning("mDNS discovery found no peers")
        else:
            logger.info("mDNS discovery found %d peer(s)", len(collector.peers))
        return collector.peers

    async def disconnect(self) -> None:
        """Gracefully close the WebSocket connection."""
        if self._ws is not None:
            try:
                await self._ws.close()
                logger.info("Disconnected from peer")
            except websockets.exceptions.WebSocketException as exc:
                logger.warning("Error during disconnect: %s", exc)
            finally:
                self._ws = None
                self._connected = False

    # ------------------------------------------------------------------
    # Protocol messages
    # ------------------------------------------------------------------

    async def send_handshake(self) -> None:
        """Send a ``handshake`` message to the connected peer."""
        if self._ws is None:
            raise RuntimeError("Not connected — call connect() first")

        manifest = await self.get_local_manifest()
        payload: dict[str, Any] = {
            "type": "handshake",
            "node_id": self.node_id,
            "version": PROTOCOL_VERSION,
            "chunk_count": len(manifest),
            "timestamp": int(time.time()),
        }
        raw = json.dumps(payload)
        await self._ws.send(raw)
        logger.info(
            "Sent handshake (node=%s, chunks=%d)", self.node_id[:12], len(manifest)
        )

        # Wait for the peer's handshake reply
        try:
            resp_raw = await asyncio.wait_for(
                self._ws.recv(), timeout=HANDSHAKE_TIMEOUT_S
            )
            resp = json.loads(resp_raw)
            if resp.get("type") == "handshake":
                logger.info(
                    "Received handshake from peer %s (v%s, %d chunks)",
                    resp.get("node_id", "?")[:12],
                    resp.get("version", "?"),
                    resp.get("chunk_count", 0),
                )
            else:
                logger.warning("Unexpected message type after handshake: %s", resp.get("type"))
        except asyncio.TimeoutError:
            logger.warning("Handshake response timed out")

    async def send_sync_request(self) -> None:
        """Send a ``sync_request`` containing the local manifest of chunk hashes."""
        if self._ws is None:
            raise RuntimeError("Not connected — call connect() first")

        manifest = await self.get_local_manifest()
        payload: dict[str, Any] = {
            "type": "sync_request",
            "manifest": manifest,
            "layer": "lan",
        }
        raw = json.dumps(payload)
        await self._ws.send(raw)
        logger.info("Sent sync_request with %d known hashes", len(manifest))

    async def handle_sync_response(self, message: dict[str, Any]) -> int:
        """Process a ``sync_response`` message from the peer.

        Each chunk is verified and, if valid, ingested into Qdrant.

        Parameters
        ----------
        message:
            Parsed JSON dict of the ``sync_response``.

        Returns
        -------
        int
            Number of chunks successfully ingested.
        """
        chunks: list[dict[str, Any]] = message.get("chunks", [])
        sent = message.get("sent", len(chunks))
        dedup = message.get("deduplicated", 0)
        logger.info(
            "Received sync_response: %d chunk(s), %d deduplicated",
            sent,
            dedup,
        )

        ingested = 0
        for chunk in chunks:
            chunk_hash = chunk.get("hash", "?")
            try:
                if not self.verify_chunk_signature(chunk):
                    logger.warning(
                        "Chunk %s failed signature verification — skipping",
                        chunk_hash[:16],
                    )
                    continue

                if await self.ingest_chunk(chunk):
                    ingested += 1
            except Exception:
                logger.exception("Error processing chunk %s", chunk_hash[:16])

        logger.info("Ingested %d / %d chunks", ingested, len(chunks))
        return ingested

    # ------------------------------------------------------------------
    # Chunk verification
    # ------------------------------------------------------------------

    @staticmethod
    def verify_chunk_signature(chunk: dict[str, Any]) -> bool:
        """Verify the Ed25519 signature of *chunk*.

        The signed message is the concatenation of:
            chunk["hash"] + chunk["contributor_id"] + str(chunk["timestamp"]) + chunk["visibility"]
        all encoded as UTF-8 bytes.

        Returns ``True`` if the signature is valid, ``False`` otherwise.
        """
        try:
            chunk_hash: str = chunk["hash"]
            contributor_id: str = chunk["contributor_id"]
            timestamp: str = str(chunk["timestamp"])
            visibility: str = chunk["visibility"]
            signature_hex: str = chunk["signature"]

            message_bytes = (
                chunk_hash.encode("utf-8")
                + contributor_id.encode("utf-8")
                + timestamp.encode("utf-8")
                + visibility.encode("utf-8")
            )
            signature_bytes = bytes.fromhex(signature_hex)

            # Derive the public key from contributor_id.
            # The contributor_id is a SHA-256 of the public key, so we cannot
            # reconstruct the key from it alone.  In production the public key
            # should be provided in the chunk or resolved from a registry.
            # Here we expect the chunk to carry "public_key" (hex-encoded raw
            # 32-byte Ed25519 public key) for verification.
            pub_key_hex: Optional[str] = chunk.get("public_key")
            if pub_key_hex is None:
                logger.warning(
                    "Chunk %s has no public_key field — cannot verify",
                    chunk_hash[:16],
                )
                return False

            pub_key = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(pub_key_hex)
            )
            pub_key.verify(signature_bytes, message_bytes)

            # Confirm contributor_id matches public key
            expected_id = hashlib.sha256(bytes.fromhex(pub_key_hex)).hexdigest()
            if expected_id != contributor_id:
                logger.warning(
                    "Chunk %s contributor_id mismatch (expected %s, got %s)",
                    chunk_hash[:16],
                    expected_id[:16],
                    contributor_id[:16],
                )
                return False

            return True

        except Exception as exc:
            logger.warning("Signature verification error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Qdrant / Ollama integration
    # ------------------------------------------------------------------

    async def ingest_chunk(self, chunk: dict[str, Any]) -> bool:
        """Embed and store a verified chunk in Qdrant.

        Steps:
        1. Check if the chunk hash already exists in the collection.
        2. Generate an embedding via Ollama (``nomic-embed-text``).
        3. Upsert the point into Qdrant with a ``source=network`` payload.

        Returns ``True`` on success, ``False`` on failure or duplicate.
        """
        chunk_hash: str = chunk["hash"]

        # --- deduplicate ---------------------------------------------------
        if await self._chunk_exists_in_qdrant(chunk_hash):
            logger.debug("Chunk %s already in Qdrant — skipping", chunk_hash[:16])
            return False

        # --- embed ---------------------------------------------------------
        content: str = chunk.get("content", "")
        embedding = await self._generate_embedding(content)
        if embedding is None:
            logger.error(
                "Failed to generate embedding for chunk %s", chunk_hash[:16]
            )
            return False

        # --- upsert --------------------------------------------------------
        point_id = chunk_hash  # use the hex hash as a named point ID
        payload: dict[str, Any] = {
            "source": "network",
            "hash": chunk_hash,
            "content": content,
            "contributor_id": chunk.get("contributor_id", ""),
            "timestamp": chunk.get("timestamp", 0),
            "visibility": chunk.get("visibility", "public"),
            "layer": chunk.get("layer", "lan"),
        }

        try:
            resp = await self._http.put(
                f"{self.qdrant_url}/collections/{QDRANT_COLLECTION}/points",
                json={
                    "points": [
                        {
                            "id": point_id,
                            "vector": embedding,
                            "payload": payload,
                        }
                    ]
                },
            )
            resp.raise_for_status()
            logger.info("Stored chunk %s in Qdrant", chunk_hash[:16])
        except httpx.HTTPError as exc:
            logger.error("Qdrant upsert failed for %s: %s", chunk_hash[:16], exc)
            return False

        # Also cache locally
        self.chunk_store.add(chunk)
        return True

    async def _chunk_exists_in_qdrant(self, chunk_hash: str) -> bool:
        """Return ``True`` if a point with this hash exists in the Qdrant collection."""
        try:
            resp = await self._http.post(
                f"{self.qdrant_url}/collections/{QDRANT_COLLECTION}/points/scroll",
                json={
                    "filter": {
                        "must": [
                            {
                                "key": "hash",
                                "match": {"value": chunk_hash},
                            }
                        ]
                    },
                    "limit": 1,
                    "with_payload": False,
                    "with_vector": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            points = data.get("result", {}).get("points", [])
            return len(points) > 0
        except httpx.HTTPError as exc:
            logger.warning("Qdrant scroll check failed: %s", exc)
            return False

    async def _generate_embedding(self, text: str) -> Optional[list[float]]:
        """Call Ollama to generate an embedding for *text*."""
        try:
            resp = await self._http.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embedding: list[float] = data.get("embedding", [])
            if not embedding:
                logger.error("Ollama returned empty embedding")
                return None
            return embedding
        except httpx.HTTPError as exc:
            logger.error("Ollama embedding request failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    async def get_local_manifest(self) -> list[str]:
        """Return a list of chunk hashes from both the local store and Qdrant.

        The Qdrant collection is scrolled to collect all stored hashes.  Hashes
        from the in-memory ``chunk_store`` are merged in.
        """
        hashes: set[str] = set(self.chunk_store.hashes())

        # Pull hashes from Qdrant
        try:
            offset: Optional[str] = None
            while True:
                body: dict[str, Any] = {
                    "limit": 500,
                    "with_payload": {"include": ["hash"]},
                    "with_vector": False,
                }
                if offset is not None:
                    body["offset"] = offset

                resp = await self._http.post(
                    f"{self.qdrant_url}/collections/{QDRANT_COLLECTION}/points/scroll",
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
                result = data.get("result", {})
                points = result.get("points", [])

                for pt in points:
                    h = pt.get("payload", {}).get("hash")
                    if h:
                        hashes.add(h)

                next_offset = result.get("next_page_offset")
                if next_offset is None or not points:
                    break
                offset = next_offset

        except httpx.HTTPError as exc:
            logger.warning(
                "Could not read Qdrant manifest (using local store only): %s",
                exc,
            )

        return list(hashes)

    # ------------------------------------------------------------------
    # High-level workflows
    # ------------------------------------------------------------------

    async def sync_with_peer(self, host: str, port: int) -> int:
        """Execute the full sync workflow with a single peer.

        1. Connect
        2. Handshake
        3. Send sync request
        4. Receive and handle sync response
        5. Disconnect

        Returns the number of chunks successfully ingested.
        """
        ingested = 0
        try:
            await self.connect(host, port)
            await self.send_handshake()
            await self.send_sync_request()

            # Receive messages until the peer sends a sync_response or closes
            assert self._ws is not None
            try:
                async for raw in self._ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("Received non-JSON message — ignoring")
                        continue

                    msg_type = msg.get("type")
                    if msg_type == "sync_response":
                        ingested = await self.handle_sync_response(msg)
                        break
                    elif msg_type == "error":
                        logger.error("Peer error: %s", msg.get("message", "unknown"))
                        break
                    else:
                        logger.debug("Ignoring message type: %s", msg_type)
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for sync_response from %s:%d", host, port)
            except websockets.exceptions.ConnectionClosed as exc:
                logger.warning("Connection closed during sync: %s", exc)

        except ConnectionError as exc:
            logger.error("Could not sync with %s:%d — %s", host, port, exc)
        finally:
            await self.disconnect()

        return ingested

    async def auto_sync(self) -> dict[str, int]:
        """Discover peers via mDNS and synchronise with each one.

        Returns a dict mapping ``"host:port"`` to the number of chunks
        ingested from that peer.
        """
        peers = await self.discover_and_connect()
        results: dict[str, int] = {}

        for host, port in peers:
            key = f"{host}:{port}"
            logger.info("--- auto_sync: syncing with %s ---", key)
            try:
                count = await self.sync_with_peer(host, port)
                results[key] = count
            except Exception:
                logger.exception("auto_sync: error syncing with %s", key)
                results[key] = 0

        total = sum(results.values())
        logger.info(
            "auto_sync complete: %d peer(s), %d total chunks ingested",
            len(results),
            total,
        )
        return results

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release all resources (HTTP client, WebSocket)."""
        await self.disconnect()
        await self._http.aclose()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    """Create a LANSyncClient and run auto_sync."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SuperBrain LAN Sync Client"
    )
    parser.add_argument(
        "--node-id",
        default="client-" + hashlib.sha256(socket.gethostname().encode()).hexdigest()[:16],
        help="Node ID (default: derived from hostname)",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help=f"Qdrant URL (default: {DEFAULT_QDRANT_URL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--peer",
        metavar="HOST:PORT",
        help="Connect to a specific peer instead of mDNS discovery",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    client = LANSyncClient(
        node_id=args.node_id,
        private_key=None,  # not needed for client-only operation
        chunk_store=ChunkStore(),
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
    )

    try:
        if args.peer:
            host, _, port_str = args.peer.rpartition(":")
            if not host or not port_str.isdigit():
                parser.error("--peer must be HOST:PORT (e.g., 192.168.1.10:9734)")
            count = await client.sync_with_peer(host, int(port_str))
            logger.info("Sync complete: ingested %d chunk(s)", count)
        else:
            results = await client.auto_sync()
            for peer_key, count in results.items():
                logger.info("  %s -> %d chunk(s)", peer_key, count)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(_main())
