"""Prefix-aware, adapter-broadcasting reverse proxy in front of N vLLM servers, for AsyncGRPO LoRA.

Runs next to the trainer on 127.0.0.1:8000 (or PROXY_PORT), routing rollouts with KV prefix affinity
and broadcasting /v1/load_lora_adapter across replicas with automatic retries for remote mount propagation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

logger = logging.getLogger("lora_proxy")

HOP_BY_HOP = {
    "host",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "content-encoding",
    "authorization",
}
ROUTED = {"/v1/completions", "/v1/chat/completions"}
BROADCAST = {
    "/v1/load_lora_adapter",
    "/v1/unload_lora_adapter",
    "/pause",
    "/resume",
    "/reset_prefix_cache",
}
FAN_IN_HEALTH = {"/health", "/ping"}


class Config:
    def __init__(self, env=os.environ):
        self.upstreams = [
            u.strip().rstrip("/") for u in env.get("UPSTREAM_URLS", "http://127.0.0.1:8000").split(",") if u.strip()
        ]
        self.token = env.get("HF_TOKEN", "")
        self.port = int(env.get("PROXY_PORT", "8000"))
        self.block_size = int(env.get("PROXY_BLOCK_SIZE", "16"))
        self.max_blocks = int(env.get("PROXY_MAX_TRACKED_BLOCKS", "500000"))
        self.imbalance = int(env.get("PROXY_IMBALANCE", "8"))
        self.lora_load_timeout_s = float(env.get("PROXY_LORA_LOAD_TIMEOUT_S", "300"))
        self.lora_retry_s = float(env.get("PROXY_LORA_RETRY_S", "2"))
        self.conn_limit = int(env.get("PROXY_CONN_LIMIT", "1024"))
        self.stats_every_s = float(env.get("PROXY_STATS_EVERY_S", "60"))


class Upstream:
    def __init__(self, index, url):
        self.index, self.url = index, url
        self.inflight = 0
        self.routed = 0


class PrefixRouter:
    def __init__(self, cfg: Config, n_upstreams: int):
        self.cfg = cfg
        self.n = n_upstreams
        self.blocks: OrderedDict[bytes, list] = OrderedDict()
        self.rr = 0
        self.stats = {"routed": 0, "affinity": 0, "spilled": 0, "unmatched": 0}

    def block_hashes(self, model: str, prompt) -> list[bytes]:
        if isinstance(prompt, str):
            ids = prompt.encode()
            size = self.cfg.block_size * 4
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            ids, size = prompt, self.cfg.block_size
        else:
            return []
        out, prev = [], hashlib.blake2b(model.encode(), digest_size=16).digest()
        for i in range(0, len(ids) - size + 1, size):
            chunk = ids[i : i + size]
            prev = hashlib.blake2b(
                prev + (bytes(chunk) if isinstance(chunk, bytes) else json.dumps(chunk).encode()),
                digest_size=16,
            ).digest()
            out.append(prev)
        return out

    def matched_prefix(self, hashes: list[bytes]) -> list[int]:
        matched = [0] * self.n
        alive = set(range(self.n))
        for h in hashes:
            entry = self.blocks.get(h)
            if not entry:
                break
            alive &= entry[0]
            if not alive:
                break
            for i in alive:
                matched[i] += 1
        return matched

    def common_prefix_len(self, hashes: list[bytes]) -> int:
        last = 0
        for i, h in enumerate(hashes):
            entry = self.blocks.get(h)
            if entry is None:
                break
            if len(entry[0]) == self.n or len(entry[1]) >= 2:
                last = i + 1
        return last

    def choose(self, upstreams: list[Upstream], model: str, prompt) -> Upstream:
        hashes = self.block_hashes(model, prompt)
        matched = self.matched_prefix(hashes)
        common = self.common_prefix_len(hashes)
        specific = [max(0, m - common) for m in matched]
        least = min(u.inflight for u in upstreams)

        best = max(range(self.n), key=lambda i: (specific[i], -upstreams[i].inflight))
        self.stats["routed"] += 1
        if specific[best] > 0 and upstreams[best].inflight - least <= self.cfg.imbalance:
            pick = best
            self.stats["affinity"] += 1
        else:
            if specific[best] > 0:
                self.stats["spilled"] += 1
            else:
                self.stats["unmatched"] += 1
            candidates = [i for i in range(self.n) if upstreams[i].inflight == least]
            pick = candidates[self.rr % len(candidates)]
            self.rr += 1
        for i, h in enumerate(hashes):
            entry = self.blocks.get(h)
            if entry is None:
                entry = self.blocks[h] = [set(), set()]
            entry[0].add(pick)
            if i + 1 < len(hashes) and len(entry[1]) < 2:
                entry[1].add(hashes[i + 1])
            self.blocks.move_to_end(h)
        while len(self.blocks) > self.cfg.max_blocks:
            self.blocks.popitem(last=False)
        return upstreams[pick]


def make_app(cfg: Config) -> web.Application:
    app = web.Application(client_max_size=1 << 30)
    ups = [Upstream(i, u) for i, u in enumerate(cfg.upstreams)]
    router = PrefixRouter(cfg, len(ups))
    app["cfg"], app["upstreams"], app["router"] = cfg, ups, router
    forward_rr = [0]

    def headers_for(request: web.Request) -> dict:
        h = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        if cfg.token:
            h["Authorization"] = f"Bearer {cfg.token}"
        return h

    async def send(up: Upstream, method: str, path_qs: str, headers: dict, body: bytes | None):
        up.inflight += 1
        try:
            async with app["session"].request(
                method,
                up.url + path_qs,
                headers=headers,
                data=body or None,
                allow_redirects=False,
            ) as r:
                return (
                    r.status,
                    {k: v for k, v in r.headers.items() if k.lower() not in HOP_BY_HOP},
                    await r.read(),
                )
        except Exception as e:
            return (
                502,
                {"Content-Type": "text/plain"},
                f"proxy: {up.url}: {type(e).__name__}: {e}".encode(),
            )
        finally:
            up.inflight -= 1

    async def handle(request: web.Request) -> web.StreamResponse:
        path = request.path
        body = await request.read()
        headers = headers_for(request)

        if path in ROUTED and request.method == "POST":
            try:
                payload = json.loads(body)
                prompt = payload.get("prompt", payload.get("messages"))
                model = str(payload.get("model", ""))
            except Exception:
                payload, prompt, model = None, None, ""
            if isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
                prompt = prompt[0]
            if not isinstance(prompt, (str, list)):
                prompt = json.dumps(prompt, sort_keys=True) if prompt is not None else ""
            up = router.choose(ups, model, prompt)
            up.routed += 1
            status, h, out = await send(up, "POST", request.path_qs, headers, body)
            h["X-Proxy-Upstream"] = str(up.index)
            return web.Response(status=status, headers=h, body=out)

        if path == "/v1/load_lora_adapter" and request.method == "POST":
            return await load_lora(request, headers, body)

        if path in BROADCAST:
            results = await asyncio.gather(*(send(u, request.method, request.path_qs, headers, body) for u in ups))
            if path == "/v1/unload_lora_adapter":
                return web.Response(status=200, body=results[0][2], content_type="application/json")
            bad = [(u, r) for u, r in zip(ups, results) if r[0] != 200]
            if bad:
                u, r = bad[0]
                return web.Response(
                    status=r[0],
                    text=f"proxy: {len(bad)}/{len(ups)} servers failed {path}; "
                    f"first: {u.url} -> {r[0]} {r[2][:300].decode(errors='replace')}",
                )
            return web.Response(status=200, headers=results[0][1], body=results[0][2])

        if path in FAN_IN_HEALTH:
            results = await asyncio.gather(*(send(u, request.method, request.path_qs, headers, body) for u in ups))
            bad = [(u, r) for u, r in zip(ups, results) if r[0] != 200]
            if bad:
                u, r = bad[0]
                return web.Response(
                    status=502 if r[0] == 502 else r[0],
                    text=f"proxy: {len(bad)}/{len(ups)} servers unhealthy; first: {u.url} -> {r[0]}",
                )
            return web.Response(status=200, headers=results[0][1], body=results[0][2])

        if path == "/proxy/stats":
            return web.json_response(
                {
                    "upstreams": [{"url": u.url, "inflight": u.inflight, "routed": u.routed} for u in ups],
                    "router": dict(router.stats, tracked_blocks=len(router.blocks)),
                }
            )

        up = ups[forward_rr[0] % len(ups)]
        forward_rr[0] += 1
        status, h, out = await send(up, request.method, request.path_qs, headers, body)
        return web.Response(status=status, headers=h, body=out)

    async def load_lora(request: web.Request, headers: dict, body: bytes) -> web.Response:
        try:
            lora_name = json.loads(body).get("lora_name", "?")
        except Exception:
            lora_name = "?"
        deadline = time.monotonic() + cfg.lora_load_timeout_s

        async def load_one(u: Upstream):
            attempts = 0
            while True:
                attempts += 1
                status, h, out = await send(u, "POST", request.path_qs, headers, body)
                text = out.decode(errors="replace")
                if status == 200:
                    return u, status, text, attempts
                if "No adapter found" in text and time.monotonic() < deadline:
                    await asyncio.sleep(cfg.lora_retry_s)
                    continue
                return u, status, text, attempts

        results = await asyncio.gather(*(load_one(u) for u in ups))
        failed = [r for r in results if r[1] != 200]
        ok = [r for r in results if r[1] == 200]
        for u, st, text, attempts in results:
            print(
                f"[proxy] load_lora_adapter {lora_name} -> {u.url} {st} after {attempts} attempt(s)",
                flush=True,
            )
        if failed:
            unload = json.dumps({"lora_name": lora_name}).encode()
            await asyncio.gather(*(send(u, "POST", "/v1/unload_lora_adapter", headers, unload) for u, *_ in ok))
            u, st, text, _ = failed[0]
            if "No adapter found" in text:
                st = 504
            return web.Response(
                status=st,
                text=f"proxy: {len(failed)}/{len(ups)} servers failed to load "
                f"{lora_name}; rolled back on the others. First: {u.url} -> {st} {text[:300]}",
            )
        return web.Response(status=200, body=results[0][2], content_type="application/json")

    app.router.add_route("*", "/{tail:.*}", handle)

    async def _session(app):
        app["session"] = ClientSession(
            connector=TCPConnector(limit=cfg.conn_limit),
            timeout=ClientTimeout(total=None, sock_connect=60, sock_read=None),
        )
        yield
        await app["session"].close()

    async def _stats_logger(app):
        async def loop():
            last = 0
            while True:
                await asyncio.sleep(cfg.stats_every_s)
                if router.stats["routed"] != last:
                    last = router.stats["routed"]
                    snapshot = dict(
                        router.stats,
                        tracked_blocks=len(router.blocks),
                        inflight=[u.inflight for u in ups],
                        routed=[u.routed for u in ups],
                    )
                    print(f"[proxy] stats {json.dumps(snapshot)}", flush=True)

        task = asyncio.create_task(loop())
        yield
        task.cancel()

    app.cleanup_ctx.append(_session)
    if cfg.stats_every_s > 0:
        app.cleanup_ctx.append(_stats_logger)
    return app


if __name__ == "__main__":
    cfg = Config()
    print(
        f"[proxy] 127.0.0.1:{cfg.port} -> {len(cfg.upstreams)} upstream(s): {', '.join(cfg.upstreams)}",
        flush=True,
    )
    web.run_app(make_app(cfg), host="127.0.0.1", port=cfg.port, print=None)
