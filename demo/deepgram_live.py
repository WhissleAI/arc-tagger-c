"""Optional live ASR for the demo's Live tab: browser audio (16 kHz PCM16 mono) -> Deepgram nova-3 streaming -> words.

Used only when the server runs with DEEPGRAM_API_KEY set in its environment. The key stays on the server (it is sent to
Deepgram only, never to the page). Transcripts are passed to the tagger in memory and never logged.
"""
import asyncio
import json
import os
import time
from urllib.parse import urlencode

from websockets.asyncio.client import connect

URL = os.environ.get("ARCTAG_DEEPGRAM_URL", "wss://api.deepgram.com/v1/listen")      # override: tests or a proxy


class DeepgramLive:
    def __init__(self, key, cfg, on_words, on_endpoint, on_status):
        """on_words(words, is_final, speech_final, latency_s); on_endpoint(); on_status(dict)"""
        self.key, self.cfg = key, cfg
        self.on_words, self.on_endpoint, self.on_status = on_words, on_endpoint, on_status
        self.ws = None
        self.sent_s = 0.0
        self.closed = False
        self.reader = self.keeper = None

    def url(self):
        ep = self.cfg.get("endpointing", 300)
        q = {"model": "nova-3", "language": "en", "encoding": "linear16", "sample_rate": 16000, "channels": 1,
             "interim_results": "true", "punctuate": "true", "smart_format": "true", "diarize": "true",
             "endpointing": int(ep) if isinstance(ep, (int, float)) and ep > 0 else 300, "utterance_end_ms": 1000, "vad_events": "false"}
        return URL + "?" + urlencode(q)

    async def start(self):
        self.ws = await connect(self.url(), additional_headers={"Authorization": "Token " + self.key}, max_size=2 ** 22,
                                open_timeout=10, ping_interval=20)
        self.reader = asyncio.create_task(self.read())
        self.keeper = asyncio.create_task(self.keepalive())

    async def audio(self, b):
        if self.ws is None or self.closed:
            return
        self.sent_s += len(b) / 32000.0
        self.last_audio = time.monotonic()
        try:
            await self.ws.send(b)
        except Exception:
            self.closed = True

    async def keepalive(self):
        try:
            while not self.closed:
                await asyncio.sleep(5)
                if time.monotonic() - getattr(self, "last_audio", 0) > 5 and self.ws is not None:
                    await self.ws.send(json.dumps({"type": "KeepAlive"}))
        except Exception:
            pass

    async def read(self):
        try:
            async for raw in self.ws:
                if isinstance(raw, bytes):
                    continue
                m = json.loads(raw)
                t = m.get("type")
                if t == "Results":
                    alt = ((m.get("channel") or {}).get("alternatives") or [{}])[0]
                    ws = []
                    fin = bool(m.get("is_final"))
                    for w in alt.get("words") or []:
                        x = {"w": w.get("punctuated_word") or w.get("word") or "", "t0": round(float(w.get("start", 0)), 3),
                             "t1": round(float(w.get("end", 0)), 3), "conf": round(float(w.get("confidence", 0.9)), 3), "final": fin}
                        if w.get("speaker") is not None:
                            x["spk"] = "dg%d" % int(w["speaker"])
                        if x["w"]:
                            ws.append(x)
                    lat = (self.sent_s - ws[-1]["t1"]) if ws else None
                    self.on_words(ws, fin, bool(m.get("speech_final")), lat)
                elif t == "UtteranceEnd":
                    self.on_endpoint()
        except Exception as e:
            if not self.closed:
                self.on_status({"level": "error", "msg": f"Deepgram connection lost ({type(e).__name__})"})
        finally:
            self.closed = True

    async def finish(self):
        """end of audio: flush Deepgram, wait for its last results"""
        if self.ws is None:
            return
        if not self.closed:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
        if self.reader is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.reader), timeout=6)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

    async def close(self):
        self.closed = True
        for t in (self.keeper, self.reader):
            if t is not None and not t.done():
                t.cancel()
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
