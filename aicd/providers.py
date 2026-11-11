"""Thin wrappers around each model provider's SDK.

Every generator in config/generation.yaml names a ``provider``; this module
turns (system prompt, user prompt) into the model's text reply, or ``None``
when the model declined. SDKs are imported lazily, so you only need to
install the ones you actually use.

Providers:
  anthropic          Claude models via the official ``anthropic`` SDK
  openai             OpenAI models via the official ``openai`` SDK
  openai_compatible  Any OpenAI-compatible server (Ollama, vLLM, LM Studio,
                     Groq, Together, OpenRouter, ...) via ``openai`` + base_url
  google             Gemini models via the ``google-genai`` SDK
  mock               Offline fake generator for testing the pipeline
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .common import stable_rng


class Usage:
    """Thread-safe running total of tokens used by one generator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += int(input_tokens or 0)
            self.output_tokens += int(output_tokens or 0)


@dataclass
class Generator:
    name: str
    provider: str
    model: str
    weight: float = 1.0
    holdout: bool = False
    temperature: Optional[float] = None
    max_tokens: int = 1024
    options: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    _call: Optional[Callable[[str, str, Optional[float]], Optional[str]]] = None
    _init_lock: threading.Lock = field(default_factory=threading.Lock)

    def generate(self, system: str, user: str, temperature: Optional[float]) -> Optional[str]:
        if self._call is None:
            with self._init_lock:
                if self._call is None:
                    self._call = _build(self)
        return self._call(system, user, temperature)

    def cost_usd(self) -> Optional[float]:
        """Estimated spend from token counts, if prices are set in the config."""
        pin, pout = self.options.get("price_in_per_mtok"), self.options.get("price_out_per_mtok")
        if pin is None or pout is None:
            return None
        return (self.usage.input_tokens * float(pin) + self.usage.output_tokens * float(pout)) / 1e6


def from_config(entry: dict) -> Generator:
    known = {"name", "provider", "model", "weight", "holdout", "temperature", "max_tokens", "enabled"}
    if entry.get("model") in (None, "", "CHANGE-ME"):
        raise ValueError(f"generator {entry.get('name')}: set `model` to a real model name in the config")
    return Generator(
        name=entry.get("name") or f"{entry['provider']}:{entry['model']}",
        provider=entry["provider"],
        model=entry["model"],
        weight=float(entry.get("weight", 1.0)),
        holdout=bool(entry.get("holdout", False)),
        temperature=entry.get("temperature"),
        max_tokens=int(entry.get("max_tokens", 1024)),
        options={k: v for k, v in entry.items() if k not in known},
    )


def _build(g: Generator):
    builders = {
        "anthropic": _anthropic,
        "openai": _openai,
        "openai_compatible": _openai_compatible,
        "google": _google,
        "mock": _mock,
    }
    if g.provider not in builders:
        raise ValueError(f"unknown provider {g.provider!r} for generator {g.name}")
    return builders[g.provider](g)


# --- Anthropic -------------------------------------------------------------

def _anthropic(g: Generator):
    import anthropic

    client = anthropic.Anthropic(max_retries=int(g.options.get("max_retries", 4)))
    effort = g.options.get("effort", "low")
    fallbacks = g.options.get("fallbacks")  # e.g. "default"; only some models accept it

    def call(system: str, user: str, temperature: Optional[float]) -> Optional[str]:
        # Current Claude models think adaptively and reject sampling params
        # (temperature/top_p), so variety comes from the prompt instead.
        kwargs: dict[str, Any] = dict(
            model=g.model,
            max_tokens=g.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort},
        )
        if fallbacks:
            resp = client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks=fallbacks, **kwargs
            )
        else:
            resp = client.messages.create(**kwargs)
        if resp.usage is not None:
            g.usage.add(resp.usage.input_tokens, resp.usage.output_tokens)
        if resp.stop_reason == "refusal":
            return None
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text or None

    return call


# --- OpenAI and OpenAI-compatible servers ------------------------------------

def _openai_call(g: Generator, client, token_param: str):
    def call(system: str, user: str, temperature: Optional[float]) -> Optional[str]:
        kwargs: dict[str, Any] = dict(
            model=g.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        kwargs[token_param] = g.max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if g.options.get("reasoning_effort"):
            kwargs["reasoning_effort"] = g.options["reasoning_effort"]
        resp = client.chat.completions.create(**kwargs)
        if getattr(resp, "usage", None) is not None:
            g.usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            return None
        return msg.content or None

    return call


def _openai(g: Generator):
    from openai import OpenAI

    client = OpenAI(max_retries=int(g.options.get("max_retries", 4)))
    # Newer OpenAI models take max_completion_tokens instead of max_tokens.
    return _openai_call(g, client, "max_completion_tokens")


def _openai_compatible(g: Generator):
    from openai import OpenAI

    base_url = g.options.get("base_url")
    if not base_url:
        raise ValueError(f"generator {g.name}: openai_compatible needs base_url")
    key_env = g.options.get("api_key_env")
    api_key = os.environ.get(key_env, "") if key_env else "not-needed"
    if key_env and not api_key:
        raise ValueError(f"generator {g.name}: environment variable {key_env} is not set")
    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=int(g.options.get("max_retries", 4)))
    return _openai_call(g, client, "max_tokens")


# --- Google Gemini -----------------------------------------------------------

def _google(g: Generator):
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY

    def call(system: str, user: str, temperature: Optional[float]) -> Optional[str]:
        cfg: dict[str, Any] = dict(system_instruction=system, max_output_tokens=g.max_tokens)
        if temperature is not None:
            cfg["temperature"] = temperature
        resp = client.models.generate_content(
            model=g.model, contents=user, config=types.GenerateContentConfig(**cfg)
        )
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            out = (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
            g.usage.add(um.prompt_token_count or 0, out)
        return resp.text or None

    return call


# --- Mock (offline testing) ---------------------------------------------------

_MOCK_OPENERS = ["Honestly", "Ok so", "Not gonna lie", "Just saying,", "Wow,", "Ugh,", "Fun fact:", ""]
_MOCK_BODIES = [
    "today has been a lot and I am ready for it to be over",
    "this is exactly the kind of thing I needed to hear right now",
    "can't believe how fast this week went by",
    "still thinking about that and it makes me smile",
    "some days you just need a nap and a snack",
    "that sounds rough, hope it gets better soon",
]


def _mock(g: Generator):
    def call(system: str, user: str, temperature: Optional[float]) -> Optional[str]:
        rng = stable_rng("mock", g.name, user)
        g.usage.add(len(system.split()) + len(user.split()), 12)
        return f'"{rng.choice(_MOCK_OPENERS)} {rng.choice(_MOCK_BODIES)}."'.replace('" ', '"')

    return call
