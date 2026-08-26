"""Telegram demo agent: a memory-backed assistant.

Listens for messages; before answering, pulls relevant context from memory;
after answering, stores notable facts. Uses OpenAI for the chat loop when a
key is set, otherwise echoes tool calls (demo mode) so the wiring is testable
without keys.

Run:
    export TELEGRAM_BOT_TOKEN=...
    uv run python -m agent_memory.demo.telegram_agent
"""

from __future__ import annotations

import asyncio
import json
import os
import re

from agent_memory.core.config import get_settings
from agent_memory.tools import MemoryToolExecutor

SYSTEM_PROMPT = (
    "You are Simon's personal memory agent. You have long-term memory tools.\n"
    "Before answering questions about the user, search your memory.\n"
    "When the user shares a durable preference, fact, or event, call memory_write.\n"
)


class MemoryAgent:
    def __init__(self, executor: MemoryToolExecutor, llm=None) -> None:
        self.executor = executor
        self.llm = llm  # OpenAI client or None (demo mode)
        self._pending_tools: list[tuple[str, dict]] = []

    async def handle_message(self, text: str) -> str:
        """Produce a reply; in demo mode, exercise tools directly."""
        if self.llm is None:
            return await self._demo_reply(text)

        # real LLM loop with function calling
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        for _ in range(3):  # bounded tool-calling rounds
            resp = await asyncio.to_thread(
                self.llm.chat.completions.create,
                model="gpt-4o-mini",
                messages=messages,
                tools=self.executor.tools,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content or ""
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = await asyncio.to_thread(self.executor.execute, tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return "(gave up after 3 rounds)"

    async def _demo_reply(self, text: str) -> str:
        """No-key demo: inject remembered context + store the message as an episode."""
        ctx = json.loads(
            await asyncio.to_thread(self.executor.execute, "memory_context", {"topic": text})
        )
        remember_match = re.search(r"\bremember that (.+)", text, re.IGNORECASE)
        stored = None
        if remember_match:
            stored = json.loads(
                await asyncio.to_thread(
                    self.executor.execute,
                    "memory_write",
                    {"content": remember_match.group(1), "kind": "semantic"},
                )
            )

        lines = []
        if stored:
            lines.append(f"Remembered: {remember_match.group(1)} (id {stored['id'][:8]}…)")
        block = ctx.get("context", "")
        if block and block != "(no relevant memories)":
            lines.append(f"From memory:\n{block}")
        if not lines:
            lines.append("(demo mode — set AM_OPENAI_API_KEY for full LLM replies)")
        return "\n\n".join(lines)


async def run_telegram() -> None:  # pragma: no cover — needs live bot token
    from telegram.ext import AIORateLimiter, Application, MessageHandler, filters

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    settings = get_settings()

    from agent_memory.core.embedding import get_embedder
    from agent_memory.ingest.pipeline import get_repo_for_url

    repo = get_repo_for_url(settings.database_url)
    executor = MemoryToolExecutor(repo, get_embedder())

    llm = None
    api_key = os.environ.get("AM_OPENAI_API_KEY")
    if api_key:
        from openai import OpenAI

        llm = OpenAI(api_key=api_key)

    agent = MemoryAgent(executor, llm)

    app = (
        Application.builder()
        .token(token)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    async def on_message(update, context) -> None:
        if not update.message or not update.message.text:
            return
        reply = await agent.handle_message(update.message.text)
        await update.message.reply_text(reply[:4000])

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    print("memory agent listening…")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(run_telegram())
