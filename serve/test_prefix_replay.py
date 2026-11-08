"""serve/test_prefix_replay.py - the prefix cache's matching logic, no GPU, no engine, no pack.

    cd ~/strata && python -m unittest serve.test_prefix_replay -v

The tests with the real Qwen tokenizer need its files (the model's `tokenizer` directory: vocab.json, merges.txt,
token_type.json); set STRATA_TEST_TOKENIZER to it, else they skip.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.frontend import ChatTemplate, OutputParser, openai_to_messages  # noqa: E402
from serve.prefix_replay import (PrefixReplay, PromptIds, common_prefix, fingerprint,  # noqa: E402
                                 message_fingerprint)
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402

STOP = 9999
TOOLS = [{"name": "bash", "description": "run a command",
          "parameters": {"type": "object", "required": ["command"],
                         "properties": {"command": {"type": "string"}, "timeout": {"type": "number"},
                                        "flags": {"type": "array"}}}}]


def asst(reasoning="r", content="c", calls=()):
    m = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if calls:
        m["tool_calls"] = [{"function": {"name": n, "arguments": a}} for n, a in calls]
    return m


def engine_resume(live, checkpoints, ids):
    """Where the engine starts reading `ids` (generate.cpp, 'where this request starts reading'): the session it holds
    if ALL of it is a prefix of the prompt, or the longest checkpoint that is - at most n - 1 tokens."""
    def starts(pre):
        return 1 <= len(pre) <= len(ids) - 1 and list(ids[:len(pre)]) == list(pre)
    resume = len(live) if starts(live) else 0
    for c in checkpoints:
        if len(c) > resume and starts(c):
            resume = len(c)
    return resume


class Logic(unittest.TestCase):
    """PrefixReplay on bare ids: the engine's rule is 'the session is an exact prefix of the prompt'."""

    def setUp(self):
        self.rp = PrefixReplay([STOP], pin_min=4)
        self.prompt = list(range(100, 130))                       # the first request's prompt (ends in the header)
        self.out = [1, 2, 3, 4]                                   # what the model generated
        self.fp = fingerprint("think", "ok", [])
        self.rp.record(self.prompt, self.out + [STOP], 1, self.fp)

    def next_prompt(self, body, tail=(7, 8)):
        return self.prompt + body + [STOP, 10] + list(tail)

    def msgs(self, reply):
        return [{"role": "user", "content": "q"}, reply, {"role": "user", "content": "next"}]

    def test_exact_reply_is_left_alone(self):
        ids, note = self.rp.apply(self.next_prompt(self.out), self.msgs(asst("think", "ok")))
        self.assertEqual(ids, self.next_prompt(self.out))
        self.assertIsNone(note)
        self.assertEqual(self.rp.stats["exact"], 1)
        self.assertEqual(ids.n_messages, 3)

    def test_rerendered_reply_gets_the_models_tokens_back(self):
        rendered = [1, 5, 4]                                       # the template's tokens for the same message
        ids, note = self.rp.apply(self.next_prompt(rendered), self.msgs(asst("think", "ok")))
        self.assertEqual(list(ids), self.next_prompt(self.out))
        self.assertIn("prefix replay", note)
        self.assertEqual(self.rp.stats["replayed"], 1)
        self.assertEqual(ids.n_messages, 3)

    def test_whitespace_the_template_trims_does_not_change_the_message(self):
        ids, _ = self.rp.apply(self.next_prompt([1, 5]), self.msgs(asst("think \n", "\nok  ")))
        self.assertEqual(list(ids), self.next_prompt(self.out))

    def test_a_message_the_client_changed_is_never_replaced(self):
        for reply in (asst("think", "ok!"), asst("other", "ok"), asst("think", "ok", [("bash", {"command": "ls"})])):
            rendered = [1, 5, 4]
            ids, note = self.rp.apply(self.next_prompt(rendered), self.msgs(reply))
            self.assertEqual(list(ids), self.next_prompt(rendered), reply)
            self.assertIn("client changed", note)
        self.assertEqual(self.rp.stats["changed"], 3)

    def test_nothing_to_do_without_the_assistant_turn_or_the_stop_token(self):
        ids, note = self.rp.apply(self.next_prompt([1, 5]), [{"role": "user", "content": "q"}] * 3)
        self.assertEqual(list(ids), self.next_prompt([1, 5]))
        ids, _ = self.rp.apply(self.prompt + [1, 5, 4], self.msgs(asst("think", "ok")))     # no stop token after it
        self.assertEqual(list(ids), self.prompt + [1, 5, 4])

    def test_a_prompt_that_does_not_start_with_the_remembered_one_is_untouched(self):
        other = [99] + self.prompt[1:] + [1, 5, STOP, 10]
        ids, note = self.rp.apply(other, self.msgs(asst("think", "ok")))
        self.assertEqual(list(ids), other)
        self.assertIsNone(note)

    def test_record_drops_the_stop_tokens_and_the_older_turn_of_the_same_chat(self):
        self.assertEqual(self.rp.records[0].out, self.out)
        longer = self.prompt + self.out + [STOP, 10, 7, 8, 11]
        self.rp.record(longer, [20, 21, STOP, STOP], 4, fingerprint("", "x", []))
        self.assertEqual([r.ids for r in self.rp.records], [longer])
        self.rp.record([5, 6, 7], [STOP], 1, self.fp)             # an empty reply is not kept
        self.assertEqual(len(self.rp.records), 1)

    def test_at_most_max_records_are_kept(self):
        rp = PrefixReplay([STOP], max_records=3)
        for k in range(6):
            rp.record([k, 0, 0], [1], 1, self.fp)
        self.assertEqual([r.ids[0] for r in rp.records], [3, 4, 5])

    def test_tool_calls_compare_by_parsed_arguments(self):
        a = fingerprint("", "", [("bash", {"command": "ls", "timeout": 1.5})])
        b = message_fingerprint(asst("", "", [("bash", {"timeout": 1.5, "command": "ls"})]))
        self.assertEqual(a, b)
        self.assertNotEqual(a, message_fingerprint(asst("", "", [("bash", {"command": "ls", "timeout": 2})])))

    def test_sibling_pin(self):
        rp = PrefixReplay([STOP], pin_min=8)
        head = list(range(50))
        rp.record(head + [900, 901, 902], [1], 1, self.fp)
        self.assertEqual(rp.sibling_pin(head + [800, 801, 802, 803]), 50)            # a new chat, 50 shared tokens
        self.assertIsNone(rp.sibling_pin(head + [900, 901, 902, 5, 6]))              # it continues the remembered one
        self.assertIsNone(rp.sibling_pin(head[:6] + [800, 801, 802, 803]))           # shares too little
        self.assertIsNone(rp.sibling_pin(head + [900, 901, 777]))                    # would leave no tail to read
        self.assertIsNone(PrefixReplay([STOP], pin_min=0).sibling_pin(head))
        self.assertIsNone(PrefixReplay([STOP], pin_min=8).sibling_pin(head))         # nothing remembered yet

    def test_overlap_tells_a_continuation_from_a_new_chat(self):
        rp = PrefixReplay([STOP], pin_min=8)
        self.assertIsNone(rp.overlap([1, 2, 3]))                                       # nothing remembered
        rp.record(list(range(40)), [1], 1, self.fp)
        self.assertEqual(rp.overlap(list(range(40)) + [5, 6]), (True, 40))
        self.assertEqual(rp.overlap(list(range(20)) + [-1, -2, -3]), (False, 20))
        self.assertEqual(rp.pin_for(list(range(20)) + [-1, -2, -3], 20), 20)
        self.assertEqual(rp.stats["new_chat"], 1)

    def test_common_prefix_across_block_boundaries(self):
        a = list(range(5000))
        for cut in (0, 1, 511, 512, 513, 1024, 4999, 5000):
            b = a[:cut] + [-1] + a[cut + 1:] if cut < 5000 else list(a)
            self.assertEqual(common_prefix(a, b), cut)
        self.assertEqual(common_prefix(a, a[:700]), 700)


class ServiceFlow(unittest.TestCase):
    """The same through Service.prepare()/run() with the byte tokenizer and the scripted engine."""

    def make(self, script, replay=True):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, script, max_context=8192), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        if replay:
            svc.replay = PrefixReplay(svc.stop_ids, pin_min=16, decode=tok.decode)
        return tok, svc

    def turn(self, svc, messages, tools=TOOLS):
        ids, thinking, max_new = svc.prepare(messages, tools, {}, 512, req={})
        reasoning, content, calls = [], [], []
        for kind, x in svc.run(ids, thinking, tools, max_new, {}, threading.Event()):
            if kind == "event":
                if x.kind == "reasoning":
                    reasoning.append(x.text)
                elif x.kind == "content":
                    content.append(x.text)
                elif x.kind == "tool_call":
                    calls.append(x.call)
        return ids, svc.engine.last_prompt, "".join(reasoning), "".join(content), calls

    def echo(self, reasoning, content, calls):
        """What a client sends back for the reply it got (OpenAI Chat Completions, parsed again by the server)."""
        m = {"role": "assistant", "content": content or None, "reasoning_content": reasoning}
        if calls:
            m["tool_calls"] = [{"id": c.id, "type": "function",
                                "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in calls]
        return m

    def second_prompt(self, svc, first_messages, reply, follow_up):
        req = {"messages": first_messages + [reply] + follow_up,
               "tools": [{"type": "function", "function": TOOLS[0]}]}
        messages, tools, kw = openai_to_messages(req)
        ids, _, _ = svc.prepare(messages, tools, kw, 512, req={})
        return ids

    CALL = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n<parameter=timeout>\n1.50\n</parameter>\n" \
           "<parameter=flags>\n[\"a\",\"b\"]\n</parameter>\n</function>\n</tool_call>"

    def test_the_templates_rerendering_misses_the_cache_and_replay_fixes_it(self):
        script = "I will look.  \n\n</think>\n\nSure.\n\n" + self.CALL      # the template trims the reasoning, 1.50 -> 1.5
        first = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
        resumed = {}
        for replay in (False, True):
            _, svc = self.make(script, replay)
            ids, sent, reasoning, content, calls = self.turn(svc, first)
            self.assertEqual(list(ids), sent)
            out = svc.tok.encode(script, parse_special=True)
            live = list(ids) + out                                   # what the engine holds after the reply
            turn_at = max(i for i, t in enumerate(ids) if t == 256)  # its checkpoint: the last <|im_start|> of the prompt
            tool = {"role": "tool", "tool_call_id": calls[0].id, "content": "file.txt"}
            second = list(self.second_prompt(svc, first, self.echo(reasoning, content, calls), [tool]))
            resumed[replay] = (engine_resume(live, [list(ids[:turn_at])], second), len(second), len(live), turn_at)
        # without the replay the session is no prefix of the prompt: the engine goes back to the checkpoint and reads
        # the whole reply again; with it the session matches and only the tool result (and the header) is read
        miss_from, miss_n, live_n, turn_at = resumed[False]
        hit_from, hit_n, _, _ = resumed[True]
        self.assertEqual(miss_from, turn_at)
        self.assertEqual(hit_from, live_n)
        self.assertGreater(miss_n - miss_from, 3 * (hit_n - hit_from))
        self.assertLess(hit_n - hit_from, 80)

    def test_a_reply_the_client_edited_is_read_as_sent(self):
        script = "think\n</think>\n\nok"
        first = [{"role": "user", "content": "go"}]
        _, svc = self.make(script)
        ids, _, reasoning, content, calls = self.turn(svc, first)
        edited = self.echo(reasoning, content + " (edited)", calls)
        second = self.second_prompt(svc, first, edited, [{"role": "user", "content": "more"}])
        self.assertIn("(edited)", svc.tok.decode(list(second)))
        self.assertEqual(svc.replay.stats["replayed"], 0)

    def test_an_untouched_continuation_is_recorded_and_stays_exact(self):
        script = "think\n</think>\n\nok"
        first = [{"role": "user", "content": "go"}]
        _, svc = self.make(script)
        ids, _, reasoning, content, calls = self.turn(svc, first)
        second = self.second_prompt(svc, first, self.echo(reasoning, content, calls), [{"role": "user", "content": "x"}])
        live = list(ids) + svc.tok.encode(script, parse_special=True)
        self.assertEqual(list(second)[:len(live)], live)
        self.assertEqual(svc.replay.stats["exact"], 1)

    def test_replay_is_off_unless_the_service_has_it(self):
        _, svc = self.make("ok", replay=False)
        ids, *_ = self.turn(svc, [{"role": "user", "content": "go"}])
        self.assertNotIsInstance(ids, PromptIds)

    def test_a_new_chat_pins_the_prefix_it_shares_with_the_last_one(self):
        _, svc = self.make("ok")
        shared = "You are an agent. " * 20
        self.turn(svc, [{"role": "system", "content": shared + "<cwd>/a</cwd>"}, {"role": "user", "content": "hi"}])
        req = {}
        ids, _, _ = svc.prepare([{"role": "system", "content": shared + "<cwd>/b</cwd>"},
                                 {"role": "user", "content": "hi"}], TOOLS, {}, 512, req=req)
        pin = req["strata_prefix"]["tokens"]
        self.assertGreater(pin, len(shared) // 2)
        self.assertLess(pin, len(ids) - 1)
        self.assertIn("/a</cwd>", svc.tok.decode(svc.replay.records[0].ids))
        a, b = svc.replay.records[0].ids, list(ids)
        self.assertEqual(a[:pin], b[:pin])
        self.assertNotEqual(a[pin], b[pin])                              # exactly the first token that differs
        req2 = {"strata_prefix": {"tokens": 5}}                           # a client's own prefix wins
        svc.prepare([{"role": "system", "content": shared + "<cwd>/c</cwd>"}, {"role": "user", "content": "hi"}],
                    TOOLS, {}, 512, req=req2)
        self.assertEqual(req2["strata_prefix"], {"tokens": 5})


TOK_DIR = Path(os.environ["STRATA_TEST_TOKENIZER"]) if os.environ.get("STRATA_TEST_TOKENIZER") else None
TOKENIZER_FILES = ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja")


@unittest.skipUnless(TOK_DIR is not None and all((TOK_DIR / f).exists() for f in TOKENIZER_FILES),
                     "set STRATA_TEST_TOKENIZER to the model's tokenizer directory")
class RealTokenizer(unittest.TestCase):
    """The Qwen tokenizer and chat template: the drifts a real agent produces, and a non-canonical token split."""

    @classmethod
    def setUpClass(cls):
        import strata_tokenizer as ST
        vocab = json.loads((TOK_DIR / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (TOK_DIR / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((TOK_DIR / "token_type.json").read_text())
        cls.tok = ST.Tokenizer(tokens, merges, types)
        cls.tpl = ChatTemplate(TOK_DIR / "chat_template.jinja")
        cls.im_end = cls.tok.encode("<|im_end|>", parse_special=True)

    def drift(self, raw, split_token=False):
        """Round trip one model output: -> (the engine's session after it, the next prompt with and without replay)."""
        tok = self.tok
        sysmsg = {"role": "system", "content": "You are an agent.\n<cwd>\n/tmp\n</cwd>"}
        first = [sysmsg, {"role": "user", "content": "do it"}]
        prompt = tok.encode(self.tpl.render(first, tools=TOOLS), parse_special=True)
        out = tok.encode(raw, parse_special=True)
        if split_token:                                                # a non-canonical split: "\n\n" as "\n", "\n"
            at = next(i for i, t in enumerate(out) if tok.decode([t]) == "\n\n")
            out = out[:at] + tok.encode("\n") * 2 + out[at + 1:]
            self.assertEqual(tok.decode(out), tok.decode(tok.encode(raw, parse_special=True)))
        live = prompt + out + self.im_end + tok.encode("\n")
        parser = OutputParser(thinking=True, tools=TOOLS)
        evs = parser.feed(raw) + parser.finish("stop")
        reasoning = "".join(e.text for e in evs if e.kind == "reasoning")
        content = "".join(e.text for e in evs if e.kind == "content")
        calls = [e.call for e in evs if e.kind == "tool_call"]
        reply = {"role": "assistant", "content": content or None, "reasoning_content": reasoning}
        if calls:
            reply["tool_calls"] = [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in calls]
        follow = ([{"role": "tool", "tool_call_id": calls[0].id, "content": "done"}] if calls
                  else [{"role": "user", "content": "next"}])
        messages, _, kw = openai_to_messages({"messages": first + [reply] + follow,
                                              "tools": [{"type": "function", "function": TOOLS[0]}]})
        plain = tok.encode(self.tpl.render(messages, tools=TOOLS, **kw), parse_special=True)
        rp = PrefixReplay(list(self.im_end))
        rp.record(prompt, out + self.im_end, len(first), fingerprint(reasoning, content,
                                                                      [(c.name, c.arguments) for c in calls]))
        replayed, _ = rp.apply(plain, messages)
        return live, plain, list(replayed)

    def check(self, raw, **kw):
        live, plain, replayed = self.drift(raw, **kw)
        self.assertNotEqual(plain[:len(live)], live, "this output no longer drifts: the case proves nothing")
        self.assertEqual(replayed[:len(live)], live)
        self.assertGreater(len(replayed), len(live))

    CALL = ("<tool_call>\n<function=bash>\n<parameter=command>\n{cmd}\n</parameter>\n{extra}</function>\n</tool_call>")

    def test_reasoning_with_trailing_blank_lines(self):
        self.check("The user wants X.  \n\n</think>\n\nok")

    def test_content_with_a_trailing_newline(self):
        self.check("think\n</think>\n\nok\n")

    def test_float_argument(self):
        self.check("think\n</think>\n\n" + self.CALL.format(cmd="ls", extra="<parameter=timeout>\n1.50\n</parameter>\n"))

    def test_compact_json_argument(self):
        self.check("think\n</think>\n\n" + self.CALL.format(cmd="ls", extra='<parameter=flags>\n["a","b"]\n</parameter>\n'))

    def test_non_canonical_token_split(self):
        self.check("think\n</think>\n\nSure.\n\n" + self.CALL.format(cmd="ls", extra=""), split_token=True)

    def test_a_faithful_round_trip_needs_no_replay(self):
        live, plain, replayed = self.drift("think\n</think>\n\n" + self.CALL.format(cmd="ls -la", extra=""))
        self.assertEqual(plain[:len(live)], live)
        self.assertEqual(replayed, plain)


if __name__ == "__main__":
    unittest.main()
