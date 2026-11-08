"""Opt-in, single-slot disk tier for long Strata conversations.

Only the active conversation is snapshotted when a different conversation takes
its place. The engine verifies restored token identity; a mismatch reads cold.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def identity(messages):
    if not isinstance(messages, list) or not messages:
        return None
    first = next((m for m in messages if m.get('role') == 'user'), None)
    if first is None:
        return None
    seed = [m for m in messages if m.get('role') in ('system', 'developer')]
    seed.append(first)
    data = json.dumps(seed, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


def prefix(messages):
    data = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


class Pool:
    def __init__(self, directory, cap_gib, min_free_gib=51):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.cap = int(cap_gib * (1 << 30))
        self.min_free_gib = min_free_gib
        self.min_free = int(min_free_gib * (1 << 30))
        self.active = None
        self.counts = {'saves': 0, 'restores': 0, 'misses': 0, 'errors': 0, 'removed': 0,
                       'last_save_ms': None, 'last_restore_ms': None}

    def _files(self):
        # Never inspect or remove unrelated sessions in slot_save_path.
        return list(self.directory.glob('convpool-*.bin'))

    def _trim(self):
        files = sorted(self._files(), key=lambda p: p.stat().st_mtime)
        size = sum(p.stat().st_size for p in files)
        while files and size > self.cap:
            old = files.pop(0)
            size -= old.stat().st_size
            old.unlink()
            old.with_suffix('.json').unlink(missing_ok=True)
            self.counts['removed'] += 1

    def before(self, engine, messages, prompt_tokens):
        key = identity(messages)
        if not key or getattr(engine, 'batch', 0):
            return
        if self.active and self.active[0] != key and self.active[2] >= 30000:
            old_key, old_digest, old_tokens, old_count = self.active
            file = self.directory / ('convpool-' + old_key + '.bin')
            # Avoid filling the rig: save only while the disk keeps the minimum free space.
            free = shutil.disk_usage(self.directory).free
            if free >= self.min_free:
                try:
                    result = engine.session_file('save', str(file))
                    if not result.get('tokens') or not file.is_file():
                        raise RuntimeError('empty native save')
                    meta = {'prefix': old_digest, 'messages': old_count, 'tokens': old_tokens}
                    temp = file.with_suffix('.json.tmp')
                    temp.write_text(json.dumps(meta))
                    os.replace(temp, file.with_suffix('.json'))
                    self.counts['saves'] += 1
                    self.counts['last_save_ms'] = result.get('ms')
                    self._trim()
                    print('[strata] convpool: save on active eviction', file=sys.stderr, flush=True)
                except Exception as exc:
                    self.counts['errors'] += 1
                    print(f'[strata] convpool: save failed ({exc})', file=sys.stderr, flush=True)
                    file.unlink(missing_ok=True)
                    file.with_suffix('.json').unlink(missing_ok=True)
            else:
                print(f'[strata] convpool: save skipped, {free / (1 << 30):.1f} GiB free is under the '
                      f'{self.min_free_gib:g} GiB floor', file=sys.stderr, flush=True)
        if self.active is None or self.active[0] != key:
            file = self.directory / ('convpool-' + key + '.bin')
            try:
                meta = json.loads(file.with_suffix('.json').read_text())
                count = meta['messages']
                if count > len(messages) or prefix(messages[:count]) != meta['prefix'] or not file.is_file():
                    raise ValueError('snapshot does not match conversation prefix')
                result = engine.session_file('restore', str(file))
                if not result.get('tokens'):
                    raise RuntimeError('empty native restore')
                self.counts['restores'] += 1
                self.counts['last_restore_ms'] = result.get('ms')
                print('[strata] convpool: restored matching conversation', file=sys.stderr, flush=True)
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                self.counts['misses'] += 1
                print(f'[strata] convpool: miss ({exc})', file=sys.stderr, flush=True)
        self.active = (key, prefix(messages), prompt_tokens, len(messages))

    def finished(self, messages, prompt_tokens, finish):
        if finish in ('stop', 'length') and self.active and self.active[0] == identity(messages):
            self.active = (self.active[0], prefix(messages), prompt_tokens, len(messages))
