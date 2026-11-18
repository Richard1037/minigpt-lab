"""Runnable reference for training, resuming and generating with the course modules."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'exercises'))
from t_checkpoint import load_checkpoint, save_checkpoint
from t_data import make_batch, tokenize
from t_generate import generate
from t_gpt import GPT
from t_loss import CrossEntropy
from t_optim import AdamW, clip_grad


def build_state(vocab):
    model = GPT(len(vocab), 8, 2, 32, 1, 12, 7)
    optimizer = AdamW(model.parameters(), lr=0.01, decay=0.01)
    return model, optimizer, np.random.default_rng(7)


def update_once(model, optimizer, loss_fn, ids, targets):
    model.zero_grad()
    loss = loss_fn(model(ids), targets)
    model.backward(loss_fn.backward())
    clip_grad(model.parameters(), 1.0)
    optimizer.step()
    return float(loss)


def train_text(text, run_dir, steps, *, resume=False):
    if steps < 1:
        raise ValueError('steps must be positive')
    vocab, ids = tokenize(text)
    model, optimizer, rng = build_state(vocab)
    context = model.context
    split = int(len(ids) * 0.8)
    train_ids, validation_ids = ids[:split], ids[split:]
    if min(len(train_ids), len(validation_ids)) <= context:
        raise ValueError('Both splits must contain more characters than the context window')

    run_dir = Path(run_dir)
    state_file = run_dir / 'checkpoint.npz'
    metadata_file = run_dir / 'metadata.json'
    metadata = {'vocab': list(vocab), 'text': text}
    if resume:
        if json.loads(metadata_file.read_text(encoding='utf-8')) != metadata:
            raise ValueError('Resume with the original training text and vocabulary')
        load_checkpoint(state_file, model, optimizer, rng)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    loss_fn = CrossEntropy()
    losses = []
    for _ in range(steps):
        starts = rng.integers(0, len(train_ids) - context, size=2)
        ids, targets = make_batch(train_ids, starts, context)
        losses.append(update_once(model, optimizer, loss_fn, ids, targets))

    ids, targets = make_batch(validation_ids, [0], context)
    validation_loss = float(loss_fn(model(ids), targets))
    save_checkpoint(state_file, model, optimizer, rng)
    if not resume:
        metadata_file.write_text(json.dumps(metadata), encoding='utf-8')
    return {'step': optimizer.t, 'losses': losses, 'validation_loss': validation_loss}


def continue_text(run_dir, prompts, steps):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / 'metadata.json').read_text(encoding='utf-8'))
    vocab = metadata['vocab']
    model, optimizer, rng = build_state(vocab)
    load_checkpoint(run_dir / 'checkpoint.npz', model, optimizer, rng)
    stoi = {char: index for index, char in enumerate(vocab)}
    continuations = []
    for prompt in prompts:
        if not prompt:
            raise ValueError('Each prompt must contain at least one character')
        unknown = sorted(set(prompt) - stoi.keys())
        if unknown:
            raise ValueError(f'Characters outside the vocabulary: {unknown!r}')
        prefix = np.array([[stoi[char] for char in prompt]], dtype=np.int64)
        result = generate(model, prefix, steps, temperature=0.8, top_k=1, seed=7)
        continuations.append(''.join(vocab[index] for index in result[0]))
    return continuations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'resume', 'generate'))
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--text', type=Path)
    parser.add_argument('--prompt', action='append')
    parser.add_argument('--steps', type=int, required=True)
    args = parser.parse_args()

    if args.mode == 'generate':
        if not args.prompt or args.steps < 0:
            parser.error('generate requires --prompt and nonnegative --steps')
        for result in continue_text(args.run_dir, args.prompt, args.steps):
            print(result)
    else:
        if args.text is None:
            parser.error('train and resume require --text')
        text = args.text.read_text(encoding='utf-8')
        print(train_text(text, args.run_dir, args.steps, resume=args.mode == 'resume'))


if __name__ == '__main__':
    main()
