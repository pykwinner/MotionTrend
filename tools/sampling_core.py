"""Numerical input hash, softmax and sequential FP32 per-video sums are unchanged.
The natural evaluator uses one existing registry key for its single model.
"""
import hashlib,json,struct,importlib.util
from pathlib import Path
import numpy as np
from tools.memory_budget import guard

def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def update_output_hash(h, probabilities, labels, indices):
    for name, value in (('indices', np.asarray(indices, np.int64)),
                        ('labels', np.asarray(labels, np.int64)),
                        ('probabilities', np.asarray(probabilities, np.float32))):
        value = np.ascontiguousarray(value)
        h.update(str((name, value.shape, str(value.dtype))).encode('utf8'))
        h.update(value.tobytes())


def canonical_sums_sha(rows):
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r['name']):
        b = r['name'].encode('utf8')
        h.update(struct.pack('<I', len(b))); h.update(b)
        h.update(np.asarray([r['label'], r['windows']], dtype='<i8').tobytes())
        h.update(np.asarray(r['probability_sum'], dtype='<f4').tobytes())
    return h.hexdigest()


def video_rows(totals, truth, counts, names):
    require(set(totals) == set(truth) == set(counts), 'Video populations differ')
    rows = []
    for i in sorted(truth):
        p, label = totals[i], truth[i]
        require(p.dtype == np.float32 and np.isfinite(p).all(), 'Invalid video sum')
        pred = int(p.argmax())
        rows.append(dict(name=str(names[i]), video_index=i, label=label, windows=counts[i],
                         prediction=pred, correct=pred == label,
                         probability_sum=p.tolist(),
                         probabilities=(p.astype(np.float64) / counts[i]).tolist()))
    return rows


def summarize(rows):
    require(rows and len(rows) == len({r['name'] for r in rows}), 'Duplicate/empty videos')
    confusion = np.zeros((20, 20), np.int64)
    for r in rows:
        require(r['correct'] == (r['prediction'] == r['label']), 'Wrong correctness flag')
        confusion[r['label'], r['prediction']] += 1
    correct = sum(bool(r['correct']) for r in rows)
    return dict(correct=correct, total=len(rows), accuracy=correct / len(rows),
                confusion_matrix=confusion.tolist(),
                incorrect_videos=[r['name'] for r in rows if not r['correct']])


def paired_counts(baseline, candidate):
    require([(r['name'], r['video_index'], r['label'], r['windows']) for r in baseline]
            == [(r['name'], r['video_index'], r['label'], r['windows']) for r in candidate],
            'Cross-model video alignment differs')
    repaired = sum(not b['correct'] and c['correct'] for b, c in zip(baseline, candidate))
    harmed = sum(b['correct'] and not c['correct'] for b, c in zip(baseline, candidate))
    kept = sum(b['correct'] and c['correct'] for b, c in zip(baseline, candidate))
    wrong = sum(not b['correct'] and not c['correct'] for b, c in zip(baseline, candidate))
    bc, cc = sum(r['correct'] for r in baseline), sum(r['correct'] for r in candidate)
    require(cc - bc == repaired - harmed, 'Repair/harm closure failed')
    return dict(repaired=int(repaired), harmed=int(harmed), kept_correct=int(kept),
                still_wrong=int(wrong), net=int(cc - bc), baseline_correct=int(bc),
                candidate_correct=int(cc), total=len(baseline))


def accumulate(models, loader, video_names, torch, *, device='cuda', model_ids=None):
    """Shared-stream numerical core; CPU fixtures exercise this same function."""
    if model_ids is None:
        require(bool(models), 'Expected at least one frozen model')
        model_ids={v:v for v in models}
    else:
        require(bool(models) and set(model_ids)==set(models)
                and all(isinstance(x,str) and x for x in model_ids.values())
                and len(set(model_ids.values()))==len(model_ids),'Invalid explicit model registry')
    for model in models.values():
        require(not any(m.training for m in model.modules())
                and not any(p.requires_grad for p in model.parameters()), 'Models must be frozen/eval')
    input_h, local_h = hashlib.sha256(), hashlib.sha256()
    out_h = {v: hashlib.sha256() for v in models}
    totals, parts = {v: {} for v in models}, {v: [] for v in models}
    truth, counts, labels_parts, index_parts, batches = {}, {}, [], [], []
    clip_correct = {v: 0 for v in models}
    with torch.no_grad():
        for batch, (inputs, labels, indices) in enumerate(loader):
            if batch%25==0:guard(torch)
            cpu_versions = {k: t._version for k, t in inputs.items()}
            for k, t in sorted(inputs.items()):
                require(t.device.type == 'cpu', 'Hash original CPU inputs before H2D')
                input_h.update(k.encode()); input_h.update(t.contiguous().numpy().tobytes())
            local_h.update(b'local'); local_h.update(inputs['local'].contiguous().numpy().tobytes())
            y, ids = labels.numpy(), indices.numpy()
            require(y.dtype == ids.dtype == np.int64 and y.shape == ids.shape,
                    'Invalid label/index batch')
            for label, index in zip(y, ids):
                label, index = int(label), int(index)
                require(0 <= label < 20 and 0 <= index < len(video_names), 'Invalid video identity')
                require(index not in truth or truth[index] == label, 'Mixed video labels')
                truth[index] = label; counts[index] = counts.get(index, 0) + 1
            labels_parts.append(y.copy()); index_parts.append(ids.copy()); batches.append(len(y))
            x = inputs.to(device, non_blocking=True) if hasattr(inputs, 'to') else {
                k: t.to(device, non_blocking=True) for k, t in inputs.items()}
            versions = {k: t._version for k, t in x.items()}
            for v, model in models.items():
                logits = model(x)
                require(logits.dtype == torch.float32 and tuple(logits.shape) == (len(y), 20)
                        and bool(torch.isfinite(logits).all()), 'Invalid frozen logits')
                require(all(t._version == versions[k] for k, t in x.items()),
                        'Model changed shared GPU inputs: ' + v)
                probability = logits.softmax(-1).cpu().numpy()
                update_output_hash(out_h[v], probability, y, ids)
                for pr, index in zip(probability, ids):
                    i = int(index)
                    if i not in totals[v]: totals[v][i] = pr.copy()
                    else: totals[v][i] += pr
                parts[v].append(probability)
                clip_correct[v] += int((probability.argmax(1) == y).sum())
            require(all(t._version == cpu_versions[k] for k, t in inputs.items()),
                    'Model changed shared CPU inputs')
            if batch % 100 == 0:
                print('EVALUATION_BATCH', batch, 'windows', sum(batches), flush=True)
    require(batches, 'Empty evaluation stream')
    rows = {v: video_rows(totals[v], truth, counts, video_names) for v in models}
    arrays = dict(labels=np.concatenate(labels_parts), indices=np.concatenate(index_parts),
                  batch_sizes=np.asarray(batches, np.int64), model_ids=np.asarray([model_ids[v] for v in models]))
    for v in models: arrays['probabilities__' + model_ids[v]] = np.concatenate(parts[v])
    record = dict(input_sha256=input_h.hexdigest(), local_input_sha256=local_h.hexdigest(),
                  output_sha256={v: h.hexdigest() for v, h in out_h.items()},
                  sums_sha256={v: canonical_sums_sha(rows[v]) for v in models},
                  clips=sum(batches), loaded_videos=len(video_names), scored_videos=len(truth),
                  clip_correct=clip_correct,
                  unscored=[str(n) for i, n in enumerate(video_names) if i not in truth])
    return record, arrays, rows
