"""Fixed54/14 single-model natural/local-reference-span evaluation with replay evidence.

Reuses the verified sampling500 numerical core and original10-worker RNG recipe.
Natural denominators follow actual T, with a second length>=36 common subset.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch


HERE=Path(__file__).resolve().parent
ANCHOR_INPUT_SHA='976c42e36dac5074468044f087cb13a46032035f49834b323e4cdb7e90f56d56'
SEED,CONTEXT=54,14


from tools import data_runtime as DATA
from tools import sampling_core as CORE


def _write(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def _state_sha(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(str((name,tuple(t.shape),str(t.dtype))).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _summary(rows):
    if rows:return CORE.summarize(rows)
    return dict(correct=0,total=0,accuracy=None,confusion_matrix=np.zeros((20,20),np.int64).tolist(),incorrect_videos=[])


class _FrameAudit:
    """Stream selected-point hashes, without caching local point coordinates."""
    def __init__(self,loader,path,index_map):
        self.loader,self.path,self.index_map=loader,path,index_map
    def __iter__(self):
        offset=0
        with self.path.open('x',encoding='utf8') as f:
            for batch,(inputs,labels,indices) in enumerate(self.loader):
                rows=[]
                local=inputs['local'].numpy();context=inputs['context'].numpy()
                for j,video_index in enumerate(indices.tolist()):
                    expected,start=self.index_map[offset+j]
                    CORE.require(expected==video_index,'Frame audit/video order differs')
                    rows.append(dict(window_index=offset+j,video_index=video_index,start=start,
                        local_frame_sha256=[hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in local[j]],
                        context_frame_sha256=[hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in context[j]],
                        local_times=inputs['local_times'][j].tolist(),
                        context_times=inputs['times'][j].tolist(),context_valid=inputs['valid'][j].tolist()))
                f.write(json.dumps(dict(batch=batch,windows=rows),separators=(',',':'))+'\n')
                offset+=len(labels)
                yield inputs,labels,indices
        CORE.require(offset==len(self.index_map),'Incomplete frame audit')


class NativeFixedEvaluator:
    """One checkpoint, all natural windows, fixedB14 and10 native Linux workers.

    ``data_full`` accepts a preverified official rawFull24 dataset or a FULL567
    directory. Optional workers=0/device='cpu' is diagnostic-only for CPU tests.
    It must not be used to claim native54/14 anchor reproduction.
    """
    def __init__(self,root,data_full,output_root,L,G,*,local_policy='natural',model_id='MoTrend',workers=10,device='cuda'):
        self.root=Path(root).resolve();self.output_root=Path(output_root).resolve()
        self.L=DATA.validate_clip_frames(L);self.G=DATA.validate_global_frames(G)
        self.local_policy=DATA.validate_local_policy(local_policy,L)
        CORE.require(isinstance(model_id,str) and bool(model_id) and model_id.strip()==model_id,'Require an explicit nonempty model ID')
        self.model_id=model_id
        CORE.require(workers in (0,10),'Use original10 workers or explicit CPU diagnostic0')
        CORE.require(workers==10 or device=='cpu','Zero workers are CPU diagnostics only')
        self.workers,self.device=workers,device
        if str(self.root) not in sys.path:sys.path.insert(0,str(self.root))
        from tools import rng
        from tools.numerics import isolated_rng,recorded_order
        self.rng,self.isolated_rng=rng,isolated_rng
        if isinstance(data_full,(str,os.PathLike)):
            from tools.official_msr import MSRAction3D
            from tools.evaluate import data_identity
            manifest,_=data_identity(self.root,data_full)
            with recorded_order(data_full,manifest['ordered_names']):
                raw=MSRAction3D(str(data_full),frames_per_clip=24,step_between_clips=1,num_points=2048,train=False)
        else:raw=data_full
        CORE.require(raw.train is False,'Evaluation requires held-out TEST source')
        self.dataset=DATA.from_raw(raw,clip_frames=L,global_frames=G,context_seed=CONTEXT,local_policy=self.local_policy)
        CORE.require(len(self.dataset)>0,'Natural evaluation has no eligible windows')
        self.output_root.mkdir(parents=True,exist_ok=True)
        self.reference_rng=self.root/'results/selected/RNG_STATE.pth'
        self.input_sha=None
        self.frame_audit_reference=None

    def _saved(self):
        include_cuda=self.device!='cpu'
        if include_cuda:
            CORE.require(self.reference_rng.is_file(),'Missing sealed RNG_STATE.pth')
            saved=torch.load(self.reference_rng,map_location='cpu',weights_only=False)
            CORE.require(saved['evaluation_seed']==SEED and saved['context_seed']==CONTEXT
                         and saved['cuda_included'],'Sealed RNG identity differs')
            installed=self.rng.installed_loader_recipe(torch,SEED)
            from tools.evaluate import same_tree
            for key in ('loader_generator_initial_state','loader_generator_after_base_seed_state',
                        'dataloader_base_seed','worker_initial_states'):
                CORE.require(same_tree(saved['loader'][key],installed[key]),'Installed worker RNG differs: '+key)
            ignore={'worker_module_path','dataloader_module_path'}
            CORE.require({k:v for k,v in saved['loader']['recipe'].items() if k not in ignore}
                         =={k:v for k,v in installed['recipe'].items() if k not in ignore},
                         'Native Torch/Python/NumPy loader recipe differs')
            CORE.require(installed['recipe']['multiprocessing_start_method']=='fork','Native original loader requires Linux fork')
            CORE.require(saved['loader']['recipe']['batch_size']==14 and saved['loader']['recipe']['num_workers']==10,
                         'Original loader B14/W10 differs')
        else:saved=self.rng.capture_pre_eval(torch,SEED,False,CONTEXT)
        saved['private_context']=self.rng.context_initial_states(self.dataset)
        saved['natural_contract']=self.dataset.contract()
        saved['reference_archive_sha256']=CORE.sha(self.reference_rng) if self.reference_rng.is_file() else None
        return saved

    def _loader(self,saved,directory,archive_sha):
        generator=torch.Generator();generator.set_state(saved['loader']['loader_generator_initial_state'])
        if self.workers==10:
            # The original function's closure writes a proof from each restored
            # worker and checks NumPy/Python/Torch installed initialization.
            from tools import evaluate as runtime
            previous=runtime.RNG_SHA;runtime.RNG_SHA=archive_sha
            try:loader,generator=runtime.fixed_loader(self.dataset,saved,directory,torch)
            finally:runtime.RNG_SHA=previous
            # Under fork, workers start when the iterator is created below, so
            # bind the runtime's global for the iterator lifetime in run().
            return loader,generator
        return torch.utils.data.DataLoader(self.dataset,batch_size=14,shuffle=False,num_workers=0,
            drop_last=False,generator=generator),generator

    def run(self,model,tag,expected_input_sha=None,write_clips=True):
        CORE.require(isinstance(tag,str) and tag and Path(tag).name==tag
                     and not any(c in tag for c in ('\\',':','\x00')),'Unsafe evaluation tag')
        directory=self.output_root/tag;directory.mkdir(exist_ok=False)
        before=_state_sha(model);modes=[(m,m.training) for m in model.modules()]
        grads=[(p,p.requires_grad) for p in model.parameters()]
        start=time.monotonic();model.eval();model.requires_grad_(False)
        try:
            with self.isolated_rng(SEED):
                saved=self._saved();self.rng.restore_current(saved['global_state'],torch)
                torch.save(saved,directory/'RNG_STATE.pth');archive_sha=CORE.sha(directory/'RNG_STATE.pth')
                _write(directory/'RNG_RECEIPT.json',dict(archive_sha256=archive_sha,**self.rng.public_receipt(saved)))
                loader,generator=self._loader(saved,directory,archive_sha)
                from tools import evaluate as runtime
                previous=runtime.RNG_SHA;runtime.RNG_SHA=archive_sha
                try:
                    frame_path=directory/'FRAME_POINT_DIGESTS.jsonl'
                    if self.frame_audit_reference is None:
                        stream=_FrameAudit(loader,frame_path,self.dataset.index_map)
                    else:
                        # Every tensor byte is still hashed by accumulate. Once
                        # that SHA matches, identical per-frame audit bytes can
                        # share storage instead of duplicating GBs over50epochs.
                        stream=loader
                    record,arrays,all_rows=CORE.accumulate({self.model_id:model},stream,self.dataset.video_names,torch,
                        device=self.device,model_ids={self.model_id:self.model_id})
                finally:runtime.RNG_SHA=previous
                CORE.require(torch.equal(generator.get_state(),saved['loader']['loader_generator_after_base_seed_state']),
                             'Loader generator consumption differs')
                if self.workers==10:
                    CORE.require(self.rng.same_state(self.rng.snapshot_current(torch,True),saved['global_state'],torch),
                                 'Native evaluation consumed main RNG')
                    for i in range(10):
                        proof=CORE.read(directory/f'WORKER_{i:02d}_RESTORED.json')
                        CORE.require(proof['passed'] and proof['source_rng_archive_sha256']==archive_sha,'Worker restore proof differs')
            rows=all_rows[self.model_id];contract=self.dataset.contract()
            CORE.require(record['clips']==len(self.dataset) and len(rows)==contract['eligible_videos'],
                         'Natural video/window coverage differs')
            common_names={name for name,n in zip(self.dataset.video_names,self.dataset.video_lengths) if n>=36}
            common=[r for r in rows if r['name'] in common_names]
            CORE.require(len(common)==len(common_names),'Common>=36 coverage differs')
            actual=record['input_sha256']
            target=expected_input_sha or self.input_sha
            if target is not None:CORE.require(actual==target,'Fixed per-epoch input SHA differs')
            if self.device!='cpu' and self.L==24 and self.G==32:
                CORE.require(actual==ANCHOR_INPUT_SHA,'Original L24/G32 input SHA differs')
            CORE.require(_state_sha(model)==before,'Evaluation changed checkpoint state')
            audit_source=self.frame_audit_reference or frame_path
            audit_reused=self.frame_audit_reference is not None
            # Publish a reused frame audit only after this epoch's actual CPU
            # input stream passed the full SHA check against its first epoch.
            if audit_reused:
                try:os.link(audit_source,frame_path)
                except OSError:shutil.copyfile(audit_source,frame_path)
            self.input_sha=actual
            if self.frame_audit_reference is None:self.frame_audit_reference=frame_path
            summary=dict(natural=_summary(rows),common192=_summary(common))
            summary['common192']['selection']='actual video length >=36; no fixed count assertion'
            summary['natural']['accuracy_percent']=100*summary['natural']['accuracy']
            if summary['common192']['accuracy'] is not None:
                summary['common192']['accuracy_percent']=100*summary['common192']['accuracy']
            record.update(model_id=self.model_id,local_frames=self.L,global_frames=self.G,local_policy=self.local_policy,evaluation_seed=SEED,context_seed=CONTEXT,
                elapsed_seconds=time.monotonic()-start,model_state_sha256=before,
                native_protocol=self.workers==10 and self.device!='cpu',batch_size=14,workers=self.workers,
                frame_point_digest_sha256=CORE.sha(frame_path),frame_audit_reused=audit_reused,
                frame_audit_source=str(audit_source.relative_to(self.output_root)),
                frame_audit_source_sha256=CORE.sha(audit_source))
            _write(directory/'CONTRACT.json',contract);_write(directory/'HASHES.json',record)
            _write(directory/'RESULT.json',dict(summary=summary,natural=rows,common192=common,record=record,
                single_model=True,ensemble=False,model_states_unchanged=True))
            if write_clips:
                with (directory/'clips.npz').open('xb') as f:np.savez_compressed(f,**arrays)
            _write(directory/'COMPLETED.json',dict(passed=True,summary=summary,input_sha256=actual,
                files_sha256={p.name:CORE.sha(p) for p in directory.iterdir() if p.is_file()},
                single_model=True,ensemble=False,no_training=True))
            return dict(summary=summary,input_sha=actual,input_sha256=actual,path=str(directory),record=record)
        except BaseException as error:
            _write(directory/'FAILED.json',dict(passed=False,error=repr(error),traceback=traceback.format_exc()))
            raise
        finally:
            for m,mode in modes:m.training=mode
            for p,flag in grads:p.requires_grad_(flag)


__all__=['NativeFixedEvaluator','ANCHOR_INPUT_SHA']
