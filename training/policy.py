"""Declared parent30-best + fresh-SRR20 policy; no Torch, filesystem or RNG.

The fixed287 score chooses among this run's30 parent checkpoints. This is
evaluation-selected development performance, not held-out model selection.
The worker must additionally verify the owned checkpoint file SHA, complete
model/optimizer metadata and zero/fresh Relation state before installing any
selected model tensor. Only model tensors rewind; the stage-boundary TRAIN
RNG, completed-step ledger and parent optimizer/scheduler do not rewind.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
import math
import re

TOTAL_EPOCHS=50
PARENT_EPOCHS=30
RELATION_EPOCHS=20
SWITCH_EPOCH=PARENT_EPOCHS
INPUT_SHA='976c42e36dac5074468044f087cb13a46032035f49834b323e4cdb7e90f56d56'
BASE_LR=.01
MOMENTUM=.9
WEIGHT_DECAY=1e-4
WARMUP_FACTOR=1e-5


def require(value,message):
    if not value:raise ValueError(message)


def _integer(value,name,minimum=0):
    require(type(value) is int and value>=minimum,f'{name} must be an integer >= {minimum}')
    return value


def phase_for_epoch(epoch):
    _integer(epoch,'epoch');require(epoch<TOTAL_EPOCHS,'Exactly epochs0..49')
    return 'parent' if epoch<PARENT_EPOCHS else 'relation'


def controller_state(epoch):
    phase_for_epoch(epoch)
    return dict(schema='content283-parent30-best-srr20-v1',total_epochs=TOTAL_EPOCHS,
        switch_epoch=SWITCH_EPOCH,dynamic=False,
        completed_parent_epochs=min(epoch+1,PARENT_EPOCHS),
        completed_relation_epochs=max(0,epoch+1-PARENT_EPOCHS),
        parent_selection='Highest fixed287 within this run after all30 parent epochs; earliest tie',
        transition_load='Complete selected model tensors only; fresh Relation required',
        training_rng_policy='Continue actual parent-epoch29-end TRAIN RNG; do not restore selected-checkpoint RNG',
        relation_lr_clock='Independent20-epoch clock, reset at actual global epoch30',
        decision_inputs='Fixed declared phase boundary; evaluation only selects the parent checkpoint')


def _step(step,batches,epochs,*,terminal=False):
    _integer(batches,'batches',1);_integer(step,'step')
    require(step<=epochs*batches if terminal else step<epochs*batches,'Step outside declared phase budget')


def parent_lr_for_step(step,batches):
    """Parent scheduler clock0..30B (terminal value included).

    All first11*B pre-update values equal the sealed11+39 recipe exactly.
    Scheduler milestones are11B and20B, with10B actual warmup updates.
    """
    _step(step,batches,PARENT_EPOCHS,terminal=True)
    factor=1.
    if step<10*batches:
        alpha=float(step)/(10*batches)
        factor=WARMUP_FACTOR*(1-alpha)+alpha
    return BASE_LR*factor*.1**bisect_right([11*batches,20*batches],step)


def relation_lr_for_step(step,batches):
    """New Relation optimizer's local clock0..20B, with no warmup."""
    _step(step,batches,RELATION_EPOCHS,terminal=True)
    return BASE_LR*.1**bisect_right([5*batches,15*batches],step)


def lr_for_step(step,batches):
    """LR before one actual global update; do not rewind to selected epoch."""
    _step(step,batches,TOTAL_EPOCHS)
    if step<PARENT_EPOCHS*batches:return parent_lr_for_step(step,batches)
    return relation_lr_for_step(step-PARENT_EPOCHS*batches,batches)


def expected_steps(epoch,batches):
    phase_for_epoch(epoch);_integer(batches,'batches',1)
    parent=min(epoch+1,PARENT_EPOCHS)*batches
    relation=max(0,epoch+1-PARENT_EPOCHS)*batches
    return dict(global_step=parent+relation,parent_optimizer_steps=parent,
                relation_optimizer_steps=relation,parent_scheduler_last_epoch=parent)


def _score(row):
    require(isinstance(row.get('evaluation'),dict),'Missing fixed evaluation')
    natural=row['evaluation'].get('natural');require(isinstance(natural,dict),'Missing natural fixed287 score')
    correct=natural.get('correct');_integer(correct,'correct')
    require(natural.get('total')==287 and correct<=287,'Parent/full selection requires exactly287 videos')
    if 'accuracy' in natural:
        accuracy=natural['accuracy']
        require(type(accuracy) in (int,float) and math.isfinite(accuracy)
                and math.isclose(accuracy,correct/287,rel_tol=0,abs_tol=1e-12),'Inconsistent fixed287 accuracy')
    return correct


def _validate_rows(rows,*,run_id=None):
    require(isinstance(rows,(list,tuple)) and len(rows)<=TOTAL_EPOCHS,'Need a closed0..N metric prefix')
    for epoch,row in enumerate(rows):
        require(isinstance(row,dict) and type(row.get('epoch')) is int and row['epoch']==epoch,
                'Metrics must be a closed ordered prefix0..N')
        require(row.get('phase')==phase_for_epoch(epoch),'Recorded phase differs from declared30+20')
        require(row.get('checkpoint')==f'model_{epoch}.pth','Selection must refer to an owned epoch filename, not an external checkpoint')
        digest=row.get('checkpoint_sha256')
        require(isinstance(digest,str) and re.fullmatch('[a-f0-9]{64}',digest),'Missing/invalid closed checkpoint SHA')
        require(row.get('input_sha256')==INPUT_SHA and row.get('evaluation_alpha')==.5,
                'Fixed54/14 input or alpha.5 selection changed')
        if run_id is not None:require(row.get('run_id')==run_id,'Selected checkpoint belongs to a different run')
        _score(row)
        steps=row.get('relation_optimizer_steps');_integer(steps,'relation_optimizer_steps')
        require(type(row.get('eligible_full')) is bool,'Full eligibility must be Boolean')
        if epoch<PARENT_EPOCHS:
            require(steps==0 and row['eligible_full'] is False,'Parent-only checkpoint cannot be eligible full')
        else:
            evidence=row.get('train',{}).get('relation_nonzero_exit') is True
            require(row['eligible_full'] is (steps>0 and evidence),
                    'Full eligibility must reflect actual Relation updates and nonzero exit')
    return rows


def _record(row):
    keys=('epoch','phase','checkpoint','checkpoint_sha256','evaluation','eligible_full',
          'relation_optimizer_steps','input_sha256','evaluation_alpha')
    result={key:deepcopy(row[key]) for key in keys}
    for key in ('run_id','source_manifest_sha256','parent_optimizer_steps'):
        if key in row:result[key]=deepcopy(row[key])
    return result


def _choose(rows):
    return _record(max(rows,key=_score)) if rows else None


def select_best(rows,*,run_id=None):
    """Report the best actual full model separately from parent selection.

    Pure function over closed records. File ownership, SHA and state contents
    remain the worker/queue's mandatory checks, not assumptions of this API.
    """
    _validate_rows(rows,run_id=run_id)
    return dict(completed_epochs=len(rows),
        best=_choose([r for r in rows if r['phase']=='relation' and r['eligible_full']]),
        parent_best=_choose([r for r in rows if r['phase']=='parent']),
        fixed_last=_choose(rows[-1:]) if len(rows)==TOTAL_EPOCHS else None,
        selection='Highest fixed287 alpha.5 among actual nonzero Relation-update epochs30..49; earliest tie',
        parent_selection='Highest fixed287 among own epochs0..29; earliest tie',
        common192_selection='Same natural-selected checkpoint, never an independently selected common192 best')


def select_parent_for_transition(rows,*,run_id=None):
    """Choose once, after30 closed parent epochs, without rewinding time/RNG."""
    require(isinstance(rows,(list,tuple)) and len(rows)==PARENT_EPOCHS,
            'Transition selection requires all30 closed parent epochs, no earlier freeze or Relation rows')
    _validate_rows(rows,run_id=run_id)
    require(all(row['phase']=='parent' for row in rows),'Only this run\'s parent stage is eligible')
    selected=_choose(rows)
    selected.update(parent_stage_completed_epochs=PARENT_EPOCHS,parent_stage_end_epoch=PARENT_EPOCHS-1,
        next_actual_epoch=PARENT_EPOCHS,load_scope='complete_model_state',
        source_checkpoint_epoch=selected['epoch'],
        restore_selected_checkpoint_rng=False,rng_source='actual_end_of_parent_epoch29',
        restore_selected_parent_optimizer=False,restore_selected_parent_scheduler=False,
        require_fresh_relation_state=True,fresh_relation_optimizer=True,
        relation_lr_local_step=0,
        selection='Own fixed287 parent maximum after30 closed epochs; earliest tie')
    return selected


__all__=['TOTAL_EPOCHS','PARENT_EPOCHS','RELATION_EPOCHS','SWITCH_EPOCH','INPUT_SHA',
    'BASE_LR','MOMENTUM','WEIGHT_DECAY','phase_for_epoch','controller_state',
    'parent_lr_for_step','relation_lr_for_step','lr_for_step','expected_steps',
    'select_best','select_parent_for_transition']
