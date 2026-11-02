"""CPU tests of the training clock, parent transition, and restart invariants."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from training import policy as P
from training import state as H
from training.checkpoints import closed_prefix, quarantine_unclosed, load_resume, restore_runtime
from training.engine import new_optimizer, parent_scheduler, fixed_evaluate
from training.loss import relation_loss
from training.runner import validate_output
from training.transition import restore_selected_parent, validate_selected_provenance


def row(epoch, correct=280):
    return dict(epoch=epoch, run_id='test', phase=P.phase_for_epoch(epoch),
        checkpoint=f'model_{epoch}.pth', checkpoint_sha256='a'*64,
        input_sha256=P.INPUT_SHA, evaluation_alpha=.5,
        eligible_full=epoch >= 30, parent_optimizer_steps=min(epoch+1,30)*320,
        relation_optimizer_steps=max(0,epoch-29)*320,
        train={'relation_nonzero_exit':epoch >= 30},
        evaluation={'natural':{'correct':correct,'total':287,'accuracy':correct/287}})


def test_parent_clock_matches_every_applied_original_lr():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    scheduler = parent_scheduler(optimizer, 320, H)
    for step in range(9600):
        assert optimizer.param_groups[0]['lr'] == P.parent_lr_for_step(step, 320)
        optimizer.step()
        scheduler.step()
    assert scheduler.last_epoch == 9600
    assert optimizer.param_groups[0]['lr'] == P.parent_lr_for_step(9600, 320)


def test_relation_lr_resets_at_real_epoch30():
    assert P.lr_for_step(30*320, 320) == .01
    assert P.lr_for_step(35*320-1, 320) == .01
    assert P.lr_for_step(35*320, 320) == .001
    assert P.lr_for_step(45*320, 320) == pytest.approx(.0001)
    assert P.expected_steps(49,320)['global_step'] == 16000


def test_best_selection_separates_parent_from_actual_srr_and_earliest_tie():
    rows = [row(e,287 if e < 30 else 283) for e in range(50)]
    result = P.select_best(rows)
    assert result['best']['epoch'] == 30
    assert result['parent_best']['epoch'] == 0
    assert result['fixed_last']['epoch'] == 49
    assert result['best']['evaluation']['natural']['correct'] == 283


@pytest.mark.parametrize('mutation', ['input','population','zero_srr','gap'])
def test_selection_rejects_invalid_evidence(mutation):
    rows = [row(e) for e in range(31)]
    if mutation == 'input': rows[0]['input_sha256'] = 'b'*64
    elif mutation == 'population': rows[0]['evaluation']['natural']['total'] = 286
    elif mutation == 'zero_srr': rows[30]['train']['relation_nonzero_exit'] = False
    else: rows[10]['epoch'] = 11
    with pytest.raises(ValueError): P.select_best(rows)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.parent = nn.ParameterDict({f'p{i}': nn.Parameter(torch.tensor([float(i)])) for i in range(125)})
        self.prepool_relation = nn.Module()
        self.prepool_relation.output_projection = nn.Linear(1,1,bias=False)
        nn.init.zeros_(self.prepool_relation.output_projection.weight)
        self.prepool_relation.hidden = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(6)])
        self.phase_epoch = 0
        self.srr_scale = 1.
        self.set_phase(0)

    def set_phase(self, epoch):
        self.phase_epoch = epoch
        for parameter in self.parent.parameters(): parameter.requires_grad_(epoch < 30)
        for parameter in self.prepool_relation.parameters(): parameter.requires_grad_(epoch >= 30)

    def set_srr_scale(self, alpha): self.srr_scale = alpha
    def parent_parameters(self): return self.parent.parameters()
    def relation_parameters(self): return self.prepool_relation.parameters()
    def phase_parameters(self):
        return self.relation_parameters() if self.phase_epoch >= 30 else self.parent_parameters()


def transition_fixture(path):
    model = TinyModel()
    configuration = dict(run_id='test', train_batches=320)
    fresh = H.cpu_tree(model.state_dict())
    H.save_checkpoint(path/'initialization.pth',dict(model=fresh,epoch=-1,from_scratch=True,
                      trained_checkpoint_loaded=False,control=configuration))
    H.atomic_json(path/'INITIALIZATION.json',dict(checkpoint='initialization.pth',
                  checkpoint_sha256=H.sha(path/'initialization.pth')))
    rows = [row(e,283 if e==10 else 280) for e in range(30)]
    selected = H.cpu_tree(fresh)
    selected['parent.p0'].fill_(7)
    payload = dict(model=selected,epoch=10,control=configuration,phase={'relation_active':False},
                   parent_optimizer_steps=11*320,relation_optimizer_steps=0)
    H.save_checkpoint(path/'model_10.pth',payload)
    rows[10]['checkpoint_sha256'] = H.sha(path/'model_10.pth')
    return model,configuration,rows,payload


def test_parent_transition_only_restores_tensors_and_preserves_boundary_rng(tmp_path):
    model,config,rows,payload = transition_fixture(tmp_path)
    torch.manual_seed(29)
    np.random.seed(29)
    before = H.capture_rng()
    result = restore_selected_parent(model,tmp_path,rows,config,torch,H)
    assert H.same_tree(before,H.capture_rng())
    assert model.parent['p0'].item() == 7
    assert result['selected_parent_epoch'] == 10
    assert result['completed_parent_budget_steps'] == 9600
    assert result['selected_parent_lineage_steps'] == 3520
    assert result['selected_checkpoint_rng_restored'] is False
    assert model.phase_epoch == 29
    assert torch.count_nonzero(model.prepool_relation.output_projection.weight) == 0


def test_relation_optimizer_does_not_inherit_parent_momentum():
    model = TinyModel()
    parent = new_optimizer(model,0,torch)
    for parameter in model.parent_parameters(): parameter.grad = torch.ones_like(parameter)
    parent.step()
    assert parent.state
    relation = new_optimizer(model,30,torch)
    assert not relation.state
    assert {id(p) for p in relation.param_groups[0]['params']} == {id(p) for p in model.relation_parameters()}
    assert all(not p.requires_grad for p in model.parent_parameters())


def test_parent_transition_rejects_changed_selected_weight(tmp_path):
    model,config,rows,payload = transition_fixture(tmp_path)
    with (tmp_path/'model_10.pth').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError): restore_selected_parent(model,tmp_path,rows,config,torch,H)


def test_parent_transition_rejects_nonfresh_srr(tmp_path):
    model,config,rows,payload = transition_fixture(tmp_path)
    payload['model']['prepool_relation.hidden.0'].fill_(4)
    H.save_checkpoint(tmp_path/'model_10.pth',payload)
    rows[10]['checkpoint_sha256'] = H.sha(tmp_path/'model_10.pth')
    with pytest.raises(ValueError): restore_selected_parent(model,tmp_path,rows,config,torch,H)


def test_loss_uses_only_train_teacher_correct_subset():
    labels = torch.tensor([0,1])
    teacher = torch.zeros(2,20,requires_grad=True)
    with torch.no_grad(): teacher[:,0] = 5
    student = torch.arange(40,dtype=torch.float32).reshape(2,20).requires_grad_()
    value,metrics = relation_loss(student,teacher,labels)
    ce = torch.nn.functional.cross_entropy(student,labels)
    kl = torch.nn.functional.kl_div(torch.log_softmax(student,dim=-1),
        torch.softmax(teacher.detach(),dim=-1),reduction='none').sum(-1)[0]
    assert torch.equal(value,ce+.25*kl)
    assert metrics['keep_samples'] == 1
    value.backward()
    assert teacher.grad is None and student.grad is not None


def test_no_teacher_correct_loss_has_differentiable_zero_keep():
    student = torch.zeros(2,20,requires_grad=True)
    teacher = torch.zeros(2,20)
    value,metrics = relation_loss(student,teacher,torch.tensor([1,2]))
    assert metrics['keep_samples'] == 0 and metrics['keep_kl'] == 0
    value.backward()
    assert torch.isfinite(student.grad).all()


def test_fixed_evaluation_restores_alpha_and_does_not_advance_training_rng():
    model = SimpleNamespace(srr_scale=1.)
    model.set_srr_scale = lambda alpha: setattr(model,'srr_scale',alpha)
    class Evaluator:
        def run(self,*args,**kwargs):
            assert model.srr_scale == .5
            with H.preserve_rng(): torch.rand(9)
            return dict(input_sha256=P.INPUT_SHA,summary={
                'natural':dict(total=287,correct=283,accuracy=283/287),
                'common192':dict(total=192,correct=190,accuracy=190/192)})
    before = H.capture_rng()
    E = fixed_evaluate(model,Evaluator(),'test',{'natural_eval_videos':287},H,P.INPUT_SHA)
    assert E['summary']['natural']['correct'] == 283
    assert model.srr_scale == 1 and H.same_tree(before,H.capture_rng())


def test_fixed_evaluation_restores_alpha_on_exception():
    model = SimpleNamespace(srr_scale=1.)
    model.set_srr_scale = lambda alpha: setattr(model,'srr_scale',alpha)
    class Evaluator:
        def run(self,*args,**kwargs): raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError): fixed_evaluate(model,Evaluator(),'test',{},H,P.INPUT_SHA)
    assert model.srr_scale == 1


def test_atomic_checkpoint_and_rng_roundtrip(tmp_path):
    before = H.capture_rng()
    path = tmp_path/'model.pth'
    H.save_checkpoint(path,dict(rng=before,value=torch.arange(5)))
    assert not (tmp_path/'model.pth.pending').exists()
    torch.rand(20)
    payload = torch.load(path,map_location='cpu',weights_only=False)
    H.restore_rng(payload['rng'])
    assert H.same_tree(before,H.capture_rng())


def test_quarantine_preserves_completed_and_unrelated_files(tmp_path):
    for name in ('model_29.pth','model_30.pth','other.txt','FAILED.json'):
        (tmp_path/name).write_text('evidence')
    (tmp_path/'fixed_epoch_030').mkdir()
    quarantine_unclosed(tmp_path,30)
    assert (tmp_path/'model_29.pth').exists() and (tmp_path/'other.txt').exists()
    retained = list(tmp_path.glob('interrupted_*'))
    assert len(retained) == 1
    assert (retained[0]/'model_30.pth').read_text() == 'evidence'
    assert not (tmp_path/'model_30.pth').exists()


@pytest.mark.parametrize('destination',['root','data','model','results','existing'])
def test_output_guard(tmp_path,destination):
    root,data = tmp_path/'repo',tmp_path/'dataset'
    root.mkdir(); data.mkdir(); (root/'model').mkdir(); (root/'results').mkdir()
    existing = tmp_path/'existing'; existing.mkdir()
    output = dict(root=root,data=data,model=root/'model/output',results=root/'results/output',existing=existing)[destination]
    with pytest.raises(RuntimeError): validate_output(root,data,output,False)


def test_output_allows_new_run_under_runs(tmp_path):
    root,data = tmp_path/'repo',tmp_path/'dataset'
    root.mkdir(); data.mkdir()
    output = validate_output(root,data,root/'runs/test',False)
    assert output.is_dir()


def _parent_update(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    # This random scalar must continue from the checkpoint RNG on restart.
    value = sum((p.square().sum() for p in model.parent_parameters())) * torch.rand(())
    value.backward()
    optimizer.step()
    scheduler.step()


def closed_parent_fixture(path):
    model = TinyModel()
    config = dict(run_id='test',train_batches=2)
    optimizer = new_optimizer(model,0,torch)
    scheduler = parent_scheduler(optimizer,2,H)
    torch.manual_seed(157)
    for _ in range(2): _parent_update(model,optimizer,scheduler)
    payload = dict(epoch=0,control=config,from_scratch=True,model=H.cpu_tree(model.state_dict()),
        parent_optimizer=H.cpu_tree(optimizer.state_dict()),parent_scheduler=H.cpu_tree(scheduler.state_dict()),
        relation_optimizer=None,global_step=2,parent_optimizer_steps=2,relation_optimizer_steps=0,
        frozen_parent_sha256=None,selected_parent=None,rng=H.cpu_tree(H.capture_rng()))
    H.atomic_json(path/'run_configuration.json',config)
    H.save_checkpoint(path/'model_0.pth',payload)
    fixed=path/'fixed_epoch_000'; fixed.mkdir()
    (fixed/'results.txt').write_text('prediction evidence')
    H.atomic_json(fixed/'COMPLETED.json',dict(passed=True,input_sha256=P.INPUT_SHA,
        files_sha256={'results.txt':H.sha(fixed/'results.txt')}))
    metrics=row(0); metrics['parent_optimizer_steps']=2
    metrics['checkpoint_sha256']=H.sha(path/'model_0.pth')
    metrics['evaluation_completion_sha256']=H.sha(fixed/'COMPLETED.json')
    H.atomic_json(path/'epoch_000_metrics.json',metrics)
    return model,config,optimizer,scheduler,payload


def test_resume_matches_uninterrupted_model_optimizer_scheduler_rng(tmp_path):
    model,config,optimizer,scheduler,payload=closed_parent_fixture(tmp_path)
    _parent_update(model,optimizer,scheduler)
    expected=dict(model=H.cpu_tree(model.state_dict()),optimizer=H.cpu_tree(optimizer.state_dict()),
                  scheduler=H.cpu_tree(scheduler.state_dict()),rng=H.capture_rng())
    resumed=TinyModel()
    saved,rows=load_resume(tmp_path,config,resumed)
    assert len(rows)==1
    new_optimizer_,new_scheduler,relation=restore_runtime(resumed,saved,2)
    assert relation is None
    _parent_update(resumed,new_optimizer_,new_scheduler)
    assert H.same_tree(expected['model'],H.cpu_tree(resumed.state_dict()))
    assert H.same_tree(expected['optimizer'],H.cpu_tree(new_optimizer_.state_dict()))
    assert H.same_tree(expected['scheduler'],H.cpu_tree(new_scheduler.state_dict()))
    assert H.same_tree(expected['rng'],H.capture_rng())


def test_resume_rejects_configuration_change(tmp_path):
    model,config,*_=closed_parent_fixture(tmp_path)
    with pytest.raises(RuntimeError): load_resume(tmp_path,dict(config,train_batches=3),model)


def test_resume_rejects_changed_closed_prediction(tmp_path):
    model,config,*_=closed_parent_fixture(tmp_path)
    (tmp_path/'fixed_epoch_000/results.txt').write_text('different prediction evidence')
    with pytest.raises(RuntimeError): load_resume(tmp_path,config,model)


def test_resume_rejects_scheduler_or_optimizer_corruption(tmp_path):
    model,config,optimizer,scheduler,payload=closed_parent_fixture(tmp_path)
    payload['parent_optimizer']['param_groups'][0]['momentum']=.5
    H.save_checkpoint(tmp_path/'model_0.pth',payload)
    metrics=H.read(tmp_path/'epoch_000_metrics.json')
    metrics['checkpoint_sha256']=H.sha(tmp_path/'model_0.pth')
    H.atomic_json(tmp_path/'epoch_000_metrics.json',metrics)
    with pytest.raises(RuntimeError,match='hyperparameter'): load_resume(tmp_path,config,model)


def test_resume_preserves_closed_epoch_and_quarantines_incomplete_next(tmp_path):
    model,config,*_=closed_parent_fixture(tmp_path)
    (tmp_path/'model_1.pth').write_text('interrupted checkpoint')
    (tmp_path/'fixed_epoch_001').mkdir()
    saved,rows=load_resume(tmp_path,config,model)
    assert len(rows)==1 and saved['epoch']==0
    assert (tmp_path/'model_0.pth').exists()
    assert not (tmp_path/'model_1.pth').exists()
    assert len(list(tmp_path.glob('interrupted_*')))==1


def test_srr_resume_preserves_frozen_parent_momentum_and_local_clock():
    model=TinyModel()
    parent=new_optimizer(model,0,torch)
    scheduler=parent_scheduler(parent,2,H)
    for _ in range(60): _parent_update(model,parent,scheduler)
    relation=new_optimizer(model,30,torch)
    def relation_update(target,optimizer,step):
        optimizer.param_groups[0]['lr']=P.relation_lr_for_step(step,2)
        optimizer.zero_grad(set_to_none=True)
        value=sum((parameter.square().sum() for parameter in target.relation_parameters()))*torch.rand(())
        value.backward(); optimizer.step()
    for step in range(10): relation_update(model,relation,step)
    payload=dict(epoch=34,model=H.cpu_tree(model.state_dict()),parent_optimizer=H.cpu_tree(parent.state_dict()),
                 parent_scheduler=H.cpu_tree(scheduler.state_dict()),relation_optimizer=H.cpu_tree(relation.state_dict()),
                 rng=H.cpu_tree(H.capture_rng()))
    before=H.state_sha(H.split_state(model))
    relation_update(model,relation,10)
    expected=H.cpu_tree(model.state_dict()),H.cpu_tree(relation.state_dict()),H.capture_rng()
    resumed=TinyModel()
    new_parent,new_clock,new_relation=restore_runtime(resumed,payload,2)
    assert resumed.phase_epoch==34 and new_clock.last_epoch==60
    assert H.same_tree(new_parent.state_dict(),payload['parent_optimizer'])
    relation_update(resumed,new_relation,10)
    assert new_relation.param_groups[0]['lr']==.001
    assert H.same_tree(expected[0],H.cpu_tree(resumed.state_dict()))
    assert H.same_tree(expected[1],H.cpu_tree(new_relation.state_dict()))
    assert H.same_tree(expected[2],H.capture_rng())
    assert H.state_sha(H.split_state(resumed))==before
