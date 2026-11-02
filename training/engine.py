"""Exact parent/SRR optimizer steps and isolated fixed-input evaluation."""
from __future__ import annotations
import hashlib
import math
import re
import time
from . import policy as P

def parent_stream_gate(optimizer,scheduler,epoch,stats,reference,torch):
    """Retain the original first11 input RNG and applied LR path."""
    require(0<=epoch<11,'Original stream gate is only applicable to first11 parent epochs')
    expected=reference['epoch_end'][epoch];actual=rng_receipt(torch)
    require(all(actual[k]==expected[k] for k in actual),'Capacity TRAIN/next-TRAIN RNG trajectory differs')
    require(stats['count_lr_sha256']==expected['train_count_lr_sha256'],'Capacity applied TRAIN LR/count differs')
    state=scheduler.state_dict()
    require(state['milestones']==[3520,6400] and state['last_epoch']==(epoch+1)*320,'Capacity parent scheduler clock differs')
    return dict(passed=True,epoch=epoch,original_cpu_numpy_cuda_rng_exact=True,
                original_applied_batch_lr_exact=True,weight_equality_required=False)


def require(value,message):
    if not value:raise RuntimeError(message)


def phase_for_epoch(epoch):
    return P.phase_for_epoch(epoch)


def lr_for_step(step,batches):
    return P.lr_for_step(step,batches)


def new_optimizer(model,epoch,torch):
    model.set_phase(epoch);active=list(model.phase_parameters())
    require({id(p) for p in active}=={id(p) for p in model.parameters() if p.requires_grad},'Phase optimizer parameter registry differs')
    return torch.optim.SGD(active,lr=.01,momentum=.9,weight_decay=1e-4)


def parent_scheduler(optimizer,batches,H):
    return H.WarmupMultiStepLR(optimizer,[11*batches,20*batches],gamma=.1,warmup_iters=10*batches,warmup_factor=1e-5)


def validate_evaluation(result,configuration,expected_sha=None):
    value=result['input_sha256'];require(isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value),'Invalid fixed-input digest')
    if expected_sha is not None:require(value==expected_sha,'Fixed evaluation input changed')
    for name,total in (('natural',configuration['natural_eval_videos']),('common192',192)):
        r=result['summary'][name]
        require(r['total']==total and type(r['correct']) is int and 0<=r['correct']<=total,'Wrong evaluation population: '+name)
        require(math.isclose(r['accuracy'],r['correct']/total,rel_tol=0,abs_tol=1e-12),'Evaluation count/accuracy differs')
    return value


def rng_receipt(torch):
    import numpy as np
    value=np.random.get_state();h=hashlib.sha256();h.update(repr((value[0],*value[2:])).encode());h.update(value[1].tobytes())
    return dict(cpu_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),numpy_rng_sha256=h.hexdigest(),
                cuda_rng_sha256=[hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for v in torch.cuda.get_rng_state_all()])


def train_epoch(model,optimizer,scheduler,loader,epoch,batches,torch,H,S,*,device='cuda'):
    import numpy as np
    phase=phase_for_epoch(epoch);model.set_phase(epoch);model.set_srr_scale(1.);model.train();loader.reset()
    require(bool(model.relation_active)==(phase=='relation'),'Parent30/SRR20 phase differs from actual graph')
    if phase=='parent':require(scheduler is not None and scheduler.last_epoch==epoch*batches,'Parent scheduler starts on wrong clock')
    else:require(scheduler is None,'SRR uses its independent20-epoch LR clock without parent momentum/scheduler')
    parent_sha=H.state_sha(H.split_state(model)) if phase=='relation' else None
    count=correct=seen=keep=hard=0;ce_sum=loss_sum=kl_sum=raw_sum=effective_sum=0.;outlet_max=hidden_max=0.;clock=[]
    tick=time.monotonic()
    for batch,(inputs,labels,_) in enumerate(loader):
        require(batch<batches,'TRAIN traversal exceeds declared epoch')
        step=epoch*batches+batch;lr=lr_for_step(step,batches)
        if phase=='parent':require(optimizer.param_groups[0]['lr']==lr,'Parent per-batch LR differs from original global clock')
        else:
            for group in optimizer.param_groups:group['lr']=lr
        x,y=inputs.to(device),labels.to(device);optimizer.zero_grad(set_to_none=True)
        if phase=='parent':
            logits=model(x);loss=torch.nn.functional.cross_entropy(logits,y);values=dict(ce=float(loss.detach()),keep_kl=0.)
        else:
            outputs=model.training_outputs(x);logits=outputs['logits']
            require(outputs['relation_active'] and outputs['base_logits'] is not None,'Missing frozen Content TRAIN teacher')
            loss,values=S.relation_loss(logits,outputs['base_logits'],y,kl_weight=.25)
            mask=outputs['base_logits'].detach().argmax(-1).eq(y);keep+=int(mask.sum());hard+=int((~mask).sum())
            raw_sum+=float(outputs['relation_raw'].detach().norm())*len(y)
            effective_sum+=float(outputs['relation_residual'].detach().norm())*len(y)
        require(bool(torch.isfinite(loss)),'Nonfinite Content TRAIN loss');loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.phase_parameters()),'Active Content parameter lacks finite gradient')
        inactive=model.relation_parameters() if phase=='parent' else model.parent_parameters()
        require(all(not p.requires_grad and p.grad is None for p in inactive),'Inactive Content phase received gradients')
        if phase=='relation':
            outlet_max=max(outlet_max,float(model.prepool_relation.output_projection.weight.grad.norm()))
            hidden_max=max(hidden_max,float(model.prepool_relation.edge_mlp[0].weight.grad.norm()))
        optimizer.step()
        if scheduler is not None:scheduler.step()
        n=len(y);count+=n;seen+=1;correct+=int(logits.detach().argmax(-1).eq(y).sum())
        loss_sum+=float(loss.detach())*n;ce_sum+=values['ce']*n;kl_sum+=values['keep_kl']*n;clock.append([n,lr])
        if batch%100==0:print('CONTENT283_TRAIN',epoch,phase,batch,float(loss.detach()),flush=True)
    require(seen==batches and count==len(loader.loader.dataset),'Incomplete Content TRAIN traversal')
    frozen=None;nonzero=False
    if phase=='relation':
        frozen=H.state_sha(H.split_state(model))==parent_sha;require(frozen,'Content frozen parent changed')
        nonzero=bool(torch.count_nonzero(model.prepool_relation.output_projection.weight)) and effective_sum>0
        require(nonzero and outlet_max>0 and hidden_max>0,'Relation failed to learn a nonzero effective exit')
    return dict(loss=loss_sum/count,ce=ce_sum/count,mean_ce=ce_sum/count,keep_kl=kl_sum/count,clips=count,batches=seen,
        correct=correct,train_correct=correct,train_count=count,keep_samples=keep,hard_samples=hard,phase=phase,
        parent_state_sha256=parent_sha,frozen_parent_byte_exact=frozen,relation_nonzero_exit=nonzero,
        raw_residual_norm=raw_sum/count,effective_residual_norm=effective_sum/count,
        outlet_gradient_max=outlet_max,hidden_gradient_max=hidden_max,first_lr=clock[0][1],last_lr=clock[-1][1],
        count_lr_sha256=hashlib.sha256(np.asarray(clock,np.float64).tobytes()).hexdigest(),seconds=time.monotonic()-tick)


def fixed_evaluate(model,evaluator,tag,configuration,H,expected_sha=None):
    before=H.capture_rng();old_alpha=float(getattr(model,'srr_scale',1.))
    try:
        model.set_srr_scale(.5);result=evaluator.run(model,tag,expected_input_sha=expected_sha,write_clips=True)
        validate_evaluation(result,configuration,expected_sha)
    finally:model.set_srr_scale(old_alpha)
    require(H.same_tree(before,H.capture_rng()),'Fixed alpha.5 evaluation changed TRAIN RNG')
    return result
