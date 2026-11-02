"""Install only this run's selected parent tensors, without rewinding its clock."""
from pathlib import Path
import hashlib,json
from . import policy as P

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def parent_state(state):return {k:v for k,v in state.items() if not k.startswith('prepool_relation.')}
def relation_state(state):return {k:v for k,v in state.items() if k.startswith('prepool_relation.')}

def _owned_checkpoint(output,record):
    output=Path(output).resolve();path=output/record['checkpoint']
    require(path.name==f"model_{record['epoch']}.pth" and path.resolve().parent==output and not path.is_symlink()
            and path.is_file() and sha(path)==record['checkpoint_sha256'],'Selected parent checkpoint is not the owned closed archive')
    return path

def restore_selected_parent(model,output,rows,configuration,torch,H):
    if 'capacity_configuration' in configuration:
        require(model.assert_capacity_config(configuration['capacity_configuration'])==configuration['name'],'Selected parent live capacity differs')
    record=P.select_parent_for_transition(rows,run_id=configuration['run_id'])
    selected_path=_owned_checkpoint(output,record);output=Path(output).resolve()
    initial_path=output/'initialization.pth'
    require(initial_path.is_file() and not initial_path.is_symlink(),'Missing owned fresh initialization')
    receipt_path=output/'INITIALIZATION.json'
    require(receipt_path.is_file() and not receipt_path.is_symlink(),'Missing fresh initialization SHA receipt')
    receipt=json.loads(receipt_path.read_text())
    require(receipt['checkpoint']=='initialization.pth' and receipt['checkpoint_sha256']==sha(initial_path),
            'Fresh initialization archive SHA differs')
    rng=H.capture_rng()
    selected=torch.load(selected_path,map_location='cpu',weights_only=False)
    initial=torch.load(initial_path,map_location='cpu',weights_only=False)
    require(selected['epoch']==record['epoch'] and selected['control']==configuration
            and selected['phase']['relation_active'] is False,'Selected checkpoint belongs to a different run/phase')
    require(selected['parent_optimizer_steps']==(record['epoch']+1)*configuration['train_batches']
            and selected['relation_optimizer_steps']==0,'Selected parent has an invalid update budget')
    require(initial['epoch']==-1 and initial['from_scratch'] is True and initial['trained_checkpoint_loaded'] is False
            and initial['control']==configuration,'Fresh archive identity differs')
    state=selected['model'];fresh=relation_state(initial['model']);actual=relation_state(state)
    require(len(fresh)==len(actual)==7 and H.same_tree(actual,fresh),'All seven selected SRR tensors must still be fresh')
    require(bool((actual['prepool_relation.output_projection.weight']==0).all()),'SRR exit must remain zero before its first update')
    target=model.state_dict();require(set(state)==set(target) and len(state)==132,'Selected full model state registry differs')
    require(all(value.shape==target[k].shape and value.dtype==target[k].dtype and bool(torch.isfinite(value).all())
                for k,value in state.items()),'Selected parent shape/dtype/finite contract differs')
    model.load_state_dict(state,strict=True);model.zero_grad(set_to_none=True);model.set_phase(29);model.set_srr_scale(1.)
    if 'capacity_configuration' in configuration:
        require(model.assert_capacity_config(configuration['capacity_configuration'])==configuration['name'],'Restored parent live capacity differs')
    require(H.same_tree(H.cpu_tree(model.state_dict()),H.cpu_tree(state)),'Selected complete model was not restored exactly')
    require(H.same_tree(rng,H.capture_rng()),'Selecting parent changed the end-of-epoch29 TRAIN RNG')
    return dict(selected_record=record,selected_parent_epoch=record['epoch'],selected_checkpoint=selected_path.name,
        selected_checkpoint_sha256=record['checkpoint_sha256'],selected_parent_state_sha256=H.state_sha(parent_state(state)),
        selected_model_state_sha256=H.state_sha(state),fresh_relation_state_sha256=H.state_sha(fresh),
        completed_parent_budget_steps=30*configuration['train_batches'],selected_parent_lineage_steps=(record['epoch']+1)*configuration['train_batches'],
        training_rng_unchanged=True,selected_checkpoint_rng_restored=False,
        selected_parent_optimizer_restored=False,selected_parent_scheduler_restored=False,
        lineage_note='Search budget30; the chosen model lineage contains only selected_epoch+1 parent epochs')

def validate_selected_provenance(cp,rows,output,configuration,H):
    import torch
    require(len(rows)>=30 and cp['epoch']>=30,'Relation resume requires a completed parent search')
    record=P.select_parent_for_transition(rows[:30],run_id=configuration['run_id'])
    provenance=cp.get('selected_parent');require(isinstance(provenance,dict),'Missing selected-parent provenance')
    require(provenance['selected_record']==record and provenance['selected_parent_epoch']==record['epoch']
            and provenance['selected_checkpoint']==record['checkpoint']
            and provenance['selected_checkpoint_sha256']==record['checkpoint_sha256'],'Resume selected parent differs from this run\'s best')
    selected_path=_owned_checkpoint(output,record)
    selected=torch.load(selected_path,map_location='cpu',weights_only=False)
    expected=H.state_sha(parent_state(selected['model']))
    require(selected['control']==configuration and provenance['selected_parent_state_sha256']==expected
            and cp['frozen_parent_sha256']==expected and H.state_sha(parent_state(cp['model']))==expected,
            'Relation checkpoint parent does not equal its selected source')
    require(provenance['training_rng_unchanged'] is True and provenance['selected_checkpoint_rng_restored'] is False
            and provenance['selected_parent_optimizer_restored'] is False and provenance['selected_parent_scheduler_restored'] is False,
            'Resume attempted to rewind parent optimizer/clock/RNG')
    require(provenance['completed_parent_budget_steps']==30*configuration['train_batches']
            and provenance['selected_parent_lineage_steps']==(record['epoch']+1)*configuration['train_batches'],
            'Search budget and selected lineage were confused')
    return provenance
