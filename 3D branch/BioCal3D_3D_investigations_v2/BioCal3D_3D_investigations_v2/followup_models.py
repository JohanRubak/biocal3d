"""Controlled input/colour pathways around the unchanged official-backend runner."""
from pathlib import Path
import copy,types
import numpy as np
from study_data import signature,digest,seed_for
import biocal3d_train_3d_investigations as core

ORIGINAL_SURFACE=core.surface_data
ORIGINAL_BUILD=core.build_model
PATHWAYS=['geometry_matched','stats_only','rgb_encoder_only','rgb_combined','rgb_shuffled','gray_combined']

def implementation_hash():
    root=Path(__file__).parent
    return signature({name:digest(root/name) for name in ['followup_models.py','followup.py','followup_report.py','approved_data.py','extensions.py','study_data.py','study_report.py','biocal3d_train_3d_investigations.py','official_backends.py']})

def alter_surface(data,pathway,row_id,shuffle_seed=123):
    s={k:v.copy() for k,v in data.items()}
    if pathway in ['geometry_matched','legacy_geometry']:s['color'][:]=0
    elif pathway=='gray_combined':
        gray=s['color']@np.array([.299,.587,.114],np.float32)
        s['color']=np.repeat(gray[:,None],3,axis=1).astype(np.float32)
    elif pathway=='rgb_shuffled':
        rng=np.random.default_rng(seed_for('rgb-permutation:'+row_id,shuffle_seed))
        s['color']=s['color'][rng.permutation(len(s['color']))]
    elif pathway not in ['stats_only','rgb_encoder_only','rgb_combined']:raise ValueError('Unknown pathway '+pathway)
    return s

def surface_data(row,job,output):
    s,key=ORIGINAL_SURFACE(row,job,output)
    pathway=job['surface'].get('pathway')
    if pathway:s=alter_surface(s,pathway,row['row_id'],job['surface']['sampling_seed'])
    return s,key

def colour_statistics_numpy(color):
    c=np.asarray(color,dtype=np.float32)
    return np.concatenate([c.mean(0)*2-1,(c*2-1).max(0)]).astype(np.float32)

def build_model(job,args,device):
    model,provenance=ORIGINAL_BUILD(job,args,device)
    pathway=job['surface'].get('pathway')
    if not pathway:return model,provenance  # Exact existing-checkpoint model.
    if pathway=='legacy_geometry':
        return model,dict(provenance,followup_implementation=implementation_hash(),pathway=pathway)
    T=core.T;channels=model.encoder.channels
    if pathway=='stats_only':
        model.frozen=True
        for p in model.encoder.parameters():p.requires_grad=False
    def embedding(self,d,seed):
        batch=d['color'].shape[0]
        if pathway=='stats_only':pooled=d['color'].new_zeros((batch,2*channels))
        else:
            with core.local_rng(seed,device):local,_=self.encoder(d)
            pooled=T.cat([local.mean(1),local.amax(1)],dim=1)
        if pathway in ['geometry_matched','rgb_encoder_only']:stats=pooled.new_zeros((batch,6))
        else:
            stats=T.cat([d['color'].mean(1)*2-1,(d['color']*2-1).amax(1)],dim=1)
        return T.cat([pooled,stats],dim=1)
    model.embedding=types.MethodType(embedding,model)
    return model,dict(provenance,followup_implementation=implementation_hash(),pathway=pathway,
        encoder_used=pathway!='stats_only',direct_colour_statistics=pathway not in ['geometry_matched','rgb_encoder_only'],
        matched_head_input_width=2*channels+6,stats='point RGB mean and maximum after 2*x-1',
        comparison='same 9-channel encoder architecture and same full-width head within each model/seed; unused pathways zeroed')

def install():
    core.surface_data=surface_data;core.build_model=build_model
