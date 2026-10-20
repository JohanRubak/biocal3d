import importlib.util, unittest
import numpy as np
import followup_models as fm
@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch unavailable here; run on training machine')
class TorchRoutingTests(unittest.TestCase):
    def test_pathway_information_and_gradient_routing(self):
        import torch
        fm.core.T=torch
        class Encoder(torch.nn.Module):
            def __init__(self):super().__init__();self.channels=4;self.layer=torch.nn.Linear(9,4)
            def forward(self,d):return self.layer(torch.cat([d['coord'],d['normal'],d['color']],-1)),d['coord']
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.encoder=Encoder();self.frozen=False;self.head=torch.nn.Linear(14,3)
            def forward(self,d,seed):return self.head(self.embedding(d,seed))
        original=fm.ORIGINAL_BUILD;fm.ORIGINAL_BUILD=lambda *args:(Model(),{});d={k:torch.rand(1,32,3) for k in ['coord','normal','color']}
        try:
            for pathway in fm.PATHWAYS:
                model,_=fm.build_model({'surface':{'pathway':pathway}},None,torch.device('cpu'));emb=model.embedding(d,123);self.assertEqual(tuple(emb.shape),(1,14))
                if pathway=='stats_only':
                    np.testing.assert_array_equal(emb.detach().numpy()[0,:8],0);model(d,123).sum().backward();self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
                if pathway in ['geometry_matched','rgb_encoder_only']:np.testing.assert_array_equal(emb.detach().numpy()[0,-6:],0)
                if pathway=='rgb_combined':np.testing.assert_allclose(emb.detach().numpy()[0,-6:],fm.colour_statistics_numpy(d['color'][0].numpy()),atol=1e-6)
        finally:fm.ORIGINAL_BUILD=original

if __name__=='__main__':unittest.main()
