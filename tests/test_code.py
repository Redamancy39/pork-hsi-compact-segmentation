"""Synthetic CPU tests only. No model forward, fitting or data readback."""
from pathlib import Path
import sys,tempfile,unittest,json,math
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from models import create_model,source_metrics,masked_loss,FAMILIES,load_preprocessing,make_schedule
from labels import decode_label
from workflow import member_path

def synthetic_prep():
    return dict(mean=np.zeros(184),scale=np.ones(184),pca_mean=np.zeros(184),
                components=np.eye(184)[:16],explained_variance=np.ones(16))

class CodeChecks(unittest.TestCase):
    def test_class_ids_and_ignore(self):
        cfg={'original_id_to_rgb':{str(i):[i,i,i] for i in range(13)},
             'derived_foreground':list(range(10)),'derived_background':[10,11,12]}
        ids=np.array([[0,9,10,11,12,-1,255]])
        _,binary,_=decode_label(ids,cfg)
        self.assertEqual(binary.tolist(),[[1,1,0,0,0,-1,-1]])
        _,rgb_binary,_=decode_label(np.array([[[0,0,0],[10,10,10],[200,200,200]]]),cfg)
        self.assertEqual(rgb_binary.tolist(),[[1,0,-1]])

    def test_metrics_and_loss(self):
        r=source_metrics(np.array([[0,1,10,11,12,-1,255]]),np.array([[1,-1,0,2,-2,10,10]],np.float32))
        self.assertEqual([r[k] for k in ['TP','FP','FN','TN','ignore_pixels']],[1,2,1,1,2])
        self.assertEqual(r['dice'],.4);self.assertEqual(r['background_fp_rate'],2/3)
        loss,n=masked_loss(torch.zeros((1,1,3)),torch.tensor([[[1,0,-1]]]))
        self.assertEqual(n,1);self.assertAlmostEqual(float(loss),math.log(2)+1/3,places=6)

    def test_nine_model_definitions(self):
        expected=dict(B0=120785,B1=116961,T=1669249,RAW184_MLP=139265,
            PCA16_DS_RESIDUAL=17441,RAW184_UNET=144977,B1_ALL_D1=116961,
            B1_NO_SKIP=116961,PCA16_T_SAME_WIDTH=1638337)
        self.assertEqual(set(expected),set(FAMILIES))
        for f in FAMILIES:
            model=create_model(synthetic_prep(),f,42)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),expected[f],f)
            self.assertTrue(torch.equal(model.mean,torch.zeros((1,184,1,1))))

    def test_external_assets_and_path_boundary(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as td:
            root=Path(td);(root/'artifacts').mkdir()
            np.savez(root/'artifacts/preprocessing.npz',**synthetic_prep())
            self.assertEqual(load_preprocessing(root)['components'].shape,(16,184))
            (root/'marker.txt').write_text('synthetic')
            self.assertEqual(member_path(root,'marker.txt'),root/'marker.txt')
            with self.assertRaises(ValueError):member_path(root,'../outside.txt')

    def test_synthetic_sampling_reproducibility(self):
        groups={'a':np.array([0,1]),'b':np.array([2,3])}
        a,af=make_schedule(groups,42,updates=4,batch=8)
        b,bf=make_schedule(groups,42,updates=4,batch=8)
        self.assertTrue(np.array_equal(a,b));self.assertTrue(np.array_equal(af,bf))
        self.assertTrue(set(a.ravel()).issubset({0,1,2,3}))

if __name__=='__main__':unittest.main()
