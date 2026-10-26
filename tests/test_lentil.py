"""Synthetic CPU contract checks, with zero optimiser updates or dataset reads."""
import sys,unittest,tempfile,json,subprocess
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lentil.common import validate_members,canonical_role,recipe_rows,contained_path
from lentil.preparation import official_mask,fit_preprocessing,project
from lentil.metrics import stitch_columns,count_frame,aggregate
from lentil.networks import make_model

class LentilContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)
    def test_published_role_mapping(self):
        self.assertEqual(canonical_role('FIT'),'FIT_PROPOSED')
        self.assertEqual(canonical_role('DAY4_EVALUATION'),'DAY4_HOLDOUT_PROPOSED')
        with self.assertRaises(ValueError):canonical_role('test_unknown')
    def test_group_leakage_and_path_escape_rejected(self):
        rows=[dict(source='data/day2/a.cu3s',index='0',group='g',role='FIT'),dict(source='data/day2/a.cu3s',index='1',group='g',role='DEV')]
        with self.assertRaises(ValueError):validate_members(rows,require_counts=False)
        with tempfile.TemporaryDirectory() as p:
            with self.assertRaises(ValueError):contained_path(Path(p),'../escape')
    def test_fixed_eight_recipes(self):
        rr=recipe_rows();self.assertEqual(len(rr),8)
        self.assertEqual({int(r['seed']) for r in rr},{42,43})
        self.assertTrue(all(r['updates']==6000 and r['effective_batch']==64 for r in rr))
    def test_cli_default_does_not_read_data_or_train(self):
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,str(root/'src/lentil_workflow.py'),'train','--config',str(root/'configs/lentil.example.json')],capture_output=True,text=True,check=True)
        data=json.loads(result.stdout);self.assertFalse(data['execute']);self.assertEqual(len(data['runs']),8)
    def test_patch_local_spatial_distance(self):
        from analyse_saved_spatial import bands
        x=np.zeros((2,20,16),bool);x[0,5,5]=True
        b=bands(x)
        self.assertEqual(int(b[0,5,6]),0);self.assertEqual(int(b[0,5,8]),1)
        self.assertEqual(int(b[0,5,9]),2);self.assertTrue((b[1]==3).all())
    def test_lentil_ID0_is_background(self):
        labels=np.zeros((1000,1080),np.uint8);labels[0,0]=1
        z=np.full(labels.shape,-1,np.float32);z[0,0]=1;z[4,4]=1
        row=count_frame(dict(source='synthetic',index='0',group='g',day='d',role='DEV',documented_normal_recording='False'),z,labels)
        self.assertEqual((row['TP'],row['FP'],row['FN'],row['BG']),(1,1,0,1079999))
        self.assertAlmostEqual(row['Dice'],2/3)
        normal=dict(row,group='normal',FG=0,positive=False,Dice=None,Recall=None,precision=None,IoU=None,AP=None)
        self.assertEqual(aggregate([row,normal])[0]['Dice'],row['Dice'])
    def test_polygon_decoder_full_canvas(self):
        coco={'annotations':[dict(image_id=0,category_id=7,segmentation=[[2,3,5,3,5,7,2,7]])]}
        y=official_mask(coco,0);self.assertEqual(y.shape,(1000,1080))
        self.assertEqual(set(np.unique(y)),{0,7});self.assertEqual(int(y[4,3]),7)
    def test_projection_and_overlap(self):
        x=np.random.default_rng(19).normal(size=(300,61)).astype(np.float32)
        prep=fit_preprocessing(x);z=project(x,prep)
        self.assertEqual(z.shape,(300,16));self.assertTrue(np.isfinite(z).all())
        self.assertTrue(np.allclose(prep['components']@prep['components'].T,np.eye(16),atol=1e-5))
        image=torch.arange(1080000,dtype=torch.float32).reshape(1,1,1000,1080)
        columns=torch.nn.functional.unfold(image,(20,16),stride=(10,8))
        self.assertTrue(torch.equal(stitch_columns(columns),image[0,0]))
    def test_initials_shapes_and_no_weight_update(self):
        b=make_model('B1',42);d=make_model('B1_ALL_D1',42)
        self.assertTrue(all(torch.equal(v,d.state_dict()[k]) for k,v in b.state_dict().items()))
        for family,n in [('B1',116961),('B1_ALL_D1',116961),('PCA16_UNET',120785),('PCA16_T_SAME_WIDTH',1638337)]:
            m=make_model(family,42).eval();self.assertEqual(sum(p.numel() for p in m.parameters()),n)
            before={k:v.clone() for k,v in m.state_dict().items()}
            with torch.no_grad():z=m(torch.zeros(1,16,20,16))
            self.assertEqual(z.shape,(1,20,16));self.assertTrue(torch.isfinite(z).all())
            self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()))

if __name__=='__main__':unittest.main()
