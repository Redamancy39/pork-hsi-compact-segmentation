"""Protocol checks with synthetic metadata; no model forwards or CUDA calls."""
from pathlib import Path
import csv, json, subprocess, sys, tempfile, unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from file_timing import protocol_conditions, read_manifest, resolve_member, validate_config

class FileTimingChecks(unittest.TestCase):
    def test_both_blocks_cover_all_conditions_and_roles(self):
        seen=set()
        for block in ['A','B']:
            rows=protocol_conditions(block)
            self.assertEqual(len(rows),63)
            self.assertEqual(sum(r['role']=='MEASURED' for r in rows),27)
            self.assertEqual(sum(r['role']=='WARMUP' for r in rows),27)
            self.assertEqual(sum(r['role']=='FIRST_USE' for r in rows),9)
            families={r['family'] for r in rows};seen|=families
            for family in families:
                for batch in [1,8,64]:
                    rs=[r for r in rows if r['family']==family and r['batch']==batch]
                    self.assertEqual([r['role'] for r in rs],['FIRST_USE','WARMUP','MEASURED','WARMUP','MEASURED','WARMUP','MEASURED'])
            for batch in [1,8,64]:
                self.assertEqual(len({r['family'] for r in rows if r['batch']==batch and r['repeat']==0}),3)
        self.assertEqual(seen,{'B0','B1','T','PCA16_T_SAME_WIDTH','PCA16_DS_RESIDUAL','B1_ALL_D1'})
        with self.assertRaises(ValueError):protocol_conditions('ALL')

    def test_fit_manifest_and_path_safety(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as td:
            root=Path(td);manifest=root/'timing.csv'
            def save(role='FIT',duplicate=False):
                with manifest.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['role','image_member']);w.writeheader()
                    w.writerows({'role':role,'image_member':f'patch_{0 if duplicate else i}.npy'} for i in range(128))
            save();self.assertEqual(len(read_manifest(manifest)),128)
            save('DEV')
            with self.assertRaises(ValueError):read_manifest(manifest)
            save(duplicate=True)
            with self.assertRaises(ValueError):read_manifest(manifest)
            (root/'patch.npy').write_bytes(b'synthetic path marker')
            self.assertEqual(resolve_member([root],'patch.npy'),root/'patch.npy')
            for member in ['../outside.npy','C:/outside.npy','/outside.npy']:
                with self.assertRaises(ValueError):resolve_member([root],member)

    def test_default_cli_only_prints_plan(self):
        example=ROOT/'configs/file_timing.example.json'
        cfg=json.loads(example.read_text());validate_config(cfg)
        bad=dict(cfg,seed=43)
        with self.assertRaises(ValueError):validate_config(bad)
        cp=subprocess.run([sys.executable,'-B',str(ROOT/'src/file_timing.py'),'--config',str(example)],capture_output=True,text=True)
        self.assertEqual(cp.returncode,0,cp.stderr)
        result=json.loads(cp.stdout)
        self.assertFalse(result['executed'])
        self.assertEqual(result['file_passes'],63)
        self.assertEqual(result['formal_passes'],27)
        self.assertIn('separate three-model blocks',result['scope'])

if __name__=='__main__':unittest.main()
