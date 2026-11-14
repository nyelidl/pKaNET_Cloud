"""Regression tests for chemical identity and safe tautomer export."""
import ast,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from rdkit import Chem
import core
ROOT=Path(__file__).parent
INPUT=ROOT/'identity_fix_audit/fixtures/HCY_ideal.sdf'

def key(m):return Chem.MolToSmiles(Chem.RemoveHs(m),isomericSmiles=True)
class IdentityTests(unittest.TestCase):
 def test_hcy_enone_is_not_aromatic_loss(self):
  a=Chem.SDMolSupplier(str(INPUT),removeHs=False)[0]
  score,parts=core.score_tautomer_plausibility(key(a),a)
  self.assertNotIn('ring_carbonyl_onaromring_former',parts);self.assertEqual(score,0)
 def test_real_aromatic_loss_still_penalized(self):
  _,parts=core.score_tautomer_plausibility('O=C1C=CC=CC1',Chem.MolFromSmiles('Oc1ccccc1'))
  self.assertLess(parts['arom_ring_lost_vs_input'],0)
 def test_reference_retained_with_hard_candidate_cap(self):
  parent='CC=O'
  def score(smi,ref_mol=None):return (-100 if smi==parent else 10),{}
  with patch.object(core,'score_tautomer_plausibility',side_effect=score):
   kept,_,_,_=core.enumerate_and_filter_tautomers(parent,max_states=1)
  self.assertEqual([x['smiles'] for x in kept],[parent]);self.assertEqual(kept[0]['score'],-100)
 def test_hcy_default_ranking(self):
  reference=Chem.SDMolSupplier(str(INPUT),removeHs=False)[0]
  top,_,rows,*_=core.generate_ranked_microstates(key(reference),pubchem_result={})
  self.assertTrue(core._DIMORPHITE_OK)
  self.assertEqual(top[0]['microstate_smiles'],key(reference))
  self.assertTrue(top[0]['strict_input_identity'])
  self.assertTrue(any(r['tautomer_changed'] for r in rows))
 def test_protonation_distinguished_from_tautomer(self):
  self.assertTrue(core._same_heavy_bond_graph(Chem.MolFromSmiles('CCN'),Chem.MolFromSmiles('CC[NH3+]')))
  self.assertFalse(core._same_heavy_bond_graph(Chem.MolFromSmiles('CC=O'),Chem.MolFromSmiles('C=CO')))
 def test_stereo_inversion_not_protonation(self):
  self.assertFalse(core._same_heavy_bond_graph(Chem.MolFromSmiles('C[C@H](F)O'),Chem.MolFromSmiles('C[C@@H](F)O')))
 def test_chiral_protonation_preserves_heavy_graph(self):
  self.assertTrue(core._same_heavy_bond_graph(Chem.MolFromSmiles('C[C@H](O)CN'),Chem.MolFromSmiles('C[C@H](O)C[NH3+]')))
 def test_3d_reference_identity(self):
  a=Chem.SDMolSupplier(str(INPUT),removeHs=False)[0];b=core.build_minimized_3d(key(a));core._require_identity(a,b,'test')
 def test_sdf_serialization_identity(self):
  a=core.build_minimized_3d('C[C@H](O)CC')
  with tempfile.TemporaryDirectory() as t:
   files=core.save_molecule_files(a,str(Path(t)/'test'),['SDF'])
   core._require_identity(a,Chem.SDMolSupplier(files['sdf'],removeHs=False)[0],'test readback')
 def test_selected_zip_tracks_actual_rank_and_includes_sdf(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t);p=d/'test_micro9_min.sdf';p.write_text('fixture')
   (d/'test_micro1_min.sdf').write_text('wrong fixture')
   (d/'identity_selection.json').write_text('[]')
   core.zip_selected_structures(t,str(d/'selected.zip'),['SDF'],[dict(export_status='PASS',minimized_sdf=str(p))])
   with zipfile.ZipFile(d/'selected.zip') as z:self.assertEqual(set(z.namelist()),{p.name,'identity_selection.json'})
 def test_app_wires_explicit_opt_in(self):
  source=(ROOT/'app.py').read_text();ast.parse(source)
  self.assertIn('"Allow alternative-tautomer exports", value=False',source)
  self.assertIn('allow_tautomer_changes = allow_tautomer_changes,',source)
  self.assertIn('zip_selected_structures(str(out_dir_path)',source)
 def test_export_policy_even_when_rank_one_is_an_alternative(self):
  # Deterministic selection-policy fixture; the HCY integration uses real ranking.
  ref='CC=O';alt='C=CO'
  _,_,rows,*_=core.generate_ranked_microstates(ref,pubchem_result={})
  refrow=next(r for r in rows if r['microstate_smiles']==ref)
  altrow=next(r for r in rows if r['microstate_smiles']==alt)
  altrow=dict(altrow,microstate_rank=1);refrow=dict(refrow,microstate_rank=9)
  fake=([altrow],True,[altrow,refrow],True,[],[])
  with tempfile.TemporaryDirectory() as t,patch.object(core,'generate_ranked_microstates',return_value=fake):
   result=core.run_job(input_type='SMILES',smiles_text=ref,uploaded_bytes=None,uploaded_name=None,target_pH=7.4,output_name='test',out_dir=t,output_formats=['SDF'],enumerate_stereoisomers=False,use_pubchem=False,write_alt_3d_for_top_k=1)
   r=result['results'][0];self.assertEqual(r['selected_microstate_smiles'],ref);self.assertEqual(r['selected_rank'],9)
   self.assertEqual(key(Chem.SDMolSupplier(r['minimized_sdf'],removeHs=False)[0]),ref)
  with tempfile.TemporaryDirectory() as t,patch.object(core,'generate_ranked_microstates',return_value=fake):
   result=core.run_job(input_type='SMILES',smiles_text=ref,uploaded_bytes=None,uploaded_name=None,target_pH=7.4,output_name='test',out_dir=t,output_formats=['SDF'],enumerate_stereoisomers=False,use_pubchem=False,write_alt_3d_for_top_k=1,allow_tautomer_changes=True)
   r=result['results'][0];self.assertEqual(r['selected_microstate_smiles'],alt);self.assertTrue(r['tautomer_changed']);self.assertTrue(r['allow_tautomer_changes'])
 def test_failed_primary_is_not_replaced_by_alternative(self):
  ref='CC=O';alt='C=CO';_,_,rows,*_=core.generate_ranked_microstates(ref,pubchem_result={})
  first=dict(next(r for r in rows if r['microstate_smiles']==ref),microstate_rank=1)
  second=dict(next(r for r in rows if r['microstate_smiles']==alt),microstate_rank=2)
  real=core.build_minimized_3d
  def build(smiles):
   if smiles==ref:raise ValueError('simulated primary failure')
   return real(smiles)
  with tempfile.TemporaryDirectory() as t,patch.object(core,'generate_ranked_microstates',return_value=([first,second],True,[first,second],True,[],[])),patch.object(core,'build_minimized_3d',side_effect=build):
   r=core.run_job(input_type='SMILES',smiles_text=ref,uploaded_bytes=None,uploaded_name=None,target_pH=7.4,output_name='test',out_dir=t,output_formats=['SDF'],enumerate_stereoisomers=False,use_pubchem=False,write_alt_3d_for_top_k=2,allow_tautomer_changes=True)['results'][0]
   self.assertIsNone(r['minimized_sdf']);self.assertEqual(r['export_status'],'BLOCKED_3D_EXPORT');self.assertEqual(r['alt3d'][0]['smiles'],alt)
if __name__=='__main__':unittest.main(verbosity=2)
