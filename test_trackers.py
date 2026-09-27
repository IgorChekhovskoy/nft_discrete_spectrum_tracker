"""Regression tests for the numerical functions in ds_tracker.ipynb.

Run: python -m unittest -v test_trackers
Only imports, constants and function definitions are loaded. No dataset is
opened and no plot is displayed on import; the real-data test reads its fixture.
"""
from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def load_kernels(notebook_path=None):
    notebook = json.loads(Path(notebook_path or ROOT / 'ds_tracker.ipynb').read_text(encoding='utf-8'))
    kernels = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'def run_discrete_tracker' not in source:
            continue
        stop = '# === Run ===' if 'def run_discrete_tracker(' in source else '# --- Run tracking and get groups ---'
        body = []
        for node in ast.parse(source.split(stop)[0]).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if not (isinstance(node, ast.Import) and any(a.name.startswith('plotly') for a in node.names)):
                    body.append(node)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                body.append(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                def base(target):
                    while isinstance(target, ast.Subscript):
                        target = target.value
                    return target.id if isinstance(target, ast.Name) else None
                if all(base(t) not in {'df', 'unique_z', 'FILENAME', 'COLS'} for t in targets):
                    body.append(node)
        ns = {}
        exec(compile(ast.Module(body=body, type_ignores=[]), str(notebook_path or 'ds_tracker.ipynb'), 'exec'), ns)
        kernels.append(ns)
    if len(kernels) != 2:
        raise ValueError('Expected two self-contained tracker cells')
    return kernels


def make_frame(z, real, r=None, imag=None, r_imag=None):
    z = np.asarray(z, float)
    real = np.asarray(real, float)
    r = np.zeros(len(z)) if r is None else np.asarray(r, float)
    imag = np.ones(len(z)) if imag is None else np.asarray(imag, float)
    r_imag = np.zeros(len(z)) if r_imag is None else np.asarray(r_imag, float)
    return pd.DataFrame(dict(z=z, zeta_real=real, zeta_imag=imag, Re_r=r, Im_r=r_imag,
                             abs_r=np.hypot(r, r_imag)))


def partition(groups):
    return sorted(tuple(sorted(tuple(map(float, p)) for p in pts)) for pts in groups.values())


def observations(df):
    cols = ['z', 'zeta_real', 'zeta_imag', 'Re_r', 'Im_r', 'abs_r']
    return Counter(tuple(map(float, p)) for p in df[cols].to_numpy())


def synthetic_case(seed=0, kind='crossing', size=80):
    """Known identities are metadata, never tracker inputs. z is always explicit."""
    rng = np.random.default_rng(seed)
    z = np.arange(size, dtype=float)
    if kind == 'irregular':
        z = np.concatenate([[0.], np.cumsum(rng.uniform(.4, 1.6, size - 1))])
    rows = []
    for frame, t in enumerate(z):
        for identity, direction in enumerate([1., -1.]):
            if kind == 'births' and (identity == 1 and (frame < 12 or frame > size - 12)):
                continue
            if kind == 'gaps' and 15 <= frame <= 17:
                continue
            if kind == 'gaps' and frame > 2 and rng.random() < .08:
                continue
            real = direction * (-.06 + .0015 * t)
            imag = .055 + .003 * identity
            if kind == 'curved':
                real += .008 * np.sin(.09 * t + identity)
                imag += .003 * np.sin(.11 * t + identity)
            rr = .03 * direction + .002 * np.cos(.07 * t)
            ri = .005 * np.sin(.04 * t + identity)
            noise = rng.normal(0, 2e-5, 4)
            vals = np.array([real, imag, rr, ri]) + noise
            rows.append((t, *vals, np.hypot(vals[2], vals[3]), identity))
        if kind == 'clutter' and frame % 9 == 0:
            vals = [rng.uniform(-.1, .1), .085, rng.uniform(-.07, .07), .025]
            rows.append((t, *vals, np.hypot(vals[2], vals[3]), -frame-1))
    df = pd.DataFrame(rows, columns=['z','zeta_real','zeta_imag','Re_r','Im_r','abs_r','truth'])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True), z


def synthetic_stress_case(seed, kind='oscillating_r', size=100):
    rng=np.random.default_rng(seed)
    z=np.r_[0., np.cumsum(rng.uniform(.7,1.3,size-1))]
    rows=[]
    for f,t in enumerate(z):
        for k,d in enumerate([1.,-1.,.2]):
            if kind=='gaps_clutter' and ((22<=f<=24) or (f>3 and rng.random()<.07)):
                continue
            if kind=='gaps_clutter' and k==2 and (f<17 or f>=86):
                continue
            x=d*(-.085+.0019*t)+.002*np.sin(.16*t+k)
            y=.06+.005*k+.0015*np.cos(.14*t+k)
            theta=(.12 if kind!='fast_phase' else .45)*t+k*2
            amp=.018+.01*k+.004*np.sin(.09*t+k)
            rr,ri=amp*np.cos(theta),amp*np.sin(theta)
            if kind=='ambiguous':
                x=d*(-.07+.0014*t); y=.06; rr=ri=0.
            noise=rng.normal(0, .00010 if kind!='ambiguous' else .00055,4)
            a=np.array([x,y,rr,ri])+noise
            rows.append((t,*a,np.hypot(a[2],a[3]),k))
        if kind=='gaps_clutter' and f%7==0:
            a=[rng.uniform(-.1,.1),rng.uniform(.02,.09),rng.uniform(-.05,.05),rng.uniform(-.05,.05)]
            rows.append((t,*a,np.hypot(a[2],a[3]),-f-1))
    df=pd.DataFrame(rows,columns=['z','zeta_real','zeta_imag','Re_r','Im_r','abs_r','truth'])
    return df.sample(frac=1,random_state=seed).reset_index(drop=True),z


def association_metrics(df, groups):
    cols = ['z', 'zeta_real', 'zeta_imag', 'Re_r', 'Im_r', 'abs_r']
    identity = {tuple(map(float, row[:6])): int(row[6]) for row in df[cols + ['truth']].to_numpy()}
    expected = set()
    for label, part in df.groupby('truth'):
        if label < 0:
            continue
        pts = [tuple(map(float, p)) for p in part.sort_values('z')[cols].to_numpy()]
        expected.update(zip(pts, pts[1:]))
    predicted = set()
    switches = 0
    for pts in groups.values():
        pts = sorted(tuple(map(float, p)) for p in pts)
        predicted.update(zip(pts, pts[1:]))
        switches += sum(identity[a] != identity[b] for a, b in zip(pts, pts[1:]))
    true = len(expected & predicted)
    return dict(correct_links=true, predicted_links=len(predicted), expected_links=len(expected),
                precision=true / len(predicted) if predicted else 1.,
                recall=true / len(expected) if expected else 1., id_switches=switches,
                tracks=len(groups), raw_points=sum(map(len, groups.values())))


def benchmark_synthetic(notebook_path=None, seeds=range(1200, 1220)):
    """Evaluate both trackers on labeled synthetic trajectories.

    Parameters come from the notebook example calls, with birth_threshold=0.01
    for every synthetic case. notebook_path selects the notebook to evaluate.
    Truth labels are used only for scoring.
    """
    path=Path(notebook_path or ROOT/'ds_tracker.ipynb')
    kernels=load_kernels(path)
    notebook=json.loads(path.read_text(encoding='utf-8'))
    cells=[c for c in notebook['cells'] if c['cell_type']=='code'
           and 'def run_discrete_tracker' in ''.join(c['source'])]
    parameters=[]
    for cell,ns in zip(cells,kernels):
        source=''.join(cell['source'])
        name='run_discrete_tracker_simple' if 'run_discrete_tracker_simple' in ns else 'run_discrete_tracker'
        call=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Name) and n.func.id==name)
        kw={k.arg:eval(compile(ast.Expression(body=k.value),'<notebook-parameter>','eval'),ns)
            for k in call.keywords}
        kw['birth_threshold']=.01
        parameters.append(kw)
    seeds=tuple(seeds)
    results=[]
    basic=('crossing','curved','irregular','gaps','clutter','births')
    difficult=('oscillating_r','fast_phase','gaps_clutter','ambiguous')
    for kind in basic+difficult:
        generator=synthetic_case if kind in basic else synthetic_stress_case
        for index,name in enumerate(('simple','kalman')):
            total=Counter()
            for seed in seeds:
                frame,z=generator(seed,kind)
                if name=='simple':
                    groups=kernels[index]['run_discrete_tracker_simple'](frame,z,**parameters[index])
                else:
                    groups,_=kernels[index]['run_discrete_tracker'](frame,z,**parameters[index])
                metrics=association_metrics(frame,groups)
                total.update({k:metrics[k] for k in ('correct_links','expected_links','predicted_links',
                                                   'id_switches','tracks','raw_points')})
            row=dict(case=kind,method=name,runs=len(seeds),**total)
            row['precision']=total['correct_links']/total['predicted_links'] if total['predicted_links'] else 1.
            row['recall']=total['correct_links']/total['expected_links'] if total['expected_links'] else 1.
            results.append(row)
    return results


class TrackersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simple, cls.kalman = load_kernels()

    def both(self, frame, levels, simple=None, kalman=None):
        return [self.simple['run_discrete_tracker_simple'](frame, levels, **(simple or {})),
                self.kalman['run_discrete_tracker'](frame, levels, **(kalman or {}))[0]]

    def assert_raw(self, frame, groups):
        self.assertEqual(observations(frame), Counter(tuple(p) for g in groups.values() for p in g))
        for pts in groups.values():
            self.assertTrue(np.all(np.diff([p[0] for p in pts]) > 0))

    def test_empty_input(self):
        for groups in self.both(make_frame([], []), []):
            self.assertEqual(groups, {})

    def test_empty_frames_only(self):
        for groups in self.both(make_frame([], []), np.arange(4)):
            self.assertEqual(groups, {})

    def test_leading_empty_frame_does_not_age_birth(self):
        for groups in self.both(make_frame([1,2,3], [0,0,0]), np.arange(4), {'max_gap':1}, {'max_gap':1}):
            self.assertEqual(sorted(map(len,groups.values())), [3])

    def test_single_observation_is_not_discarded(self):
        f = make_frame([2],[.2])
        for g in self.both(f,np.arange(5)):
            self.assert_raw(f,g)
            self.assertEqual(list(map(len,g.values())),[1])

    def test_max_gap_is_inclusive(self):
        f = make_frame([0,1,6],[0,0,0])
        for g in self.both(f,np.arange(7), {'max_gap':4}, {'max_gap':4}):
            self.assertEqual(list(map(len,g.values())),[3])

    def test_expired_track_is_not_resurrected(self):
        for g in self.both(make_frame([0,1,7],[0,0,0]),np.arange(8),{'max_gap':4},{'max_gap':4}):
            self.assertEqual(sorted(map(len,g.values())),[1,2])

    def test_zero_gap_allows_adjacent_frames(self):
        for g in self.both(make_frame([0,1,2],[0,0,0]),np.arange(3),{'max_gap':0},{'max_gap':0}):
            self.assertEqual(list(map(len,g.values())),[3])

    def test_zero_gap_breaks_at_empty_frame(self):
        for g in self.both(make_frame([0,1,3],[0,0,0]),np.arange(4),{'max_gap':0},{'max_gap':0}):
            self.assertEqual(sorted(map(len,g.values())),[1,2])

    def test_disabled_r_channel_does_not_gate(self):
        for g in self.both(make_frame([0,1,2],[0,0,0],[0,1,2]),np.arange(3),{'weight_r':0},{'meas_w_r':0}):
            self.assertEqual(list(map(len,g.values())),[3])

    def test_disabled_zeta_channel_does_not_gate(self):
        f = make_frame([0,1,2],[0,100,-100])
        g,_=self.kalman['run_discrete_tracker'](f,np.arange(3),meas_w_zeta=0)
        self.assertEqual(list(map(len,g.values())),[3])

    def test_stationary_simple_track_cannot_teleport(self):
        f = make_frame([0,1,2],[0,0,100])
        g=self.simple['run_discrete_tracker_simple'](f,np.arange(3))
        self.assertEqual(sorted(map(len,g.values())),[1,2])

    def test_kalman_amplitude_startup_uses_uncertainty(self):
        z=np.arange(5.)
        f=make_frame(z,.0005*z, .004*z)
        g,_=self.kalman['run_discrete_tracker'](f,z)
        self.assertEqual(list(map(len,g.values())),[5])

    def test_newborn_eigenvalue_cannot_jump_despite_large_covariance(self):
        f=make_frame([0.,1.],[0.,.02])
        g,_=self.kalman['run_discrete_tracker'](f,[0.,1.],P0_vel_scale=100.,birth_threshold=.001)
        self.assertEqual(sorted(map(len,g.values())),[1,1])

    def test_simple_first_velocity_is_not_damped(self):
        z=np.arange(8.)
        f=make_frame(z,.0008*z)
        g=self.simple['run_discrete_tracker_simple'](f,z,K=1,alpha=.01)
        self.assertEqual(list(map(len,g.values())),[8])

    def test_no_raw_points_are_lost_or_invented(self):
        f,z=synthetic_case(17,'gaps')
        for g in self.both(f,z,{'birth_threshold':.01},{'birth_threshold':.01}):
            self.assert_raw(f,g)

    def test_duplicate_detections_are_preserved(self):
        f=make_frame([0,0,1,1],[0,0,0,0])
        for g in self.both(f,[0,1]):
            self.assert_raw(f,g)
            self.assertEqual(sorted(map(len,g.values())),[2,2])

    def test_input_rows_can_be_shuffled(self):
        f,z=synthetic_case(22,'crossing',40)
        a=self.both(f,z,{'birth_threshold':.01},{'birth_threshold':.01})
        b=self.both(f.sample(frac=1,random_state=1),z,{'birth_threshold':.01},{'birth_threshold':.01})
        for x,y in zip(a,b): self.assertEqual(partition(x),partition(y))

    def test_inputs_and_global_covariances_are_not_mutated(self):
        f,z=synthetic_case(23,'crossing',20)
        old=f.copy(deep=True)
        matrices={n:self.kalman[n].copy() for n in ('R_init','Q_init','P0','H')}
        self.both(f,z,{'birth_threshold':.01},{'birth_threshold':.01})
        pd.testing.assert_frame_equal(f,old)
        for n,v in matrices.items(): np.testing.assert_array_equal(self.kalman[n],v)

    def test_repeated_call_is_identical(self):
        f,z=synthetic_case(24,'crossing',20)
        a=self.both(f,z,{'birth_threshold':.01},{'birth_threshold':.01})
        b=self.both(f,z,{'birth_threshold':.01},{'birth_threshold':.01})
        self.assertEqual([partition(g) for g in a],[partition(g) for g in b])

    def test_appending_future_data_does_not_change_prefix(self):
        f,z=synthetic_case(25,'crossing',30)
        short=f[f.z<15].copy()
        a=self.both(short,z[:15],{'birth_threshold':.01},{'birth_threshold':.01,'inv_reg':1e-5})
        b=self.both(f,z,{'birth_threshold':.01},{'birth_threshold':.01,'inv_reg':1e-5})
        for x,y in zip(a,b):
            cropped={k:[p for p in v if p[0]<15] for k,v in y.items()}
            cropped={k:v for k,v in cropped.items() if v}
            self.assertEqual(partition(x),partition(cropped))

    def test_unsorted_frames_are_rejected(self):
        for ns,name in [(self.simple,'run_discrete_tracker_simple'),(self.kalman,'run_discrete_tracker')]:
            with self.subTest(method=name),self.assertRaises(ValueError): ns[name](make_frame([0,1],[0,0]),[1,0])

    def test_duplicate_frames_are_rejected(self):
        for ns,name in [(self.simple,'run_discrete_tracker_simple'),(self.kalman,'run_discrete_tracker')]:
            with self.subTest(method=name),self.assertRaises(ValueError): ns[name](make_frame([0],[0]),[0,0])

    def test_missing_frame_coordinates_are_rejected(self):
        for ns,name in [(self.simple,'run_discrete_tracker_simple'),(self.kalman,'run_discrete_tracker')]:
            with self.subTest(method=name),self.assertRaises(ValueError): ns[name](make_frame([0,1],[0,0]),[0])

    def test_nonfinite_observations_are_rejected(self):
        for value in (np.nan,np.inf):
            for ns,name in [(self.simple,'run_discrete_tracker_simple'),(self.kalman,'run_discrete_tracker')]:
                with self.subTest(method=name,value=value),self.assertRaises(ValueError): ns[name](make_frame([0],[value]),[0])

    def test_missing_columns_are_rejected(self):
        for ns,name in [(self.simple,'run_discrete_tracker_simple'),(self.kalman,'run_discrete_tracker')]:
            with self.subTest(method=name),self.assertRaises(ValueError): ns[name](pd.DataFrame({'z':[0]}),[0])

    def test_invalid_parameters_are_rejected(self):
        f=make_frame([0],[0])
        for method,ns,name,options in [
            ('simple',self.simple,'run_discrete_tracker_simple',[{'max_gap':-1},{'max_gap':1.5},{'alpha':2},{'weight_r':-1},{'birth_threshold':0},{'K':0}]),
            ('kalman',self.kalman,'run_discrete_tracker',[{'max_gap':-1},{'birth_confirm':0},{'gate_p':1},{'alpha_R':2},{'R_scale_r':0},{'meas_w_zeta':0,'meas_w_r':0}])]:
            for kw in options:
                with self.subTest(method=method,kw=kw),self.assertRaises(ValueError): ns[name](f,[0],**kw)

    def test_optional_assignment_can_reject_a_real_pair(self):
        for ns in (self.simple,self.kalman):
            self.assertEqual(ns['_tracking_assignment'](np.array([[.1,100.],[.2,100.]]),.5),[(0,0)])

    def test_all_gated_pairs_can_be_unmatched(self):
        for ns in (self.simple,self.kalman):
            self.assertEqual(ns['_tracking_assignment'](np.full((2,3),np.inf),1),[])

    def test_covariance_prediction_composes_over_irregular_steps(self):
        rng=np.random.default_rng(37)
        B=rng.normal(size=(8,8)); P=B@B.T
        B=rng.normal(size=(8,8)); Q=B@B.T
        x=rng.normal(size=8)
        predict=self.kalman['kf_predict']
        xa,Pa=predict(x,P,Q,2.3)
        xb,Pb=predict(x,P,Q,.7); xb,Pb=predict(xb,Pb,Q,1.6)
        np.testing.assert_allclose(xa,xb,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(Pa,Pb,rtol=1e-12,atol=1e-12)

    def test_zero_step_does_not_add_process_noise(self):
        x=np.arange(8.); P=np.eye(8)
        xx,PP=self.kalman['kf_predict'](x,P,np.eye(8),0)
        np.testing.assert_array_equal(xx,x); np.testing.assert_array_equal(PP,P)

    def test_negative_step_is_rejected(self):
        with self.assertRaises(ValueError): self.kalman['make_F'](-1)

    def test_R_adaptation_subtracts_prediction_variance(self):
        R=np.eye(4); S=np.eye(4)*10
        out=self.kalman['adapt_R'](R,np.ones(4)*np.sqrt(10),S,alpha=1)
        np.testing.assert_allclose(out,R,atol=1e-12)

    def test_zero_adaptation_keeps_noise_matrices(self):
        R=np.eye(4); Q=np.eye(8)
        np.testing.assert_array_equal(self.kalman['adapt_R'](R,np.ones(4),R*3,alpha=0),R)
        out,_=self.kalman['adapt_Q'](Q,np.zeros(4),np.ones(4),1,alpha=0)
        np.testing.assert_array_equal(out,Q)

    def test_Joseph_update_keeps_covariance_positive(self):
        rng=np.random.default_rng(44)
        x=np.zeros(8); P=np.eye(8); Q=np.eye(8)*1e-5; R=np.eye(4)*1e-3
        for _ in range(60):
            x,P=self.kalman['kf_predict'](x,P,Q,rng.uniform(.1,2))
            x,P,_,_=self.kalman['kf_update'](x,P,rng.normal(size=4),R)
            self.assertGreaterEqual(np.linalg.eigvalsh(P).min(),-1e-12)
            np.testing.assert_allclose(P,P.T,atol=1e-12)

    def test_synthetic_crossing_has_known_identity(self):
        f,z=synthetic_case(130,'crossing')
        for g in self.both(f,z,{'birth_threshold':.01},{'birth_threshold':.01}):
            self.assert_raw(f,g)
            m=association_metrics(f,g)
            self.assertEqual(m['id_switches'],0)
            self.assertGreaterEqual(m['recall'],.98)

    def test_synthetic_irregular_frames(self):
        f,z=synthetic_case(131,'irregular')
        for g in self.both(f,z,{'birth_threshold':.01},{'birth_threshold':.01}):
            self.assert_raw(f,g)
            self.assertEqual(association_metrics(f,g)['id_switches'],0)

    def test_synthetic_missed_frames(self):
        f,z=synthetic_case(132,'gaps')
        for g in self.both(f,z,{'birth_threshold':.01},{'birth_threshold':.01}):
            self.assert_raw(f,g)
            m=association_metrics(f,g)
            self.assertEqual(m['id_switches'],0)
            self.assertGreaterEqual(m['recall'],.95)

    def test_simple_tracking_is_independent_of_z_units(self):
        z=np.arange(8.)*1e-16
        f=make_frame(z,.0008*np.arange(8.))
        g=self.simple['run_discrete_tracker_simple'](f,z)
        self.assertEqual(list(map(len,g.values())),[8])

    def test_equal_eigenvalues_are_distinguished_by_discrete_amplitudes(self):
        rows=[]
        z=np.arange(11.)
        for t in z:
            for identity, direction in enumerate([1.,-1.]):
                rows.append((t,direction*.0005*(t-5),.05,direction*.03,0.,.03,identity))
        f=pd.DataFrame(rows,columns=['z','zeta_real','zeta_imag','Re_r','Im_r','abs_r','truth'])
        for g in self.both(f,z,{'birth_threshold':.01},{'birth_threshold':.01}):
            self.assert_raw(f,g)
            self.assertEqual(association_metrics(f,g)['id_switches'],0)
            self.assertEqual(association_metrics(f,g)['recall'],1.)

    def test_kalman_fast_complex_amplitude_rotation(self):
        f,z=synthetic_stress_case(381,'fast_phase')
        g,_=self.kalman['run_discrete_tracker'](f,z,birth_threshold=.01)
        self.assert_raw(f,g)
        m=association_metrics(f,g)
        self.assertGreaterEqual(m['precision'],.99)
        self.assertGreaterEqual(m['recall'],.95)

    def test_kalman_gaps_births_and_clutter(self):
        f,z=synthetic_stress_case(382,'gaps_clutter')
        g,_=self.kalman['run_discrete_tracker'](f,z,birth_threshold=.01)
        self.assert_raw(f,g)
        m=association_metrics(f,g)
        self.assertGreaterEqual(m['precision'],.98)
        self.assertGreaterEqual(m['recall'],.95)

    def test_tentative_timeout_preserves_raw_detections(self):
        f=make_frame([0,4],[0,0])
        g,_=self.kalman['run_discrete_tracker'](f,np.arange(5.),max_gap=10,birth_confirm=2)
        self.assert_raw(f,g)
        self.assertEqual(sorted(map(len,g.values())),[1,1])

    def test_real_data_preserves_all_observations(self):
        path=ROOT/'NFT_DiscreteSpectrum.dat'
        if not path.exists(): self.skipTest('Example data file is not installed')
        cols=['z','x','h','Re_r','Im_r','T','f','abs_r','abs_a','abs_b','abs_aprime','ds_count','energyRelativeError']
        f=pd.read_csv(path,sep=r'\s+',skiprows=1,names=cols)
        f['zeta_real']=f.x; f['zeta_imag']=f.h
        for g in self.both(f,np.sort(f.z.unique())):
            self.assert_raw(f,g)


if __name__=='__main__':
    unittest.main()
