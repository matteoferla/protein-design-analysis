"""
Common pyrosetta operations
"""
import numpy as np
import pyrosetta
import pyrosetta_help as ph
from types import ModuleType
from typing import TypeAlias, Dict, Any, Tuple, List, Optional
prc: ModuleType = pyrosetta.rosetta.core
prp: ModuleType = pyrosetta.rosetta.protocols
pru: ModuleType = pyrosetta.rosetta.utility  # noqa
prn: ModuleType = pyrosetta.rosetta.numeric
prs: ModuleType = pyrosetta.rosetta.std  # noqa
pr_conf: ModuleType = pyrosetta.rosetta.core.conformation
pr_scoring: ModuleType = pyrosetta.rosetta.core.scoring
pr_options: ModuleType = pyrosetta.rosetta.basic.options
pr_res: ModuleType = pyrosetta.rosetta.core.select.residue_selector
FTypeIdx: TypeAlias = int  # one-based index
CTypeIdx: TypeAlias = int  # zero-based index

def init_pyrosetta(detect_disulf=False):
    """
    Initialize pyrosetta with the given options
    """
    logger = ph.configure_logger()
    pyrosetta.distributed.maybe_init(extra_options=ph.make_option_string(no_optH=False,
                                                                         ex1=None,
                                                                         ex2=None,
                                                                         # mute='all',
                                                                         ignore_unrecognized_res=True,
                                                                         load_PDB_components=False,
                                                                         ignore_waters=True,
                                                                         )
                                     )
    pr_options.set_boolean_option('in:detect_disulf', detect_disulf)
    return logger

def get_correct_ref(trb, template: pyrosetta.Pose) -> List[int]:
    """
    The idx0 numbering in 'con_ref_idx0' is the sequential numbering of the reference parts
    of the template in rfdiffusion used. Therefore, where a deletion to be present
    the original will actually have extra residues relative to 'reference'.
    """
    pi = template.pdb_info()
    return [pi.pdb2pose(res=pidx, chain=chain) - 1 for chain, pidx in trb['con_ref_pdb_idx']]


def steal_frozen_sidechains(acceptor: pyrosetta.Pose,
                 donor: pyrosetta.Pose, trb: Dict[str, Any],
                 move_acceptor: bool = False
                 ):
    """
    Copy all the conserved coordinates from the donor to the acceptor.
    These are determined by `trb` dict from RFdiffusion.

    The hallucination is the acceptor, the parent is the donor.
    The RFdiffused pose is skeleton, but when imported the sidechains are added.
    The theft is done in 3 steps.

    1. A mapping of residue idx, atom idx to residue idx, atom idx is made.
    2. The hallucination is superimposed on the parent if move_acceptor is True, else vice versa.
    3. The coordinates are copied from the parent to the hallucination.

    The term 'ref' gets confusing.
     hallucination is fixed / mutanda, parent is mobile
    fixed is called ref, but ref means parent for RFdiffusion so it is flipped)


    :param acceptor:
    :param donor:
    :param trb:
    :param move_acceptor:
    :return:
    """

    # ## Make mapping of all atoms of conserved residues
    donor2acceptor_idx1s: Dict[Tuple[FTypeIdx, FTypeIdx], Tuple[FTypeIdx, FTypeIdx, str]] = {}  # noqa
    if 'complex_con_ref_idx0' not in trb:
        # single chain
        trb['complex_con_ref_idx0'] = trb['con_ref_idx0']
        trb['complex_con_hal_idx0'] = trb['con_hal_idx0']
    # these run off 0-based indices
    for donor_res_idx0, acceptor_res_idx0 in zip(get_correct_ref(trb, donor), trb['complex_con_hal_idx0']):
        donor_res_idx1 = donor_res_idx0 + 1
        acceptor_res_idx1 = acceptor_res_idx0 + 1
        acceptor_res = acceptor.residue(acceptor_res_idx1)
        donor_res = donor.residue(donor_res_idx1)
        assert donor_res.name3() == acceptor_res.name3(), \
            f'donor {donor_res.name3()}{donor_res_idx1} != acceptor {acceptor_res.name3()}{acceptor_res_idx1}'
        mob_atomnames = [donor_res.atom_name(ai1) for ai1 in range(1, donor_res.natoms() + 1)]
        for fixed_atm_idx1 in range(1, acceptor_res.natoms() + 1):
            aname = acceptor_res.atom_name(fixed_atm_idx1)  # key to map one to other: overkill bar for HIE/HID
            if aname not in mob_atomnames:
                print(f'Template residue {donor_res.annotated_name()}{donor_res_idx1} lacks atom {aname}')
                continue
            donor_atm_idx1 = donor_res.atom_index(aname)
            donor2acceptor_idx1s[(donor_res_idx1, donor_atm_idx1)] = (acceptor_res_idx1, fixed_atm_idx1, aname) # noqa hungarian typehinting

    # ## Align
    atom_map = prs.map_core_id_AtomID_core_id_AtomID()
    if move_acceptor:
        mobile: pyrosetta.Pose = acceptor
        fixed: pyrosetta.Pose = donor
    else:
        mobile: pyrosetta.Pose = donor
        fixed: pyrosetta.Pose = acceptor
    for (donor_res_idx1, donor_atm_idx1), (acceptor_res_idx1, acceptor_atm_idx1, aname) in donor2acceptor_idx1s.items():
        if move_acceptor:
            mob_res_idx1, mob_atm_idx1 = acceptor_res_idx1, acceptor_atm_idx1
            fixed_res_idx1, fixed_atm_idx1 = donor_res_idx1, donor_atm_idx1
        else:
            mob_res_idx1, mob_atm_idx1 = donor_res_idx1, donor_atm_idx1
            fixed_res_idx1, fixed_atm_idx1 = acceptor_res_idx1, acceptor_atm_idx1
        if aname.strip() not in ('N', 'CA', 'C', 'O'):  # BB alignment
            continue
        fixed_atom = pyrosetta.AtomID(fixed_atm_idx1, fixed_res_idx1)
        mobile_atom = pyrosetta.AtomID(mob_atm_idx1, mob_res_idx1)
        atom_map[mobile_atom] = fixed_atom
    rmsd = prc.scoring.superimpose_pose(mod_pose=mobile, ref_pose=fixed, atom_map=atom_map)
    # I am unsure why this is not near zero but around 0.1–0.3
    assert rmsd < 1, f'RMSD {rmsd} is too high'

    # ## Copy coordinates
    to_move_atomIDs = pru.vector1_core_id_AtomID()
    to_move_to_xyz = pru.vector1_numeric_xyzVector_double_t()
    for (donor_res_idx1, donor_atm_idx1), (acceptor_res_idx1, acceptor_atm_idx1, aname) in donor2acceptor_idx1s.items():
        # if aname in ('N', 'CA', 'C', 'O'):  # BB if common
        #     continue
        # this does not stick: fixed_res.set_xyz( fixed_ai1, mob_res.xyz(mob_ai1) )
        to_move_atomIDs.append(pyrosetta.AtomID(acceptor_atm_idx1, acceptor_res_idx1))
        to_move_to_xyz.append(donor.residue(donor_res_idx1).xyz(donor_atm_idx1))

    acceptor.batch_set_xyz(to_move_atomIDs, to_move_to_xyz)

    # ## Fix HIE/HID the brutal way
    v = prc.select.residue_selector.ResidueNameSelector('HIS').apply(acceptor)
    relax = prp.relax.FastRelax(pyrosetta.get_score_function(), 1)
    movemap = pyrosetta.MoveMap()
    movemap.set_bb(False)
    movemap.set_chi(v)
    movemap.set_jump(False)
    relax.apply(acceptor)
    return rmsd, donor2acceptor_idx1s

def appraise_itxns(pose,
                   max_clashes=0,
                   clash_dist_cutoff=1.5,
                   bond_dist_cutoff=1.7,
                   trb: Optional[Dict[str, Any]] = None
                   ) -> Tuple[int, int]:
    """
    Assumes the chains have been fixed already.
    if `trb` is not None, the reference is used to check only stretching in the designed bits.
    NB. if the PDB had a gap and no trb is passed, it will be considered a stretch!

    :param pose:
    :param max_clashes:
    :param clash_dist_cutoff:
    :param bond_dist_cutoff:
    :param trb:
    :return:
    """
    n_clashing = 0
    chains = pose.split_by_chain()
    xyz_model = extract_coords(chains[1])
    for idx0 in range(1, pose.num_chains()):  # chain 0 is the full AHIR, designed
        xyz_other = extract_coords(chains[idx0 + 1])
        distances = np.sqrt(np.sum((xyz_other[:, np.newaxis, :] - xyz_model[np.newaxis, :, :]) ** 2, axis=-1))
        # 1.5 Å is too close
        n_clashing += np.count_nonzero(distances < clash_dist_cutoff)
    if n_clashing > max_clashes:
        raise ValueError(f'{n_clashing} clashes')
    # check no stretch
    n_warning_stretch = 0
    if trb is None:
        termini = [pose.chain_begin(c+1) for c in range(pose.num_chains())] + \
                  [pose.chain_end(c+1) for c in range(pose.num_chains())]
        # I could be doing pose.residue(i).is_terminus() but that is a needless amount of calls
        indices1 = [idx0+1 for idx0 in range(pose.total_residue() - 1) if idx0+1 not in termini]
    else:
        # get the not conserved and the preceding residue (unless it is the first)
        indices1 = [idx0 + 1 for idx0, conned in enumerate(trb['inpaint_seq']) if not conned] + \
                   [idx0 for idx0, conned in enumerate(trb['inpaint_seq']) if not conned if idx0 != 0]
        indices1 = list(set(indices1))
    for idx1 in indices1:
        d: float = pose.residue(idx1).xyz('C').distance(pose.residue(idx1 + 1).xyz('N'))
        if d > bond_dist_cutoff:
            raise ValueError(f'Stretch {d:.1}')
        if d > 1.36:
            n_warning_stretch += 1
    return n_clashing, n_warning_stretch

def extract_coords(pose: pyrosetta.Pose) -> np.ndarray:
    # this seems to be present in the docs but not in my version?
    # pyrosetta.toolbox.extract_coords_pose.pose_coords_as_row
    return np.array([list(pose.xyz(pyrosetta.AtomID(a, r))) for r in range(1, 1+pose.total_residue()) for a in range(1, 1+pose.residue(r).natoms())])

def fix_starts(pose, chain_letters: str, start_seqs: List[str]):
    """
    Fix the chains
    In anything based on pentakaimer it is

    .. code-block:: python

        strep_seq = 'MEAGIT'
        start_seqs = ['MKIYY', strep_seq, strep_seq, 'GEFAR', strep_seq, strep_seq, 'FKDET']
        fix_starts(pose, chain_letters='ACDEFGB', start_seq=start_seq)

    :param pose:
    :param chain_letters:
    :param start_seqs: Confusingly, the first is ignored: the start of the pose is the start of the first chain.
    :return:
    """
    pi = pose.pdb_info()
    seq = pose.sequence()
    seq_iter = iter(start_seqs[1:]+[None])
    chain_iter = iter(chain_letters)
    start_idx = 1
    while True:
        this_chain = next(chain_iter)
        next_seq = next(seq_iter)
        if next_seq is None:
            for i in range(start_idx, len(seq)+1):
                pi.chain(i, this_chain)
            break
        else:
            next_start = seq.find(next_seq, start_idx) + 1
            for i in range(start_idx, next_start):
                pi.chain(i, this_chain)
            start_idx = next_start
    pose.update_pose_chains_from_pdb_chains()
    assert pose.num_chains() == len(chain_letters), f'{pose.num_chains()} != {len(chain_letters)}'

# -------- constraints
def constrain_chainbreak(pose: pyrosetta.Pose, chain_break: FTypeIdx, x0_in=1.334, sd_in=0.2, tol_in=0.02):
    """
    Add a constraint to squeeze an unwanted chain break between residue (F-type idx) at ``chain_break``
    and succeeding residue.
    """
    AtomPairConstraint = pr_scoring.constraints.AtomPairConstraint  # noqa
    fore_c = pyrosetta.AtomID(atomno_in=pose.residue(chain_break).atom_index('C'),
                                rsd_in=chain_break)
    aft_n = pyrosetta.AtomID(atomno_in=pose.residue(chain_break + 1).atom_index('N'),
                              rsd_in=chain_break + 1)
    fun = pr_scoring.func.FlatHarmonicFunc(x0_in=x0_in, sd_in=sd_in, tol_in=tol_in)
    con = AtomPairConstraint(fore_c, aft_n, fun)
    pose.add_constraint(con)

def freeze_atom(pose: pyrosetta.Pose, frozen_index: int, ref_index: int, x0_in=0., sd_in=0.01):
    ref_ca = pyrosetta.AtomID(atomno_in=pose.residue(ref_index).atom_index('CA'), rsd_in=ref_index)
    frozen_ca = pyrosetta.AtomID(atomno_in=pose.residue(frozen_index).atom_index('CA'), rsd_in=frozen_index)
    frozen_xyz = pose.residue(frozen_index).xyz(frozen_ca.atomno())
    fun = pr_scoring.func.HarmonicFunc(x0_in=x0_in, sd_in=sd_in)
    con = pr_scoring.constraints.CoordinateConstraint(a1=frozen_ca, fixed_atom_in=ref_ca, xyz_target_in=frozen_xyz,
                                                      func=fun, scotype=pr_scoring.ScoreType.coordinate_constraint)
    pose.add_constraint(con)


def create_design_tf(pose:pyrosetta.Pose, design_sele: pr_res.ResidueSelector, distance:int) -> prc.pack.task.TaskFactory:
    """
    Create design task factory for relax.
    Designs the ``design_sele`` and repacks around ``distance`` of it.

    Remember to do

    ... code-block:: python

        relax.set_enable_design(True)
        relax.set_task_factory(task_factory)
    """
    #residues_to_design = design_sele.apply(pose)
    # this is default:
    # design_ops = prc.pack.task.operation.OperateOnResidueSubset(????, residues_to_design)
    no_cys = pru.vector1_std_string(1)
    no_cys[1] = 'CYS'
    no_cys_ops =  prc.pack.task.operation.ProhibitSpecifiedBaseResidueTypes(no_cys)
    # No design, but repack
    repack_sele = pr_res.NeighborhoodResidueSelector(design_sele, distance, False)
    residues_to_repack = repack_sele.apply(pose)
    repack_rtl = prc.pack.task.operation.RestrictToRepackingRLT()
    repack_ops = prc.pack.task.operation.OperateOnResidueSubset(repack_rtl, residues_to_repack)
    # No repack, no design
    frozen_sele = pr_res.NotResidueSelector(pr_res.OrResidueSelector(design_sele, repack_sele))
    residues_to_freeze = frozen_sele.apply(pose)
    prevent_rtl = prc.pack.task.operation.PreventRepackingRLT()
    frozen_ops = prc.pack.task.operation.OperateOnResidueSubset(prevent_rtl, residues_to_freeze)
    # pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT
    # pyrosetta.rosetta.core.pack.task.operation.PreserveCBetaRLT
    task_factory = prc.pack.task.TaskFactory()
    task_factory.push_back(no_cys_ops)
    task_factory.push_back(repack_ops)
    task_factory.push_back(frozen_ops)
    return task_factory




