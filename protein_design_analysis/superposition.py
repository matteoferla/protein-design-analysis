from types import ModuleType
from typing import Dict, Optional

import pyrosetta
from Bio.Align import PairwiseAligner, Alignment

prc: ModuleType = pyrosetta.rosetta.core
prp: ModuleType = pyrosetta.rosetta.protocols
pru: ModuleType = pyrosetta.rosetta.utility  # noqa
prn: ModuleType = pyrosetta.rosetta.numeric
prs: ModuleType = pyrosetta.rosetta.std  # noqa
pr_conf: ModuleType = pyrosetta.rosetta.core.conformation
pr_scoring: ModuleType = pyrosetta.rosetta.core.scoring
pr_options: ModuleType = pyrosetta.rosetta.basic.options
pr_res: ModuleType = pyrosetta.rosetta.core.select.residue_selector

FTypeIdx = int  # one-based index
CTypeIdx = int  # zero-based index

def align_for_atom_map(mobile: pyrosetta.Pose, ref: pyrosetta.Pose) -> Dict[int, int]:
    """
    Pairwise alignment of the sequences of the poses.
    return  (ref_index, mobile_index)
    :param mobile:
    :param ref:
    :return:
    """
    # pad with '-' to make it faux-local alignment and deal with Fortran counting does not work '-' is a match not gap
    # hence the silly +1s and the PairwiseAligner settings
    aligner = PairwiseAligner()
    aligner.internal_gap_score = -10
    aligner.extend_gap_score = -0.01
    aligner.end_gap_score = -0.01
    # pose is longer and right does not matter. left aligned!
    aligner.target_right_gap_score = 0.
    aligner.target_right_extend_gap_score = 0.
    ref_seq: str = ref.sequence()
    pose_seq: str = mobile.sequence()
    aln: Alignment = aligner.align(ref_seq, pose_seq)[0]
    return {t: q for t, q in zip(aln.indices[0], aln.indices[1]) if
               q != -1 and t != -1 and ref_seq[t] == pose_seq[q]}

def superpose_pose_by_chain(pose, ref, chain: str, strict: bool=True) -> float:
    """
    superpose by PDB chain letter

    :param pose:
    :param ref:
    :param chain:
    :return:
    """
    atom_map = prs.map_core_id_AtomID_core_id_AtomID()
    chain_sele: pr_res.ResidueSelector = pr_res.ChainSelector(chain)
    for r, m in zip(pr_res.selection_positions(chain_sele.apply(ref)),
                    pr_res.selection_positions(chain_sele.apply(pose))
                    ):
        if strict:
            assert pose.residue(m).name3() == ref.residue(r).name3(), 'Mismatching residue positions!'
        ref_atom = pyrosetta.AtomID(ref.residue(r).atom_index("CA"), r)
        mobile_atom = pyrosetta.AtomID(pose.residue(m).atom_index("CA"), m)
        atom_map[mobile_atom] = ref_atom
    return prc.scoring.superimpose_pose(mod_pose=pose, ref_pose=ref, atom_map=atom_map)

def superpose_pose_by_alt_chains(pose, ref, pose_chain: str, ref_chain: str) -> float:
    """
    superpose by PDB chain letter

    :param pose:
    :param ref:
    :return:
    """
    atom_map = prs.map_core_id_AtomID_core_id_AtomID()
    pose_chain_sele: pr_res.ResidueSelector = pr_res.ChainSelector(pose_chain)
    ref_chain_sele: pr_res.ResidueSelector = pr_res.ChainSelector(ref_chain)
    for r, m in zip(pr_res.selection_positions(ref_chain_sele.apply(ref)),
                    pr_res.selection_positions(pose_chain_sele.apply(pose))
                    ):
        assert pose.residue(m) == ref.residue(r), 'Mismatching residue positions!'
        ref_atom = pyrosetta.AtomID(ref.residue(r).atom_index("CA"), r)
        mobile_atom = pyrosetta.AtomID(pose.residue(m).atom_index("CA"), m)
        atom_map[mobile_atom] = ref_atom
    return prc.scoring.superimpose_pose(mod_pose=pose, ref_pose=ref, atom_map=atom_map)

def superpose_by_seq_alignment(mobile: pyrosetta.Pose, ref: pyrosetta.Pose) -> float:
    """
    Superpose ``pose`` on ``ref`` based on Pairwise alignment and superposition of CA

    :param mobile:
    :param ref:
    :return:
    """

    aln_map = align_for_atom_map(mobile, ref)
    rmsd: float = superpose(ref=ref, mobile=mobile, aln_map=aln_map)
    return rmsd

def superpose(ref: pyrosetta.Pose, mobile: pyrosetta.Pose, aln_map: Optional[Dict[int, int]] = None, zero_based=True) -> float:
    """
    Superpose ``mobile`` on ``ref`` based on CA of indices in ``aln_map`` (ref_indices, mobile_indices).
    Indices are 0-based.
    :param ref:
    :param mobile:
    :param aln_map:
    :return:
    """
    offset = 1 if zero_based else 0
    if aln_map is None:
        aln_map = dict(zip(range(ref.total_residue()), range(mobile.total_residue())))
    # ## make pyrosetta map
    atom_map = prs.map_core_id_AtomID_core_id_AtomID()
    for r, m in aln_map.items():
        ref_atom = pyrosetta.AtomID(ref.residue(r + offset).atom_index("CA"), r + offset)
        mobile_atom = pyrosetta.AtomID(mobile.residue(m + offset).atom_index("CA"), m + offset)
        atom_map[mobile_atom] = ref_atom
    # return RMSD
    return prc.scoring.superimpose_pose(mod_pose=mobile, ref_pose=ref, atom_map=atom_map)
