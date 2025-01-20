# -*- coding: utf-8 -*-
"""
TODO clean up this

first record is input:

    >beta_0-checkpoint, score=2.4184, global_score=1.5778, fixed_chains=['B', 'C', 'D'], designed_chains=['A'], model_name=v_48_020, git_hash=unknown, seed=37

herein I will call the input sample zero and use it as a negative control
others are:

    >T=0.1, sample=1, score=1.0129, global_score=1.5088, seq_recovery=0.0312

Setting from crysalin (to be fixed):

        SETTINGS = {
    'chain_letters': 'ACDEFGB',
    # full AHIR, strep x2, CTD AHIR, strep x2, AHIR snippet
    'start_seqs': ['MKIYY', strep_seq, strep_seq, 'GEFAR', strep_seq, strep_seq, 'FKDET'],
    'domain_end': 'KDETET',
    }
"""

import gzip
import json
import os
import pickle
import random
import re
import warnings
import time
import traceback
import functools
from logging import warning

from filelock import FileLock
from concurrent.futures import TimeoutError
from pathlib import Path
from types import ModuleType
from typing import Union, TypeAlias, List, Any, Dict, Tuple, Optional, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyrosetta
import pyrosetta_help as ph
from Bio import SeqIO
from pebble import ProcessPool, ProcessFuture
prc: ModuleType = pyrosetta.rosetta.core
prp: ModuleType = pyrosetta.rosetta.protocols
pru: ModuleType = pyrosetta.rosetta.utility  # noqa
prn: ModuleType = pyrosetta.rosetta.numeric
prs: ModuleType = pyrosetta.rosetta.std  # noqa
pr_conf: ModuleType = pyrosetta.rosetta.core.conformation
pr_res: ModuleType = pyrosetta.rosetta.core.select.residue_selector
pr_scoring: ModuleType = pyrosetta.rosetta.core.scoring
pr_options: ModuleType = pyrosetta.rosetta.basic.options
from .utils import (init_pyrosetta, steal_frozen_sidechains, appraise_itxns,
                    constrain_chainbreak, fix_starts, freeze_atom, create_design_tf,)
from .superposition import superpose

# from .common import (thread, extract_coords, create_design_tf,
#                               freeze_atom, init_pyrosetta, constrain_chainbreak,
#                               fix_starts, steal_frozen,
#                               appraise_itxns)
from .schemas import _Settings, _Paths, CTypeIdx, FTypeIdx, SeqType


class Tuner:

    settings = _Settings()

    def __init__(self,
                 design_name: str,
                 design_sequence: SeqType,
                 hallucination_path: Union[str, Path],
                 original_path: Union[str, Path],
                 particles_to_avoid: Optional[Union[str, np.array]]=None,
                 output_folder: Union[str, Path]=None,
                 chain_letters: Optional[str]=None,
                 start_seqs: Optional[List[str]]=None,
                 metadata: Optional[dict]=None,
                 ):
        """
        Instantiation does not do all the analysis, but sets up the object
        that can be called to do so later (within a try-except block say).

        :param design_name: name of the hallucinated construct with a given sequence
        :param design_sequence: ProteinMPNN sequence
        :param hallucination_path: RFdiffusion output PDB 'hallucination'
        :param original_path: the template pdb used for the hallucination, the original pdb
        :param output_folder: the folder where the output files are written
        :param metadata: a dictionary of useless information
        :param chain_letters: the chain letters in the hallucination (A_BC will have become A_B)
        :param start_seqs: the start sequences of the hallucination (start seq to figure out the chain breaks)
        """
        # ## Path independent attributes
        self.design_name = str(design_name)               #: name of the RFdiffusion+ProteinMPNN design
        self.design_sequence = str(design_sequence)       #: ProteinMPNN sequence
        self.chain_letters = chain_letters
        self.start_seqs = start_seqs
        self._initialized = False
        self.particles_to_avoid: np.array = self.parse_particles_to_avoid(particles_to_avoid)
        # ----------------------------------------------
        # ## Paths
        self.paths = _Paths(hallucination=hallucination_path,
                            original=original_path,
                            output_folder=output_folder,
                            )
        # when ``self.settings['*_folder_name']`` is set to None, the path is None and the file is not written.
        # files are checked in ``self.initialize``
        if self.settings.unrelaxed_folder_name:
            self.paths.raw = self.paths.input_folder / self.settings.unrelaxed_folder_name / f'{self.design_name}.pdb.gz'
        if self.settings.relaxed_folder_name:
            self.paths.relaxed = self.paths.input_folder / self.settings.relaxed_folder_name / f'{self.design_name}.pdb.gz'
        if self.settings.tuned_folder_name:
            self.paths.tuned = self.paths.input_folder / self.settings.tuned_folder_name / f'{self.design_name}.pdb.gz'
        else:
            warnings.warn('No tuned folder_name set, the tuned pdb will not be written... Is that what you want?')
        # path derived
        self.hallucination_name = self.paths.hallucination.stem  #: name of the seq-threaded hallucination
        # ----------------------------------------------
        # ## read by initialize
        self.trb: Dict[str, Any] = {}          #: the trb file w/ details about the hallucination
        self.original: Union[pyrosetta.Pose, None] = None   #: the template pdb used for the hallucination
        # ----------------------------------------------
        # ## Info
        # start info: this is the dict that gets returned by the instance call
        self.info = metadata.copy() if metadata else {}
        self.info['name'] = design_name
        self.info['folder'] = self.paths.input_folder.as_posix()
        self.info['design_sequence'] = design_sequence  # the seq wanted by proteinMPNN not the seq of the hallucination
        self.info['hallucination_name'] = self.paths.hallucination.stem
        self.info['start'] = time.time()
        self.info['status'] = 'ongoing'
        self.info['is_already_done'] = False

    def parse_particles_to_avoid(self, particles_to_avoid) -> np.array:

        if isinstance(particles_to_avoid, np.array):
            return particles_to_avoid
        elif isinstance(particles_to_avoid, list):
            return list(particles_to_avoid)
        elif particles_to_avoid is None:
            return np.array([])
        elif isinstance(particles_to_avoid, Path) and particles_to_avoid.suffix == '.npy':
            return np.load(str(particles_to_avoid))
        elif isinstance(particles_to_avoid, Path) and particles_to_avoid.suffix == '.pdb':
            return self.extract_coordinates_from_pdbblock(particles_to_avoid.read_text())
        else:
            raise ValueError(f'Not coded for particles_to_avoid type ({type(particles_to_avoid)}')


    def __call__(self) -> Dict[str, Any]:
        """
        Performs the analysis. See the following steps

        * ``.read_hallucination`` —reads the output PDB of RFdiffusion and fixes the chains
        * ``.thread`` —threads the hallucination to the target sequence (proteinMPNN say)
        * ``.freeze`` —adds coordinate constraints to the conserved residue's CA atoms
        * ``.squeze_gaps`` —adds atom pair constraints to abolish unexpected chainbreaks
        """
        try:
            self.initialize()
            hallucination: pyrosetta.Pose = self.read_hallucination()
            self.appraise(hallucination, is_final=False)
            self.custom_after_reading(hallucination)
            threaded: pyrosetta.Pose = self.thread(hallucination)
            self.custom_after_threading(threaded)
            self.freeze(threaded, ref_index=1)
            self.squeze_gaps(threaded)
            relaxed: pyrosetta.Pose  = self.cartesian_relax(threaded) # technically it is the same object -> readability
            self.relax(relaxed)
            self.score(relaxed)
            self.custom_after_relaxation(relaxed)
            tuned = self.tune(relaxed)
            self.custom_after_tuning(tuned)
            self.info['status'] = 'complete'
        except self.settings.exception_to_catch as e:
            self.info['error_type'] = e.__class__.__name__
            self.info['error'] = str(e)
            self.info['traceback'] = traceback.format_exception(e)
            self.info['status'] = 'error'
        # wrap up
        self.info['end'] = time.time()
        self.write_log()
        return self.info

    def write_log(self):
        lock = FileLock(self.paths.log.with_suffix('.lock'))
        with lock:
            with self.paths.log.open('a') as f:
                f.write(json.dumps(self.info) + '\n')
        return self.info

    def get_log(self):
        if not self.paths.log.exists():
            return None
        with self.paths.log.open('r') as f:
            logs = [json.loads(line) for line in f]
        for log in logs:
            if log['name'] == self.design_name:
                return log
        else:
            return None

    def initialize(self):
        """
        1. check the folders
        2. check if the target is already done (``cls.settings['override']``)
        3. Start PyRosetta
        4. Read the original and metadata
        """
        self.paths.check_folders()
        init_pyrosetta()
        # ## Check if it's already done
        prior_info = self.get_log()
        if prior_info and not self.settings.override:
            print(f'is_already_done: {prior_info} ')
            prior_info['is_already_done'] = True
            return prior_info
        self.write_log()
        # ## housekeeping
        # gz --> `dump_pdbgz`
        # ## read parent
        self.original: pyrosetta.Pose = pyrosetta.pose_from_file(self.paths.original.as_posix())
        # ## read metadata
        self.trb: Dict[str, Any] = pickle.load(self.paths.trb.open('rb'))
        self._initialized = True

    def read_hallucination(self, copy_sidechains=True) -> pyrosetta.Pose:
        """
        Reads the hallucination and fixes the chains (if set)
        """
        if not self._initialized:
            self.initialize()
        # ## read hallucination
        hallucination: pyrosetta.Pose = pyrosetta.pose_from_file(self.paths.hallucination.as_posix())
        self.fix_chains(hallucination)
        self.info['status'] = 'hallucination_read'
        if copy_sidechains:
            # steal the frozen sidechain atoms from the parent
            rmsd, tem2hal_idx1s = steal_frozen_sidechains(hallucination, self.original, self.trb, move_acceptor=False)
            self.info['parent_hallucination_RMSD'] = rmsd
            self.info['N_conserved_parent_hallucination_atoms'] = len(tem2hal_idx1s)
            self.info['status'] = 'sidechain_fixed'
        return hallucination

    def fix_chains(self, pose: pyrosetta.Pose):
        if self.chain_letters:
            fix_starts(pose, chain_letters=self.chain_letters, start_seqs=self.start_seqs)

    def thread(self, hallucination: pyrosetta.Pose, fragment_sets=None, temp_folder='/tmp') -> pyrosetta.Pose:
        # the seq from proteinMPNN is only chain A
        # warning! using the full sequence slows down the threading from 16s to 1m 41s
        full_target_seq = self.design_sequence + hallucination.sequence()[len(hallucination.chain_sequence(1)):]
        # ## Generate alignment file
        aln_filename = f'{temp_folder}/{self.design_name}.grishin'
        ph.write_grishin(target_name=self.design_name,
                         target_sequence=full_target_seq,
                         template_name=self.hallucination_name,
                         template_sequence=hallucination.sequence(),
                         outfile=aln_filename
                         )
        aln: prc.sequence.SequenceAlignment = prc.sequence.read_aln(format='grishin', filename=aln_filename)[1]
        # ## Thread proper
        threaded: pyrosetta.Pose
        threader: prp.comparative_modeling.ThreadingMover
        threadites: pru.vector1_bool
        threaded, threader, threadites = ph.thread(target_sequence=full_target_seq,
                                                   template_pose=hallucination,
                                                   target_name=self.design_name,
                                                   template_name=self.hallucination_name,
                                                   align=aln,
                                                   fragment_sets=fragment_sets
                                                   )
        # ## Finish up
        # TODO I need to check that there is a need to superpose: I think it is already aligned
        self.fix_chains(threaded)
        superpose(ref=hallucination, mobile=threaded)
        self.dump_pdbgz(threaded, self.paths.raw)
        self.info['status'] = 'threaded'
        self.write_log()
        return threaded

    def dump_pdbgz(self, pose: pyrosetta.Pose, path: Optional[Path]=None):
        """
        Dumps the pose to a gzipped pdb file if a path is given,
        else nothing happens. Say you changed ``Tuner(...).paths.👾`` to None,
        or altered ``Tuner.settings.👾_folder_name`` to None (which controls the `.paths`` of the instances,
        then the file is not written.
        """
        if path:
            with gzip.open(path, 'wt') as f:
                f.write(ph.get_pdbstr(pose))

    def freeze(self, pose: pyrosetta.Pose, ref_index=1):
        """
        Add ``CoordinateConstraint`` constraints to the conserved atoms
        """
        # complex_ is absent in single chain designs
        for idx0 in self.trb.get('complex_con_hal_idx0', self.trb['con_hal_idx0']):
            if idx0 == 0:
                # pointless/troublesome constrain to self, kind of
                continue
            freeze_atom(pose=pose, frozen_index=idx0 + 1, ref_index=ref_index)  # coordinate constraint

    def squeze_gaps(self, pose: pyrosetta.Pose):
        """
        Enforce abolition of chainbreaks in the designed sequence via ``AtomPairConstraint`` constraints
        """
        indices1 = {idx0 + 1 for idx0, is_conserved in enumerate(self.trb['inpaint_seq']) if not is_conserved} | \
                   {idx0 for idx0, is_conserved in enumerate(self.trb['inpaint_seq']) if not is_conserved and idx0 != 0}
        for idx1 in indices1:
                # Missing density = gaps are okay only in the conserved regions
                # jumps between chains: also okay
                # idx1 to idx1 + 1
                if pose.chain(idx1) != pose.chain(idx1 + 1):
                    continue
                constrain_chainbreak(pose, idx1)

    @property
    def cart_scorefxn(self) -> pr_scoring.ScoreFunction:
        scorefxn: pr_scoring.ScoreFunction = pyrosetta.create_score_function('ref2015_cart')
        scorefxn.set_weight(pr_scoring.ScoreType.coordinate_constraint, self.settings.coordinate_constraint_weight)
        scorefxn.set_weight(pr_scoring.ScoreType.atom_pair_constraint, self.settings.atom_pair_constraint_weight)
        scorefxn.set_weight(pr_scoring.ScoreType.res_type_constraint, self.settings.res_type_constraint_weight)
        return scorefxn

    @property
    def internal_scorefxn(self) -> pr_scoring.ScoreFunction:
        """
        Internal space. dihedral space. Rotational space.
        """
        scorefxn: pr_scoring.ScoreFunction = pyrosetta.create_score_function('ref2015')
        scorefxn.set_weight(pr_scoring.ScoreType.coordinate_constraint, self.settings.coordinate_constraint_weight)
        scorefxn.set_weight(pr_scoring.ScoreType.atom_pair_constraint, self.settings.atom_pair_constraint_weight)
        scorefxn.set_weight(pr_scoring.ScoreType.res_type_constraint, self.settings.res_type_constraint_weight)
        return scorefxn

    @property
    def design_selector(self) -> pr_res.ResidueSelector:
        resi_sele = prc.select.residue_selector.ResidueIndexSelector()
        for idx0, b in enumerate(self.trb['inpaint_seq']):
            if b:
                continue
            resi_sele.append_index(idx0 + 1)
        return resi_sele

    def cartesian_relax(self, pose: pyrosetta.Pose):
        """
        The carbonyl oxygens can be funky in hallucinated models.
        """
        if int(self.settings.cartesian_relax_cycles) <= 0:
            return pose
        relax = prp.relax.FastRelax(self.cart_scorefxn, self.settings.cartesian_relax_cycles)
        relax.dualspace(True)
        relax.minimize_bond_angles(True)
        relax.minimize_bond_lengths(True)
        movemap = pyrosetta.MoveMap()
        v: pru.vector1_bool = self.design_selector.apply(pose)
        movemap.set_bb(v)
        movemap.set_chi(v)
        movemap.set_jump(False)
        relax.set_movemap(movemap)
        relax.apply(pose)
        self.info['status'] = 'cart_relaxed'
        return pose

    def relax(self, pose: pyrosetta.Pose, neighbor_distance=8):
        """
        Internal space relaxation
        """
        neigh_sele = prc.select.residue_selector.NeighborhoodResidueSelector(self.design_selector, neighbor_distance, True)
        v: pru.vector1_bool = neigh_sele.apply(pose)
        movemap = pyrosetta.MoveMap()
        movemap.set_bb(v)
        movemap.set_chi(v)
        movemap.set_jump(False)
        if int(self.settings.internal_relax_cycles) <= 0:
            return pose
        relax = prp.relax.FastRelax(self.internal_scorefxn,self.settings.internal_relax_cycles)
        relax.set_movemap(movemap)
        relax.apply(pose)
        self.info['status'] = 'relaxed'
        self.dump_pdbgz(pose, self.paths.relaxed)
        self.write_log()
        return pose

    def score(self, pose):
        vanilla_scorefxn: pr_scoring.ScoreFunction = pyrosetta.get_fa_scorefxn()
        self.info['dG'] = vanilla_scorefxn(pose)
        self.info['dG_monomer'] = vanilla_scorefxn(pose.split_by_chain(1))

    def safe_late_stop(self, pose):
        """
        Stop if 90% close to timeout: better to have metadata than nothing
        """
        if (time.time() - self.info['start']) * 0.90 > self.settings.timeout:
            self.info['status'] = 'close to timeout'
            self.appraise(pose, is_final=True)
            raise ValueError('Close to timeout')

    def appraise(self, pose, is_final=False):
        """
        TODO refactor the function as a method
        """
        n_clashing, n_warning_stretch =  appraise_itxns(pose,
                                           max_clashes=0 if is_final else self.settings.initial_max_clashes,
                                           clash_dist_cutoff=self.settings.clash_dist_cutoff,
                                           bond_dist_cutoff=self.settings.bond_dist_cutoff,
                                           trb=self.trb)
        self.info['N_clashes'] = n_clashing
        self.info['N_warning_stretches'] = n_warning_stretch
        if self.particles_to_avoid:
            bin_tally = self.tally_avoidance_by_bin(pose)
            if bin_tally[0] > self.settings.avoidance_cutoff:
                raise ValueError('Too much overlap with particles to avoid')
        if not is_final:
            return None
        pose2pdb = pose.pdb_info().pose2pdb
        for distance in (1, 2, 3, 4, 5, 6, 8, 10, 12):
            v = pr_res.NeighborhoodResidueSelector(self.design_selector, distance, False).apply(pose)
            close_residues = prc.select.get_residues_from_subset(v)
            self.info[f'N_close_residues_{distance}'] = len(close_residues)
            self.info[f'close_residues_{distance}'] = [pose2pdb(r) for r in close_residues]
        # per residue scores
        vanilla_scorefxn: pr_scoring.ScoreFunction = pyrosetta.get_fa_scorefxn()
        res_scores = []
        monomer = pose.split_by_chain(1)
        for i in range(1, monomer.total_residue() + 1):
            v = pru.vector1_bool(pose.total_residue())
            v[i] = 1
            # score only monomer residues, but on oligomer
            res_scores.append(vanilla_scorefxn.get_sub_score(pose, v))
        self.info['per_res_score'] = res_scores
        self.info['max_per_res_scores'] = max(res_scores)
        self.info['camel_seq'] = self.get_camelized_seq(pose)

    def tally_avoidance_by_bin(self, pose: pyrosetta.Pose) -> List[int]:
        """
        This is all numpy based. The ``self.particles_to_avoid`` is a np.array
        """
        query: np.array = self.extract_coordinates_from_pose(pose)
        # ``self.settings.avoidance_bins`` is by default ((0, 2.), (2., 3.), (2, 5.), (5., np.inf)))
        distances = np.min(np.linalg.norm(query[:, np.newaxis] - self.particles_to_avoid, axis=2), axis=1)
        bin_counts = [0] * len(self.settings.avoidance_bins)
        for i, bin_range in enumerate(self.settings.avoidance_bins):
            mask = (distances > bin_range[0]) & (distances <= bin_range[1])
            bin_counts[i] = np.sum(mask)
        return bin_counts

    def tune(self, pose: pyrosetta.Pose) -> pyrosetta.Pose:
        """
        Run FastDesign on the pose
        """
        vanilla_scorefxn: pr_scoring.ScoreFunction = pyrosetta.get_fa_scorefxn()
        # res_type_constraint is already set in the scorefxn
        prp.protein_interface_design.FavorNativeResidue(pose, 1)
        # make the ref sequence: need for safeguarding against Relax changing
        ref_seq = ''.join([resn if b else '-' for b, resn in zip(self.trb['inpaint_seq'], pose.sequence())])
        previous_design = pose
        previous_complex_dG = vanilla_scorefxn(previous_design)
        previous_mono_dG = vanilla_scorefxn(previous_design.split_by_chain(1))
        current_design = previous_design # just in case self.settings.design_cycles == 0
        for cycle in range(self.settings.design_cycles):
            current_design = previous_design.clone()
            task_factory: prc.pack.task.TaskFactory = create_design_tf(current_design,
                                                                       design_sele=self.design_selector,
                                                                       distance=0)
            relax = pyrosetta.rosetta.protocols.relax.FastRelax(self.internal_scorefxn, 1)  # one cycle at a time
            relax.set_enable_design(True)
            relax.set_task_factory(task_factory)
            relax.apply(current_design)
            current_complex_dG = vanilla_scorefxn(current_design)
            chain = current_design.split_by_chain(1)
            current_mono_dG = vanilla_scorefxn(chain)
            self.info[f'design_cycle{cycle}_seq'] = chain.sequence()
            self.info[f'design_cycle{cycle}_dG_complex'] = current_complex_dG
            self.info[f'design_cycle{cycle}_dG_monomer'] = current_mono_dG
            if any([have != expected and expected != '-' for have, expected in
                    zip(current_design.sequence(), ref_seq)]):
                # this is a weird glitch in Relax that happens rarely
                #print('Mismatch happened: reverting!')
                current_design = previous_design
                self.info[f'design_cycle{cycle}_outcome'] = 'mismatched'
            elif current_complex_dG > previous_complex_dG:
                #print('Design is worse: reverting!')
                current_design = previous_design
                self.info[f'design_cycle{cycle}_outcome'] = 'worse complex'
            elif current_mono_dG > previous_mono_dG:
                #print('Design is worse: reverting!')
                current_design = previous_design
                self.info[f'design_cycle{cycle}_outcome'] = 'worse monomer'
            else:
                #info[f'design_cycle{cycle}_outcome'] = 'success'
                previous_design = current_design
                previous_complex_dG = current_complex_dG
                self.dump_pdbgz(current_design, self.paths.tuned)
                self.write_log()
                self.safe_late_stop(current_design)
        self.appraise(current_design, is_final=True)
        # hurray:
        self.info['status'] = 'tuned'
        self.write_log()
        return current_design

    def get_camelized_seq(self, pose: pyrosetta.Pose) -> str:
        """
        Returns the sequence with the conserved residues in lowercase
        """
        v = self.design_selector.apply(pose)
        seq = pose.sequence()
        return ''.join([r if d else r.lower() for r, d in zip(seq, v)])

    # --------------------------------------------------------
    def custom_after_reading(self, pose):
        """
        Method called after reading and fixing hallucination
        intended for subclasses
        """
        pass

    def custom_after_threading(self, pose):
        """
        Method called after threading
        intended for subclasses
        """
        pass

    def custom_after_relaxation(self, pose):
        """
        Method called after minimisations
        intended for subclasses
        """
        pass

    def custom_after_tuning(self, pose):
        """
        Method called after tuning
        intended for subclasses
        """
        pass

    # --------------------------------------------------------

    @staticmethod
    def extract_coordinates_from_pdbblock(block: str, atom_names: Optional[Sequence[str]]=None) -> np.ndarray:
        coordinates = []
        for line in block.split('\n'):
            if line.startswith(('ATOM', 'HETATM')) and (not atom_names or line[12:16] in atom_names):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coordinates.append([x, y, z])
                except ValueError:
                    continue
        return np.array(coordinates)

    @staticmethod
    def extract_coordinates_from_pose(pose: pyrosetta.Pose, atom_names = (' N  ', ' CA ', ' C  ', ' O  ')) -> np.ndarray:
        """
        Extracts the coordinates from a pose
        """
        return np.array([pose.xyz(prc.id.NamedAtomID(atom_in=n, rsd_in=idx1))
                         for n in atom_names for idx1 in range(1, pose.total_residue() + 1)])

    @classmethod
    def process_task(cls, **kwargs):
        tuner = cls(**kwargs)
        return tuner()