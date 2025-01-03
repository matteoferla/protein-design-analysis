"""
The .paths and .settings attributes are dataclass instances.
The classes are defined here in schemas.py
"""

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Union, TypeAlias

CTypeIdx: TypeAlias = int  # 0-indexed
FTypeIdx: TypeAlias = int  # 1-indexed
SeqType: TypeAlias = str # no check for AA though!

@dataclass
class _Settings:
    timeout: int = 60 * 60 * 24  #: 24 hours
    exception_to_catch: Exception = Exception
    override: bool = False
    cartesian_relax_cycles: int = 3
    internal_relax_cycles: int = 5
    design_cycles: int = 15
    clash_dist_cutoff: float = 1.5
    bond_dist_cutoff: float = 1.7  #: N-C ideally is 1.32 Å
    atom_pair_constraint_weight: float = 3
    coordinate_constraint_weight: float = 1
    res_type_constraint_weight: float = 1
    initial_max_clashes: int = 3  #: a clash or two is fine for now
    tuned_folder_name: str = 'tuned_pdbs'
    relaxed_folder_name: str = 'relaxed_pdbs'
    unrelaxed_folder_name: str = 'unrelaxed_pdbs'

class _Paths:
    """
    A dataclass to hold the paths of the output files.
    The instance is found as ``Tuner(...).paths``.

    :param input_folder:  input_folder is the input folder with the RFdiffusion pdb and trb
                            named ``{target_name}.pdb`` and ``{hallucination_name}.trb``
    :param original: the template pdb used for the hallucination
    :param raw: the unrelaxed pdb output file
    :param relaxed: the relaxed pdb output file
    :param tuned: the tuned pdb output file
    """
    def __init__(self,
                 hallucination: Union[str, Path],
                 original: Union[str, Path],
                 output_folder: Union[str, Path]=None,
                 raw: Union[str, Path, None]=None,
                 relaxed: Union[str, Path, None]=None,
                 tuned: Union[str, Path, None]=None,
                 ):
        """
        """
        self.hallucination = Path(hallucination)   #: the hallucination pdb (output of RFdiffusion)
        self.input_folder = self.hallucination.parent  #: the folder with the pdb and trb
        self.trb = self.hallucination.with_suffix('.trb')  #: the metadata file
        self.output_folder = output_folder if output_folder else self.input_folder
        self.original = Path(original)              #: the template pdb used for the hallucination
        self.raw = Path(raw) if raw else None    #: the unrelaxed/threaded pdb file path
        self.relaxed = Path(relaxed) if relaxed else None  #: the relaxed pdb file path
        self.tuned = Path(tuned) if tuned else None      #: the tuned pdb file path
        self.log = self.output_folder / 'log.jsonl'  #: the log file path

    def check_folders(self):
        """
        Called by `Tuner(...).initialize` to check if the folders exist,
        the output folder are created if they don't
        files are checked here and not at Tuner(...) init because 
        the info object would be available if the call is in a try-catching context
        """
        assert self.input_folder.exists(), f'{self.input_folder} does not exist'
        assert self.original.exists(), f'{self.original} does not exist'
        assert self.trb.exists(), f'{self.trb} does not exist'
        assert self.hallucination.exists(), f'{self.hallucination} does not exist'
        if self.raw is not None:
            os.makedirs(self.raw.parent, exist_ok=True)
        if self.relaxed is not None:
            os.makedirs(self.relaxed.parent, exist_ok=True)
        if self.tuned is not None:
            os.makedirs(self.tuned.parent, exist_ok=True)
