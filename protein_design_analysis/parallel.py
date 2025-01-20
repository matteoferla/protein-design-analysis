"""
Ongoing refactoring...
"""

import gzip
import json
import os
import pickle
import random
import re
import time
import traceback
from filelock import FileLock
from concurrent.futures import TimeoutError
from pathlib import Path
from types import ModuleType
from typing import Union, Dict, List, Callable, Type, Any
from dataclasses import dataclass, fields


import numpy as np
import pandas as pd
from Bio import SeqIO
from pebble import ProcessPool, ProcessFuture

from .tuning import Tuner

class Dispatcher:

    exceptions_to_catch = (Exception,)

    def __init__(self,
                 hallucination_folder: Path,
                 sequence_folder: Path,
                 task_func=None,
                 Tuner_cls=Tuner,
                 **settings):
        self.hallucination_folder = hallucination_folder
        self.sequence_folder = sequence_folder
        self.Tuner: Type[Tuner] = Tuner_cls  # in case a Tuner subclass is used
        self.settings: Dict = settings
        tuner_settings, tuner_args = self.partition_settings()
        self.tuner_args: Dict[str, Any] = tuner_args
        self.update_tuner_settings(tuner_settings)  # Tuner has its settings attribute, this is to override them
        self.task_func: Callable = task_func  # this is a function as an attribute, not a method
        self.max_workers = self.settings.get( 'max_workers', self.max_cores - 1)  # override if needed
        self.timeout = self.settings.get('timeout', self.Tuner.settings.timeout)
        self.futuredex: Dict[str, ProcessFuture] = {}

    def partition_settings(self):
        tuner_settings = {}
        tuner_args = {}
        for key in self.settings:
            if key in ('design_name', 'design_sequence', 'hallucination_path', 'original_path',
                     'output_folder', 'chain_letters', 'start_seqs', 'metadata'):
                tuner_args[key] = self.settings[key]
            elif key in self.Tuner.settings.__dataclass_fields__:
                tuner_settings[key] = self.settings[key]
            else:
                print(f'Key {key} is not a setting for `Tuner`: skipping')
        return tuner_settings, tuner_args

    def update_tuner_settings(self, settings):
        """
        ``Tuner.settings`` is an instance of ``_Settings`` dataclass so is a bit more convoluted.
        """
        for key in settings:
            assert key in self.Tuner.settings.__dataclass_fields__, f'Tuner.settings has no value {key}'
            setattr(self.Tuner.settings, key, self.settings[key])  # no typechecks

    @property
    def max_cores(self):
        """
        the number of cores to use.
        Called by init
        To override the max N of workers use ``.max_workers``
        """
        return int(os.environ.get('SLURM_JOB_CPUS_PER_NODE', os.cpu_count()) )

    # ------------------------------------------------------------------------------------------------

    def __call__(self) -> pd.DataFrame:
        # submit
        self.futuredex: Dict[str, ProcessFuture] = {}
        self.futuredex = self._submit()
        if len(self.futuredex) == 0:
            raise ValueError(f'No jobs added to process pool. '+\
                             f'N of fasta files {len(list(self.sequence_folder.glob("*.fa")))} in {self.sequence_folder}')
        print(f'Queued {len(self.futuredex)} processes')
        # retrieve
        results: List[dict] = self._parse(self.futuredex)
        self.futuredex = {}  # the garbage collector is hungry!
        # return
        df = pd.DataFrame(results)
        # df.to_pickle(target_folder / 'tuned.pkl.gz')
        return df

    def __iter__(self):
        seq_paths = list(self.sequence_folder.glob('*.fa'))
        for seq_path in seq_paths:
            # ## Read sequences
            records: List[dict] = self.read_sequences(seq_path)
            # ``records`` keys: design_name design_sequence hallucination_name metadata
            for record in records:
                # ## hallucination check
                # Input requires hallucination (RFdiffusion output) so checking it exists,
                # just in case it was manually deleted
                # not used: record['hallucination_folder'] = self.hallucination_folder
                record['hallucination_path'] = self.hallucination_folder / (seq_path.stem + '.pdb')
                assert record['hallucination_path'].exists()
                del record['hallucination_name']  # not needed
                # ## Return
                yield {**record, **self.tuner_args}

    def _submit(self) -> Dict[str, ProcessFuture]:
        futuredex: Dict[str, ProcessFuture] = {}

        with ProcessPool(max_workers=self.max_workers, max_tasks=0) as pool:
            job_args: Dict
            for job_args in self:
                future: ProcessFuture = pool.schedule(self.task_func,
                                                      # target = seq-variant of hallucination
                                                      # parent = WT template
                                                      # hallucination = RFdiffused skeleton
                                                      kwargs=job_args,
                                                      timeout=self.timeout)
                design_name = job_args['design_name']
                futuredex[design_name] = future
        return futuredex

    def _parse(self, futuredex: Dict[str, ProcessFuture]):
        results: List[dict] = []
        for name, future in futuredex.items():
            try:
                result = future.result()  # blocks until results are ready
                print(result['name'], result['status'])
                results.append(result)
            except KeyboardInterrupt as error:
                print('Attempting to kill all children')
                unfuture: ProcessFuture
                for unfuture in reversed(futuredex.values()):
                    unfuture.cancel()
                    break
            except self.exceptions_to_catch as error:
                error_msg = str(error)
                error_type=error.__class__.__name__
                result = dict(name=name, target_name=name, error=error_msg, error_type=error_type, )
                results.append(result)
                if isinstance(error, TimeoutError):
                    print(f'Function took longer than {self.timeout} seconds {error}')
                else:
                    print(f"Function raised {error_type}:{error_msg}")
                    traceback.print_tb(error.__traceback__)  # traceback of the function
        return results

    @staticmethod
    def read_sequences(fasta_path: Path) -> List[Dict]:
        """
        Read proteinMPNN output sequences and return a list of dict w/ keys where
        'design' = the new sequence and 'parent' is the hallucination.
        The seq are sequentially labelled w/ suffix Ø (ref poly G), A, B, C, D etc.

        NB. proteinMPNN returns only the designed chain sequence
        """
        records = []
        for seq_record in SeqIO.parse(fasta_path, 'fasta'):
            # ## Read metadata
            metadata = {k: float(v) for k, v in re.findall(r'([\w_]+)=([\d.]+)', seq_record.description)}
            hallucination_name: str = fasta_path.stem  # noqa
            target_sequence: str = str(seq_record.seq)  # noqa
            target_name = f"{hallucination_name}{'ØABCDEFGHIJKLMNOPQRSTUVWXYZ'[int(metadata.get('sample', 0))]}"
            records.append(
                dict(design_name=target_name, design_sequence=target_sequence, hallucination_name=hallucination_name,
                     metadata=metadata))
        return records