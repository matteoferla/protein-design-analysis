"""
These are to check the base operations are working fine
"""
import unittest
import inspect
import random
from pathlib import Path

try:
    from protein_design_analysis import Tuner, Dispatcher
except ModuleNotFoundError as error:
    print(f'protein_design_analysis not installed - {error}')
    import sys
    repo_path = Path('.').absolute().parent
    sys.path.append(repo_path.as_posix())
    from protein_design_analysis import Tuner, Dispatcher



class MockTuner:

    settings = Tuner.settings

    def __init__(self, **kwargs):
        params = inspect.signature(Tuner.__init__).parameters
        for p_name, param in params.items():
            if param.default != inspect._empty:
                assert p_name in kwargs, f'{p_name} is mandatory in Tuner'

    @classmethod
    def process_task(cls, **kwargs):
        tuner = cls(**kwargs)
        return tuner()

def do_nothing_task(*args, **kwargs):
    kwargs['args'] = args
    return {'name': 'test', 'status': 'success', **kwargs}

def do_failsome_task(*args, **kwargs):
    kwargs['args'] = args
    roll = random.randint(1, 20)
    if roll == 1:
        raise Exception('crit fail')
    elif roll <10:
        raise ValueError('Made up')
    return {'name': 'test', 'status': 'success', **kwargs}

# ================================================================================================================

class TestPool(unittest.TestCase):

    test_settings = dict(hallucination_folder=Path('data'),
                        sequence_folder=Path('data'),
                        output_folder=Path('output'),
                        original_path=Path('data/wt.pdb'),
                        timeout=60*5,
                        override=True,
                        cartesian_relax_cycles=1,
                        internal_relax_cycles=1,
                        design_cycles=1,)

    def test_nothing(self):
        df = Dispatcher(hallucination_folder=Path('data'),
                        sequence_folder=Path('data'),
                        output_folder=Path('output'),
                        task_func=do_nothing_task)()
        print(df)

    def test_fail(self):
        df = Dispatcher(hallucination_folder=Path('data'),
                        sequence_folder=Path('data'),
                        output_folder=Path('output'),
                        task_func=do_failsome_task)()
        print(df)

    def test_mock_tuner(self):
        df = Dispatcher(Tuner_cls=MockTuner,
                        task_func=Tuner.process_task,
                        **self.test_settings)()
        print(df)


    def test_tuner(self):
        df = Dispatcher(task_func=Tuner.process_task, **self.test_settings)()
        print(df)

# ================================================================================================================

class TestTuner(unittest.TestCase):

    def test_tuner(self):
        tuner = Tuner(design_name='test_001',
                      design_sequence='SRYFIEFEELQLLGKGAFGAVIKVQNKLDGCCYAVKRIPINPASRQFRRIKGEVTLLSRLHHENIVRYYNAWIENAVHYLYIQMEYCEASTLRDTIDQGLYRDTVRLWRLFREILDGLAYIHEKGMIHRNLKPVNIFLDSDDHVKIGAIGIGDSAEDIATKSLKKLDDSMPEEEKLALQKVDLFSLGIIFFEMSYHPMVTASERIFVLNQLRDPTSPKFPEDFDDGEHAKQKSVISWLLNHDPAKRPTATELLKSELLPPP',
                      hallucination_path=Path('data/test.pdb'),
                      original_path=Path('data/wt.pdb'),
                      output_folder=Path('output')
                      )
        tuner.settings.exception_to_catch = ()
        tuner.settings.timeout=60*5
        tuner.settings.override=True
        tuner.settings.cartesian_relax_cycles=1
        tuner.settings.internal_relax_cycles=1
        tuner.settings.design_cycles=1
        info = tuner()
        print(info)


if __name__ == '__main__':
    unittest.main()