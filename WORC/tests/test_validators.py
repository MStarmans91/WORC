from pathlib import Path
import pytest
from unittest.mock import MagicMock
import tempfile

from WORC.validators.preflightcheck import InvalidLabelsValidator
import WORC.addexceptions as ae


"""
Test to see what happens if a valid configuration is given to the InvalidLabelsValidator.
Under normal circumstances this should not throw any exception and thus return as a passed test.
"""
def test_invalidlabelsvalidator_validconfig():
    with tempfile.TemporaryDirectory() as tmpdirname:
        validator = InvalidLabelsValidator()

        experiment = MagicMock()
        experiment._labels_file_train = str(Path(tmpdirname) / 'patients.csv')

        with open(experiment._labels_file_train, 'w') as fh:
            fh.write("""Patient,mock
pat001,1
pat002,1
pat003,0
pat004,0""")

        res = validator.do_validation(experiment)

        assert res == True


"""
Test to see what happens if an invalid configuration is given to the InvalidLabelsValidator.
Under normal circumstances this test should pass due to an exception being thrown as the label file contains patient
as the first column instead of Patient.
"""
def test_invalidlabelsvalidator_patientcolumn():
    with tempfile.TemporaryDirectory() as tmpdirname:
        validator = InvalidLabelsValidator()

        experiment = MagicMock()
        experiment._labels_file_train = str(Path(tmpdirname) / 'patients.csv')

        with open(experiment._labels_file_train, 'w') as fh:
            fh.write("""patient,mock
pat001,1
pat002,1
pat003,0
pat004,0""")

        with pytest.raises(ae.WORCValueError) as e_info:
            res = validator.do_validation(experiment)

        assert "needs to be named Patient" in str(e_info.value)


"""
Test to see what happens if an invalid configuration is given to the InvalidLabelsValidator.
Under normal circumstances this test should pass due to an exception being thrown as the patient names are
substrings of eachother.
"""
def test_invalidlabelsvalidator_patientsubstring():
    with tempfile.TemporaryDirectory() as tmpdirname:
        validator = InvalidLabelsValidator()

        experiment = MagicMock()
        experiment._labels_file_train = str(Path(tmpdirname) / 'patients.csv')

        with open(experiment._labels_file_train, 'w') as fh:
            fh.write("""Patient,mock
pat111,1
pat11,1
pat1,0""")

        with pytest.raises(ae.WORCValueError) as e_info:
            res = validator.do_validation(experiment)

        assert "Found subject(s) that are a substring of other subject(s)" in str(e_info.value)


"""
Test to see what happens if an invalid configuration is given to the InvalidLabelsValidator.
Under normal circumstances this test should pass due to an exception being thrown as the columns / labels are
substrings of eachother.
"""
def test_invalidlabelsvalidator_columnsubstring():
    with tempfile.TemporaryDirectory() as tmpdirname:
        validator = InvalidLabelsValidator()

        experiment = MagicMock()
        experiment._labels_file_train = str(Path(tmpdirname) / 'patients.csv')

        with open(experiment._labels_file_train, 'w') as fh:
            fh.write("""Patient,mock1,mock111,mock1111,mock01
pat01,1,1,1,1
pat02,1,1,1,1
pat03,1,1,1,1""")

        with pytest.raises(ae.WORCValueError) as e_info:
            res = validator.do_validation(experiment)

        assert "Found label(s) that are a substring of other label(s)" in str(e_info.value)