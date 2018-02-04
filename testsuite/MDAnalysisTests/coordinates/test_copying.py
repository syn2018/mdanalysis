# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# MDAnalysis --- http://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
from __future__ import absolute_import

import numpy as np
from numpy.testing import assert_equal
import pytest
import MDAnalysis as mda

from MDAnalysisTests.datafiles import (
    CRD,
    DCD,
    DMS,
    DLP_CONFIG,
    DLP_HISTORY,
    INPCRD,
    GMS_ASYMOPT,
    GRO,
    LAMMPSdata_mini,
    mol2_molecules,
    MMTF,
    NCDF,
    PDB_small,
    PDBQT_input,
    PQR,
    TRR,
    TRJ,
    TRZ,
    XTC,
    XPDB_small,
    XYZ,
)
from MDAnalysis.coordinates.core import get_reader_for


@pytest.fixture(params=[
    # formatname, filename
    ('CRD', CRD, dict()),
    ('DATA', LAMMPSdata_mini, dict(n_atoms=1)),
    ('DCD', DCD, dict()),
    ('DMS', DMS, dict()),
    ('CONFIG', DLP_CONFIG, dict()),
    ('HISTORY', DLP_HISTORY, dict()),
    ('INPCRD', INPCRD, dict()),
    ('GMS', GMS_ASYMOPT, dict()),
    ('GRO', GRO, dict()),
    ('MMTF', MMTF, dict()),
    ('MOL2', mol2_molecules, dict()),
    ('PDB', PDB_small, dict()),
    ('PQR', PQR, dict()),
    ('PDBQT', PDBQT_input, dict()),
    ('TRR', TRR, dict()),
    ('TRZ', TRZ, dict(n_atoms=8184)),
    ('TRJ', TRJ, dict(n_atoms=252)),
    ('XTC', XTC, dict()),
    ('XPDB', XPDB_small, dict()),
    ('XYZ', XYZ, dict()),
    ('NCDF', NCDF, dict()),
    ('memory', np.arange(60).reshape(2, 10, 3).astype(np.float64), dict()),
], scope='module')
def refReader(request):
    fmt_name, filename, extras = request.param

    r = get_reader_for(filename, format=fmt_name)(filename, **extras)
    try:
        yield r
        # make sure file handle is closed afterwards
    finally:
        r.close()


@pytest.fixture()
def original_and_copy(refReader):
    new = refReader.copy()
    try:
        yield refReader, new
    finally:
        new.close()


def test_reader_n_atoms(original_and_copy):
    original, copy = original_and_copy
    assert original.n_atoms == copy.n_atoms

def test_reader_filename(original_and_copy):
    original, copy = original_and_copy
    assert original.filename == copy.filename

def test_reader_independent_iteration(original_and_copy):
    original, copy = original_and_copy
    if len(original) < 2:
        pytest.skip('Single frame reader')
    # initially at same frame
    assert original.ts.frame == copy.ts.frame

    copy[1]
    assert original.ts.frame == 0
    assert copy.ts.frame == 1

def test_timestep_copied(refReader):
    # modify the positions from original
    refReader.ts.positions *= 2

    new = refReader.copy()

    assert_equal(refReader.ts.positions, new.ts.positions)

def test_positions_share_memory(original_and_copy):
    original, copy = original_and_copy
    assert not np.shares_memory(original.ts.positions, copy.ts.positions)

    original.ts.positions *= 2

    with pytest.raises(AssertionError):
        assert_equal(original.ts.positions, copy.ts.positions)

def test_chainreader_NIE():
    u = mda.Universe(GRO, [GRO, GRO])

    with pytest.raises(NotImplementedError) as e:
        u.trajectory.copy()
    assert 'Copy not implemented for ChainReader' in str(e.value)
