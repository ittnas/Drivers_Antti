#!/usr/bin/env python3
import logging
import random as rnd

import numpy as np
import cliffords
import copy

import itertools

import gates
from sequence import Sequence

log = logging.getLogger('LabberDriver')
import os
path_currentdir  = os.path.dirname(os.path.realpath(__file__)) # curret directory

def CheckIdentity(matrix):
    """
    Check whether the matrix is identity by calculating it numerically.

    Parameters
    ----------
    matrix: 4x4 np.matrix
        matrix

    Returns
    -------
    result: bool
        True, if identity.
    """

    #Check all the diagonal entries.
    if ((np.abs(np.abs(matrix[0,0])-1)< 1e-6) and
        (np.abs(np.abs(matrix[1,1])-1)< 1e-6) and
        (np.abs(np.abs(matrix[2,2])-1)< 1e-6) and
        (np.abs(np.abs(matrix[3,3])-1)< 1e-6) and
        (np.abs(matrix[1,1]/matrix[0,0]-1)<1e-6) and
        (np.abs(matrix[2,2]/matrix[0,0]-1)<1e-6) and
        (np.abs(matrix[3,3]/matrix[0,0]-1)<1e-6)):
        return True
    else:
        return False

def add_singleQ_clifford(index, gate_seq, pad_with_I=True):
    """Add single qubit clifford (24)."""
    length_before = len(gate_seq)
    # Paulis
    if index == 0:
        gate_seq.append(gates.I)
    elif index == 1:
        gate_seq.append(gates.Xp)
    elif index == 2:
        gate_seq.append(gates.Yp)
    elif index == 3:
        gate_seq.append(gates.Xp)
        gate_seq.append(gates.Yp)

    # 2pi/3 rotations
    elif index == 4:
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.X2p)
    elif index == 5:
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.X2p)
    elif index == 6:
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.X2m)
    elif index == 7:
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.X2m)
    elif index == 8:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Y2p)
    elif index == 9:
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.Y2p)
    elif index == 10:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Y2m)
    elif index == 11:
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.Y2m)

    # pi/2 rotations
    elif index == 12:
        gate_seq.append(gates.X2p)
    elif index == 13:
        gate_seq.append(gates.X2m)
    elif index == 14:
        gate_seq.append(gates.Y2p)
    elif index == 15:
        gate_seq.append(gates.Y2m)
    elif index == 16:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.X2m)
    elif index == 17:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.X2m)

    # Hadamard-Like
    elif index == 18:
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.Xp)
    elif index == 19:
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.Xp)
    elif index == 20:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Yp)
    elif index == 21:
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.Yp)
    elif index == 22:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.X2p)
    elif index == 23:
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.X2m)
    else:
        raise ValueError(
            'index is out of range. it should be smaller than 24 and greater'
            ' or equal to 0: ', str(index))

    length_after = len(gate_seq)
    if pad_with_I:
        # Force the clifford to have a length of 3 gates
        for i in range(3-(length_after-length_before)):
            gate_seq.append(gates.I)


def add_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = [], generator = 'CZ'):
    """Add single qubit clifford (11520 = 576 + 5184 + 5184 + 576)."""
    if (index < 0):
        raise ValueError(
            'index is out of range. it should be smaller than 11520 and '
            'greater or equal to 0: ', str(index))
    elif (index < 576):
        add_singleQ_based_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = gate_seq_Cplr, generator = generator)
    elif (index < 5184 + 576):
        add_CNOT_like_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = gate_seq_Cplr, generator = generator)
    elif (index < 5184 + 5184 + 576):
        add_iSWAP_like_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = gate_seq_Cplr, generator = generator)
    elif (index < 576 + 5184 + 5184 + 576):
        add_SWAP_like_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = gate_seq_Cplr, generator = generator)
    else:
        raise ValueError(
            'index is out of range. it should be smaller than 11520 and '
            'greater or equal to 0: ', str(index))

    pass


def add_singleQ_S1(index, gate_seq):
    """Add single qubit clifford from S1.

    (I-like-subset of single qubit clifford group) (3)
    """
    if index == 0:
        gate_seq.append(gates.I)
        gate_seq.append(gates.I)  # auxiliary
        gate_seq.append(gates.I)  # auxiliary
    elif index == 1:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.I)  # auxiliary
    elif index == 2:
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.I)  # auxiliary


def add_singleQ_S1_X2p(index, gate_seq):
    """Add single qubit clifford from S1_X2p.

    (X2p-like-subset of single qubit clifford group) (3)
    """
    if index == 0:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.I)  # auxiliary
        gate_seq.append(gates.I)  # auxiliary
    elif index == 1:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.X2p)
    elif index == 2:
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.I)  # auxiliary
        gate_seq.append(gates.I)  # auxiliary


def add_singleQ_S1_Y2p(index, gate_seq):
    """Add single qubit clifford from S1_Y2p.

    (Y2p-like-subset of single qubit clifford group) (3)
    """
    if index == 0:
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.I)  # auxiliary
        gate_seq.append(gates.I)  # auxiliary
    elif index == 1:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Yp)
        gate_seq.append(gates.I)  # auxiliary
    elif index == 2:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.X2m)

def add_singleQ_S1_Z2p(index, gate_seq):
    """Add single qubit clifford from S1_Z2p.

    (Z2p-like-subset of single qubit clifford group) (3)
    """
    if index == 0:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.X2m)
    elif index == 1:
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.I)  # auxiliary
        gate_seq.append(gates.I)  # auxiliary
    elif index == 2:
        gate_seq.append(gates.Ym)
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.I)  # auxiliary

def add_singleQ_based_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = [], generator = 'CZ'):
    """Add single-qubit-gates-only-based two Qubit Clifford.

    (24*24 = 576)
    (gate_seq_1: gate seq. of qubit #1, gate_seq_t: gate seq. of qubit #2)
    """
    index_1 = index % 24  # randomly sample from single qubit cliffords (24)
    # randomly sample from single qubit cliffords (24)
    index_2 = (index // 24) % 24
    add_singleQ_clifford(index_1, gate_seq_1)
    add_singleQ_clifford(index_2, gate_seq_2)
    if generator == 'iSWAP_Cplr':
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler


def add_CNOT_like_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = [], generator = 'CZ'):
    """Add CNOT like two Qubit Clifford.

    (24*24*3*3 = 5184)
    (gate_seq_1: gate seq. of qubit #1, gate_seq_t: gate seq. of qubit #2)
    """
    index_1 = index % 3  # randomly sample from S1 (3)
    index_2 = (index // 3) % 3  # randomly sample from S1 (3)
    # randomly sample from single qubit cliffords (24)
    index_3 = (index // 3 // 3) % 24
    # randomly sample from single qubit cliffords (24)
    index_4 = (index // 3 // 3 // 24) % 24

    if generator == 'CZ':
        add_singleQ_S1(index_1, gate_seq_1)
        add_singleQ_S1_Y2p(index_2, gate_seq_2)
        gate_seq_1.append(gates.I)
        gate_seq_2.append(gates.CZ)
        add_singleQ_clifford(index_3, gate_seq_1)
        add_singleQ_clifford(index_4, gate_seq_2)

    elif generator == 'iSWAP_Cplr':
        add_singleQ_S1(index_1, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_S1_Z2p(index_2, gate_seq_2)
        
        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.iSWAP_Cplr)
        gate_seq_2.append(gates.I)

        gate_seq_1.append(gates.X2p)
        gate_seq_Cplr.append(gates.I)
        gate_seq_2.append(gates.I)
        
        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.iSWAP_Cplr)
        gate_seq_2.append(gates.I)

        add_singleQ_clifford(index_3, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_clifford(index_4, gate_seq_2)




def add_iSWAP_like_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = [], generator = 'CZ'):
    """Add iSWAP like two Qubit Clifford.

    (24*24*3*3 = 5184)
    (gate_seq_1: gate seq. of qubit #1, gate_seq_t: gate seq. of qubit #2)
    """
    index_1 = index % 3  # randomly sample from S1_Y2p (3)
    index_2 = (index // 3) % 3  # randomly sample from S1_X2p(3)
    # randomly sample from single qubit cliffords (24)
    index_3 = (index // 3 // 3) % 24
    # randomly sample from single qubit cliffords (24)
    index_4 = (index // 3 // 3 // 24) % 24

    if generator == 'CZ':
        add_singleQ_S1_Y2p(index_1, gate_seq_1)
        add_singleQ_S1_X2p(index_2, gate_seq_2)

        gate_seq_1.append(gates.I)
        gate_seq_2.append(gates.CZ)

        gate_seq_1.append(gates.Y2p)
        gate_seq_2.append(gates.X2m)

        gate_seq_1.append(gates.I)
        gate_seq_2.append(gates.CZ)

        add_singleQ_clifford(index_3, gate_seq_1)
        add_singleQ_clifford(index_4, gate_seq_2)

    elif generator == 'iSWAP_Cplr':
        add_singleQ_S1(index_1, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_S1(index_2, gate_seq_2)

        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.iSWAP_Cplr)
        gate_seq_2.append(gates.I)

        add_singleQ_clifford(index_3, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_clifford(index_4, gate_seq_2)




def add_SWAP_like_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = [], generator = 'CZ'):
    """Add SWAP like two Qubit Clifford.

    (24*24*= 576)
    (gate_seq_1: gate seq. of qubit #1, gate_seq_t: gate seq. of qubit #2)
    """
    index_1 = index % 24  # randomly sample from single qubit cliffords (24)
    # randomly sample from single qubit cliffords (24)
    index_2 = (index // 24) % 24
    if generator == 'CZ':
        gate_seq_1.append(gates.I)
        gate_seq_2.append(gates.Y2p)

        gate_seq_1.append(gates.I)
        gate_seq_2.append(gates.CZ)

        gate_seq_1.append(gates.Y2p)
        gate_seq_2.append(gates.Y2m)

        gate_seq_1.append(gates.I)
        gate_seq_2.append(gates.CZ)

        gate_seq_1.append(gates.Y2m)
        gate_seq_2.append(gates.Y2p)

        gate_seq_1.append(gates.I)
        gate_seq_2.append(gates.CZ)

        add_singleQ_clifford(index_1, gate_seq_1)
        add_singleQ_clifford(index_2, gate_seq_2)

    elif generator == 'iSWAP_Cplr':
        gate_seq_1.append(gates.I)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        gate_seq_2.append(gates.X2m)

        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.iSWAP_Cplr)
        gate_seq_2.append(gates.I)
        
        gate_seq_1.append(gates.X2m)
        gate_seq_Cplr.append(gates.I)
        gate_seq_2.append(gates.I)
        
        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.iSWAP_Cplr)
        gate_seq_2.append(gates.I)
        
        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.I)
        gate_seq_2.append(gates.X2m)
        
        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.iSWAP_Cplr)
        gate_seq_2.append(gates.I)

        add_singleQ_clifford(index_1, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_clifford(index_2, gate_seq_2)



class SingleQubit_RB(Sequence):
    """Single qubit randomized benchmarking."""

    prev_randomize = np.inf  # store the previous value
    prev_N_cliffords = np.inf  # store the previous value
    prev_interleave = np.inf  # store the previous value
    prev_interleaved_gate = np.inf  # store the previous value
    prev_sequence = ''
    prev_gate_seq = []

    def generate_sequence(self, config):
        """Generate sequence by adding gates/pulses to waveforms."""
        # get parameters
        sequence = config['Sequence']
        # Number of Cliffords to generate
        N_cliffords = int(config['Number of Cliffords'])
        randomize = config['Randomize']
        interleave = config['Interleave 1-QB Gate']
        multi_seq = config.get('Output multiple sequences', False)
        write_seq = config.get('Write sequence as txt file', False)

        log.info('Assign seed %d' %(randomize))
        rnd.seed(randomize)

        if interleave is True:
            interleaved_gate = config['Interleaved 1-QB Gate']
        else:
            interleaved_gate = np.inf
        
        # generate new randomized clifford gates only if configuration changes
        if (self.prev_sequence != sequence or
                self.prev_randomize != randomize or
                self.prev_N_cliffords != N_cliffords or
                self.prev_interleave != interleave or
                multi_seq or
                self.prev_interleaved_gate != interleaved_gate or
                self.prev_n_qubit != self.n_qubit):

            self.prev_randomize = randomize
            self.prev_N_cliffords = N_cliffords
            self.prev_interleave = interleave
            self.prev_sequence = sequence
            self.prev_n_qubit = self.n_qubit

            multi_gate_seq = []
            for n in range(self.n_qubit):
                # Generate 1QB RB sequence
                single_gate_seq = []

                for i in range(N_cliffords):
                    rndnum = rnd.randint(0, 23)
                    log.info('Random number %d' %(rndnum))
                    add_singleQ_clifford(rndnum, single_gate_seq,
                                         pad_with_I=False)
                    # If interleave gate,
                    if interleave is True:
                        self.prev_interleaved_gate = interleaved_gate
                        single_gate_seq.append(getattr(gates, interleaved_gate))

                recovery_gate = self.get_recovery_gate(single_gate_seq)

                # print 1QB-RB sequence
                if write_seq == True:
                    import os
                    from datetime import datetime
                    directory = os.path.join(path_currentdir,'1QB_RBseq')
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S-f')[:-3] + '_qbNum=%d_N_cliffords=%d_seed=%d.txt'%(n, N_cliffords,randomize)
                    filepath = os.path.join(directory,filename)
                    log.info('make file: ' + filepath)
                    with open(filepath, "w") as text_file:
                        print('New Sequence', file=text_file)
                        for i in range(len(single_gate_seq)):
                             print("CliffordIndex: %d, Gate: ["%(i) + cliffords.Gate_to_strGate(single_gate_seq[i]) +"]", file=text_file)
                        print("Corresponding recovery sequence:")
                        print("Recovery Gate: [" + cliffords.Gate_to_strGate(recovery_gate) +"]", file=text_file)
                single_gate_seq.append(recovery_gate)
                multi_gate_seq.append(single_gate_seq)
            
            # transpose list of lists
            multi_gate_seq = list(map(list, itertools.zip_longest(*multi_gate_seq, fillvalue=gates.I))) # Not to chop
            self.add_gates(multi_gate_seq)
            self.prev_gate_seq = multi_gate_seq
        else:
            self.add_gates(self.prev_gate_seq)

    def evaluate_sequence(self, gate_seq):
        """
        Evaluate a single qubit gate sequence.
        (Reference: http://www.vcpc.univie.ac.at/~ian/hotlist/qc/talks/bloch-sphere-rotations.pdf)

        Parameters
        ----------
        gate_seq_1: list of class Gate (defined in "gates.py")
            The gate sequence applied to a qubit

        Returns
        -------
        singleQ_gate: np.matrix
            The evaulation result.
        """

        singleQ_gate = np.matrix([[1, 0], [0, 1]])
        for i in range(len(gate_seq)):
            if (gate_seq[i] == gates.I):
                pass
            elif (gate_seq[i] == gates.X2p):
                singleQ_gate = np.matmul(
                    np.matrix([[1, -1j], [-1j, 1]]) / np.sqrt(2), singleQ_gate)
            elif (gate_seq[i] == gates.X2m):
                singleQ_gate = np.matmul(
                    np.matrix([[1, 1j], [1j, 1]]) / np.sqrt(2), singleQ_gate)
            elif (gate_seq[i] == gates.Y2p):
                singleQ_gate = np.matmul(
                    np.matrix([[1, -1], [1, 1]]) / np.sqrt(2), singleQ_gate)
            elif (gate_seq[i] == gates.Y2m):
                singleQ_gate = np.matmul(
                    np.matrix([[1, 1], [-1, 1]]) / np.sqrt(2), singleQ_gate)
            elif (gate_seq[i] == gates.Xp):
                singleQ_gate = np.matmul(
                    np.matrix([[0, -1j], [-1j, 0]]), singleQ_gate)
            elif (gate_seq[i] == gates.Xm):
                singleQ_gate = np.matmul(
                    np.matrix([[0, 1j], [1j, 0]]), singleQ_gate)
            elif (gate_seq[i] == gates.Yp):
                singleQ_gate = np.matmul(
                    np.matrix([[0, -1], [1, 0]]), singleQ_gate)
            elif (gate_seq[i] == gates.Ym):
                singleQ_gate = np.matmul(
                    np.matrix([[0, 1], [-1, 0]]), singleQ_gate)
            elif (gate_seq[i] in (gates.Zp, gates.VZp)):
                singleQ_gate = np.matmul(
                    np.matrix([[-1j, 0], [0, 1j]]), singleQ_gate)
        return singleQ_gate

    def get_recovery_gate(self, gate_seq):
        """
        Get the recovery (the inverse) gate

        Parameters
        ----------
        gate_seq: list of class Gate
            The gate sequence applied to a qubit

        Returns
        -------
        recovery_gate: Gate
            The recovery gate
        """

        qubit_state = np.matrix('1; 0')
        # initial state: ground state, following the QC community's convention
        qubit_state = np.matmul(self.evaluate_sequence(gate_seq), qubit_state)
        # find recovery gate which makes qubit_state return to initial state
        if (np.abs(np.linalg.norm(qubit_state.item((0, 0))) - 1) < 0.1):
            # ground state -> I
            recovery_gate = gates.I
        elif (np.abs(np.linalg.norm(qubit_state.item((1, 0))) - 1) < 0.1):
            # excited state -> X Pi
            recovery_gate = gates.Xp
        elif (np.linalg.norm(qubit_state.item((1, 0)) /
                             qubit_state.item((0, 0)) + 1) < 0.1):
            # X State  -> Y +Pi/2
            recovery_gate = gates.Y2p
        elif (np.linalg.norm(qubit_state.item((1, 0)) /
                             qubit_state.item((0, 0)) - 1) < 0.1):
            # -X State -> Y -Pi/2
            recovery_gate = gates.Y2m
        elif (np.linalg.norm(qubit_state.item((1, 0)) /
                             qubit_state.item((0, 0)) + 1j) < 0.1):
            # Y State -> X -Pi/2
            recovery_gate = gates.X2m
        elif (np.linalg.norm(qubit_state.item((1, 0)) /
                             qubit_state.item((0, 0)) - 1j) < 0.1):
            # -Y State -> X +Pi/2
            recovery_gate = gates.X2p
        else:
            raise InstrumentDriver.Error(
                'Error in calculating recovery gates. qubit state:' +
                str(qubit_state))
        return recovery_gate


class TwoQubit_RB(Sequence):
    """Two qubit randomized benchmarking."""

    prev_randomize = np.inf  # store the previous value
    prev_N_cliffords = np.inf  # store the previous value
    prev_interleave = np.inf  # store the previous value
    prev_interleaved_gate = np.inf  # store the previous value
    prev_sequence = ''
    prev_generator = ''
    prev_gate_seq = []

    filepath_lookup_table = ""

    def generate_sequence(self, config):
        """
        Generate sequence by adding gates/pulses to waveforms.

        Parameters
        ----------
        config: dict
            configuration

        Returns
        -------
        """

        # get parameters

        sequence = config['Sequence']
        # Number of Cliffords to generate
        N_cliffords = int(config['Number of Cliffords'])
        randomize = config['Randomize']
        interleave = config['Interleave 2-QB Gate']
        multi_seq = config.get('Output multiple sequences', False)
        write_seq = config.get('Write sequence as txt file', False)
        generator = config.get('Native 2-QB gate', 'CZ')

        rnd.seed(randomize)
        if interleave is True:
            interleaved_gate = config['Interleaved 2-QB Gate']
        else:
            interleaved_gate = np.inf

        # generate new randomized clifford gates only if configuration changes
        if (self.prev_sequence != sequence or
                self.prev_randomize != randomize or
                self.prev_N_cliffords != N_cliffords or
                self.prev_interleave != interleave or
                self.prev_generator != generator or 
                multi_seq or
                self.prev_interleaved_gate != interleaved_gate):

            self.prev_randomize = randomize
            self.prev_N_cliffords = N_cliffords
            self.prev_interleave = interleave
            self.prev_sequence = sequence

            multi_gate_seq = []

            # Generate 2QB RB sequence
            if generator == 'CZ':
                # qubits_to_benchmark = np.fromstring(
                #     config['Qubits to Benchmark'], dtype=int, sep='-') - 1
                qubits_to_benchmark = [int(config['Qubits to Benchmark'][0]) - 1, 
                    int(config['Qubits to Benchmark'][2]) - 1] 

            elif generator == 'iSWAP_Cplr':
                qubits_to_benchmark = [int(config['Qubits to Benchmark (Coupler)'][0]) - 1, 
                    int(config['Qubits to Benchmark (Coupler)'][2]) - 1,
                    int(config['Qubits to Benchmark (Coupler)'][4]) - 1,] 
                # qubits_to_benchmark = np.fromstring(
                #     config['Qubits to Benchmark (Coupler)'], dtype=int, sep='-') - 1

            clifford_seq_1 = []
            clifford_seq_2 = []
            clifford_seq_Cplr = []

            for j in range(N_cliffords):
                log.info('Seed number: %d'%(randomize))
                rndnum = rnd.randint(0, 11519)
                # rndnum = rnd.randint(0, 576) #Only applying single qubit gates
                add_twoQ_clifford(rndnum, clifford_seq_1, clifford_seq_2, gate_seq_Cplr = clifford_seq_Cplr, generator = generator)
                # If interleave gate,
                if interleave is True:
                    self.prev_interleaved_gate = interleaved_gate
                    if interleaved_gate == 'CZ':
                        clifford_seq_1.append(gates.I)
                        interleaved_gate.append(gates.CZ)
                    elif interleaved_gate == 'CZEcho':
                        # CZEcho is a composite gate, so get each gate
                        gate = gates.CZEcho
                        for g in gate.sequence:
                            clifford_seq_1.append(g[1])
                            clifford_seq_2.append(g[0])
                    elif interleaved_gate == 'iSWAP_Cplr':
                        gate = gates.iSWAP_Cplr
                        clifford_seq_1.append(g[2])
                        clifford_seq_Cplr.append(g[1])
                        clifford_seq_2.append(g[0])
                    elif interleaved_gate == 'I':
                        # TBA: adjust the duration of I gates?
                        # log.info('Qubits to benchmark: ' + str(qubits_to_benchmark))
                        # gate = gates.I(width = self.pulses_2qb[qubit]).value
                        clifford_seq_1.append(gates.I)
                        clifford_seq_2.append(gates.I)
                        if not clifford_seq_Cplr: # If clifford_seq_Cplr is an empty list,
                            clifford_seq_Cplr.append(Gate.I)

            # remove redundant Identity gates for clifford_seq_1
            index_identity_clifford = [] # find where Identity gates are
            for p in range(len(clifford_seq_1)):
                if not clifford_seq_Cplr: # If clifford_seq_Cplr is an empty list,
                    if (clifford_seq_1[p] == gates.I and clifford_seq_2[p] == gates.I):
                        index_identity_clifford.append(p)
                else:
                    if (clifford_seq_1[p] == gates.I and clifford_seq_2[p] == gates.I and clifford_seq_Cplr[p] == gates.I):
                        index_identity_clifford.append(p)

            clifford_seq_1 = [m for n, m in enumerate(clifford_seq_1) if n not in index_identity_clifford]
            clifford_seq_2 = [m for n, m in enumerate(clifford_seq_2) if n not in index_identity_clifford]

            if clifford_seq_Cplr: 
                clifford_seq_Cplr = [m for n, m in enumerate(clifford_seq_Cplr) if n not in index_identity_clifford]


            log.info('*** clifford sequence *** ')
            log.info("QB1 clifford gate sequence: " + str([cliffords.Gate_to_strGate(g) for g in clifford_seq_1]))
            log.info("QB2 clifford gate sequence: " + str([cliffords.Gate_to_strGate(g) for g in clifford_seq_2]))
            if generator == 'iSWAP_Cplr':
                log.info("Coupler clifford gate sequence: " + str([cliffords.Gate_to_strGate(g) for g in clifford_seq_Cplr]))
            log.info("=================================================")
            # get recovery gate seq
            (recovery_seq_1, recovery_seq_2, recovery_seq_Cplr) = self.get_recovery_gate(
                clifford_seq_1, clifford_seq_2, config, gate_seq_Cplr = clifford_seq_Cplr, generator = generator)

            # Remove redundant identity gates in recovery gate seq
            index_identity_recovery = [] # find where Identity gates are
            for p in range(len(recovery_seq_1)):
                if not recovery_seq_Cplr: # If recovery_seq_Cplr is an empty list,
                    if (recovery_seq_1[p] == gates.I and recovery_seq_2[p] == gates.I):
                        index_identity_recovery.append(p)
                else:
                    if (recovery_seq_1[p] == gates.I and recovery_seq_2[p] == gates.I and recovery_seq_Cplr[p] == gates.I):
                        index_identity_recovery.append(p)

            recovery_seq_1 = [m for n, m in enumerate(recovery_seq_1) if n not in index_identity_recovery]
            recovery_seq_2 = [m for n, m in enumerate(recovery_seq_2) if n not in index_identity_recovery]
            if recovery_seq_Cplr:
                recovery_seq_Cplr = [m for n, m in enumerate(recovery_seq_Cplr) if n not in index_identity_recovery]

            # Construct the total gate sequence.
            gate_seq_1 = []
            gate_seq_2 = []
            gate_seq_Cplr = []
            gate_seq_1.extend(clifford_seq_1)
            gate_seq_1.extend(recovery_seq_1)
            gate_seq_2.extend(clifford_seq_2)
            gate_seq_2.extend(recovery_seq_2)
            gate_seq_Cplr.extend(clifford_seq_Cplr)
            gate_seq_Cplr.extend(recovery_seq_Cplr)

            # test the recovery gate
            psi_gnd = np.matrix('1; 0; 0; 0') # ground state |00>
            if write_seq == True:
                import os
                from datetime import datetime
                directory = os.path.join(path_currentdir,'2QB_RBseq')
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S-f')[:-3] + '_N_cliffords=%d_seed=%d.txt'%(N_cliffords,randomize)
                filepath = os.path.join(directory,filename)
                log.info('make file: ' + filepath)
                with open(filepath, "w") as text_file:

                    # print total gate sequence
                    for i in range(len(gate_seq_1)):
                        str_Gate = "Index: %d, Gate: ["%(i) + cliffords.Gate_to_strGate(gate_seq_1[i]) + ", " + cliffords.Gate_to_strGate(gate_seq_2[i]) 
                        if (generator == 'iSWAP_Cplr'):
                            str_Gate = str_Gate + ', ' + cliffords.Gate_to_strGate(gate_seq_Cplr[i]) + ']' 
                        else:
                            str_Gate = str_Gate + ']'
                        print(str_Gate, file=text_file)

                    # print clifford gate sequence
                    for i in range(len(clifford_seq_1)):
                        str_Clifford = "CliffordIndex: %d, Gate: ["%(i) + cliffords.Gate_to_strGate(clifford_seq_1[i]) + ", " + cliffords.Gate_to_strGate(clifford_seq_2[i])
                        if (generator == 'iSWAP_Cplr'):
                            str_Clifford = str_Clifford + ', ' + cliffords.Gate_to_strGate(clifford_seq_Cplr[i]) + ']' 
                        else:
                            str_Clifford = str_Clifford + ']'
                        print(str_Clifford, file=text_file)

                    # print recovery clifford gate sequence
                    for i in range(len(recovery_seq_1)):
                        str_Recovery = "RecoveryIndex: %d, Gate: ["%(i) + cliffords.Gate_to_strGate(recovery_seq_1[i]) + ", " + cliffords.Gate_to_strGate(recovery_seq_2[i])
                        if (generator == 'iSWAP_Cplr'):
                            str_Recovery = str_Recovery + ', ' + cliffords.Gate_to_strGate(recovery_seq_Cplr[i]) + ']'
                        else:
                            str_Recovery = str_Recovery + ']'
                        print(str_Recovery, file=text_file)

            psi = np.matmul(self.evaluate_sequence(gate_seq_1, gate_seq_2, gate_seq_Cplr = gate_seq_Cplr, generator = generator), psi_gnd)

            np.set_printoptions(precision=2)
            log.info('The matrix of the overall gate sequence:')
            log.info(self.evaluate_sequence(gate_seq_1, gate_seq_2, gate_seq_Cplr = gate_seq_Cplr, generator = generator))

            log.info('--- TESTING THE RECOVERY GATE ---')
            log.info('The probability amplitude of the final state vector: ' + str(np.matrix(psi).flatten()))
            log.info('The population of the ground state after the gate sequence: %.4f'%(np.abs(psi[0,0])**2))
            log.info('-------------------------------------------')

            # Assign two qubit gate sequence to where we want
            if (generator == 'CZ'):
                # for i in range(qubits_to_benchmark[0] - 1):
                #     multi_gate_seq.append([None] * len(gate_seq_1))
                multi_gate_seq.append(gate_seq_2)
                multi_gate_seq.append(gate_seq_1)
                # for i in range(self.n_qubit - qubits_to_benchmark[1]):
                #     multi_gate_seq.append([None] * len(gate_seq_1))
            elif (generator == 'iSWAP_Cplr'):
                # for i in range(qubits_to_benchmark[0] - 1):
                #     multi_gate_seq.append([None] * len(gate_seq_1))
                multi_gate_seq.append(gate_seq_2)
                multi_gate_seq.append(gate_seq_Cplr)
                multi_gate_seq.append(gate_seq_1)
                # for i in range(self.n_qubit - qubits_to_benchmark[2]):
                #     multi_gate_seq.append([None] * len(gate_seq_1))


            # transpose list of lists
            multi_gate_seq = list(map(list, itertools.zip_longest(*multi_gate_seq, fillvalue=gates.I))) # Not to chop

            for gate_seq in multi_gate_seq:
                if generator == 'CZ':
                    if gate_seq[0] == gates.CZ:
                        # self.add_gate(qubit=[0, 1], gate=gate_seq[0])
                        self.add_gate(qubit = qubits_to_benchmark, gate = gate_seq[0])
                    else:
                        # self.add_gate(qubit=[0, 1], gate=gate_seq)
                        self.add_gate(qubit = qubits_to_benchmark, gate = gate_seq)
                elif generator == 'iSWAP_Cplr':
                    if gate_seq[1] == gates.iSWAP_Cplr:
                        # self.add_gate(qubit=[0, 1, 2], gate=gate_seq[1])
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[1])
                    else:
                        # self.add_gate(qubit=[0, 1, 2], gate=gate_seq)
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)


            self.prev_gate_seq = multi_gate_seq
        else:
            for gate_seq in self.prev_gate_seq:
                if generator == 'CZ':
                    if gate_seq[0] == gates.CZ:
                        # self.add_gate(qubit=[0, 1], gate=gate_seq[0])
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[0])
                    else:
                        # self.add_gate(qubit=[0, 1], gate=gate_seq)
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)
                elif generator == 'iSWAP_Cplr':
                    if gate_seq[1] == gates.iSWAP_Cplr:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[1])
                    else:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)

    def evaluate_sequence(self, gate_seq_1, gate_seq_2, gate_seq_Cplr = None, generator = 'CZ'):
        """
        Evaluate the two qubit gate sequence.

        Parameters
        ----------
        gate_seq_1: list of class Gate (defined in "gates.py")
            The gate sequence applied to Qubit "1"

        gate_seq_2: list of class Gate (defined in "gates.py")
            The gate sequence applied to Qubit "2"

        gate_seq_Cplr: list of class Gate (defined in "gates.py")
            The gate sequence applied to Coupler (optional)

        generator: string
            Type of Native 2QB gate (optional)

        Returns
        -------
        twoQ_gate: np.matrix (shape = (4,4))
            The evaulation result.
        """
        twoQ_gate = np.matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for i in range(len(gate_seq_1)):
            gate_1 = np.matrix([[1, 0], [0, 1]])
            gate_2 = np.matrix([[1, 0], [0, 1]])
            if (gate_seq_1[i] == gates.I):
                pass
            elif (gate_seq_1[i] == gates.X2p):
                gate_1 = np.matmul(
                    np.matrix([[1, -1j], [-1j, 1]]) / np.sqrt(2), gate_1)
            elif (gate_seq_1[i] == gates.X2m):
                gate_1 = np.matmul(
                    np.matrix([[1, 1j], [1j, 1]]) / np.sqrt(2), gate_1)
            elif (gate_seq_1[i] == gates.Y2p):
                gate_1 = np.matmul(
                    np.matrix([[1, -1], [1, 1]]) / np.sqrt(2), gate_1)
            elif (gate_seq_1[i] == gates.Y2m):
                gate_1 = np.matmul(
                    np.matrix([[1, 1], [-1, 1]]) / np.sqrt(2), gate_1)
            elif (gate_seq_1[i] == gates.Xp):
                gate_1 = np.matmul(np.matrix([[0, -1j], [-1j, 0]]), gate_1)
            elif (gate_seq_1[i] == gates.Xm):
                gate_1 = np.matmul(np.matrix([[0, 1j], [1j, 0]]), gate_1)
            elif (gate_seq_1[i] == gates.Yp):
                gate_1 = np.matmul(np.matrix([[0, -1], [1, 0]]), gate_1)
            elif (gate_seq_1[i] == gates.Ym):
                gate_1 = np.matmul(np.matrix([[0, 1], [-1, 0]]), gate_1)

            if (gate_seq_2[i] == gates.I):
                pass
            elif (gate_seq_2[i] == gates.X2p):
                gate_2 = np.matmul(
                    np.matrix([[1, -1j], [-1j, 1]]) / np.sqrt(2), gate_2)
            elif (gate_seq_2[i] == gates.X2m):
                gate_2 = np.matmul(
                    np.matrix([[1, 1j], [1j, 1]]) / np.sqrt(2), gate_2)
            elif (gate_seq_2[i] == gates.Y2p):
                gate_2 = np.matmul(
                    np.matrix([[1, -1], [1, 1]]) / np.sqrt(2), gate_2)
            elif (gate_seq_2[i] == gates.Y2m):
                gate_2 = np.matmul(
                    np.matrix([[1, 1], [-1, 1]]) / np.sqrt(2), gate_2)
            elif (gate_seq_2[i] == gates.Xp):
                gate_2 = np.matmul(np.matrix([[0, -1j], [-1j, 0]]), gate_2)
            elif (gate_seq_2[i] == gates.Xm):
                gate_2 = np.matmul(np.matrix([[0, 1j], [1j, 0]]), gate_2)
            elif (gate_seq_2[i] == gates.Yp):
                gate_2 = np.matmul(np.matrix([[0, -1], [1, 0]]), gate_2)
            elif (gate_seq_2[i] == gates.Ym):
                gate_2 = np.matmul(np.matrix([[0, 1], [-1, 0]]), gate_2)

            gate_12 = np.kron(gate_1, gate_2)
            if generator == 'CZ':
                if (gate_seq_1[i] == gates.CZ or gate_seq_2[i] == gates.CZ):
                    gate_12 = np.matmul(
                        np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                   [0, 0, 0, -1]]), gate_12)
            elif generator == 'iSWAP_Cplr':
                if gate_seq_Cplr[i] == gates.iSWAP_Cplr:
                    gate_12 = np.matmul(
                        np.matrix([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0],
                                   [0, 0, 0, 1]]), gate_12)
            twoQ_gate = np.matmul(gate_12, twoQ_gate)
        # log.info('two qubit gate: ' + str(twoQ_gate))
        return twoQ_gate

    def get_recovery_gate(self, gate_seq_1, gate_seq_2, config, gate_seq_Cplr = [], generator = 'CZ'):
        """
        Get the recovery (the inverse) gate

        Parameters
        ----------
        gate_seq_1: list of class Gate
            The gate sequence applied to Qubit "1"

        gate_seq_2: list of class Gate
            The gate sequence applied to Qubit "2"

        config: dict
            The configuration

        gate_seq_Cplr: list of class Gate (defined in "gates.py")
            The gate sequence applied to Coupler (optional)

        generator: string
            Type of Native 2QB gate (optional)

        Returns
        -------
        (recovery_seq_1, recovery_seq_2, cheapest_recovery_seq_Cplr): tuple of the lists
            The recovery gate
        """


        qubit_state = np.matrix(
            '1; 0; 0; 0')  # initial state: ground state |00>

        qubit_state = np.matmul(self.evaluate_sequence(
            gate_seq_1, gate_seq_2, gate_seq_Cplr = gate_seq_Cplr, generator = generator), qubit_state)

        # find recovery gate which makes qubit_state return to initial state
        total_num_cliffords = 11520
        recovery_seq_1 = []
        recovery_seq_2 = []
        recovery_seq_Cplr = []

        # Search the recovery gate in two Qubit clifford group
        find_cheapest = config['Find the cheapest recovery Clifford']

        cheapest_recovery_seq_1 = []
        cheapest_recovery_seq_2 = []
        cheapest_recovery_seq_Cplr = []

        log.info('*** get recovery gate *** ')
        if (find_cheapest == True):
            min_N_2QB_gate = np.inf
            min_N_1QB_gate = np.inf
            max_N_I_gate = -np.inf
            cheapest_index = None

            use_lookup_table = config['Use a look-up table']
            if (use_lookup_table == True):
                filepath_lookup_table = config['File path of the look-up table']
                if len(filepath_lookup_table) == 0:
                    if (generator == 'CZ'):
                        filepath_lookup_table = os.path.join(path_currentdir, 'recovery_rb_table_CZ.pickle')
                    elif (generator == 'iSWAP_Cplr'):
                        filepath_lookup_table = os.path.join(path_currentdir, 'recovery_rb_table_iSWAP_Cplr.pickle')
                if filepath_lookup_table != self.filepath_lookup_table:
                    log.info("Load Look-up table.")
                    self.filepath_lookup_table = filepath_lookup_table
                    self.dict_lookup_table = cliffords.loadData(filepath_lookup_table)
                stabilizer = cliffords.get_stabilizer(qubit_state)
                for index, item in enumerate(self.dict_lookup_table['psi_stabilizer']):
                    if stabilizer == item:
                        seq1 = self.dict_lookup_table['recovery_gates_QB1'][index]
                        for str_Gate in seq1:
                            cheapest_recovery_seq_1.append(cliffords.strGate_to_Gate(str_Gate))
                        seq2 = self.dict_lookup_table['recovery_gates_QB2'][index]
                        for str_Gate in seq2:
                            cheapest_recovery_seq_2.append(cliffords.strGate_to_Gate(str_Gate))

                        if (generator == 'iSWAP_Cplr'):
                            seqCplr = self.dict_lookup_table['recovery_gates_Cplr'][index]
                            for str_Gate in seqCplr:
                                cheapest_recovery_seq_Cplr.append(cliffords.strGate_to_Gate(str_Gate))

                        log.info("=== FOUND THE CHEAPEST RECOVERY GATE IN THE LOOK-UP TABLE. ===")
                        log.info("QB1 recovery gate sequence: " + str(seq1))
                        log.info("QB2 recovery gate sequence: " + str(seq2))
                        if generator == 'iSWAP_Cplr':
                            log.info("Coupler recovery gate sequence: " + str (seqCplr))
                        log.info("=================================================")
                        return (cheapest_recovery_seq_1, cheapest_recovery_seq_2, cheapest_recovery_seq_Cplr)

            log.info("--- COULDN'T FIND THE RECOVERY GATE IN THE LOOK-UP TABLE... ---")


        # Calculate the matrix of the clifford sequence
        matrix_cliffords = self.evaluate_sequence(gate_seq_1,gate_seq_2, gate_seq_Cplr = gate_seq_Cplr, generator = generator)

        for i in range(total_num_cliffords):
            recovery_seq_1 = []
            recovery_seq_2 = []
            recovery_seq_Cplr = []
            add_twoQ_clifford(i, recovery_seq_1, recovery_seq_2, gate_seq_Cplr = recovery_seq_Cplr, generator = generator)

            # Calculate the matrix of the recovery clifford
            matrix_recovery = self.evaluate_sequence(recovery_seq_1, recovery_seq_2, gate_seq_Cplr = recovery_seq_Cplr, generator = generator)

            # Calculate the matrix of the total clifford sequence
            matrix_total = np.matmul(matrix_recovery,matrix_cliffords)
            if (CheckIdentity(matrix_total)):
                log.info("QB1 recovery gate sequence: " + str([cliffords.Gate_to_strGate(g) for g in recovery_seq_1]))
                log.info("QB2 recovery gate sequence: " + str([cliffords.Gate_to_strGate(g) for g in recovery_seq_2]))
                if generator == 'iSWAP_Cplr':
                    log.info("Coupler recovery gate sequence: " + str([cliffords.Gate_to_strGate(g) for g in recovery_seq_Cplr]))
                return (recovery_seq_1, recovery_seq_2, recovery_seq_Cplr)

        raise ValueErorr('Recovery sequence not found!')

if __name__ == '__main__':
    pass
