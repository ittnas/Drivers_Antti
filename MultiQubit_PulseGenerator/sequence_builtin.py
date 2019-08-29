#!/usr/bin/env python3
# add logger, to allow logging to Labber's instrument log
import logging

import numpy as np

import gates
from sequence import Sequence

log = logging.getLogger('LabberDriver')


class Rabi(Sequence):
    """Sequence for driving Rabi oscillations in multiple qubits."""

    def generate_sequence(self, config):
        """Generate sequence by adding gates/pulses to waveforms."""
        # just add pi-pulses for the number of available qubits
        self.add_gate_to_all(gates.Xp, align='right')


class CPMG(Sequence):
    """Sequence for multi-qubit Ramsey/Echo/CMPG experiments."""

    def generate_sequence(self, config):
        """Generate sequence by adding gates/pulses to waveforms."""
        # get parameters
        n_pulse = int(config['# of pi pulses'])
        pi_to_q = config['Add pi pulses to Q']
        duration = config['Sequence duration']
        edge_to_edge = config['Edge-to-edge pulses']
        measure_2nd = bool(config['Measure T1 for 2nd Excited State'])
        # select type of refocusing pi pulse
        if (measure_2nd == True):
            gate_pi = gates.Yp_12 if pi_to_q else gates.Xp_12
        else:
            gate_pi = gates.Yp if pi_to_q else gates.Xp

        # always do T1 same way, regardless if edge-to-edge or center-center
        if n_pulse < 0:
            self.add_gate_to_all(gates.Xp)
            if (measure_2nd == True):
                self.add_gate_to_all(gates.Xp_12)
                # self.add_gate_to_all(gate_pi)
                # self.add_gate(qubit=2, gate=gates.Xp_12)
            self.add_gate_to_all(gates.IdentityGate(width=duration), dt=0)

        elif edge_to_edge:
            # edge-to-edge pulsing, set pulse separations
            if (measure_2nd == True):
                self.add_gate_to_all(gates.Xp)
                self.add_gate_to_all(gates.X2p_12)
            else:
                self.add_gate_to_all(gates.X2p)
            # for ramsey, just add final pulse
            if n_pulse == 0:
                if (measure_2nd == True):
                    self.add_gate_to_all(gates.X2p_12, dt=duration)
                else:
                    self.add_gate_to_all(gates.X2p, dt=duration)
            else:
                dt = duration / n_pulse
                # add first pi pulse after half duration
                self.add_gate_to_all(gate_pi, dt=dt/2)
                # add rest of pi pulses
                for i in range(n_pulse - 1):
                    self.add_gate_to_all(gate_pi, dt=dt)

                if (measure_2nd == True):
                    # add final pi/2 pulse
                    self.add_gate_to_all(gates.X2p_12, dt=dt/2)
                else:
                    # add final pi/2 pulse
                    self.add_gate_to_all(gates.X2p, dt=dt/2)

        else:
            if (measure_2nd == True):
                self.add_gate_to_all(gates.Xp)
                # center-to-center spacing, set absolute pulse positions
                self.add_gate_to_all(gates.X2p_12, t0=0)
                # add pi pulses at right position
                for i in range(n_pulse):
                    self.add_gate_to_all(gate_pi,
                                         t0=(i + 0.5) * (duration / n_pulse))
                # add final pi/2 pulse
                self.add_gate_to_all(gates.X2p_12, t0=duration)

            else:
                # center-to-center spacing, set absolute pulse positions
                self.add_gate_to_all(gates.X2p, t0=0)
                # add pi pulses at right position
                for i in range(n_pulse):
                    self.add_gate_to_all(gate_pi,
                                         t0=(i + 0.5) * (duration / n_pulse))
                # add final pi/2 pulse
                self.add_gate_to_all(gates.X2p, t0=duration)


class PulseTrain(Sequence):
    """Sequence for multi-qubit pulse trains, for pulse calibrations."""

    def generate_sequence(self, config):
        """Generate sequence by adding gates/pulses to waveforms."""
        # get parameters
        n_pulse = int(config['# of pulses'])
        alternate = config['Alternate pulse direction']

        if n_pulse == 0:
            self.add_gate_to_all(gates.I)
        for n in range(n_pulse):
            pulse_type = config['Pulse']
            if pulse_type == 'CPh':
                if alternate and (n % 2) == 1:
                    gate = gates.CPHASE(negative_amplitude=True)
                else:
                    gate = gates.CPHASE(negative_amplitude=False)
                for i in range(self.n_qubit-1):
                    self.add_gate([i, i+1], gate)
            elif pulse_type == 'NetZero':
                for i in range(self.n_qubit-1):
                    self.add_gate([i, i+1], gates.NetZero)
            else:
                if alternate and (n % 2) == 1:
                    pulse_type = pulse_type.replace('p', 'm')
                gate = getattr(gates, pulse_type)
                self.add_gate_to_all(gate)


class SpinLocking(Sequence):
    """ Sequence for spin-locking experiment.

    """

    def generate_sequence(self, config):
        """Generate sequence by adding gates/pulses to waveforms."""

        # pulse_amps = []
        # for ii in range(9):
        #     pulse_amps.append(
        #         float(config['Drive pulse amplitude #' + str(ii + 1)]))
        # pulse_duration = float(config['Drive pulse duration'])
        # pulse_phase = float(config['Drive pulse phase']) / 180.0 * np.pi
        pulse_sequence = config['Pulse sequence']
        SL_level = config['Spin-locking Level']
        if (SL_level == '0-1'):
            if (pulse_sequence == 'SL-3' or pulse_sequence == 'SL-3Y' or pulse_sequence == 'SL-3Z') :
                self.add_gate_to_all(gates.Y2p)
            elif pulse_sequence == 'SL-5a':
                self.add_gate_to_all(gates.Y2m)
            elif pulse_sequence == 'SL-5b':
                self.add_gate_to_all(gates.Y2p)

            if (pulse_sequence == 'SL-1'):
                pass
            elif (pulse_sequence == 'SL-3' or pulse_sequence == 'SL-3Y' or pulse_sequence == 'SL-3Z'):
                pass
            else:
                self.add_gate_to_all(gates.Xp)

            # rabi_gates = []
            # for ii in range(self.n_qubit):
            #     rabi_gates.append(
            #         gates.RabiGate(pulse_amps[ii], pulse_duration, pulse_phase))
            # self.add_gate(list(range(self.n_qubit)), rabi_gates)

            # Use custom-made spin locking gate
            self.add_gate_to_all(gates.SL_X)
            
            if (pulse_sequence == 'SL-1'):
                pass
            elif (pulse_sequence == 'SL-3' or pulse_sequence == 'SL-3Y' or pulse_sequence == 'SL-3Z'):
                pass
            else:
                self.add_gate_to_all(gates.Xp)

            if pulse_sequence == 'SL-3':
                self.add_gate_to_all(gates.Y2p)
            elif pulse_sequence == 'SL-5a':
                self.add_gate_to_all(gates.Y2m)
            elif pulse_sequence == 'SL-5b':
                self.add_gate_to_all(gates.Y2p)
            elif pulse_sequence == 'SL-3Y':
                self.add_gate_to_all(gates.X2p)
            elif pulse_sequence == 'SL-3Z':
                pass
        if (SL_level == '1-2'):
            if pulse_sequence == 'SL-3':
                self.add_gate_to_all(gates.Y2p_12)
            if pulse_sequence == 'SL-5a':
                self.add_gate_to_all(gates.Y2m_12)
            if pulse_sequence == 'SL-5b':
                self.add_gate_to_all(gates.Y2p_12)

            if (pulse_sequence == 'SL-1'):
                pass
            elif (pulse_sequence == 'SL-3' or pulse_sequence == 'SL-3Y' or pulse_sequence == 'SL-3Z'):
                pass
            else:
                self.add_gate_to_all(gates.Xp_12)

            # Use custom-made spin locking gate
            self.add_gate_to_all(gates.SL_X)
            
            if (pulse_sequence == 'SL-1'):
                pass
            elif (pulse_sequence == 'SL-3' or pulse_sequence == 'SL-3Y' or pulse_sequence == 'SL-3Z'):
                pass
            else:
                self.add_gate_to_all(gates.Xp_12)

            if pulse_sequence == 'SL-3':
                self.add_gate_to_all(gates.Y2p_12)
            elif pulse_sequence == 'SL-5a':
                self.add_gate_to_all(gates.Y2m_12)
            elif pulse_sequence == 'SL-5b':
                self.add_gate_to_all(gates.Y2p_12)
            elif pulse_sequence == 'SL-3Y':
                self.add_gate_to_all(gates.X2p_12)
            elif pulse_sequence == 'SL-3Z':
                pass

        return

class iSWAP_Cplr(Sequence):
    """ Sequence for iSWAP using tunable couplergate.
    """

    def generate_sequence(self, config):
        """Generate sequence by adding gates/pulses to waveforms."""
        # just add pi-pulses for the number of available qubits

        self.add_gate(qubit=[0, 1, 2], gate=gates.iSWAP_Cplr)

if __name__ == '__main__':
    pass
