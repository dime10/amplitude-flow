import sys
sys.path.insert(0, './intel-qs/build/lib/')
import intelqs_py as sim

import numpy as np
from pennylane import QubitDevice
from matplotlib import pyplot as plt, patches, patheffects, colors


bmcg_map_colors = ['#000000', '#030103', '#070206', '#0b030a', '#0f050d', '#130611', '#170714', '#1a0818', '#1e0a1b', '#220b1f', '#260c22', '#2a0e26', '#2e0f29', '#31102d', '#351130', '#391333', '#3d1437', '#41153a', '#45163e', '#491841', '#4c1945', '#501a48', '#541c4c', '#581d4f', '#5c1e53', '#601f56', '#63215a', '#67225d', '#6b2360', '#6f2464', '#732667', '#77276b', '#7a286e', '#7e2a72', '#822b75', '#862c79', '#8a2d7c', '#8e2f80', '#923083', '#953187', '#99328a', '#9d348e', '#a13591', '#a53694', '#a93898', '#ab399b', '#a83b9c', '#a63c9c', '#a43e9d', '#a1409e', '#9f429f', '#9c44a0', '#9a45a0', '#9847a1', '#9549a2', '#934ba3', '#904ca4', '#8e4ea4', '#8c50a5', '#8952a6', '#8753a7', '#8455a8', '#8257a8', '#7f59a9', '#7d5aaa', '#7b5cab', '#785eac', '#7660ac', '#7362ad', '#7163ae', '#6f65af', '#6c67b0', '#6a69b0', '#676ab1', '#656cb2', '#626eb3', '#6070b4', '#5e71b4', '#5b73b5', '#5975b6', '#5677b7', '#5478b8', '#527ab8', '#4f7cb9', '#4d7eba', '#4a80bb', '#4881bc', '#4683bc', '#4385bd', '#4187be', '#4088bd', '#4088ba', '#4188b7', '#4189b4', '#4189b0', '#428aad', '#428aaa', '#438aa7', '#438ba4', '#448ba1', '#448b9e', '#448c9b', '#458c97', '#458d94', '#468d91', '#468d8e', '#478e8b', '#478e88', '#478f85', '#488f81', '#488f7e', '#49907b', '#499078', '#499075', '#4a9172', '#4a916f', '#4b926c', '#4b9268', '#4c9265', '#4c9362', '#4c935f', '#4d935c', '#4d9459', '#4e9456', '#4e9553', '#4f954f', '#4f954c', '#4f9649', '#509646', '#509743', '#519740', '#51973d', '#52983a', '#529836', '#529833', '#519632', '#4f9331', '#4d8f2f', '#4c8c2e', '#4a882d', '#48852c', '#46812b', '#447e2a', '#427b29', '#407727', '#3f7426', '#3d7025', '#3b6d24', '#396923', '#376622', '#356321', '#335f1f', '#325c1e', '#30581d', '#2e551c', '#2c521b', '#2a4e1a', '#284b19', '#264717', '#254416', '#234015', '#213d14', '#1f3a13', '#1d3612', '#1b3311', '#192f0f', '#182c0e', '#16290d', '#14250c', '#12220b', '#101e0a', '#0e1b09', '#0c1707', '#0b1406', '#091105', '#070d04', '#050a03', '#030602', '#010301', '#000000']
bmcg_map = colors.ListedColormap(bmcg_map_colors)
color_map = bmcg_map


class AmplitudeFlow(QubitDevice):
    """The Amplitude Flow Drawer visualizes a quantum circuit by drawing a wire for each
       computational basis state. The wires carry the complex amplitudes associated to the basis
       states by way of contrast (magnitude) and color (phase). Operations change the amplitudes
       via 3 fundamental mechanims: amplitude transfer, wire permutations, and phase shifts.
    """
    name = 'Amplitude Flow Drawer'
    short_name = 'ampflow'
    pennylane_requires = '0.20.0'
    version = '0.1'
    author = 'David Ittah'
    operations = {
        "Identity",
        "PauliX", "PauliY", "PauliZ",
        "Hadamard",
        "RY", "RZ",
        "S", "T",
        "PhaseShift",
        "CNOT", "Toffoli",
    }
    observables = {
        "Identity"
    }

    def __init__(self, n_qubits, n_layers=32):
        super().__init__(wires=n_qubits, shots=None)
        self.drawer = AmplitudeFlowDrawer(n_qubits, n_layers)
        self.drawer.draw_layer(*self.drawer.draw_none())

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_analytic_computation=False,
            supports_finite_shots=False,
            returns_probs=False,
            returns_state=False,
            supports_reversible_diff=False,
            supports_inverse_operations=False,
            supports_tensor_observables=False,
            supports_tracker=False,
        )
        return capabilities

    def apply(self, operations, **kwargs):
        for op in operations:
            if op.name == "Identity":
                self.drawer.draw_layer(*self.drawer.draw_id())
            elif op.name == "PauliX":
                self.drawer.draw_layer(*self.drawer.draw_x(op.wires[0]))
            elif op.name == "PauliY":
                self.drawer.draw_layer(*self.drawer.draw_y(op.wires[0]))
            elif op.name == "PauliZ":
                self.drawer.draw_layer(*self.drawer.draw_z(op.wires[0]))
            elif op.name == "Hadamard":
                self.drawer.draw_layer(*self.drawer.draw_h(op.wires[0]))
            elif op.name == "S":
                self.drawer.draw_layer(*self.drawer.draw_s(op.wires[0]))
            elif op.name == "T":
                self.drawer.draw_layer(*self.drawer.draw_t(op.wires[0]))
            elif op.name == "PhaseShift":
                self.drawer.draw_layer(*self.drawer.draw_p(op.wires[0]))
            elif op.name == "RY":
                self.drawer.draw_layer(*self.drawer.draw_ry(op.wires[0], op.parameters[0]))
            elif op.name == "RZ":
                self.drawer.draw_layer(*self.drawer.draw_rz(op.wires[0], op.parameters[0]))
            elif op.name == "CNOT":
                self.drawer.draw_layer(*self.drawer.draw_cnot(*op.wires))
            elif op.name == "Toffoli":
                self.drawer.draw_layer(*self.drawer.draw_toffoli(*op.wires))
            else:
                raise NotImplementedError

    def analytic_probability(self, wires=None):
        return self.marginal_prob(np.abs(self.drawer._get_state_vector()) ** 2, wires if wires else list(range(self.drawer.n_bits)))


class AmplitudeFlowDrawer:
    stylings = {
        'linewidth': 3,
        'solid_capstyle': 'butt',
        'path_effects': [
            patheffects.Stroke(linewidth=5, foreground='black'),
            patheffects.Stroke(linewidth=3, foreground='white'),
            patheffects.Normal(),
        ]
    }

    def __init__(self, n_qubits, n_layers) -> None:
        self.n_bits = n_qubits
        self.n_wires = 2**n_qubits
        self.n_layers = n_layers
        self.layer = 0
        self.ax = self._init_figure()
        self._draw_bitstring_labels()
        self.state = self._init_simulator()

    def _init_figure(self):
        fig = plt.figure(figsize=(self.n_layers + 2, self.n_wires + 2))
        fig.patch.set_facecolor('white')

        ax = fig.add_axes(
            [0, 0, 1, 1],
            xlim=(-1, self.n_layers + 1),
            ylim=(-2, self.n_wires),
            xticks=[],
            yticks=[],
        )
        ax.axis("off")
        ax.invert_yaxis()

        return ax

    def _init_simulator(self):
        return sim.QubitRegister(self.n_bits, "base", 0, 0)

    def _get_state_vector(self):
        return [self.state[wire] for wire in range(self.n_wires)]

    def _draw_bitstring_labels(self):
        for wire in range(self.n_wires):
            bitstring = '∣' + bin(wire)[2:].zfill(self.n_bits) + '⟩'
            self.ax.text(0.7, wire, bitstring, ha='right', va='center', size=24)

    def _state_to_style(self, statevector):
        magnitude = np.abs(statevector)
        phase = np.angle(statevector)

        styles = []
        for m, p in zip(magnitude, phase):
            c = color_map((p % (2*np.pi)) / (2*np.pi))
            style = {'color': colors.to_rgba(c, m), **self.stylings}
            styles.append(style)

        return styles

    def _draw_wire(self, wire, i, style=None, length=1):
        line = plt.Line2D((i, i+length), (wire, wire))
        if style:
            line.set(**style)
        self.ax.add_line(line)

    def _draw_wire_padding(self, i, styles=None, length=1):
        for wire in range(self.n_wires):
            self._draw_wire(wire, i, styles[wire], length)

    def _draw_swap(self, x, y, i, style_x=None, style_y=None):
        line1 = plt.Line2D((i, i+1), (x, y))
        line2 = plt.Line2D((i, i+1), (y, x))
        if style_x and style_y:
            line1.set(**style_y)
            line2.set(**style_x)
        self.ax.add_line(line1)
        self.ax.add_line(line2)

    def _draw_permutation(self, perm, i, styles=None):
        assert len(perm) == self.n_wires == len(set(perm))

        for wire, new_wire in enumerate(perm):
            line = plt.Line2D((i, i+1), (wire, new_wire))
            if styles:
                line.set(**styles[new_wire])
            self.ax.add_line(line)

    def _draw_phase(self, wire, angle, size, i):
        box = patches.FancyBboxPatch((i+0.35, wire-0.15), 0.3, 0.3, boxstyle="Round, pad=0.1")
        self.ax.add_patch(box)

        self.ax.text(i+0.5, wire+0.03, angle, ha='center', va='center', size=size)

    def _draw_wire_mixing_middle(self, targets, n_sets, i, styles_A=None, styles_B=None):
        mixes_per_set = int(self.n_wires / n_sets / 2)
        curr_mix = 0
        curr_set = 0
        for wire1, wire2 in targets:
            d = (wire2 - wire1) / 2
            start_top = plt.Line2D((i, i+0.4), (wire1, wire1+d))
            start_bottom = plt.Line2D((i, i+0.4), (wire2, wire2-d))
            middle_top = plt.Line2D((i+0.4, i+0.6), (wire1+d, wire1+d))
            middle_bottom = plt.Line2D((i+0.4, i+0.6), (wire2-d, wire2-d))
            end_top = plt.Line2D((i+0.6, i+1), (wire1+d, wire1))
            end_bottom = plt.Line2D((i+0.6, i+1), (wire2-d, wire2))
            if styles_A and styles_B:
                start_top.set(**styles_A[wire1])
                start_bottom.set(**styles_A[wire2])

                middle_top.set(**styles_A[wire1])
                middle_bottom_style = styles_A[wire2]
                del middle_bottom_style['path_effects']
                middle_bottom.set(**middle_bottom_style)

                end_top.set(**styles_B[wire1])
                end_bottom.set(**styles_B[wire2])
            self.ax.add_line(start_top)
            self.ax.add_line(start_bottom)
            self.ax.add_line(middle_top)
            self.ax.add_line(middle_bottom)
            self.ax.add_line(end_top)
            self.ax.add_line(end_bottom)

            curr_mix += 1
            if curr_mix == mixes_per_set:
                w = 0.5
                h = 2*mixes_per_set-1.5
                x = i + 0.5
                y = 2*mixes_per_set*curr_set + mixes_per_set - 0.5
                box = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, fill=False, ec='white', lw=5, boxstyle="Round, pad=0.1", zorder=100)
                self.ax.add_patch(box)
                box = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, fc=(0, 0, 0, 0.05), lw=1, boxstyle="Round, pad=0.1", zorder=101)
                self.ax.add_patch(box)
                curr_mix = 0
                curr_set += 1

    def draw_multicontrolled_x(self, ctrls, t, label="MCX"):
        # ctrls fix the bits of certain qubits to 1
        base_bitstring = 0
        for c in ctrls:
            base_bitstring |= 1 << c
        # get all bitsprings with non-ctrls and non-targets = {0, 1}
        bitstrings = [base_bitstring]
        for nc in set(range(self.n_bits)) - set(ctrls) - set([t]):
            for i in range(len(bitstrings)):
                bitstrings.append(bitstrings[i] | 1 << nc)
        # index pairs of swapped wires are with t = 0 and t = 1
        swap_pairs = []
        for index in bitstrings:
            swap_pairs.append((index, index | 1 << t))

        def draw_fn(i, styles):
            _, styles_B = styles
            remaining_wires = set(range(self.n_wires))
            for wire1, wire2 in swap_pairs:
                self._draw_swap(wire1, wire2, i, styles_B[wire1], styles_B[wire2])
                remaining_wires -= set([wire1, wire2])
            for wire in remaining_wires:
                self._draw_wire(wire, i, styles_B[wire])
            self.ax.text(i+0.5, -1, label, ha='center', va='center', size=24)

        return draw_fn

    def draw_multicontrolled_phase(self, ctrls, t, angle_0=None, angle_1=None, size=24, label="MCP"):
        # ctrls fix the bits of certain qubits to 1
        base_bitstring = 0
        for c in ctrls:
            base_bitstring |= 1 << c
        # get all bitsprings with non-ctrls and non-targets = {0, 1}
        bitstrings = [base_bitstring]
        for nc in set(range(self.n_bits)) - set(ctrls) - set([t]):
            for i in range(len(bitstrings)):
                bitstrings.append(bitstrings[i] | 1 << nc)
        # index pairs of swapped wires are with t = 0 and t = 1
        phase_targets = []
        for index in bitstrings:
            phase_targets.append((index, index | 1 << t))

        def draw_fn(i, styles):
            _, styles_B = styles
            remaining_wires = set(range(self.n_wires))
            for wire1, wire2 in phase_targets:
                if angle_0:
                    self._draw_phase(wire1, angle_0, size, i)
                    remaining_wires -= set([wire1])
                if angle_1:
                    self._draw_phase(wire2, angle_1, size, i)
                    remaining_wires -= set([wire2])
            for wire in remaining_wires:
                self._draw_wire(wire, i, styles_B[wire])
            self.ax.text(i+0.5, -1, label, ha='center', va='center', size=24)

        return draw_fn

    def draw_none(self):
        return lambda i, styles: ...,

    def draw_id(self, no_label=False):
        def draw_fn(i, styles):
            styles = styles[0]
            for wire in range(self.n_wires):
                self._draw_wire(wire, i, styles[wire])
            if not no_label:
                self.ax.text(i+0.5, -1, "Id", ha='center', va='center', size=24)

        return draw_fn,

    def draw_x(self, qubit, no_label=False):
        draw_fn = self.draw_multicontrolled_x([], qubit, label=None if no_label else "$X_"+str(qubit)+"$")
        return draw_fn, lambda: self.state.ApplyPauliX(qubit)

    def draw_y(self, qubit, no_label=False):
        def draw_fn(i, styles):
            self.draw_x(qubit, no_label=True)[0](i, styles[0:2])
            self.layer += 1
            self._draw_wire_padding(i+1, styles[1], length=0.25)
            self.layer += 0.25
            self.draw_multicontrolled_phase([], qubit, angle_0="-π/2", angle_1="π/2", size=16, label=None)(i+1.25, styles[1:3])

            if not no_label:
                self.ax.add_line(plt.Line2D((i+0.1, i+2.15), (-0.6, -0.6), linewidth=1.5, color='black'))
                self.ax.add_line(plt.Line2D((i+0.1, i+0.1), (-0.6, -0.5), linewidth=1.5, color='black'))
                self.ax.add_line(plt.Line2D((i+2.15, i+2.15), (-0.6, -0.5), linewidth=1.5, color='black'))
                self.ax.text(i+1.125, -1, "$Y_"+str(qubit)+"$", ha='center', va='center', size=24)

        return draw_fn, lambda: self.state.ApplyPauliX(qubit), lambda: self.state.ApplyRotationZ(qubit, np.pi)

    def draw_z(self, qubit, no_label=False):
        draw_fn = self.draw_multicontrolled_phase([], qubit, angle_1="π", label=None if no_label else "$Z_"+str(qubit)+"$")
        return draw_fn, lambda: self.state.ApplyPauliZ(qubit)

    def draw_s(self, qubit, no_label=False):
        draw_fn = self.draw_multicontrolled_phase([], qubit, angle_1="π/2", size=16, label=None if no_label else "$S_"+str(qubit)+"$")
        return draw_fn, lambda: self.state.ApplyS(qubit)

    def draw_t(self, qubit, no_label=False):
        draw_fn = self.draw_multicontrolled_phase([], qubit, angle_1="π/4", size=16, label=None if no_label else "$T_"+str(qubit)+"$")
        return draw_fn, lambda: self.state.ApplyT(qubit)

    def draw_p(self, qubit, angle, no_label=False):
        draw_fn = self.draw_multicontrolled_phase([], qubit, angle_1="θ", label=None if no_label else "$P_"+str(qubit)+"$")
        return draw_fn, lambda: self.state.ApplyPhase(qubit, angle)

    def draw_rz(self, qubit, angle, no_label=False):
        draw_fn = self.draw_multicontrolled_phase([], qubit, angle_0="-θ/2", angle_1="θ/2", size=16, label=None if no_label else "$RZ_"+str(qubit)+"$")
        return draw_fn, lambda: self.state.ApplyRotationZ(qubit, angle)

    def draw_ry(self, qubit, angle, no_label=False):
        # build up indices for mixing
        d = 2**qubit
        mixing_pairs = []
        n_sets, wire, j = 0, 0, 1
        while wire < self.n_wires:
            mixing_pairs.append((wire, wire+d))
            if j < d:
                wire += 1
                j += 1
            else:
                wire += d+1
                n_sets += 1
                j = 1

        def draw_fn(i, styles):
            styles_A, styles_B = styles
            self._draw_wire_mixing_middle(mixing_pairs, n_sets, i, styles_A, styles_B)
            if not no_label:
                self.ax.text(i+0.5, -1, "$RY_"+str(qubit)+"$", ha='center', va='center', size=24)

        return draw_fn, lambda: self.state.ApplyRotationY(qubit, angle)

    def draw_h(self, qubit, no_label=False):
        def draw_fn(i, styles):
            self.draw_ry(qubit, np.pi/2, no_label=True)[0](i, styles[0:2])
            self.layer += 1
            self._draw_wire_padding(i+1, styles[1], length=0.25)
            self.layer += 0.25
            self.draw_x(qubit, no_label=True)[0](i+1.25, styles[1:3])

            if not no_label:
                self.ax.add_line(plt.Line2D((i+0.1, i+2.15), (-0.6, -0.6), linewidth=1.5, color='black'))
                self.ax.add_line(plt.Line2D((i+0.1, i+0.1), (-0.6, -0.5), linewidth=1.5, color='black'))
                self.ax.add_line(plt.Line2D((i+2.15, i+2.15), (-0.6, -0.5), linewidth=1.5, color='black'))
                self.ax.text(i+1.125, -1, "$H_"+str(qubit)+"$", ha='center', va='center', size=24)

        return draw_fn, lambda: self.state.ApplyRotationY(qubit, np.pi/2), lambda: self.state.ApplyPauliX(qubit)

    def draw_cnot(self, c, t, no_label=False):
        draw_fn = self.draw_multicontrolled_x([c], t, label=None if no_label else "$CNOT_{"+str(c)+","+str(t)+"}$")
        return draw_fn, lambda: self.state.ApplyCPauliX(c, t)

    def draw_toffoli(self, c1, c2, t, no_label=False):
        draw_fn = self.draw_multicontrolled_x([c1, c2], t, label=None if no_label else "$Toffoli_{"+str(c1)+","+str(c2)+","+str(t)+"}$")
        return draw_fn, lambda: self.state.ApplyToffoli(c1, c2, t)

    def draw_layer(self, draw_fn, *sim_fns):
        # simulate
        states = [self._get_state_vector()]
        for sim_fn in sim_fns:
            sim_fn()
            states.append(self._get_state_vector())
        styles = []
        for state in states:
            styles.append(self._state_to_style(state))

        # draw operation
        i = self.layer
        draw_fn(i, styles)
        self.layer += 1

        # draw intermediate wire
        i = self.layer
        self._draw_wire_padding(i, styles=styles[-1], length=1)
        self.layer += 1
