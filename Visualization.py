import numpy as np
from matplotlib import cm
from matplotlib import colors as c
from chord import Chord
import plotly.graph_objects as go


def get_colors(gray=(), color=(), map='viridis'):
    if type(gray) == int: gray = list(range(gray))
    if type(color) == int: color = list(range(color))

    cmap = cm.get_cmap(map)
    colors = ["#e5e5e5" for j in range(len(gray) - len(color))] + [c.to_hex(cmap(i / len(color))[:3]) for i, y in
                                                                   enumerate(color)]
    return colors


class Chords:
    def __init__(self, matrix, filename='Chords', wrap=False, width=700, margin=0):
        self.matrix = matrix
        self.wrap = wrap
        self.width = width
        self.margin = margin
        self.filename = filename

    def moduloAB(self, x, a, b):  # maps a real number onto the unit circle identified with
        # the interval [a,b), b-a=2*PI
        if a >= b:
            raise ValueError('Incorrect interval ends')
        y = (x - a) % (b - a)
        return y + b if y < 0 else y + a

    def test_2PI(self, x):
        return 0 <= x < 2 * np.pi

    def make_layout(self, title, p_size):
        axis = dict(
            showline=False,  # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
        )

        margin = 0
        return go.Layout(
            title=title,
            xaxis=dict(axis),
            yaxis=dict(axis),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            width=p_size[0],
            height=p_size[1],
            margin=dict(t=margin, b=margin, l=margin, r=margin),
            hovermode='closest',
            shapes=[]  # to this list one appends below the dicts defining the ribbon,
            # respectively the ideogram shapes
        )

    def make_ideo_shape(self, path, line_color, fill_color):
        # line_color is the color of the shape boundary
        # fill_collor is the color assigned to an ideogram
        return dict(
            line=dict(
                color=line_color,
                width=0.45
            ),

            path=path,
            type='path',
            fillcolor=fill_color,
            layer='below'
        )

    def make_ideograms(self, row_sum, radius=1.3):
        print(f"Creating arcs for features")

        # Get ideogram lengths
        gap = 2 * np.pi * 0.002
        ideogram_len = 2 * np.pi * row_sum / row_sum.sum() - gap * np.ones(len(row_sum))

        # Get ideogram start and end coordinates
        ideo_ends = []
        left = 0
        for k in ideogram_len.index:
            right = left + ideogram_len[k]
            ideo_ends.append([left, right])
            left = right + gap

        # Make ideogram arcs
        arcs = []
        shapes = []
        a = 50
        for phi in ideo_ends:
            if not self.test_2PI(phi[0]) or not self.test_2PI(phi[1]):
                phi = [self.moduloAB(t, 0, 2 * np.pi) for t in phi]
            length = (phi[1] - phi[0]) % 2 * np.pi
            nr = 5 if length <= np.pi / 4 else int(a * length / np.pi)

            if phi[0] < phi[1]:
                theta = np.linspace(phi[0], phi[1], nr)
            else:
                phi = [self.moduloAB(t, -np.pi, np.pi) for t in phi]
                theta = np.linspace(phi[0], phi[1], nr)
            arcs.append(radius * np.exp(1j * theta))

        # print(f"Arcs for feature {group_by} successfully created")
        return {'ideo_len': ideogram_len.sum(),'ideo_ends': ideo_ends, 'arcs': arcs, 'row_sum': row_sum, 'labels': list(ideogram_len.index)}

    def map_data(self, data_matrix, default_w=0):

        init_cols = len(data_matrix.columns)
        dm = data_matrix.reset_index(drop=True)
        dm = dm.fillna(0)

        for j in range(len(dm.index)):
            dm.insert(j, j, 0, True)

        for col in data_matrix.columns:
            matrix_len = len(dm)
            if col in data_matrix.index: pass
            else: dm.at[matrix_len, :] = 0

        dm.columns = range(len(dm.columns))
        tail = range(len(dm) - init_cols, len(dm))

        dm_t = dm.transpose()
        for t in tail:
            dm.loc[int(t), :] = dm_t.loc[int(t), :]

        # mapped = np.zeros(dm.shape)
        # row_value = row_value.reset_index(drop=True)
        # for j in range(len(dm)-1):
        #     mapped[:, j] = ideogram_length * dm.loc[:, j] / row_value
        dm = dm * 1000
        return dm.astype(int).values.tolist()

    def make_ribbon_ends(self, mapped_data, ideo_ends, idx_sort):
        L = mapped_data.shape[0]
        ribbon_boundary = np.zeros((L, L + 1))
        for k in range(L):
            start = ideo_ends[k][0]
            ribbon_boundary[k][0] = start
            for j in range(1, L + 1):
                J = idx_sort[k][j - 1]
                ribbon_boundary[k][j] = start + mapped_data[k][J]
                start = ribbon_boundary[k][j]
        return [[(ribbon_boundary[k][j], ribbon_boundary[k][j + 1]) for j in range(L)] for k in range(L)]

    def control_pts(self, angle, radius):
        # angle is a  3-list containing angular coordinates of the control points b0, b1, b2
        # radius is the distance from b1 to the  origin O(0,0)

        if len(angle) != 3:
            raise ('angle must have len =3')
        b_cplx = np.array([np.exp(1j * angle[k]) for k in range(3)])
        b_cplx[1] = radius * b_cplx[1]
        return [tup for tup in zip(b_cplx.real, b_cplx.imag)]

    def ctrl_rib_chords(self, l, r, radius):
        # this function returns a 2-list containing control polygons of the two quadratic Bezier
        # curves that are opposite sides in a ribbon
        # l (r) the list of angular variables of the ribbon arc ends defining
        # the ribbon starting (ending) arc
        # radius is a common parameter for both control polygons
        if len(l) != 2 or len(r) != 2:
            raise ValueError('the arc ends must be elements in a list of len 2')
        return [self.control_pts([l[j], (l[j] + r[j]) / 2, r[j]], radius) for j in range(2)]

    def make_q_bezier(self, b):  # defines the Plotly SVG path for a quadratic Bezier curve defined by the
        # list of its control points
        b_list = [tup for tup in b]
        if len(b_list) != 3:
            raise ValueError('control polygon must have 3 points')
        A, B, C = b_list
        return f"'M {A[0]},{A[1]} Q {B[0]}, {B[1]} {C[0]}, {C[1]}"

    def make_ribbon_arc(self, theta0, theta1):

        #if self.test_2PI(theta0) and self.test_2PI(theta1):
        if theta0 < theta1:
            theta0 = self.moduloAB(theta0, -np.pi, np.pi)
            theta1 = self.moduloAB(theta1, -np.pi, np.pi)
            if theta0 * theta1 > 0:
                raise ValueError('incorrect angle coordinates for ribbon')

        nr = int(40 * (theta0 - theta1) / np.pi)
        if nr <= 2: nr = 3
        theta = np.linspace(theta0, theta1, nr)
        pts = np.exp(1j * theta)  # points on arc in polar complex form

        string_arc = ''
        for k in range(len(theta)):
            string_arc += 'L ' + str(pts.real[k]) + ', ' + str(pts.imag[k]) + ' '
        #else:
        #    raise ValueError('the angle coordinates for an arc side of a ribbon must be in [0, 2*pi]')

        return string_arc

    def make_ribbon(self, l, r, line_color, fill_color, radius=0.2):
        # l=[l[0], l[1]], r=[r[0], r[1]]  represent the opposite arcs in the ribbon
        # line_color is the color of the shape boundary
        # fill_color is the fill color for the ribbon shape
        poligon = self.ctrl_rib_chords(l, r, radius)
        b, c = poligon

        return dict(
            line=dict(
                color=line_color, width=0.5
            ),
            path=self.make_q_bezier(b) + self.make_ribbon_arc(r[0], r[1]) +
                 self.make_q_bezier(c[::-1]) + self.make_ribbon_arc(l[1], l[0]),
            type='path',
            fillcolor=fill_color,
            layer='below'
        )

    def make_self_rel(self, l, line_color, fill_color, radius):
        # radius is the radius of Bezier control point b_1
        b = self.control_pts([l[0], (l[0] + l[1]) / 2, l[1]], radius)
        return dict(
            line=dict(
                color=line_color, width=0.5
            ),
            path=self.make_q_bezier(b) + self.make_ribbon_arc(l[1], l[0]),
            type='path',
            fillcolor=fill_color,
            layer='below'
        )

    def invPerm(self, perm):
        # function that returns the inverse of a permutation, perm
        inv = [0] * len(perm)
        for i, s in enumerate(perm):
            inv[s] = i
        return inv

    def plot_diagrams(self, matrices, names, l_labels):
        for i, (matrix, name) in enumerate(zip(matrices, names)):
            colors = (get_colors(gray=name, color=l_labels))
            diagram = Chord(
                matrix,
                name,
                colors=colors, #"d3.schemeSet1",
                opacity=0.9,
                padding=0.01,
                width=self.width,
                font_size="10px",
                font_size_large="10px",
                label_color="#454545",
                wrap_labels=self.wrap,
                credit=False,
                margin=self.margin,
            )
            diagram.show()
            diagram.to_html(f'{self.filename}_{i}.html')


class Sankey:
    def __init__(self, matrix):
        self.matrix = matrix*1000

        sources = []
        targets = []
        values = []
        labels = list(self.matrix.columns)
        for c, col in enumerate(self.matrix.columns):
            for i, ind in enumerate(self.matrix.index):
                sources.append(c)
                targets.append(i+len(self.matrix.columns))
                values.append(self.matrix.at[ind, col])
                labels.append(ind)

        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad=15,
                thickness=20,
                line=dict(color = "black", width = 0.5),
                label=labels,
                color='gray'
            ),
            link = dict(
                source=sources, # indices correspond to labels, eg A1, A2, A2, B1, ...
                target=targets,
                value=values,
                color=[get_colors(color=self.matrix.columns)[i] for i in sources]
          ))])

        fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
        fig.write_html('sankey.html')
        fig.show()
        return
