import pandas as pd
import plotly.graph_objects as go
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
from plotly import offline

def plot_sankey(df, cols, group_by='importance', rename_dict=None, q=0, color_by='', transparency=0.6):
    """
    Plot sankey of DataFrame columns grouped by a single column
    :param df: DataFrame to be plotted
    :param cols: List of columns in the DataFrame to be plotted, each column represents one horizontal cluster of sankey nodes
    :param group_by: The name of one column contained in 'cols' to perform the group operation
    :param rename_dict: Optional rename dictionary to rename features
    :param q: Filter the X % highest features
    """

    # Visualize importance separated by categories
    source = []
    target = []
    values = []

    # Group importance by feature
    imp_dfs_cl = df.groupby(cols, as_index=False).mean()
    if q > 0: imp_dfs_cl = imp_dfs_cl[imp_dfs_cl[group_by] > imp_dfs_cl[group_by].quantile(q)].reset_index(drop=True)
    imp_dfs_cl = imp_dfs_cl.sort_values(by=group_by)

    # Map colors on color map based on color_by parameter
    clr_by_un = len(imp_dfs_cl[color_by].unique())
    c_map = get_cmap('terrain')
    c_mapped = [to_hex(list(c_map(i / clr_by_un)[0:3])+[(c_map(i / clr_by_un)[3]-transparency)]) for i in range(clr_by_un)]

    print(f"> Iterating over {cols} to construct nodes")
    nodes = pd.DataFrame(columns=['label', 'value', 'color'])
    for col in cols:
        for i, un in enumerate(imp_dfs_cl[col].unique()):
            ind = len(nodes)
            nodes.at[ind, 'label'] = un
            nodes.at[ind, 'value'] = imp_dfs_cl[imp_dfs_cl[col] == un][group_by].sum()
            if col == color_by: nodes.at[ind, 'color'] = c_mapped[i]
            else: nodes.at[ind, 'color'] = '#245C7C'

    print(f"> Iterating over {cols} to construct links by sources, targets and values")
    l_colors = []
    for i, col in enumerate(cols):
        if i != len(cols) - 1:
            n_col = cols[i + 1]
            for j, col_un in enumerate(imp_dfs_cl[col].unique()):
                for n_col_un in imp_dfs_cl[n_col].unique():
                    if (col_un in list(nodes['label'])) and (n_col_un in list(nodes['label'])):
                        source.append(nodes[nodes['label'] == col_un].index[0])
                        target.append(nodes[nodes['label'] == n_col_un].index[0])
                        values.append(imp_dfs_cl.loc[(imp_dfs_cl[col] == col_un) & (imp_dfs_cl[n_col] == n_col_un)][group_by].sum())
                        # if col == color_by: l_colors.append(c_mapped[j])
                        # else: l_colors.append('#D9D9D9')

    # Re-loop through links to map colors from colored nodes

    for i, col in enumerate(cols):
        if col == color_by:
            for src, tgt in zip(source, target):
                src_label = nodes.loc[src, 'label']
                if src_label in imp_dfs_cl[color_by].unique():
                    lbs = imp_dfs_cl[imp_dfs_cl[col] == src_label][cols[i+1]]
                    for j, l in enumerate(lbs):
                        nodes.loc[nodes['label'] == l].at[:, 'color'] = list(nodes[nodes['label'] == l]['color'])[0]
                        # nodes.at[:, nodes['color']] = list(nodes[nodes['label'] == src_label]['color'])[0]

    # for src, tgt in zip(source, target):
    #     src_label = nodes.loc[src, 'label']
    #     if src_label in imp_dfs_cl[color_by].unique():
    #         nodes.at[tgt, 'color'] = list(nodes[nodes['label'] == src_label]['color'])[0]

    print(f"> Renaming columns")
    nodes = nodes.replace(rename_dict)
    nodes['label'] = [str(l).upper() for l in nodes['label']]

    print(f"> Creating sankey diagram with {len(source)} sources, {len(target)} targets and {len(values)} values")
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes['label'],  # ["A1", "A2", "B1", "B2", "C1", "C2"],
            color= nodes['color'],
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            # color=l_colors,
        ))])
    fig.update_layout(title_text=f"Summed {group_by.title()}", font=dict(size=12, family="Roboto, light"),)
    offline.plot(fig, filename='Content/sankey.html')
    print("Finished")
