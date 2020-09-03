import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.cm import get_cmap
from pandas.plotting import parallel_coordinates
from plotly import offline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from GeoLearning import Regression, Clustering

directory = '/Volumes/Samsung_T5/Databases'
gpkg = f'{directory}/Metro Vancouver, British Columbia.gpkg'
radius = [4800, 3200, 1600, 800, 400]
dpi = 800

# Define variables
dependent = {
    'den': (
    ),
    'acc': ([
        ("acc_frequency_r400_ave_f", "Frequency of public transit"),
        ("acc_walk_r400_ave_f", "Ratio of population that walk to work"),
        ("acc_bike_r400_ave_f", "Ratio of population that bike to work")], []),  #  "gps_traces_density"], []),
    'div': ([
        ("div_educ_div_sh_r400_ave_f", "Educational diversity"),
        ("div_income_div_sh_r400_ave_f", "Income diversity"),
        ("div_age_div_sh_r400_ave_f", "Age diversity"),
        ("div_ethnic_div_sh_r400_ave_f", "Ethnic diversity")], []),
    'aff': ([], [
        ("aff_price_sqft_r400_ave_f", "Average rent price per square foot"),
        ("aff_owned_ave_cost_r400_ave_f", "Owned dwelling cost"),
        ("aff_owned_ave_dwe_value_r400_ave_f", "Owned dwelling value"),
        ("aff_more30%income_rat_r400_ave_f", "Population that spends more than 30% of income on rent")]),
    'vit': ([
        ("vit_employment rate_r400_ave_f", "Employment rate"),
        ("vit_indeed_employment_ct_r400_sum_f", "Number of job posts"),
        ("aff_craigslist_rent_r400_cnt_f", "Number of housing posts"),
        ("vit_retail_r400_sum_f", "Number of retail spaces")], [])
}
control = "den_population density per square kilometre, 2016_r4800_ave_l"

rename_mask = {
    'acc': 'Accessibility',
    'div': 'Social Diversity',
    'aff': 'Affordability',
    'vit': 'Economic Vitality',
    '1600': '1600m radius',
    '400': '400m radius',
    '4800': '4800m radius',
    '800': '800m radius',
    'ctr': 'Centrality',
    'ctr_axial_degree': 'Axial degree centrality',
    'ctr_axial_closeness': 'Axial closeness centrality',
    'ctr_axial_closeness_r3200_ave_f': 'Average axial closeness within 3200m',
    'ctr_axial_eigenvector': 'Axial eigenvector centrality',
    'ctr_axial_katz': 'Axial katz centrality',
    'ctr_axial_hits1': 'Axial hits centrality',
    'ctr_axial_betweenness': 'Axial betweenness centrality',
    'ctr_axial_length': 'Axial line length',
    'ctr_axial_n_betweenness': 'Normalized axial betweenness centrality',
    'ctr_node_closeness': 'Street intersection closeness centrality',
    'ctr_length': 'Street length',
    'ctr_link_betweenness': 'Street segment betweenness centrality',
    'ctr_link_n_betweenness': 'Normalized street segment betweenness',
    'ctr_network_drive_ct': 'Number of streets',
    'ctr_node_betweenness': 'Intersection betweenness',
    'f': 'No distance-decay function',
    'int': 'Intensity',
    'int_area': 'Parcel area',
    'int_area_r3200_ave_f': 'Average parcel area within 3200m',
    'int_area_r4800_ave_f': 'Average parcel area within 4800m',
    'int_ave_n_rooms': 'Number of rooms',
    'int_ave_n_rooms_r400_f': 'Average number of rooms within 400m',
    'int_cc_per': 'Tree canopy cover',
    'int_cc_per_r4800_rng_f': 'Range of tree canopy cover within 4800m',
    'int_gross_building_area': 'Gross building area',
    'int_gross_building_area_r4800_sum_f': 'Total gross building area within 4800m',
    'int_imp_per': 'Impervious surfaces',
    'int_imp_per_r1600_sum_l': 'Impervious surfaces within 1600m',
    'int_land_assessment_fabric_r1600_cnt_l': 'Number of parcels within 1600m',
    'int_land_assessment_parcels': 'Number of parcels',
    'int_land_assessment_parcels_ct': 'Number of parcels',
    'int_n_dwellings': 'Number of dwellings',
    'int_n_dwellings_r1600_sum_f': 'Number of dwellings within 1600m',
    'int_number_of_bathrooms': 'Number of bathrooms',
    'int_number_of_bedrooms': 'Number of bedrooms',
    'int_number_of_storeys': 'Number of storeys',
    'int_number_of_storeys_r4800_sum_f': 'Total number of storeys within 4800m',
    'int_pcc_veg': 'Potential planting area',
    'int_pcc_veg_r4800_ave_l': 'Average potential planting area within 4800m',
    'int_total_bedr': 'Number of bedrooms',
    'int_total_bedr_r400_ave_f': 'Average number of bedrooms within 400m',
    'int_total_bedr_r1600_sum_f': 'Number of bedrooms within 1600m',
    'int_total_finished_area': 'Total finished area',
    'int_total_finished_area_r1600_ave_f': 'Average finished area within 1600m',
    'int_year_built': 'Year built',
    'l': 'Linear distance-decay function',
    'mix': 'Diversity',
    'mix_400 to 800': 'Small-sized parcels (400 - 800m2)',
    'mix_800 to 1600': 'Medium-sized parcels (800 - 1600m2)',
    'mix_building_age_div_sh': 'Building age diversity (Shannon)',
    'mix_building_age_div_sh_r400_ave_f': 'Average building age diversity (Shannon)',
    'mix_building_age_div_si': 'Building age diversity (Simpson)',
    'mix_dwelling_div_bedrooms_sh': 'Dwelling diversity (Shannon index, number of bedrooms)',
    'mix_dwelling_div_rooms_sh': 'Dwelling diversity (Shannon index, number of rooms)',
    'mix_dwelling_div_rooms_si': 'Dwelling diversity (Simpson index, number of rooms)',
    'mix_dwelling_div_rooms_si_r4800_rng_f': 'Dwelling diversity, number of rooms (Simpson) within 4800m',
    'mix_dwelling_div_rooms_si_r4800_rng_l': 'Dwelling diversity, number of rooms (Simpson) within 4800m',
    'mix_dwelling_div_bedrooms_si': 'Dwelling diversity (number of bedrooms)',
    'mix_n_size': 'Diversity of parcel size',
    'mix_more than 6400': 'Large parcels (more than 6400m2)',
    'mob_gross_building_area_r4800_sum_f': 'Total gross building area within 4800m',
    'mob_node_closeness_r800_sum_f': 'Total intersection closeness within 800m',
    'mob_population density per square kilometer, 2016_r800_sum_f': 'Total population density within 800m',
    'mob_population density per square kilometer, 2016_r1600_sum_f': 'Total population density within 1600m',
    'mob_population density per square kilometer, 2016_r3200_sum_f': 'Average population density within 3200m',
    'mob_population density per square kilometer, 2016_r4800_sum_f': 'Total population density within 4800m',
    'mob_n_dwellings_r1600_sum_f': 'Total number of dwellings within 1600m',
    'mob_n_dwellings_r800_sum_f': 'Total number of dwellings within 800m',
    'mob_number_of_bathrooms_r4800_ave_f': 'Average number of bathrooms within 4800m',
}

rename_mask2 = {}
for k, v in rename_mask.items():
    for r in radius:
        for d in ['l', 'f']:
            for op in ['ave', 'sum', 'cnt', 'rng']:
                if op == 'ave': t = 'Average'
                elif op == 'sum': t = 'Total'
                elif op == 'rng': t = 'Range'
                else: t = ''
                renamed = f'{t} '
                item = f'{renamed}{v.lower()} within {r}m'
                rename_mask2[f"{k}_r{r}_{op}_{d}"] = f'{item.strip()[0].upper()}{item.strip()[1:]}'
rename_mask = {**rename_mask, **rename_mask2}

# Read dependent variables
print("> Reading dependent variable maps")
ctrl_gdf = gpd.read_file(f'{directory}/Network/Metro Vancouver, British Columbia_den_na.geojson')
geom = ctrl_gdf['geometry']
gdfs = {d: gpd.read_file(f'{directory}/Network/Metro Vancouver, British Columbia_{d}_na.geojson').\
    transpose().drop(['geometry']) for d in dependent.keys()}
sample4geom = gpd.read_file(f'{directory}/Network/Metro Vancouver, British Columbia_{list(dependent.keys())[0]}_na.geojson')
dep_gdf_raw = pd.concat(gdfs.values()).transpose()
dep_gdf_raw.columns = [c.strip() for c in dep_gdf_raw.columns]
dep_columns = [e[0] for sl in [e for sl in dependent.values() for e in sl] for e in sl]

# Extract dependent values rename dictionary
dep_rename = {e[0]:e[1] for sl in [e for sl in dependent.values() for e in sl] for e in sl}
[print(f"{col} not found") for col in [control]+dep_columns if col not in dep_gdf_raw.columns]
dep_gdf = dep_gdf_raw.loc[:, [control]+dep_columns]

# Find clusters
rng = (0.2, 1)
fig, ax = plt.subplots()
cluster = Clustering(dep_gdf, ax=ax, by=control, minim=300)
fig2, ax2 = plt.subplots()
f_gdf = gpd.GeoDataFrame(cluster.jenks_break(n_class=5).sort_values(by='break').dropna(),
                         geometry=sample4geom.loc[dep_gdf.index, 'geometry'])
f_gdf.plot(column='break', scheme='natural_breaks', cmap='Blues', ax=ax2, legend=True)
ax2.axis('off')
plt.tight_layout()
plt.savefig('Content/clusters.png', dpi=dpi)

# Calculate indices and export to GeoPackage
dependent.pop("den")
s_gdf = cluster.calculate_index(dependent=dependent)
s_gdf['break'] = f_gdf['break']
s_gdf['group'] = f_gdf['group']
s_gdf['geometry'] = geom
s_gdf = gpd.GeoDataFrame(s_gdf, geometry='geometry')
s_gdf = s_gdf.replace([np.inf, -np.inf], np.nan)
s_gdf = s_gdf.dropna()
s_gdf.to_file(gpkg, layer='land_dissemination_area_na_cl', driver='GPKG')

# Rename columns and create dictionary by cluster
r_gdf = s_gdf.drop(['group', 'break'] + list(dependent.keys()), axis=1)
r_gdf.columns = r_gdf.columns.to_series().map(dep_rename)
r_gdf['group'] = s_gdf['group']
ind = {g: r_gdf.loc[r_gdf['group'] == g].mean().drop('group') for g in f_gdf['group'].unique()}
ind_radar = pd.DataFrame(ind).transpose()
scl = MinMaxScaler()
ind_radar = pd.DataFrame(scl.fit_transform(ind_radar).transpose(),
                         columns=ind_radar.transpose().columns,
                         index=ind_radar.transpose().index)

# Summarize indices by density group
indices = cluster.index_by_group(f_gdf, dependent=dependent)
indices_t = indices.transpose()
indices_t['break'] = pd.to_numeric(indices_t.index)
indices_s = indices_t.sort_values(by='break').transpose()
indices_s.columns = indices_t['break']
indices_s = indices_s.drop(['break'])

# Plot ranking
ticks = [b for b in indices_s.columns]
indices_s = indices_s.rename({
    'acc': 'Accessibility', 'div': 'Diversity',
    'aff': 'Affordability', 'vit': 'Vitality',
    #  'hth': 'Health', 'sft': 'Safety'
})
indices_s['class'] = indices.index

# Plot parallel coordinates chart
spectral = [get_cmap('Spectral')((i+1)/len(ind_radar.columns)) for i in range(len(ind_radar.columns))]
blues = [get_cmap('Blues')((i+1)/len(ind_radar.columns)) for i in range(len(ind_radar.columns))]
bl = cluster.plot_clusters_hist(f_gdf)
parallel_coordinates(
    indices_s.replace(rename_mask), class_column='class', color=spectral, xticks=ticks, ax=cluster.ax2, linewidth=4, zorder=2)
plt.axis('off')
cluster.ax2.legend(bbox_to_anchor=(0, 0), loc=2, borderaxespad=1, columnspacing=1, handlelength=1)
fig.savefig('Content/ranking.png', bbox_inches='tight', dpi=dpi)

# Plot radar chart
num_vars = len(ind[0])
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
labels = list(ind_radar.index)
fig, ax = plt.subplots(figsize=(12, 6),  subplot_kw=dict(polar=True))
ax.set_rlabel_position(180 / num_vars)

# Fix axis to go in the right order and start at 12 o'clock.
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label.
ax.set_thetagrids(np.degrees(angles), labels)

# Go through labels and adjust alignment based on where it is in the circle.
for label, angle in zip(ax.get_xticklabels(), angles):
  if angle in (0, np.pi):
    label.set_horizontalalignment('center')
  elif 0 < angle < np.pi:
    label.set_horizontalalignment('left')
  else:
    label.set_horizontalalignment('right')

# Scale 0.2 to 1
scl = MinMaxScaler((0.2, 1))
ind_radar = pd.DataFrame(scl.fit_transform(ind_radar), columns=ind_radar.columns)

# Plot radar chart
patches = []
for g, color in zip(ind_radar.columns, blues):
    patch = mpatches.Patch(color=color, label=g)
    ax.plot(angles, ind_radar[g], color=color, linewidth=1)
    ax.fill(angles, ind_radar[g], color=color, alpha=0.25)
    patches.append(patch)
fig.legend(handles=patches)
fig.savefig('Content/radar.png', dpi=dpi)

# Read independent variables
dep_cols = list(dependent.keys())
prefixes = ['int', 'mix', 'ctr']
int_gdf = gpd.read_file(f'{directory}/Network/Metro Vancouver, British Columbia_int_na.geojson').transpose()
div_gdf = gpd.read_file(f'{directory}/Network/Metro Vancouver, British Columbia_mix_na.geojson').transpose()
ctr_gdf = gpd.read_file(f'{directory}/Network/Metro Vancouver, British Columbia_ctr_na.geojson').transpose()
gdf = pd.concat([int_gdf, div_gdf, ctr_gdf]).drop(['geometry']).transpose()
gdf = gdf.loc[:, [c for c in gdf.columns if c[:3] in prefixes]]
gdf = pd.concat([gdf, s_gdf.loc[:, dep_cols]], axis=1)

to_drop = ['int_land_ecohealth_ct', 'int_land_assessment_fabric_ct', 'mix_land_assessment_parcels_ct',
    'int_land_assessment_fabric', 'int_length', 'int_network_drive_ct', 'int_number_of_bedrooms',
    'int_land_ecohealth_r3200_cnt_f', 'int_land_ecohealth_r3200_cnt_l', 'int_year_built_r400_sum_l',
    'int_year_built_r400_sum_f', 'ctr_axial_hits2', 'int_land_dissemination_area_ct', 'int_land_ecohealth',
    'mix_400 to 800', 'mix_800 to 1600', 'mix_more than 6400', 'mix_land_dissemination_area_ct', 'mix_3200 to 6400',
    'mix_less than 400', 'mix_1600 to 3200'
]
for name in to_drop:
    for col in gdf.columns:
        if name in col:
            try: gdf.drop(col, axis=1, inplace=True)
            except: print(f"!!! {col} not dropped !!!")
gdf = [gdf]

reg = Regression(
    test_split=0.3,
    n_indicators = 5,
    round_f=4,
    norm_x=False,
    norm_y=True,
    data=gdf,
    directory=directory,
    predicted=dep_cols,
    rename=rename_mask,
    dpi=dpi
)

# Plot maps of dependent variables
fig3, ax3 = plt.subplots(ncols=len(dep_cols), figsize=reg.figsize)
for i, col in enumerate(dep_cols):
    s_gdf.plot(column=col, scheme='natural_breaks', cmap='viridis', ax=ax3[i])
    ax3[i].set_title(reg.titles[i])
    ax3[i].axis('off')
plt.tight_layout()
plt.savefig('Content/map_indices.png', dpi=dpi)

# Plot maps of dependent variable by category
reg.plot_dependent_maps(f_gdf, dependent, run=True)

# Run random forest and partial dependence plots
method = reg.non_linear(method=RandomForestRegressor)
reg.test_non_linear(i_method='regular')
reg.partial_dependence(n_features=9)

# Define data to be aggregated on Sankey
radius = list(radius)
imp_dfs = pd.DataFrame()
for key, imp_df in reg.feat_imp.items():
    # Set index to be feature code
    imp_df.index = imp_df['feature'].values
    # Assign y-variable
    imp_df['dependent'] = key
    # Extract morph indicator type
    imp_df['ind_type'] = [f[:3] for f in imp_df['feature']]
    for f in imp_df.feature:
        for r in radius:
            if f"_r{r}_" in f:
                imp_df.at[f, 'feat'] = f.split(f"_r{r}_")[0]
                imp_df.at[f, 'radius'] = r
                imp_df.at[f, 'decay'] = f[-1:]
    imp_dfs = pd.concat([imp_dfs, imp_df])

# Visualize importance separated by categories
sankey = go.Figure()
source = []
target = []
values = []
cols = ['dependent', 'ind_type', 'feat', 'radius', 'decay']
to_group = 'importance'
color_by = ('dependent', spectral)

# Group importance by feature
imp_dfs_cl = imp_dfs.drop(['feature'], axis=1).groupby(cols, as_index=False).mean()
imp_dfs_cl = imp_dfs_cl[imp_dfs_cl[to_group] > imp_dfs_cl[to_group].quantile(0.9)].reset_index(drop=True)
imp_dfs_cl = imp_dfs_cl.sort_values(by=to_group)

print(f"> Iterating over {cols} to construct nodes")
nodes = []
imps = []
for col in cols:
    for un in imp_dfs_cl[col].unique():
        nodes.append(un)
        imps.append(imp_dfs_cl[imp_dfs_cl[col] == un][to_group].sum())
nodes = pd.DataFrame({'label': nodes, 'value': imps})

print(f"> Iterating over {cols} to construct links by sources, targets and values")
for i, col in enumerate(cols):
    if i != len(cols) - 1:
        n_col = cols[i + 1]
        for col_un in imp_dfs_cl[col].unique():
            for n_col_un in imp_dfs_cl[n_col].unique():
                if (col_un in list(nodes['label'])) and (n_col_un in list(nodes['label'])):
                    source.append(nodes[nodes['label'] == col_un].index[0])
                    target.append(nodes[nodes['label'] == n_col_un].index[0])
                    values.append(imp_dfs_cl.loc[(imp_dfs_cl[col] == col_un) & (imp_dfs_cl[n_col] == n_col_un)][to_group].sum())

print(f"> Renaming columns")
nodes = nodes.replace(rename_mask)

print(f"> Creating sankey diagram with {len(source)} sources, {len(target)} targets and {len(values)} values")
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes['label'],  # ["A1", "A2", "B1", "B2", "C1", "C2"],
        color='#245C7C'
    ),
    link=dict(
        source=source,
        target=target,
        value=values
    ))])
fig.update_layout(title_text=f"Summed {to_group.title()}", font_size=10)
offline.plot(fig, filename='Content/sankey.html')
print("Finished")
