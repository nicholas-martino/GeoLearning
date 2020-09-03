import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from Sankey import plot_sankey
from GeoLearning import Regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import *
import matplotlib.font_manager as fm
from matplotlib import rc
from stepwise_regression.step_reg import forward_regression

fm.fontManager.ttflist += fm.createFontList(['/Volumes/Samsung_T5/Fonts/roboto/Roboto-Light.ttf'])
rc('font', family='Roboto', weight='light')

rename_dict = {
    'mob_n_use': ('Use diversity', 'landuse'),
    'mob_frequency': ('Transit frequency', 'network'),
    'mob_cycle_length': ('Cycling network length', 'network'),
    'mob_gross_building_area': ('Gross building area', 'density'),
    'mob_population, 2016': ('Population', 'density'),
    'mob_CM': ('Commercial', 'landuse'),
    'mob_SFD': ('Single-Family Detached', 'landuse'),
    'mob_SFA': ('Single-Family Attached', 'landuse'),
    'mob_MFL': ('Multi-Family Low-Rise', 'landuse'),
    'mob_MFH': ('Multi-Family High-Rise', 'landuse'),
    'mob_MX': ('Mixed Use', 'landuse'),
    'mob_population density per square kilometre, 2016': ('Population density', 'density'),
    'mob_n_dwellings': ('Number of dwellings', 'density'),
    'mob_total_finished_area': ('Total finished area', 'density'),
    'mob_network_stops_ct': ('Public transit stops', 'network'),
    'mob_number_of_bedrooms': ('Number of bedrooms', 'density'),
}

radius = [400, 600, 800, 1000, 1200]
rename_dict2 = {}
rename_dict3 = {}
type_dict2 = {}
for k, value in rename_dict.items():
    v = value[0]
    rename_dict3[k] = v
    for r in radius:
        for d in ['l', 'f']:
            for op in ['ave', 'sum', 'cnt', 'rng']:
                if op == 'ave': t = 'Average'
                elif op == 'sum': t = 'Total'
                else: t = ''
                renamed = f'{t} '
                item = f'{renamed}{v.lower()} within {r}m'
                rename_dict2[f"{k}_r{r}_{op}_{d}"] = f'{item.strip()[0].upper()}{item.strip()[1:]}'
                type_dict2[f"{k}_r{r}_{op}_{d}"] = value[1]
rename_dict = {**rename_dict2, **rename_dict3}

directory = '/Volumes/Samsung_T5/Databases'
predictors = pd.read_csv(f'{directory}/Network/Features_Mobility.csv')
predictors['predictors'] = predictors['0']
predictors = predictors.drop(['Unnamed: 0', '0'], axis=1)

# Read databases
gdfs = [
    gpd.read_file(f'{directory}/Network/Capital Regional District, British Columbia_mob_na.geojson'),
    gpd.read_file(f'{directory}/Network/Metro Vancouver, British Columbia_mob_na.geojson')
]
proxy_files = [
    f'{directory}/Network/Hillside Quadra Sandbox_mob_e0_na.geojson',
    f'{directory}/Network/Hillside Quadra Sandbox_mob_e1_na.geojson',
    f'{directory}/Network/Hillside Quadra Sandbox_mob_e2_na.geojson',
    f'{directory}/Network/Hillside Quadra Sandbox_mob_e3_na.geojson'
]

label_cols = ['walk', 'bike', 'drive', 'bus']

for rs in [10]:
    # Filter columns common to proxy and dissemination areas
    ind_cols = [set(gdf.columns) for gdf in gdfs]+[set(gpd.read_file(f).columns) for f in proxy_files]
    common_cols = list(set.intersection(*ind_cols))
    final_cols = [col for col in common_cols if col in list(rename_dict2.keys())]
    gdfs = [gdf.loc[:, final_cols + label_cols] for gdf in gdfs]

    print(f"\nStarting regression with random seed {rs}")
    reg = Regression(
        r_seed=rs,
        test_split=0.2,
        n_indicators=5,
        round_f=4,
        norm_x=False,
        norm_y=False,
        data=gdfs,
        directory=directory,
        predictors=predictors,
        predicted=label_cols,
        prefix='mob',
        rename=rename_dict,
        plot=True,
        pv=0.05,
        color_by="Population density per square kilometre, 2016",
    )

    # Run random forest and partial dependence plots
    method = reg.non_linear(method=RandomForestRegressor)
    importance = reg.test_non_linear(i_method='regular')
    features = reg.partial_dependence(n_features=9)

    # Iterate over proxy files
    fig, ax = plt.subplots(ncols=len(proxy_files), figsize=reg.figsize)
    for i, proxy_file in enumerate(proxy_files):
        title = proxy_file[-13:].split('_')[0].title()

        # Read GeoDataFrame
        proxy_gdf = gpd.read_file(proxy_file)
        """
        # Drop non numeric columns
        proxy_gdf_geom = proxy_gdf.geometry
        to_drop = ['landuse', 'LANDUSE', 'geometry']
        for col in proxy_gdf.columns:
            if not proxy_gdf[col]._is_numeric_mixed_type:
                proxy_gdf = proxy_gdf.drop(col, axis=1)

        # Normalize proxy X values
        proxy_gdf = pd.DataFrame(reg.scaler.fit_transform(proxy_gdf), columns=proxy_gdf.columns)
        proxy_gdf['geometry'] = proxy_gdf_geom
        proxy_gdf = gpd.GeoDataFrame(proxy_gdf, geometry='geometry')
        """
        # Predict sandbox using random forest
        proxy_gdf = reg.pre_norm_exp(proxy_gdf, prefix=f'{title}_rf_{rs}')

        # Plot most important features
        tup = []
        for k in range(3):
            for l in range(3):
                tup.append((k, l))
        for label_col in label_cols:
            fig2, ax2 = plt.subplots(ncols=3, nrows=3, figsize=(15, 15))
            for j, t in enumerate(tup):
                proxy_gdf.plot(column=features[label_col][j], cmap='viridis', ax=ax2[t], linewidth=0, legend=True)
                ax2[t].set_title(rename_dict2[features[label_col][j]])
                ax2[t].set_axis_off()
            fig2.savefig(f'{title}_important_{label_col}.png')

        # # Export processed parcels
        # print(f"> Exporting {title} results to GeoJSON")
        # proxy_gdf.to_file(f'{proxy_file}_s{reg.r_seed}.geojson', driver='GeoJSON')

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
                imp_df.loc[f, 'ind_type2'] = type_dict2[f]
    imp_dfs = pd.concat([imp_dfs, imp_df])
    imp_dfs = imp_dfs.drop(['feature', 'decay'], axis=1)

# Plot sankey diagram
plot_sankey(imp_dfs, cols=['ind_type2', 'feat', 'dependent'], group_by='importance', color_by='ind_type2', rename_dict=rename_dict)
print("done")

"""
# Select regression features using stepwise method
stepwise = {}
for label_col in reg.label_cols:
    stepwise[label_col] = forward_regression(reg.train_data_n[label_col], reg.train_labels[label_col], threshold_in=0.05)
for col in reg.label_cols:
    with open(f"stepwise_{col}.txt", 'w') as f:
        f.write(str(stepwise[col]))

# Perform linear model with variables selected by stepwise
reg_train_data_n = {label_col: reg.train_data_n[label_col].loc[:, stepwise[label_col]] for label_col in reg.label_cols}
coefficients = reg.linear(method=LinearRegression, x_crr_thr=0.8, y_crr_thr=1)
"""
