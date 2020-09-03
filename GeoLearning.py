import ast
import math
import os
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.layers import BatchNormalization
import warnings
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import geopandas as gpd
import jenkspy
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from boruta import BorutaPy
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from numpy.random import seed
from plotly.subplots import make_subplots
from rfpimp import permutation_importances
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from VarianceANN import VarImpVIANN

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Regression:
    def __init__(self, directory, data, predicted, predictors=None, scale=10, r_seed=1, n_indicators=4, n_weights=6,
        test_split=0.2, norm_x=True, norm_y=True, color_by="column", same_scale=True, percentile=99, plot=True,
        unique=None, prefix="", round_f=4, chdir=os.curdir, rename=None, pv=0.05, dpi=300):

        print("Pre processing data for regression")
        seed(r_seed)

        self.chdir = chdir
        if os.path.exists(self.chdir): pass
        else: os.mkdir(self.chdir)
        os.chdir(self.chdir)

        self.dpi = dpi
        self.scale = scale
        self.pv = pv
        self.label_cols = predicted
        self.r_seed = r_seed
        self.color_by = color_by
        self.n_weights = n_weights
        self.figsize = (len(self.label_cols)*4, 4)
        self.same_scale = same_scale
        self.split = test_split
        self.percentile = percentile
        self.n_indicators = n_indicators
        self.norm_x = norm_x
        self.norm_y = norm_y
        self.cur_method = None
        self.round_f = round_f
        self.rename = rename
        self.titles = []

        self.methods = {}
        self.fitted = {}
        self.predicted = {}
        self.features_d = {}
        self.feat_imp = {}

        gdf = pd.concat(data, axis=0)
        raw_len = len(gdf)
        print(f"> Data concatenated with a total of {raw_len} samples and {len(gdf.columns)} features")

        gdf.columns = [col.strip() for col in gdf.columns if type(col) != int]
        if unique is not None:
            gdf = gdf.drop_duplicates(subset=[unique])
            gdf = gdf.loc[:, [unique]+[col for col in gdf.columns if prefix in col]+self.label_cols]
        else:
            gdf = gdf.loc[:, [col for col in gdf.columns if prefix in col]+self.label_cols]

        # If predictor is not defined use all columns except dependent
        if predictors is None:
            self.predictors = pd.DataFrame({
                'feature': self.label_cols,
                'predictors': [[c for c in gdf.columns if c not in predicted] for i in self.label_cols]})
        else: self.predictors = predictors

        # Drop invalid values
        for col, na in zip(list(gdf.columns), gdf.isna().sum()):
            if na > 0: print(f"{col}: {na} nans / {len(gdf)} rows")

        gdf = gdf.replace([np.inf, -np.inf], np.nan)
        gdf = gdf.dropna(how='any', axis=0)
        if norm_y: self.range = (0, 1)
        else: self.range = (gdf.loc[:, self.label_cols].min().min(), gdf.loc[:, self.label_cols].max().max())

        # Convert columns to numeric
        for col in gdf.columns:
            try: gdf[col] = gdf[col].astype(float)
            except:
                print(f"> Could not convert {col} to float, dropping")
                gdf.drop(col, axis=1, inplace=True)

        # Normalize 0-1
        self.scaler = MinMaxScaler()
        self.n_gdf = pd.DataFrame(self.scaler.fit_transform(gdf), columns=gdf.columns)
        print(f"> Samples reduced from {raw_len} to {len(gdf)}")

        gdfs = {}
        train_data = {}
        train_data_n = {}
        train_labels = {}
        test_data = {}
        test_data_n = {}
        test_labels = {}
        label_features = {}

        if plot: fig, ax = plt.subplots(ncols=len(self.label_cols), figsize=self.figsize)
        else: fig, ax = None, None
        for i, label_col in enumerate(self.label_cols):
            prd = [col for col in gdf.columns if col not in self.label_cols]
            l_gdf = gdf.copy(deep=False).loc[:, prd+[label_col]]
            if len(l_gdf) == 0: print("!!! Length of GeoDataFrame is 0, predictors probably not found !!!")
            l_gdf = l_gdf.fillna(0)
            l_gdf = l_gdf.dropna()
            try: [l_gdf.drop(label, axis=1, inplace=True) for label in self.label_cols if label != label_col]
            except: pass

            # Calculate and filter head and tails
            p5 = round(np.percentile(l_gdf[label_col], 1), 2)
            p95 = round(np.percentile(l_gdf[label_col], 99), 2)
            l_gdf = l_gdf.loc[:,~l_gdf.columns.duplicated()]
            l_gdf = l_gdf.loc[(l_gdf[label_col] > p5) & (l_gdf[label_col] < p95)]

            # Filter low p-values
            linear_reg = sm.OLS(l_gdf[label_col], l_gdf[prd])
            fitted = linear_reg.fit()
            prd_f = [v for v, m in zip (prd, list(fitted.pvalues < self.pv)) if m]
            print(f"{label_col} features reduced from {len(prd)} to {len(prd_f)}, {len(prd) - len(prd_f)} relations with p < {self.pv}")
            l_gdf = l_gdf.loc[:, prd_f+[label_col]]

            # Normalize labels 0-1
            if norm_y:
                y = l_gdf[label_col].values.reshape(-1, 1)
                scaler = MinMaxScaler()
                l_gdf[label_col] = scaler.fit_transform(y)

            mean = l_gdf[label_col].mean()
            median = l_gdf[label_col].median()

            # Display results
            if plot:
                if self.same_scale: ax[i].set(ylim=(0, 350), xlim=self.range)

                # Rename columns if rename dictionary is provided
                if self.rename is not None:
                    if label_col in list(self.rename.keys()): title = self.rename[label_col]
                    else: title = f"{label_col.upper()[0]}{label_col[1:]}"
                else:
                    if len(label_col[0].upper()) > 1: title = label_col[0].upper()
                    else: title = f"{label_col.upper()[0]}{label_col[1:]}"

                ax[i].set_title(f"{title} ({len(l_gdf)} samples, {p5}-{p95})")
                sns.distplot(l_gdf[label_col], ax=ax[i], kde=False)
                ax[i].axvline(mean, color='b', linestyle='--')
                ax[i].axvline(median, color='b', linestyle='-')
                self.titles.append(title)

            gdfs[label_col] = l_gdf

            # Split dataset
            train_data[label_col] = l_gdf.sample(frac=1 - test_split, random_state=r_seed)
            train_stats = train_data[label_col].describe()
            test_data[label_col] = l_gdf.drop(train_data[label_col].index)
            print(f"> {len(train_data[label_col])} training and {len(test_data[label_col])} testing samples for {label_col} column")

            # Remove label col from stats df
            train_stats.pop(label_col)
            train_stats_tr = train_stats.transpose()

            # Extract train and test labels
            train_labels[label_col] = train_data[label_col].pop(label_col)
            test_labels[label_col] = test_data[label_col].pop(label_col)

            scl = MinMaxScaler()
            cols = train_data[label_col].columns
            train_data_n[label_col] = pd.DataFrame(scl.fit_transform(train_data[label_col]), columns=cols)
            test_data_n[label_col] = pd.DataFrame(scl.fit_transform(test_data[label_col]), columns=cols)

            # Get list of predictor features
            label_features[label_col] = train_data[label_col].columns

        self.train_cols = [col for col in gdf.columns if col not in self.label_cols]
        self.gdf = gdf
        self.gdfs = gdfs
        self.train_data = train_data
        self.train_data_n = train_data_n
        self.test_data = test_data
        self.test_data_n = test_data_n
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.plot = plot
        self.label_features = label_features

        if plot: plt.legend({'Mean': mean, 'Median': median})
        if plot:
            fig.savefig(f"Content/y_histogram.png", height=1024, width=2048, dpi=self.dpi)
        return

    def plot_dependent_maps(self, gdf, dependent, x=10, y=5, run=True):
        if run:
            # Plot dependent variables on map
            print("Plotting dependent variable maps")
            dep2 = {}
            for d in dependent.keys():
                cols = [e for sl in dependent[d] for e in sl]
                if len(cols) > 0:
                    dep2[d] = cols

            for d, cols in dep2.items():
                if len(cols) == 1:
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(x, len(cols) * y))
                elif (len(cols) % 2) == 0:
                    n_rows = math.ceil(len(cols) / 2)
                    fig, ax = plt.subplots(ncols=2, nrows=n_rows, figsize=(x, n_rows * y))
                else:
                    n_rows = math.ceil(len(cols) / 3)
                    fig, ax = plt.subplots(ncols=3, nrows=n_rows, figsize=(x, n_rows * y))

                for i, col in enumerate(cols):
                    if col[0] in gdf.columns:
                        if len(cols) == 1:
                            axis = ax
                        elif len(cols) == 2:
                            axis = ax[i]
                        elif (len(cols) % 2) == 0:
                            k = i % 2
                            j = math.floor(i / 2)
                            axis = ax[j][k]
                        elif len(cols) == 3:
                            axis = ax[i]
                        else:
                            k = i % 3
                            j = math.floor(i / 3)
                            axis = ax[j][k]
                        gdf.plot(column=col[0], cmap='viridis', legend=True, scheme='natural_breaks', ax=axis)
                        axis.set_title(col[1])
                        axis.axis('off')

                plt.tight_layout()
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                plt.savefig(f'Content/{d}.png', dpi=self.dpi)

    def neural_network(self, epochs):
        clean = True
        split = 0.2
        tf.random.set_seed(self.r_seed)

        if clean: [os.remove(f"Checkpoints/{file}") for file in os.listdir('Checkpoints')]

        # Setup visualization
        fig = make_subplots(
            rows=len(self.label_cols), cols=2, column_widths=[0.7, 0.3],
            row_heights=[300 for row in self.label_cols],
            # row_titles=self.label_cols,
            column_titles=[
                f'Predictions ({int(len(self.gdf) * split)} rows, {int(split*100)}% of the data)',
                f'Feature Importance (seed {self.r_seed})'  # ({n_weights}/{len(self.gdf.columns) - len(self.label_cols)} features)'
            ])

        # Iterate over labels to get weights and validations
        row_names = []
        validations = pd.DataFrame()
        weights_d = {}
        rmses = []

        p_fig, ax = plt.subplots(ncols=len(self.label_cols), figsize=self.figsize, dpi=300)
        for i, label_col in enumerate(self.label_cols):
            print(f"Predicting {label_col}")
            train_data = self.train_data[label_col]
            test_labels = self.test_labels[label_col]
            train_labels = self.train_labels[label_col]
            normed_train_data = self.train_data_n[label_col]
            normed_test_data = self.test_data_n[label_col]

            # Split train and test data
            gdf = self.gdfs[label_col].copy(deep=False).loc[:, ast.literal_eval(self.predictors.at[i, 'predictors'])+[label_col]]
            gdf = gdf.dropna()
            try: [gdf.drop(label, axis=1, inplace=True) for label in self.label_cols if label != label_col]
            except: pass
            row_names.append(f'{label_col} ({len(gdf.columns)}f)')

            # Filter according to z-score (remove outliers)
            filter_z = False
            if filter_z:
                z_threshold = 4
                scaler = StandardScaler()
                scaler.fit(gdf)
                normed_data = gpd.GeoDataFrame(scaler.transform(gdf), columns=gdf.columns)
                normed_data = normed_data.apply(lambda x: [y if y < z_threshold else np.nan for y in x])
                normed_data_cl = normed_data.dropna(axis=0)
                print(f"> {len(normed_data) - len(normed_data_cl)} outlier(s) dropped")
                gdf = gdf.iloc[normed_data_cl.index]

            # Build NN model
            def build_model():
                n_nodes = 128
                model = keras.Sequential([
                    layers.Dense(n_nodes, activation='relu', input_shape=[len(train_data.keys())]),
                    layers.Dense(n_nodes, activation='relu'),
                    layers.Dropout(0.2),
                    layers.Dense(1)
                ])
                model.add(BatchNormalization())
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
                return model

            m = build_model()
            print(m.summary())

            # Create checkpoint
            cp_name = 'Checkpoints/Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
            checkpoint = ModelCheckpoint(cp_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
            viann = VarImpVIANN(verbose=1)

            # Train the neural network
            history = m.fit(normed_train_data.values, train_labels.values, batch_size=5,
                  epochs=epochs, validation_split=split, verbose=0, callbacks=[checkpoint])

            imp = pd.DataFrame({'features': [col for col in gdf.columns if col != label_col], 'importance':viann.varScores})
            imp = imp.sort_values(by=['importance'], ascending=False)

            # Visualize training progress
            hist = pd.DataFrame(history.history)
            hist['epoch'] = history.epoch
            hist.to_csv(f"{self.chdir}/NeuralNets/{label_col}_history.csv")

            # Predict to validate the model
            validation = pd.DataFrame()
            validation['labels'] = test_labels
            validation['predictions_nn'] = m.predict(normed_test_data.astype(float)).flatten()
            validation = validation.sort_values('labels').reset_index(drop=True)
            validations[f'{label_col}_labels'] = validation['labels']
            validations[f'{label_col}_predictions'] = validation['predictions_nn']

            # Flatten between 0-1
            n01 = True
            if n01:
                validation.loc[validation['predictions_nn'] > 1].at[:, 'predictions_nn'] = 1
                validation.loc[validation['predictions_nn'] < 0].at[:, 'predictions_nn'] = 0

            # Visualize predictions
            if self.plot:
                if i == 0: lgd=True
                else: lgd=False
                fig.add_trace(
                    go.Scatter(x=validation.index,y=validation.predictions_nn,
                        name='predictions', showlegend=lgd, line=dict(color='lightgray')),
                    row=i+1, col=1)
                fig.add_trace(
                    go.Scatter(x=validation.index,y=validation.predictions_nn.sort_values(),
                        name='sorted predictions', showlegend=lgd, line=dict(color='royalblue')),
                    row=i+1, col=1)
                fig.add_trace(
                    go.Scatter(x=validation.index, y=validation.labels,
                        name='labels', showlegend=lgd, line=dict(color='mediumturquoise')),
                    row=i+1, col=1)
                fig["layout"][f"yaxis{(i * 2) + 1}"].update(range=[0, 1], showgrid=False)
                fig["layout"][f"xaxis{(i * 2) + 1}"].update(showgrid=False)

            rmse = round(np.sqrt(((validation.predictions_nn - validation.labels) ** 2).mean()), 3)
            rmses.append(rmse)
            if self.plot:
                sns.scatterplot(validation.labels, validation.predictions_nn, ax=ax[i], alpha=0.2)
                ax[i].set_title(f"{label_col} (rmse: {rmse})")
            if self.same_scale: ax[i].set(xlim=self.range, ylim=self.range)

            # Extract weights and biases from neural network
            weights_raw = pd.DataFrame(m.layers[0].get_weights()[0].transpose(), columns=train_data.columns)
            weight = pd.concat([weights_raw.sum().abs(), weights_raw.sum()], axis=1)
            weight.columns = ['positive', 'raw']
            weight = weight.sort_values(by=['positive'], ascending=False)

            weight.to_csv(f'{self.chdir}/NeuralNets/{label_col}_weights_s{self.r_seed}.csv')

            weight_t = weight.head(self.n_weights)
            weights_d[label_col] = list(weight_t.index)

            # Visualize weights
            imp_f = imp.head(self.n_weights)
            fig.append_trace(
                go.Bar(x=imp_f['importance'], showlegend=False, marker={'color':'lightgray'}, orientation='h',
                    text=imp_f['features'], textfont=dict(color='royalblue'), textposition="inside"),
                row=i+1, col=2
            )

            example_batch = normed_train_data[:10]
            example_result = m.predict(example_batch)

        # Pre process data for R2 calculation
        scaler = MinMaxScaler()

        r2s_d = {}
        s_titles_df = pd.DataFrame()
        for i, label_col in enumerate(self.label_cols):
            gdf = self.gdf.copy(deep=False).loc[:, ast.literal_eval(self.predictors.at[i, 'predictors'])+[label_col]]
            gdf = gdf.dropna()

            # Calculate R2
            print(f"> Calculating R2s for {label_col}")

            r2s = pd.DataFrame(index=gdf.columns, columns=['r2'])
            for k, w_col in enumerate(weights_d[label_col]):
                scaler.fit(gdf[w_col].values.reshape(-1, 1))
                col_norm = scaler.transform(gdf[w_col].values.reshape(-1, 1))
                r2 = r2_score(gdf[label_col].values.reshape(-1, 1), col_norm)
                s_titles_df.at[k, label_col] = f"{w_col}: R2={round(r2, 2)}"

            plt.figure(figsize=(22, 17))
            sns.set(font_scale=0.7)
            sns_plot = sns.pairplot(gdf.loc[:, weights_d[label_col]+[label_col]],
                height=3, aspect=1, corner=True, markers="+", diag_kind="kde")
            sns_plot.savefig(f"Content/{label_col}_{self.n_weights}weights_seed{self.r_seed}.png", dpi=self.dpi)

        sp_titles = []
        for ind in list(s_titles_df.index):
            for label_col in self.label_cols:
                sp_titles.append(s_titles_df.at[ind, label_col])

        fig_scp = make_subplots(
            rows=self.n_weights,
            cols=len(self.label_cols),
            row_heights=[300 for col in range(self.n_weights)],
            subplot_titles=sp_titles
        )

        for i, label_col in enumerate(self.label_cols):
            gdf = self.gdf.copy(deep=False).loc[:, ast.literal_eval(self.predictors.at[i, 'predictors'])+[label_col]]
            gdf = gdf.dropna()

            for j, col in enumerate(weights_d[label_col]):
                fig_scp.add_trace(
                    go.Scatter(x=gdf[col], y=gdf[label_col],
                        name=f"{col}", mode='markers', hoverinfo='skip', marker={'color':'royalblue'}),
                    row=j + 1, col=i + 1
                )

        if self.plot:
            [fig.update_yaxes(title_text=name, row=i+1, col=1) for i, name in enumerate(row_names)]
            fig.update_layout(paper_bgcolor='rgba(255,255,255,1)', plot_bgcolor='rgba(255,255,255,1)',
                xaxis_showgrid=False, yaxis_showgrid=False)

            # [fig_scp.update_yaxes(title_text=name, row=i + 1, col=1) for i, name in enumerate(scp_rows)]
            fig_scp.update_layout(height=self.n_weights*300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_showgrid=False, yaxis_showgrid=False, showlegend=False,
                font=dict(
                    family="Calibri",
                    size=10,
                    color="#7f7f7f")
                )

            p_fig.savefig(f"Content/reg_rmse_{round(sum(rmses), 3)}_KerasANN.png", height=1024, width=2048, dpi=self.dpi)
            fig.write_image("predictions.png", height=1024, width=2048)
        print("Model trained and tested")

    def linear(self, method, y_crr_thr=0, x_crr_thr=1, poly=False):
        self.cur_method = method()
        name = self.cur_method.__class__.__name__
        print(f"\n### {name} regression model from Scikit-Learn ###")
        rmses = []
        coefficients = {}
        sns.set()
        fig, ax = plt.subplots(ncols=len(self.label_cols), figsize=self.figsize)
        for i, label_col in enumerate(self.label_cols):
            print(f"Calculating linear regresison for column {label_col}")

            # Create correlation matrix
            l_gdf = self.train_data[label_col]
            corr_mat = l_gdf.corr().abs()

            # Select upper triangle of correlation matrix
            upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))

            # Find index of feature columns with correlation greater than threshold
            to_drop = [column for column in upper.columns if any(upper[column] > x_crr_thr) and column != label_col]
            print(f"> Dropping {len(to_drop)} features that are more than {x_crr_thr} correlated among themselves")

            # Drop features
            l_gdf = l_gdf.drop(l_gdf[to_drop], axis=1)
            corr_mat = l_gdf.corr().abs()
            print(f"Dropping features more than {y_crr_thr} correlated to dependent variable")
            if label_col in corr_mat.columns:
                features = corr_mat.loc[corr_mat[label_col] > y_crr_thr][label_col].drop(label_col, axis=0).index
            else: features = corr_mat.columns
            self.features_d[label_col] = features
            print(f"> {len(features)} features extracted to predict {label_col}")

            self.cur_method.random_state = self.r_seed

            # Check if x DataFrame should be normalized
            if self.norm_x:
                x_train = self.train_data_n[label_col]
                x_test = self.test_data_n[label_col]

            else:
                x_train = self.train_data[label_col]
                x_test = self.test_data[label_col]

            # Add polynomial function, if specified
            if poly:
                transformer = PolynomialFeatures(degree=2, include_bias=False)
                x_train = transformer.fit(x_train)
                x_train = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x_train)

            # Fit and predict model
            self.fitted[label_col] = self.cur_method.fit(X=x_train.loc[:, features], y=self.train_labels[label_col])
            predictions = self.cur_method.predict(x_test.loc[:, features])

            # Verify error
            rmse = round(np.sqrt(((predictions - self.test_labels[label_col]) ** 2).mean()), 3)
            rmses.append(rmse)
            r2 = r2_score(self.test_labels[label_col], predictions)
            print(f"> {label_col} linear model tested with rmse {rmse} and R2 {round(r2, 2)}")

            print(f"> Constructing index from coefficients")
            c_df = pd.DataFrame(columns=['coefficient'], index=features)
            for f, c in zip(features, self.cur_method.coef_):
                c_df.at[f, 'coefficient'] = c
            c_df = c_df.sort_values(by='coefficient', ascending=False)
            coefficients[label_col] = c_df

            # Plot results
            if self.plot:
                if self.same_scale: ax[i].set(xlim=self.range, ylim=self.range)
                tl = self.test_labels
                tl['predictions'] = predictions
                sns.scatterplot(
                    x=label_col, y='predictions', data=tl, ax=ax[i], alpha=0.5)
                title = f"{self.titles[i]} (rmse: {rmse} | R2: {round(r2, 2)})"
                ax[i].set_title(title)
                ax[i].set_xlabel('')
                t = pd.DataFrame()
                t['Feature'] = coefficients[label_col].index
                t['Coef.'] = [round(v, self.round_f) for v in coefficients[label_col]['coefficient'].values]

                if self.rename is not None:
                    t['Feature'] = [self.rename[f] if f in self.rename.keys() else f for f in t['Feature']]

                table = ax[i].table(
                    cellText=t.values, colLabels=t.columns, colWidths=[0.9, 0.1], edges='open',
                    loc='bottom', bbox=[0,-0.5,1,0.3])
                plt.subplots_adjust(bottom=0.3)
                table.auto_set_font_size(False)
                table.set_fontsize(7)

        fig.savefig(
            f"Content/reg_rmse_{round(sum(rmses), 3)}_s{self.r_seed}_{self.cur_method.__class__.__name__}.png",
            bbox_inches='tight', dpi=self.dpi
        )
        return coefficients

    def test_linear(self, coefficient_dfs, prefix=''):

        corrs = []
        fig, ax = plt.subplots(ncols=len(self.label_cols), figsize=self.figsize)
        for i, label_col in enumerate(self.label_cols):
            c_df = coefficient_dfs[label_col]
            scaler = MinMaxScaler()
            n_gdf = pd.DataFrame(scaler.fit_transform(self.gdfs[label_col]),
                                 columns=self.gdfs[label_col].columns).loc[:, list(self.features_d[label_col]) + [label_col]]

            n_gdf['index'] = sum([n_gdf[c_df.index[i]] * c_df['coefficient'][i] for i in range(0, self.n_indicators-1)])

            correlation = n_gdf.corr()[label_col]['index']
            corrs.append(correlation)
            print(f"> Correlation between index and {label_col} label is: {round(correlation, 2)}")

            r2 = r2_score(n_gdf[label_col], n_gdf['index'])
            title = f"{label_col} (r2: {round(r2, 2)})"
            sns.regplot(n_gdf['index'], n_gdf[label_col], ax=ax[i])
            ax[i].set_title(title)

        plt.savefig(f'Content/{prefix}i{round(sum(corrs)/len(corrs), 2)}_s{self.r_seed}_{self.cur_method.__class__.__name__}.png', dpi=self.dpi)
        return corrs

    def non_linear(self, method=RandomForestRegressor):
        self.cur_method = method
        if method is None:
            methods = [
                RandomForestRegressor(), TransformedTargetRegressor(),
                AdaBoostRegressor(), BaggingRegressor(), ExtraTreesRegressor(), GradientBoostingRegressor(),
            ]
        else: methods = self.methods

        name = method().__class__.__name__
        print(f"\n### {name} regression model from Scikit-Learn ###")

        for i, label_col in enumerate(self.label_cols):
            # Initiate regressor
            r_method = method()
            print(f"> Regressing {len(self.label_features[label_col])} features on column {label_col}"
                  f" using {len(self.train_data[label_col])+len(self.test_data[label_col])} samples")

            # Train and predict
            r_method.random_state = self.r_seed
            self.fitted[label_col] = r_method.fit(self.train_data[label_col], self.train_labels[label_col])
            self.methods[label_col] = r_method

            predictions = self.fitted[label_col].predict(self.test_data[label_col])
            self.predicted[label_col] = predictions
        return name

    def test_non_linear(self, radius=(400, 800, 1600, 3200, 4800), i_method='regular'):

        columns = [col for col in self.gdf.columns if (col not in self.label_cols) and (f"_r" in col)]

        rmses = []

        for i, label_col in enumerate(self.label_cols):

            predictions = self.predicted[label_col]

            # Verify error
            rmse = round(np.sqrt(((predictions - self.test_labels[label_col]) ** 2).mean()), 3)
            rmses.append(rmse)

            # Calculate feature importance
            print("> Calculating feature importance")
            name = self.methods[label_col].__class__.__name__
            if name == 'RandomForestRegressor':
                if i_method == 'regular':
                    self.feat_imp[label_col] = pd.DataFrame({
                        'feature': self.label_features[label_col],
                        'importance': self.methods[label_col].feature_importances_
                    })
                elif i_method == 'permutation':
                    def r2(rf, x_train, y_train):
                        return r2_score(y_train, rf.predict(x_train))
                    perm_imp_rfpimp = permutation_importances(self.methods[label_col], self.train_data[label_col],
                                                              self.train_labels[label_col], r2)
                    self.feat_imp[label_col] = pd.DataFrame(self.label_features[label_col], perm_imp_rfpimp, columns=['feature', 'importance'])

            if i_method == 'boruta':
                # define Boruta feature selection method
                print(f"Iterating {label_col} column on Boruta algortihm to rank feature importance")
                feat_selector = BorutaPy(self.fitted[label_col], n_estimators='auto', verbose=2, random_state=self.r_seed)

                # find all relevant features - 5 features should be selected
                feat_selector.fit(self.train_data[label_col].values, self.train_labels[label_col].values)

                # check selected features - first 5 features are selected
                print(f"{label_col} support: {feat_selector.support_}")

                # check ranking of features
                # print(f"{label_col} ranking: {feat_selector.ranking_}")
                self.feat_imp[label_col] = pd.DataFrame({
                    'feature': [f for f in self.label_features[label_col]],
                    'importance': [r for r in feat_selector.ranking_]})
                self.feat_imp[label_col] = self.feat_imp[label_col].sort_values(by='rank')
                self.feat_imp[label_col].to_csv(f'boruta_rank_{label_col}_{name}.png')

            # out[label_col] = imp_df.loc[label_features, label_col].dropna()

        # Plot results
        if self.plot:
            sns.set()

            fig, ax = plt.subplots(ncols=len(self.label_cols), nrows=2, figsize=self.figsize, dpi=300)
            for i, label_col in enumerate(self.label_cols):

                if self.same_scale: ax[0][i].set(xlim=self.range, ylim=self.range)
                sns.regplot(self.test_labels[label_col], self.predicted[label_col], ax=ax[0][i])

                # Calculate R2
                predictions = self.predicted[label_col]
                r2 = r2_score(self.test_labels[label_col], predictions)
                title = f"{label_col.title()} (rmse: {rmses[i]} | r2: {round(r2, 2)})"
                print(title)
                ax[0][i].set_title(title)
                ax[1][i].set_title('')

                importance = self.feat_imp[label_col].sort_values(by='importance', ascending=False)
                importance = importance.replace(self.rename)
                importance.index = importance.feature
                importance.head(self.n_indicators).plot(kind='bar', ax=ax[1][i])
                total_rmse = round(sum(rmses), 3)
                plt.savefig(f"Content/reg_rmse_seed{self.r_seed}_{name}.png", bbox_inches='tight', dpi=self.dpi)

    def predict(self, x_dict):
        return {label_col: self.fitted[label_col].predict(x_dict[label_col]) for label_col in self.label_cols}

    def partial_dependence (self, n_features=0, high_feat=0.01):
        dfs = []
        features = {}
        for i, label_col in enumerate(self.label_cols):
            if label_col not in self.feat_imp.keys():
                print(f"!!! Feature importance {label_col} not found !!!")

            else:
                # Get 1% most important features
                importance = self.feat_imp[label_col].sort_values(by='importance', ascending=False)

                if n_features > 0:
                    filtered_imp = importance.head(n_features)
                else:
                    filtered_imp = importance[importance['importance'] > (np.percentile(
                        importance['importance'], 100- (high_feat * 100)))]

                # Rename features if rename dictionary is provided
                if self.rename is not None:
                    filtered_imp['names'] = [self.rename[f] if f in self.rename.keys() else f for f in filtered_imp.feature]
                else:
                    filtered_imp['names'] = filtered_imp['feature']
                features[label_col] = list(filtered_imp['feature'])

                # Plot dependencies
                ppd = plot_partial_dependence(self.fitted[label_col], self.train_data[label_col], features=filtered_imp.feature,
                                        n_jobs=3, grid_resolution=20)

                # Create DataFrame with dependency data
                df = pd.DataFrame()
                for tpl, name in zip(ppd.pd_results, filtered_imp['names']):
                    try: df[name] = tpl[0][0]
                    except: pass
                    df['x'] = range(len(tpl[0][0]))
                dfs.append(df)

        pd_fig, pd_axs = plt.subplots(ncols=len(self.label_cols))
        pd_fig.set_size_inches(self.figsize[0]*1.5, self.figsize[1]*1.5)
        font_prop = font_manager.FontProperties(size=13)

        for i, df in enumerate(dfs):
            # Get colors
            cmap = get_cmap('Set2')
            colors = [cmap((j + 1) / (len(df.columns) - 1)) for j in range(len(df.columns) - 1)]

            # Plot lines
            if self.plot:
                df.plot(x='x', y=df.loc[:, df.columns != 'x'].columns, kind='line', color=colors, ax=pd_axs[i], lw=2)
                pd_axs[i].set_title(f'Partial dependencies for {self.titles[i].title()}')
                pd_axs[i].legend(bbox_to_anchor=(0, 0), loc=2, borderaxespad=0, prop=font_prop, columnspacing=1, handlelength=1)

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.savefig(f'Content/pd{n_features}.png', bbox_inches='tight', dpi=self.dpi)

        return features

    def multi_output(self, methods=None):
        for method in self.methods:
            print("> Processing multi output data")
            gdf = self.gdf.copy()

            # Pre process data
            for label_col in self.label_cols:
                p5 = round(np.percentile(gdf[label_col], 100-self.percentile), 2)
                p95 = round(np.percentile(gdf[label_col], self.percentile), 2)
                gdf = gdf.loc[(gdf[label_col] > p5) & (gdf[label_col] < p95)]

            print(f"> GeoDataFrame pre processed with {len(gdf)} rows")

            # Split training and testing sets
            s_gdf = gdf.sample(frac=1 - self.split, random_state=self.r_seed)
            l_gdf = gdf.drop(s_gdf.index)
            y = s_gdf.loc[:, self.label_cols]
            X = s_gdf.loc[:, self.train_cols]

            # Train and test the model
            mor = MultiOutputRegressor(method).fit(X=X, y=y)
            predictions = pd.DataFrame(mor.predict(l_gdf.loc[:, self.train_cols]), columns=self.label_cols)

            rmses = []
            fig, ax = plt.subplots(ncols=len(self.label_cols), figsize=self.figsize, dpi=300)
            for i, label_col in enumerate(self.label_cols):
                # Calculate error
                rmse = round(np.sqrt(((predictions[label_col] - l_gdf[label_col]) ** 2).mean()), 3)
                rmses.append(rmse)

                # Display results
                sns.regplot(l_gdf[label_col], predictions[label_col], ax=ax[i])
                ax[i].set_title(f"{label_col} (rmse: {rmse})")

            if self.plot: plt.savefig(f"Content/reg_rmse_{round(sum(rmses), 3)}_{method.__class__.__name__}.png", dpi=self.dpi)
        return

    def group_drop_op (self, df, feature='feature', drop='radius', op_type='sum'):

        if op_type == 'sum': matrix = df.groupby(feature).sum()
        elif op_type == 'mean': matrix = df.groupby(feature).mean()
        else: matrix = df

        try: matrix = matrix.drop(drop, axis=1)
        except: print(f"> Column(s) {drop} not dropped")
        matrix_sum = matrix.sum(axis=1).sort_values(ascending=False)
        matrix_sum = matrix_sum[matrix_sum.apply(lambda x: x > 0)]

        return {'grouped': matrix, 'summed': matrix_sum}

    def weights(self, imp_df, n_outputs=10, corr_tsh=0.3, ftr=lambda col: (col != 'feature') and (col != 'radius')):
        sns.set()
        features = {}
        weights = {}

        for label_col in self.label_cols:
            imp_df = imp_df[label_col].dropna()
            indices = imp_df.abs().sort_values(ascending=False).index
            f_gdf = self.gdf.loc[:, indices]
            corr = f_gdf.corr()
            pot_ftr = list(corr.columns)
            for j, c_col in enumerate(corr.columns):
                check = corr[c_col][j+1:]
                to_drop = check[(check > corr_tsh) | (check < -corr_tsh)]
                for ind in to_drop.index:
                    try: pot_ftr.remove(ind)
                    except: pass
            features[label_col] = pot_ftr[:n_outputs]

        for label_col in self.label_cols:
            fig, ax = plt.subplots()
            weights[label_col] = imp_df.loc[features[label_col], label_col]
            weights[label_col].plot(kind='barh', ax=ax)
            plt.savefig(f'Content/weights_{label_col}_{self.methods[0].__class__.__name__}.png', bbox_inches='tight', dpi=self.dpi)

        return weights

    def validate_weights(self, weights):

        feature2 = []
        predictors2 = []
        for p, (key, values) in zip(self.label_cols, weights.items()):
            feature2.append(p)
            predictors2.append(list(values.index))

        predictors2 = pd.DataFrame({
            'feature': [p for p in self.label_cols],
            'predictors': predictors2
        })

        # Print correlation matrices
        for col in self.label_cols:
            plt.figure(figsize=(22, 17))
            l = predictors2[predictors2.feature == col].predictors
            sns_plot = sns.pairplot(self.gdf.loc[:, l.values[0] + [col]], height=3, aspect=1, corner=True, markers="+",
                                    diag_kind="kde")
            sns_plot.savefig(f"Content/indicators_{col}_{self.methods[0].__class__.__name__}.png", dpi=self.dpi)

        # From cleaned weights
        sns.set()
        indices = []
        for col, w_df in weights.items():
            df_coef = pd.DataFrame()
            z_df = self.n_gdf.loc[:, w_df.index].transpose()
            for col2 in z_df.columns:
                df_coef[col2] = w_df

            product = df_coef * z_df
            # reg.gdf[f'index_{col}'] = reg.n_gdf.loc[:, df.index].sum(axis=1)
            self.gdf[f'index_{col}'] = product.sum(axis=0)

            # print(reg.gdf[f'index_{key}'])
            indices.append(f'index_{col}')

        print("> Saving regression indices")
        f_gdf = self.gdf.loc[:, indices + self.label_cols]
        corr = f_gdf.corr()
        total = corr.loc[self.label_cols, indices].sum().sum()
        corr.to_csv(f'indexes_{round(total, 3)}_{self.methods[0].__class__.__name__}.csv')

    def pre_norm_exp(self, gdf, prefix='rf'):

        # Use random forest to predict mobility on sandbox
        predicted_proxy = self.predict(
            x_dict={label_col: gdf.loc[:, self.train_data[label_col].columns] for label_col in self.label_cols})
        for label_col in self.label_cols: gdf[f"{label_col}_{prefix}"] = predicted_proxy[label_col]

        # Normalize predictions to 0-1
        total = gdf.loc[:, [f"{lc}_{prefix}" for lc in self.label_cols]].sum(axis=1)
        for label_col in self.label_cols: gdf[f"{label_col}_{prefix}_n"] = gdf[f"{label_col}_{prefix}"] / total

        # Export results
        fig, ax = plt.subplots(ncols=len(self.label_cols), figsize=self.figsize)
        for i, label_col in enumerate(self.label_cols):
            gdf.plot(column=f"{label_col}_{prefix}_n", ax=ax[i], colormap='viridis',
                              legend=True,
                              legend_kwds={
                                  'orientation': "horizontal",
                              },
                              linewidth=0.05)
            ax[i].set_title(f"{label_col.upper()} | MEAN: {round(gdf[f'{label_col}_{prefix}_n'].mean()*100, 0)}")
            ax[i].axis('off')

        plt.savefig(f'mobility_{prefix}.png', dpi=self.dpi)
        return gdf

class Clustering:
    def __init__(self, df, features=None, rng=(0.2, 1), ax=None, by=None, minim=None, maxim=None):
        if features is not None:
            self.features = [[tp[0] for tp in features[0]], [tp[0] for tp in features[1]]]
        self.rng = rng
        self.ax = ax
        self.by = by
        if ax is not None: self.ax2 = ax.twinx()
        if by is not None:
            if minim is None: minim = df[by].quantile(0.99)
            if maxim is None: maxim = df[by].quantile(0.99)
            self.df = df[(df[by] > minim) & (df[by] < maxim)]
        else: self.df = df
        return

    def jenks_break(self, n_class):
        value = self.by

        # Break by density
        if value not in self.df.columns: print(f"!!! {value} not found on GeoDataFrame !!!")
        else:
            f_gdf = self.df
            breaks = jenkspy.jenks_breaks(f_gdf[value].values, nb_class=n_class)
            for i, b in enumerate(breaks[:n_class]):
                if i == len(breaks) - 1:
                    f_gdf.loc[f_gdf[value] >= b, 'group'] = i
                    f_gdf.loc[f_gdf[value] >= b, 'break'] = b
                else:
                    f_gdf.loc[(f_gdf[value] >= b) & (f_gdf[value] < breaks[i + 1]), 'group'] = i
                    f_gdf.loc[(f_gdf[value] >= b) & (f_gdf[value] < breaks[i + 1]), 'break'] = b

            return f_gdf

    def plot_clusters_hist(self, f_gdf, colormap='Blues'):
        cmap = get_cmap(colormap)
        colors = [cmap((i + 1) / len(f_gdf['group'].unique())) for i in range(len(f_gdf['group'].unique()))]
        df = pd.DataFrame([len(f_gdf[f_gdf['group'] == g]) for g in f_gdf['group'].unique()])
        bl = df.plot(kind='bar', stacked=True, color=colors, zorder=0, ax=self.ax, legend=False)
        [p.set_color(c) for p, c in zip(bl.patches, colors)]
        return bl

    def trim_correlated(self, x_crr_thr=0):
        # Create correlation matrix
        corr_mat = self.df.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))

        # List features to drop
        to_drop = [column for column in upper.columns if any(upper[column] > x_crr_thr)]
        print(f"> Dropping {len(to_drop)} features that are more than {x_crr_thr} correlated among themselves")

        # Drop features
        self.features.append(self.df.drop(self.df[to_drop], axis=1).columns)
        return self.features

    def norm_sum(self):
        df = self.df.fillna(0)
        if df.__class__.__name__ == 'Series':
            df = pd.DataFrame(df).transpose()

        if len(self.features) > 0:
            [print(f"!!! {f} not found on DataFrame!!!") for f in self.features[0] if f not in df.columns]
            [print(f"!!! {f} not found on DataFrame!!!") for f in self.features[1] if f not in df.columns]

            if len(self.features[0]) != 0:
                df2max = pd.DataFrame(df.loc[:, [f for f in self.features[0] if f in df.columns]])
            else: df2max = None

            if len(self.features[1]) != 0:
                df2min = pd.DataFrame(df.loc[:, [f for f in self.features[1] if f in df.columns]])
            else: df2min = None

            if (df2max is not None) & (df2min is not None): index = df2max.mean().mean()/df2min.mean().mean()
            elif df2max is None: index = 1/df2min.mean().mean()
            else: index = df2max.mean().mean()
            if index > 0: pass
            return index

    def calculate_index(self, dependent):

        # Calculate indices
        scl = StandardScaler()
        cols = [item[0] for sublist in [variable[0] + variable[1] for kind, variable in dependent.items()] for item in
                sublist if item[0] in self.df.columns]
        s_gdf = scl.fit_transform(self.df.loc[:, cols])
        scl = MinMaxScaler()
        s_gdf = pd.DataFrame(scl.fit_transform(s_gdf), columns=cols)
        s_gdf.index = self.df.index

        for i in s_gdf.index:
            if s_gdf.loc[i, cols].isna().sum() == 0:
                for kind, variable in dependent.items():
                    s_gdf.at[i, kind] = Clustering(s_gdf.loc[i, :], features=variable).norm_sum()

        return s_gdf

    def index_by_group(self, f_gdf, dependent):

        s_gdf = self.calculate_index(dependent=dependent)

        df = pd.DataFrame()
        for group in f_gdf['group'].unique():
            gdf = s_gdf.loc[f_gdf[f_gdf['group'] == group].index]
            if len(gdf) != 0:
                df[group] = [Clustering(gdf, features=variable).norm_sum() for kind, variable in dependent.items()]

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any', axis=0)

        scl = MinMaxScaler(self.rng)
        df = pd.DataFrame(scl.fit_transform(df.transpose())).transpose()
        df.index = [kind for kind in dependent.keys()]

        return df

