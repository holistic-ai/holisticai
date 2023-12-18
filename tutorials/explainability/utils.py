from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from holisticai.datasets import load_adult
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from holisticai.explainability import Explainer
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

def load_processed_diabetes(seed):
    dataset = load_diabetes() # load dataset
    
    X = dataset.data # features
    y = dataset.target # target 
    y = StandardScaler().fit_transform(y.reshape([-1, 1])).reshape([-1])
    feature_names = dataset.feature_names # feature names

    X = pd.DataFrame(X, columns=feature_names) # convert to dataframe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed) # train test split
    return X_train, X_test, y_train, y_test
    
    
    
def load_processed_adult(seed):
    # Dataset
    dataset = load_adult()

    # Dataframe
    df = pd.concat([dataset["data"], dataset["target"]], axis=1)
    protected_variables = ["sex", "race"]
    output_variable = ["class"]

    # Simple preprocessing
    y = df[output_variable].replace({">50K": 1, "<=50K": 0})
    X = pd.get_dummies(df.drop(protected_variables + output_variable, axis=1), dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed) # train test split
    return X_train, X_test, y_train, y_test

def train_classifier_model(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from tqdm import tqdm
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from holisticai.efficacy.metrics import classification_efficacy_metrics

    outputs = []
    for model_type in tqdm([LogisticRegression, RandomForestClassifier, GradientBoostingClassifier]):
        
        model = Pipeline(steps=[
            ('scaler', StandardScaler()), 
            ('model', model_type())]) # instantiate model
        model.fit(X_train, y_train) # fit model

        y_pred = model.predict(X_test) # compute predictions

        # compute efficacy metrics
        res = classification_efficacy_metrics(y_test, y_pred)
        outputs.append({'y_pred': y_pred, 'X_test':X_test, 'res': res, 'model': model})
        
    return outputs

def  train_regression_model(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from tqdm import tqdm
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from holisticai.efficacy.metrics import regression_efficacy_metrics

    outputs = []
    for model_type in tqdm([LinearRegression, RandomForestRegressor, GradientBoostingRegressor]):
        
        model = Pipeline(steps=[
            ('scaler', StandardScaler()), 
            ('model', model_type())]) # instantiate model
        model.fit(X_train, y_train) # fit model

        y_pred = model.predict(X_test) # compute predictions

        # compute efficacy metrics
        res = regression_efficacy_metrics(y_test, y_pred)
        outputs.append({'y_pred': y_pred, 'X_test':X_test, 'res': res, 'model': model})
        
    return outputs

def save_detailed_metrics(metric_names, explainers, output_path):
    xai_metrics = [p.metrics(metric_names=metric_names, detailed=True) for p in explainers]
    names = ['LR','RF','GB']
    
    m_names = names + ['Reference']
    xai_res = pd.concat([o.iloc[:,:1] for o in xai_metrics]+[xai_metrics[0].iloc[:,1:]],axis=1)
    xai_res.columns = m_names
        
    metric_latex = xai_res.reset_index().to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format)
    
    with open(os.path.join(output_path, 'xai_metrics_detailed.tex'), 'w') as f:
        f.write(metric_latex)
        
def run_permutation_explainability(model_type, outputs):
    output_path=f'{model_type}_plots_permutation'
    os.makedirs(output_path, exist_ok=True)
    # permutation feature importance
    permutation_fi_params = {
            'max_samples' : 5000,
            'n_repeats' : 20,
            'random_state' : np.random.randint(0, 1000)
        }

    permutation_explainers = []
    for output in tqdm(outputs):
        permutation_explainer = Explainer(based_on='feature_importance',
                            strategy_type='permutation',
                            model_type=model_type,
                            model = output['model'], 
                            x = output['X_test'],
                            y = output['y_pred'],
                            **permutation_fi_params)
        permutation_explainers.append(permutation_explainer)
    
    metric_names = ["Explainability Ease","Fourth Fifths", "Position Parity", "Rank Alignment", "Region Similarity", "Spread Divergence", "Spread Ratio"]
    save_detailed_metrics(metric_names, permutation_explainers, output_path)
    
    xai_metrics = [p.metrics(metric_names=metric_names, detailed=False) for p in permutation_explainers]
    
    names = ['LR','RF','GB']
    m_names = names + ['Reference']
    xai_res = pd.concat([o.iloc[:,:1] for o in xai_metrics]+[xai_metrics[0].iloc[:,1:]],axis=1)
    xai_res.columns = m_names

    eff_res = pd.concat([o['res'].iloc[:,:1] for o in outputs]+[outputs[0]['res'].iloc[:,1:]],axis=1)
    eff_res.columns = m_names
    
    metric_latex = xai_res.reset_index().to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format)
    
    with open(os.path.join(output_path, 'xai_metrics.tex'), 'w') as f:
        f.write(metric_latex)
        
    metric_latex = eff_res.reset_index().to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format)
    
    with open(os.path.join(output_path, 'eff_metrics.tex'), 'w') as f:
        f.write(metric_latex)
        
    for p,n in zip(permutation_explainers,names):
        fig,ax = plt.subplots(1,1,figsize=(10,3))
        pdp = p.partial_dependence_plot(last=3, ax=ax, kind='both', n_cols=3)
        score = xai_res.loc['Explainability Ease'][n]
        pdp.figure_.suptitle(f"{n} (Explainability Ease : {score:.3f})")   
        fig.savefig(os.path.join(output_path,f'{n}.png'), dpi=300)
        
    fig,axs = plt.subplots(1,3, figsize=(15,6))
    for ax,p,n in zip(axs,permutation_explainers,names):
        feat_imp = p.bar_plot(alpha=0.8, max_display=30, ax=ax)
        ff_score = xai_res.loc['Fourth Fifths'][n]
        sr_score = xai_res.loc['Spread Ratio'][n]
        ax.set_title(f"{n} (Fourth Fifths : {ff_score:.3f} | Spread Ratio: {sr_score:.3f})")
        
    fig.savefig(os.path.join(output_path,f'Fourth Fifths.png'), dpi=300)
    
    fig,axs = plt.subplots(1,3, figsize=(10,3))
    for i,(n,ax) in enumerate(zip(names,axs)):
        permutation_explainers[i].contrast_visualization(show_connections=False, ax=ax)
        ax.set_title(n)
    fig.savefig(os.path.join(output_path,f'Contrast.png'), dpi=300)
    

def run_surrogate_explainability(model_type, outputs):
    output_path=f'{model_type}_plots_surrogacy'
    os.makedirs(output_path, exist_ok=True)
    surrogate_explainers = []
    for output in tqdm(outputs):
        surrogate_explainer = Explainer(based_on='feature_importance',
                            strategy_type='surrogate',
                            model_type=model_type,
                            model = output['model'], 
                            x = output['X_test'], 
                            y = output['y_pred'])
        surrogate_explainers.append(surrogate_explainer)
        
    metric_names = ["Explainability Ease","Fourth Fifths", "Spread Divergence", "Spread Ratio", "Surrogacy Efficacy"]
    save_detailed_metrics(metric_names, surrogate_explainers, output_path)
    xai_surrogate_metrics = [p.metrics(metric_names=metric_names, detailed=False) for p in surrogate_explainers]
        
    names = ['LR','RF','GB']
    m_names = names + ['Reference']
    xai_res = pd.concat([o.iloc[:,:1] for o in xai_surrogate_metrics]+[xai_surrogate_metrics[0].iloc[:,1:]],axis=1)
    xai_res.columns = m_names

    eff_res = pd.concat([o['res'].iloc[:,:1] for o in outputs]+[outputs[0]['res'].iloc[:,1:]],axis=1)
    eff_res.columns = m_names
    
    metric_latex = xai_res.reset_index().to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format)
    
    with open(os.path.join(output_path, 'xai_metrics.tex'), 'w') as f:
        f.write(metric_latex)
        
    metric_latex = eff_res.reset_index().to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format)
    
    with open(os.path.join(output_path, 'eff_metrics.tex'), 'w') as f:
        f.write(metric_latex)
        
    for p,n in zip(surrogate_explainers,names):
        fig,ax = plt.subplots(1,1,figsize=(10,3))
        pdp = p.partial_dependence_plot(last=3, ax=ax, kind='both', n_cols=3)
        score = xai_res.loc['Explainability Ease'][n]
        pdp.figure_.suptitle(f"{n} (Explainability Ease : {score:.3f})")   
        fig.savefig(os.path.join(output_path,f'{n}.png'), dpi=300)
        
    fig,axs = plt.subplots(1,3, figsize=(15,6))
    for ax,p,n in zip(axs,surrogate_explainers,names):
        p.bar_plot(alpha=0.8, max_display=10, ax=ax)
        ff_score = xai_res.loc['Fourth Fifths'][n]
        sr_score = xai_res.loc['Spread Ratio'][n]
        ax.set_title(f"{n} (Fourth Fifth : {ff_score:.3f} | Spread Ratio : {sr_score:.3f})")   
        
    fig.savefig(os.path.join(output_path,f'Fourth Fifths.png'), dpi=300)
    
    for i,n in enumerate(names):
        vis = surrogate_explainers[i].tree_visualization('pydotplus')
        vis.save(os.path.join(output_path,f'Tree_{n}.png'))
        
def run_lime_explainability(model_type, outputs):
    output_path=f'{model_type}_plots_lime'
    os.makedirs(output_path, exist_ok=True)
    
    lime_explainers = []
    for output in tqdm(outputs):
        lime_explainer = Explainer(based_on='feature_importance',
                            strategy_type='lime',
                            model_type=model_type,
                            model = output['model'], 
                            x = output['X_test'], 
                            y = output['y_pred'])
        lime_explainers.append(lime_explainer)
        
    save_detailed_metrics(None, lime_explainers, output_path)
    xai_metrics = [p.metrics(detailed=False) for p in lime_explainers]
    
    names = ['LR','RF','GB']
    m_names = names + ['Reference']
    xai_res = pd.concat([o.iloc[:,:1] for o in xai_metrics]+[xai_metrics[0].iloc[:,1:]],axis=1)
    xai_res.columns = m_names

    eff_res = pd.concat([o['res'].iloc[:,:1] for o in outputs]+[outputs[0]['res'].iloc[:,1:]],axis=1)
    eff_res.columns = m_names
    
    metric_latex = xai_res.reset_index().to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format)
    
    with open(os.path.join(output_path, 'xai_metrics.tex'), 'w') as f:
        f.write(metric_latex)
        
    metric_latex = eff_res.reset_index().to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format)
    
    with open(os.path.join(output_path, 'eff_metrics.tex'), 'w') as f:
        f.write(metric_latex)
        
    if model_type.startswith('binary'):
        index = [0,1,2]
        dslabels = ["DS", "DS [label=0]", "DS [label=1]"]
        fslabels = ["FS", "FS [label=0]", "FS [label=1]"]
    else:
        index = [0,1,2,3,4]
        dslabels = ["DS", "DS [Q0]", "DS [Q1]", "DS [Q2]", "DS [Q3]"]
        fslabels = ["FS", "FS [Q0]", "FS [Q1]", "FS [Q2]", "FS [Q3]"]
        
    names = ['LR','RF','GB']
    xmin0,xmax0 = [],[]
    xmin1,xmax1 = [],[]
    fig,axes = plt.subplots(3, 2, figsize=(20,15))
    for i,(n,p) in enumerate(zip(names,lime_explainers)):
        p.show_importance_stability(axes=axes[i])
        axes[i][0].set_yticks(index,dslabels)
        xmin,xmax = axes[i][0].get_xlim()
        xmin0.append(xmin)
        xmax0.append(xmax)
        axes[i][0].set_title(f"{n} (DS={xai_res.loc['Data Stability'][n]:.3f})")
        axes[i][1].set_yticks(index,fslabels)
        xmin,xmax = axes[i][1].get_xlim()
        xmin1.append(xmin)
        xmax1.append(xmax)
        axes[i][1].set_title(f"{n} (FS={xai_res.loc['Feature Stability'][n]:.3f})")
        
    for i in range(len(axes)):
        axes[i][0].set_xlim([min(xmin0),max(xmax0)])
        axes[i][1].set_xlim([min(xmin1),max(xmax1)])
    fig.savefig(os.path.join(output_path,'Stability.png'), dpi=300)
    
def run_shap_explainability(model_type, outputs):
    output_path=f'{model_type}_plots_shap'
    os.makedirs(output_path, exist_ok=True)
    
    lime_explainers = []
    for output in tqdm(outputs):
        lime_explainer = Explainer(based_on='feature_importance',
                            strategy_type='shap',
                            model_type=model_type,
                            model = output['model'], 
                            x = output['X_test'], 
                            y = output['y_pred'])
        lime_explainers.append(lime_explainer)
        
    save_detailed_metrics(None, lime_explainers, output_path)
    xai_metrics = [p.metrics(detailed=False) for p in lime_explainers]
    
    names = ['LR','RF','GB']
    m_names = names + ['Reference']
    xai_res = pd.concat([o.iloc[:,:1] for o in xai_metrics]+[xai_metrics[0].iloc[:,1:]],axis=1)
    xai_res.columns = m_names

    eff_res = pd.concat([o['res'].iloc[:,:1] for o in outputs]+[outputs[0]['res'].iloc[:,1:]],axis=1)
    eff_res.columns = m_names
   
    metric_latex = xai_res.reset_index().to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format)
    
    with open(os.path.join(output_path, 'xai_metrics.tex'), 'w') as f:
        f.write(metric_latex)
        
    metric_latex = eff_res.reset_index().to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format)
    
    with open(os.path.join(output_path, 'eff_metrics.tex'), 'w') as f:
        f.write(metric_latex)
        
    if model_type.startswith('binary'):
        index = [0,1,2]
        dslabels = ["DS", "DS [label=0]", "DS [label=1]"]
        fslabels = ["FS", "FS [label=0]", "FS [label=1]"]
    else:
        index = [0,1,2,3,4]
        dslabels = ["DS", "DS [Q0]", "DS [Q1]", "DS [Q2]", "DS [Q3]"]
        fslabels = ["FS", "FS [Q0]", "FS [Q1]", "FS [Q2]", "FS [Q3]"]
        
    xmin0,xmax0 = [],[]
    xmin1,xmax1 = [],[]
    names = ['LR','RF','GB']
    fig,axes = plt.subplots(3, 2, figsize=(20,15))
    for i,(n,p) in enumerate(zip(names,lime_explainers)):
        p.show_importance_stability(axes=axes[i])
        axes[i][0].set_yticks(index,dslabels)
        xmin,xmax = axes[i][0].get_xlim()
        xmin0.append(xmin)
        xmax0.append(xmax)
        axes[i][0].set_title(f"{n} (DS={xai_res.loc['Data Stability'][n]:.3f})")
        axes[i][1].set_yticks(index,fslabels)
        xmin,xmax = axes[i][1].get_xlim()
        xmin1.append(xmin)
        xmax1.append(xmax)
        axes[i][1].set_title(f"{n} (FS={xai_res.loc['Feature Stability'][n]:.3f})")
        
    for i in range(len(axes)):
        axes[i][0].set_xlim([min(xmin0),max(xmax0)])
        axes[i][1].set_xlim([min(xmin1),max(xmax1)])
    fig.savefig(os.path.join(output_path,'Stability.png'), dpi=300)
    