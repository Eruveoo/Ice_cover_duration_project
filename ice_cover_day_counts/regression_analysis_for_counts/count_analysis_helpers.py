import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
import matplotlib.image as mpimg


def get_dataset_count_distribution(model,X,return_probs=False):
    '''
    This helper function gives estimation for pseudo-probability distribution for ice-cover duration based on given fitted model and set of observations.
    NOTE: Poisson-distributions are not restricted to the range we are intrested in, so the output distribution is capped and rescaled to fit the annual
            day range requirements. In other words, pseudo-probabilities are estimated only for ice-cover duration range of [0,365].
            More realistic distribution requires tailor made model.

    Inputs:
    -model: Statsmodels fitted model, for which we define distribution.
    -X: This is 'X-data frame' in Stasmodels dmatrices format, which we build when creating the model.
        Input X can be full data frame to which we fitted our model, or a sub-frame of it.
    -return_probs as optional boolean value. If true, then we return ruffly estimated probability matrix as output.
    Ouputs:
    -model_distribution is Scipy frozen distribution object based on given model and X-data.
    -density_frame includes ruff estimates for ice-cover count probability density for full dataset.
        Note that we give estimates only for the count range of [0,365], but Poisson models do not actually have this restriction.
        These pseudo-probabilities are re-scaled densities for capped prediction range.
    - If we gave return_probs=True as input, this function returns probability matrix.
        'A pseudo-probability matrix' includes ruffly estimated probabilities for ice-cover day counts between 0-365 days. 
        Estimation id given for every row observation in our input matrix X.
    '''
    model_distribution = model.get_distribution(X) # This is so called scipy.stats frozen distribution.
    # For some reason, Scipy frozen distribution did not work for matrices, but it can handle vectors.
    prob_matrix = np.full(shape=(X.shape[0],366),fill_value=np.nan)
    for count in range(366):
        prob_matrix[:,count] = model_distribution.pmf(count)
    prob_matrix = prob_matrix*(1/prob_matrix.sum(axis=1)).reshape((X.shape[0],1))
    density_frame = pd.DataFrame(data=prob_matrix.sum(axis=0),columns=['probability_density'])
    density_frame = density_frame/density_frame.sum()
    density_frame.index = density_frame.index.rename('ice_cover_duration')
    if return_probs:
        return model_distribution,prob_matrix,density_frame
    else:
        return model_distribution,density_frame
    
def get_annual_count_distribution(model,vector):
    '''
    This helper function gives estimation for pseudo-probability distribution for ice-cover duration based on given fitted model a annual observation of features.
    NOTE: Poisson-distributions are not restricted to the range we are intrested in, so the output distribution is capped and rescaled to fit the annual
            day range requirements. In other words, pseudo-probabilities are estimated only for ice-cover duration range of [0,365].
            More realistic distribution requires tailor made model.

    Inputs:
    -model: Statsmodels fitted model, for which we define distribution.
    -vector: This is a single row in our dataset, which is annual observation for some lake.
    Ouputs:
    -model_distribution is Scipy frozen distribution object based on given model and vector data.
    -density_frame includes ruff estimates for ice-cover count probability densities.
        Note that we give estimates only for the count range of [0,365], but Poisson models do not actually have this restriction.
        These pseudo-probabilites are re-scaled densities for capped prediction range.
    '''
    model_distribution = model.get_distribution(vector)
    prob_vec = model_distribution.pmf(np.array([count for count in range(0,366)]))
    prob_vec = prob_vec/prob_vec.sum()
    density_frame = pd.DataFrame(data=prob_vec,columns=['probability_density'])
    density_frame.index = density_frame.index.rename('ice_cover_duration')
    return model_distribution,density_frame

def preds_vs_true_distribution(model_preds,
                               true,
                               titles:dict,
                               colors:dict,
                               return_fig=False,
                               file_path=None):
    
    '''
    Inputs:
    model_preds: Statsmodels output for predictions.
    true: true observations as column of Pandas data frame.
    title: Plot titles in dictionary format as
            {'title':'Title text for plot',
            'xlabel':'xlabel text for plot',
            'ylabel':'ylabel text for plot'}
    colors: Visualization colors in dictionary format as
            {'title':'Color for title',
            'preds':'Histogram color for predictions',
            'true':'Histogram color for true values',
            'xlabel':'color for xlabel text',
            'ylabel':'color for ylabel text'}

    Output:
    Prints fancy histogram for distributions od predictions vs real values.
    Visualization is saved to current directory if you give the filename
    as input in a format of 'my_fancy_histogram.png'

    '''
    
    plt.style.use('dark_background')
    fig = plt.figure()
    plt.grid(linestyle='--',
            alpha=0.5)

    _, bins,_ = plt.hist(true,
                        bins=366,
                        density=True,
                        color=colors['true'],
                        alpha=0.5,
                        label='True')
    _ = plt.hist(model_preds,
                bins = bins,
                density=True,
                color=colors['preds'],
                alpha=0.5,
                label='Preds')

    plt.title(titles['title'],
            fontsize=18,
            color=colors['title'],
            fontweight='bold'
          )
    plt.xlabel(titles['xlabel'],
                color=colors['xlabel'],
                fontsize=14)
    plt.ylabel(titles['ylabel'],
                color=colors['ylabel'],
                fontsize=14)
    plt.legend(fontsize=14)

    if return_fig:
        return fig

    if file_path is not None:
        try:
            plt.savefig(file_path,
                       bbox_inches='tight')
        except:
            print('Saving file did not succeed.')

    plt.show()

def visualize_densities(prediction_density,
                        true_observations,
                        colors:dict,
                        titles:dict,
                        return_fig=False,
                        file_path=None):
    ''' 
    Inputs:
    Here vector-type of format means numpy array or Pandas Series/DataFrame object. 
    -prediction_density: A density estimation in vector-type of format.
    -true_observations: True day count observations in vector-type of format.
    Outputs:
    Function shows a visualization including the density as line and a histrogram density of true observations.
    If file_path is given in correct format, the image file is saved to this path.
    '''
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8,6))
    plt.grid(linestyle='--',alpha=0.5)
    plt.hist(true_observations,
             bins=366,
             density=True,
             color=colors['true'],
             label='True')
    
    plt.plot(prediction_density,
             color=colors['preds'],
             label='Preds')
    plt.title(titles['title'],
            fontsize=18,
            color=colors['title'],
            fontweight='bold'
          )
    plt.xlabel(titles['xlabel'],
                color=colors['xlabel'],
                fontsize=14)
    plt.ylabel(titles['ylabel'],
                color=colors['ylabel'],
                fontsize=14)
    plt.legend(fontsize=14)

    if return_fig:
        return fig

    if file_path is not None:
        try:
            plt.savefig(file_path,
                       bbox_inches='tight')
        except:
            print('Saving file did not succeed.')

    plt.show()



def combine_plot_images(density_path,ev_path,title=None,file_path=None,return_fig=None):
    ''' 
    This function combines 2 images. Function always shows the visualization, 
    if all arguments are correctly formed. Besides this, one can choose either to save the file
    into given file_path, or to return the created matplotlob-object for visualization.
    
    Arguments:
    -density_path: path to image created with function visualize_densities()
    -ev_path: path to image created with function preds_vs_true_distribution()
    -title: optional to give main title for combined image as a string.
    -file_path: optional to give saving path for file, if one wants to save the image.
    -return_fig: optional if one wants to return figure-object with this function.
                If we choose to return figure, then file will not be saved, even if file_path is given.
    
    '''
    plt.style.use('dark_background')
    combined_figure, axes  = plt.subplots(1,2,facecolor='#111111',figsize=(10,8))
    img1 = mpimg.imread(density_path)
    axes[0].imshow(img1)
    axes[0].axis('off')

    img2 = mpimg.imread(ev_path)
    axes[1].imshow(img2)
    axes[1].axis('off')
    if title is not None:
        combined_figure.suptitle(title,color='yellow',fontsize=20)
        plt.tight_layout(rect=[0,0.2,1,1.27])
    else:
        plt.tight_layout()

    if return_fig is not None:
        return combined_figure
    if file_path is not None:
        try:
            plt.savefig(file_path,
                       bbox_inches='tight')
        except:
            print('Saving file did not succeed.')
    plt.show()



def short_mse_analysis(preds,true,return_variables=False):
    '''
    Inputs:
    preds: Statsmodels predictions output.
    true: The true target values in Patsy transformation format. 
        This is the target 'y' which you used for modelling.

    Output:
    Function prints short analysis about the squared error and its distribution.
    '''
    se_vec = (preds-true.values.reshape(len(true)))**2
    mse_from_vec =se_vec.mean()
    se_std_scaled = se_vec.std()/np.sqrt(len(se_vec))
    lower = mse_from_vec - 1.96*se_std_scaled
    upper = mse_from_vec + 1.96*se_std_scaled
    print('MSE:',mse_from_vec)
    print('Squared error std scaled:', se_std_scaled)
    print('95% lower bound:', lower)
    print('95% upper bound:', upper)

    if return_variables is True:
        return {'mse':mse_from_vec,
                'SE_of_mse':se_std_scaled,
                'squared_errors':se_vec,
                '95%_lower':lower,
                '95%_upper':upper}
    

def create_model_equation(X_cols,y):
    '''
    This function creates modelling equation for Statsmodels.
    Give column names for data and targets and this returns string equation needed in building the model.

    Inputs:
    X_cols: Column names for covariates in data matrix.
    y: Column name of target variable.

    Input column names are expected to include both tuples and strings, as it is in this case.

    Output:
    Statsmodels model equation in string format.
    '''
    if type(X_cols[0]) is tuple:
        eq = y+'~'+X_cols[0][0]
    else:
        eq = y+'~'+X_cols[0]
    for i in range(1,len(X_cols)):
        if type(X_cols[i])==tuple:
            add = '+' + X_cols[i][0]
        else:
            add = '+' + X_cols[i]
        eq += add

    return eq

def columns_for_model_equation(columns):
    '''
    Input:
    colums: Column names after polynomial transformation.
    Ouput:
    Function returns column names in a format that fits for Statsmodels modelling equation.
    '''
    new_cols = []
    for i in range(len(columns)):
        if type(columns[i]) is tuple:
            new_cols.append(columns[i][0])
        else:
            new_string = columns[i].replace(' ','_x_')
            if new_string[-2:] == '^2':
                new_string = new_string[:-2] + '_2nd'
            new_cols.append(new_string)

    return new_cols

