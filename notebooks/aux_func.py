#Métricas 
from sklearn.metrics import r2_score, classification_report, fbeta_score, ConfusionMatrixDisplay, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as ss
import pandas as pd

def metrics(y_test, y_pred):

    print(f'\nF2 Score: {fbeta_score(y_test, y_pred, beta=2)}\n') # F2-Score

    print(f'R2 Score: {r2_score(y_test, y_pred)}\n') # R2

    print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}\n') # Accuracy

    print(classification_report(y_test, y_pred, labels= [0, 1])) # Tabla con precision, recall, f1-score, support

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred) # Confusion Matrix
    disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}") # devuelve la matrix en texto



def get_corr_matrix(dataset = None, metodo='pearson', size_figure=[10,8]):
    # Para obtener la correlación de Spearman, sólo cambiar el metodo por 'spearman'

    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    # Compute the correlation matrix
    corr = dataset.corr(method=metodo) 
    # Set self-correlation to zero to avoid distraction
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5,  cmap ='viridis' , linecolor="white",vmin=-1, vmax=1) #cbar_kws={"shrink": .5}
    plt.show()
    
    return 0

def cramers_v(var1,var2):

    """ 
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: tabla creada con pd.crosstab()
    
    """
    crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None))
    chi2 = ss.chi2_contingency(crosstab)[0]
    n = crosstab.sum()
    phi2 = chi2 / n
    r, k = crosstab.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))