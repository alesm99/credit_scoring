# funciones auxiliares para el proyecto

import numpy as np
import pandas as pd

def woe(df:pd.DataFrame,var:str,target:str)->pd.DataFrame:
    """ Calcula el woe de una variable

    Args:
        df (pd.DataFrame): dataframe con las variables
        var (str): nombre de la variable
        target (str): nombre de la variable target

    Returns:
        pd.DataFrame: dataframe con el woe de la variable
    """
    n=df.shape[0]
    woe=(df[[var,target]].reset_index().groupby([var,target]).size()).unstack()
    woe.rename(columns={0:'bueno',1:'malo'},inplace=True)
    woe['total']=woe['bueno']+woe['malo']
    woe['g']=woe['bueno']/sum(woe['bueno'])
    woe['b']=woe['malo']/sum(woe['malo'])
    woe['woe']=np.log(woe['g']/woe['b'])
    return woe

def iv(df:pd.DataFrame,var:str,target:str)->float:
    vals=woe(df,var,target)
    return np.sum((vals['g']-vals['b'])*vals['woe'])



def categorizar(df:pd.DataFrame,col:str,umbral:float=.05,valor:str='other')->pd.Series:
    """ Categoriza una variable en base a un umbral

    Args:
        df (pd.DataFrame): dataframe con la variable
        col (str): nombre de la variable
        umbral (float, optional): umbral para categorizar. Defaults to .05.
        valor (str, optional): valor para categorizar. Defaults to 'other'.

    Returns:
        pd.Series: serie categorizada
    """
    aux=df[col].value_counts(normalize=True)<umbral
    mapa={v:valor for v in aux[aux].index}
    return df[col].replace(mapa)
