import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def stratified_sample(df, strata, size=None, seed=None, keep_index= True):
    '''
    It samples data from a pandas dataframe using strata. These functions use
    proportionate stratification:
    n1 = (N1/N) * n
    where:
        - n1 is the sample size of stratum 1
        - N1 is the population size of stratum 1
        - N is the total population size
        - n is the sampling size
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    :seed: sampling seed
    :keep_index: if True, it keeps a column with the original population index indicator
    
    Returns
    -------
    A sampled pandas dataframe based in a set of strata.
    Examples
    --------
    >> df.head()
    	id  sex age city 
    0	123 M   20  XYZ
    1	456 M   25  XYZ
    2	789 M   21  YZX
    3	987 F   40  ZXY
    4	654 M   45  ZXY
    ...
    # This returns a sample stratified by sex and city containing 30% of the size of
    # the original data
    >> stratified = stratified_sample(df=df, strata=['sex', 'city'], size=0.3)
    Requirements
    ------------
    - pandas
    - numpy
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)

    # controlling variable to create the dataframe or append to it
    first = True 
    for i in range(len(tmp_grpd)):
        # query generator for each iteration
        qry=''
        for s in range(len(strata)):
            stratum = strata[s]
            value = tmp_grpd.iloc[i][stratum]
            n = tmp_grpd.iloc[i]['samp_size']

            if type(value) == str:
                value = "'" + str(value) + "'"
            
            if s != len(strata)-1:
                qry = qry + stratum + ' == ' + str(value) +' & '
            else:
                qry = qry + stratum + ' == ' + str(value)
        
        # final dataframe
        if first:
            stratified_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            first = False
        else:
            tmp_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            stratified_df = stratified_df.append(tmp_df, ignore_index=True)
    
    return stratified_df



def stratified_sample_report(df, strata, size=None):
    '''
    Generates a dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Returns
    -------
    A dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)
    return tmp_grpd


def __smpl_size(population, size):
    '''
    A function to compute the sample size. If not informed, a sampling 
    size will be calculated using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Parameters
    ----------
        :population: population size
        :size: sample size (default = None)
    Returns
    -------
    Calculated sample size to be used in the functions:
        - stratified_sample
        - stratified_sample_report
    '''
    if size is None:
        cochran_n = round(((1.96)**2 * 0.5 * 0.5)/ 0.02**2)
        n = round(cochran_n/(1+((cochran_n -1) /population)))
    elif size >= 0 and size < 1:
        n = round(population * size)
    elif size < 0:
        raise ValueError('Parameter "size" must be an integer or a proportion between 0 and 0.99.')
    elif size >= 1:
        n = size
    return n
## funcion de mapeo de variables
def mapeo_de_variables(df : pd.DataFrame) -> pd.DataFrame:
    """genera una tabla donde muestra, para cada variable en un dataframe, su número de nulos, tipo, valores únicos y porcentaje de nulos"""
    dimension=df.shape
    variables = df.columns.to_list()
    nulos = []
    variable = []
    tipo_variable = []
    valores_unicos = []
    unicos = []
    for variable in variables:
        nulos.append(df[variable].isnull().sum())
        tipo_variable.append(df[variable].dtype)
        valores_unicos.append(len(df[variable].dropna().unique()))
        unicos.append(df[variable].dropna().unique().tolist())
    tabla = pd.DataFrame({"Variable":variables,"Nulos":nulos,"Tipo Variable":tipo_variable,"Valores Unicos": valores_unicos,"Unicos":unicos})
    tabla["Porcentaje Nulos"]=(tabla["Nulos"]/len(df))*100
    tabla.sort_values("Porcentaje Nulos",ascending=False, inplace = True)
    return tabla

def identifica_la_lista_de_variables_constantes_y_las_elimina(df : pd.DataFrame, tabla_de_variables : pd.DataFrame) -> pd.DataFrame:
    """Elimina las columnas de un dataframe que tienen un único valor y retorna el nombre de esas columnas"""
    las_variables_constantes = tabla_de_variables[tabla_de_variables["Valores Unicos"] <= 1]["Variable"]
    df1=df.drop(las_variables_constantes.to_list(), axis=1, inplace = False)
    return df1,las_variables_constantes.to_list()

def identifica_la_lista_de_variables_con_altos_nulos(df : pd.DataFrame, tabla_de_variables : pd.DataFrame, las_variables_con_valores_unicos : pd.DataFrame,por=80) -> pd.DataFrame:
    """Elimina las columnas de un dataframe que tienen mas de 80% valores nulos --y que no sean constantes-- y retorna el nombre de esas columnas"""
    las_variables_con_muchos_nulos = tabla_de_variables[tabla_de_variables["Porcentaje Nulos"] >= por]["Variable"]
    las_variables_con_muchos_nulos = list(set(las_variables_con_muchos_nulos.values) - set(las_variables_con_valores_unicos .values)) 
    df1=df.drop(las_variables_con_muchos_nulos, axis=1, inplace = False)
    return df1,las_variables_con_muchos_nulos


def recodificacion_variables(df,df_tmp,n_cat):
    
# Para Variables Numericas, Valores Unicos hasta 5 la vamos a llamar tipo categórica, recodificamos como category
    variables_cat = df_tmp[df_tmp["Valores Unicos"] <= n_cat]["Variable"]
    for variable in list(variables_cat):
        df[variable]=df[variable].astype('category')    
        
# Recodificar todo lo string a category. Revisar antes de recodificar
    tabla_info_df_final= mapeo_de_variables(df)
    for variable in list(df.dtypes[df.dtypes == 'object'].index):
        df[variable]=df[variable].astype('category')
        
    return df


def conteo_tipos_variable(df):
    tmp_for_plot=pd.DataFrame(df.dtypes.value_counts())
    tmp_for_plot.reset_index(inplace=True)
    tmp_for_plot['index']=tmp_for_plot['index'].astype('string')
    tabla=tmp_for_plot.groupby(['index']).sum().sort_values(0,ascending=False)
    return(tabla)

def type_cols(df):  
    numeric_cols=list(df.select_dtypes(include='number').columns)
    cat_cols=list(df.dtypes[df.dtypes == 'category'].index)
    return (numeric_cols,cat_cols)

def curva_roc(y,modelo,X):
    y_pprob = modelo.predict_proba(X)[::,1]
    false_positive, true_positive, _ = roc_curve(y,  y_pprob)


    print('Modelo AUC:',roc_auc_score(y,  y_pprob))

    Probabilidad=modelo.predict_proba(X=X)
    fpr, tpr, thresholds = roc_curve(y, Probabilidad[:,1])
    AUC_AD=round(roc_auc_score(y, Probabilidad[:,1]),4)
   
    random_probs = [0 for i in range(len(y))]
    p_fpr, p_tpr, _ = roc_curve(y, random_probs, pos_label=1)
    plt.plot(fpr,tpr,linestyle="--",color="green",label="Decision Tree, AUC="+str(AUC_AD))
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    plt.title("CURVA ROC")
    plt.xlabel("Tasa Falsos Positivos")
    plt.ylabel("Tasa Verdaderos Positivos")
    plt.legend()
    plt.show()
def MetricasMatrixConfusion(y,modelo,X):
    Predicciones_test=modelo.predict(X=X)
    Accuracy_RF_test=metrics.accuracy_score(y,Predicciones_test)
    print(f'Accuracy: {Accuracy_RF_test}')
    Reporte_RF_test=metrics.classification_report(y,Predicciones_test)
    print(Reporte_RF_test)
    data1=pd.concat([y,pd.DataFrame(Predicciones_test,columns=["y_Predicted"])],axis=1)
    
    td=data1.rename(columns={"target":"y_Actual","y_Predicted":"y_Predicted"})
    confusion_matrix1 = pd.crosstab(td['y_Actual'], td['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    confusion_matrix_Normalizada=confusion_matrix1.astype('float') / confusion_matrix1.sum(axis=1)[:, np.newaxis]
    sns.heatmap(confusion_matrix_Normalizada, annot=True,linewidths = 0.01, cmap = "Blues")
    plt.show()
    
    
def bar_plot(marcasp,fontsize,a,anchor,fig_size):
    sns.set_theme(style="whitegrid")
    #Grafica de barras 
    ax = marcasp.plot(kind='bar', stacked=True, figsize=fig_size, rot=0, color=["#003677",'#08C6A1'])
    
    # Configuracion de numeros de las barras
    color = ['white','black']
    for c in range(0,2):
        ax.bar_label(ax.containers[c] , label_type='center', style="normal", weight="bold", color = color[c], fontsize=fontsize)

    ax.grid(axis='x')
    
    # leyenda ubicacion en la grafica 
    plt.legend(loc='lower center',bbox_to_anchor=(a, anchor), ncol=2, title=None)    
    
    # Y label para que aparezca con %
    list1 = [0,20,40,60,80,100]
    ax.set_yticks(list1)
    ax.set_yticklabels([str(y)+ "%" for y in list1], fontsize=fontsize)
    
    
    # Aumentar tamaño de letra del xlabel
    fontsize = fontsize
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    return 




def Resultados(test,Predicciones_Test,Probabilidades_Test):
    test["Prediccion"]=Predicciones_Test
    test["Probabilidad"]=Probabilidades_Test[:,1]
    Resultados=test[["date","team","opponent","target","Prediccion","Probabilidad","result","gf","ga"]]
    #Resultados[["target","Prediccion","gf","ga"]]=Resultados[["target","Prediccion","gf","ga"]].astype(int)
    Resultados1=Resultados
    Resultados1["new_team"] = Resultados1["team"]
    merged = Resultados.merge(Resultados1[["date","team","opponent","Prediccion","Probabilidad","target","result"]], left_on=["date", "new_team"], right_on=["date", "opponent"]).drop(columns=["new_team"])
    merged.sort_values(["date","Probabilidad_x"],ascending = [True, True])

    ResultadoFinal=merged.sort_values(["date","Probabilidad_x"],ascending = [False, False])#ResultadoFinal
    ResultadoFinal["EquipoGanador"]="Sin Resultado"
    ResultadoFinal["Target"]="Sin Resultado"
    ResultadoFinal["ResultadoFinal"]="Sin Resultado"
    ResultadoFinal["PrediccionFinal"]="Sin Resultado"
    ResultadoFinal["PreditFinal1"]=ResultadoFinal.Probabilidad_x-ResultadoFinal.Probabilidad_y

    ResultadoFinal["EquipoGanador"][ResultadoFinal.PreditFinal1>0]=ResultadoFinal["team_x"][ResultadoFinal.PreditFinal1>0]
    ResultadoFinal["PrediccionFinal"][ResultadoFinal.PreditFinal1>0]=ResultadoFinal["Prediccion_x"][ResultadoFinal.PreditFinal1>0]
    ResultadoFinal["Target"][ResultadoFinal.PreditFinal1>0]=ResultadoFinal["target_x"][ResultadoFinal.PreditFinal1>0]
    ResultadoFinal["ResultadoFinal"][ResultadoFinal.PreditFinal1>0]=ResultadoFinal["result_x"][ResultadoFinal.PreditFinal1>0]

    ResultadoFinal["EquipoGanador"][ResultadoFinal.PreditFinal1<0]=ResultadoFinal["team_y"][ResultadoFinal.PreditFinal1<0]
    ResultadoFinal["PrediccionFinal"][ResultadoFinal.PreditFinal1<0]=ResultadoFinal["Prediccion_y"][ResultadoFinal.PreditFinal1<0]
    ResultadoFinal["Target"][ResultadoFinal.PreditFinal1<0]=ResultadoFinal["target_y"][ResultadoFinal.PreditFinal1<0]
    ResultadoFinal["ResultadoFinal"][ResultadoFinal.PreditFinal1<0]=ResultadoFinal["result_y"][ResultadoFinal.PreditFinal1<0]

    ResultadoFinal["Validacion"]="Sin Resultado"
    ResultadoFinal["Validacion"][ResultadoFinal.PrediccionFinal==ResultadoFinal.Target]="Acerto"
    ResultadoFinal["Validacion"][ResultadoFinal.PrediccionFinal!=ResultadoFinal.Target]="NoAcerto"

    ResultadoFinal["PreditFinal1"]=ResultadoFinal["PreditFinal1"].abs()

    ResultadoFinal["llave"]=ResultadoFinal.EquipoGanador+ResultadoFinal.date.astype(str)
    ResultadoFinal=ResultadoFinal.drop_duplicates(['llave']).reset_index(drop=True)
    ResultadoFinal=ResultadoFinal[["date","team_x","opponent_x","gf","ga","Probabilidad_x","Probabilidad_y","EquipoGanador","PrediccionFinal","Target","ResultadoFinal","PreditFinal1","Validacion"]]
    
    return ResultadoFinal