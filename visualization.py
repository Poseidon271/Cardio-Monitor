from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import io
import random
import numpy as np

def visualizationpreprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal,result):
    sex_lower = str(sex).strip().lower()
    if sex_lower in ["male", "1"]:
        sex = 1
    else:
        sex = 0
    cp_lower = str(cp).strip().lower()
    if cp_lower in ["typical angina", "0"]:
        cp = 0
    elif cp_lower in ["atypical angina", "1"]:
        cp = 1
    elif cp_lower in ["non-anginal pain", "2"]:
        cp = 2
    elif cp_lower in ["asymptomatic", "3"]:
        cp = 2
    exang_lower = str(exang).strip().lower()
    if exang_lower in ["yes", "1"]:
        exang = 1
    else:
        exang = 0
    fbs_lower = str(fbs).strip().lower()
    if fbs_lower in ["yes", "1"]:
        fbs = 1
    else:
        fbs = 0
    slope_lower = str(slope).strip().lower()
    if "upsloping" in slope_lower or slope_lower == "0":
        slope = 0
    elif "flatsloping" in slope_lower or slope_lower == "1":
        slope = 1
    elif "downsloping" in slope_lower or slope_lower == "2":
        slope = 2
    thal_lower = str(thal).strip().lower()
    if "fixed" in thal_lower or thal_lower in ["2", "6"]:
        thal = 2
    elif "reversable" in thal_lower or "freversable" in thal_lower or thal_lower in ["3", "7"]:
        thal = 3
    elif "normal" in thal_lower or thal_lower in ["1", "2.31"]:
        thal = 1
    restecg_lower = str(restecg).strip().lower()
    if "nothing to note" in restecg_lower or restecg_lower == "0":
        restecg = 0
    elif "st-t" in restecg_lower or restecg_lower == "1":
        restecg = 1
    elif "hypertrophy" in restecg_lower or restecg_lower == "2":
        restecg = 2
    
    #final_list=[int(cp),int(trestbps),int(restecg),int(chol),int(fbs),int(thalach),int(exang),float(oldpeak),int(slope),int(ca),int(thal)]
    normal_value1=[0.478261,0.159420,0.449275,0.550725,1.585507,1.166667,1.166667,2.543478]
    user_value1=[float(cp),float(fbs),float(restecg),float(exang),float(oldpeak),float(slope),float(ca),float(thal)]
    normal_value2=[134.398551,251.086957,139.101449]
    user_value2=[float(trestbps),float(chol),float(thalach)]
    list1=[normal_value1,user_value1]
    list2=[normal_value2,user_value2]

    return list1,list2

# def create_figure1(data1):
#     fig = plt.figure()
#     axis = fig.add_axes([0,0,1,1])
#     y1 = data1[0]
#     y2 = data1[1]
#     width = 0.30
#     x=np.arange(8)
#     axis.bar(x-0.3, y1, width, color='cyan')
#     axis.bar(x, y2, width, color='orange')
#     # axis.bar(xs, ys)
#     # axis.xticks(x, ['cp','chol','fbs','exang','oldpeak','slope','ca','thal'])
#     # axis.xlabel("Heart health defining attributes")
#     axis.set_ylabel("values")
#     # axis.legend(["Normal", "Yours"])
#     axis.set_title('Your data corresponding to normal data')
#     return fig




