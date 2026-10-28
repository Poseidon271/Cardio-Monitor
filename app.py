import os
from flask import Flask, render_template, url_for, request, Response
import numpy as np
import prediction
import visualization
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


app = Flask(__name__)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
PLOT1_PATH = os.path.join(STATIC_DIR, 'plotng.png')
PLOT2_PATH = os.path.join(STATIC_DIR, 'plotng2.png')

def create_figure1(data1):
    fig = Figure(figsize=(7, 4.2), dpi=80)
    ax = fig.subplots()
    barWidth = 0.25
    normal = data1[0]
    user = data1[1]
    br1 = np.arange(len(normal))
    br2 = [x + barWidth for x in br1]
    ax.bar(br1, normal, color='#10b981', width=barWidth, edgecolor='grey', label='Normal Value')
    ax.bar(br2, user, color='#ef4444', width=barWidth, edgecolor='grey', label='Yours Value')
    ax.set_xlabel('Health status defining attributes', fontweight='bold', fontsize=11)
    ax.set_ylabel('respective values', fontweight='bold', fontsize=11)
    ax.set_xticks([r + barWidth for r in range(len(normal))])
    ax.set_xticklabels(['cp', 'chol', 'fbs', 'exang', 'oldpeak', 'slope', 'ca', 'thal'], fontsize=9)
    ax.legend()
    fig.tight_layout()
    FigureCanvas(fig).print_png(PLOT1_PATH)

def create_figure2(data2):
    fig = Figure(figsize=(7, 4.2), dpi=80)
    ax = fig.subplots()
    barWidth = 0.25
    normal = data2[0]
    user = data2[1]
    br1 = np.arange(len(normal))
    br2 = [x + barWidth for x in br1]
    ax.bar(br1, normal, color='#10b981', width=barWidth, edgecolor='grey', label='Normal Value')
    ax.bar(br2, user, color='#ef4444', width=barWidth, edgecolor='grey', label='Yours Value')
    ax.set_xlabel('Health status defining attributes', fontweight='bold', fontsize=11)
    ax.set_ylabel('respective values', fontweight='bold', fontsize=11)
    ax.set_xticks([r + barWidth for r in range(len(normal))])
    ax.set_xticklabels(['trestbps', 'chol', 'thalach'], fontsize=9)
    ax.legend()
    fig.tight_layout()
    FigureCanvas(fig).print_png(PLOT2_PATH)

@app.route('/')
def home():
    global counter2
    counter2+=1
    return render_template('home.html',all_count=counter2)


global counter
counter=0
global counter2
counter2=0

@app.route('/predict', methods=['POST'])
def predict():
    global data1
    global data2
    global counter
    global counter2
    if request.method == 'POST':
        nameofpatient = request.form.get('name', '').strip()
        if not nameofpatient:
            nameofpatient = "Patient"

        age = request.form.get('age', '')
        sex = request.form.get('sex', '')
        cp = request.form.get('cp', '')
        trestbps = request.form.get('trestbps', '')
        chol = request.form.get('chol', '')
        fbs = request.form.get('fbs', '')
        restecg = request.form.get('restecg', '')
        thalach = request.form.get('thalach', '')
        exang = request.form.get('exang', '')
        oldpeak = request.form.get('oldpeak', '')
        slope = request.form.get('slope', '')
        ca = request.form.get('ca', '')
        thal = request.form.get('thal', '')

        try:
            validated = prediction.validate_and_convert_features(
                age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal
            )
        except ValueError as err:
            return render_template(
                'home.html',
                all_count=counter2,
                error_msg=str(err),
                form_data=request.form
            ), 400

        counter += 1
        try:
            result = prediction.predict_from_validated(validated)
            if counter > 50:
                counter = 0

            # database.crudOperation(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal,result)
            data1, data2 = visualization.visualizationpreprocess(age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal, result)
            create_figure1(data1)
            create_figure2(data2)
            return render_template('result.html', prediction=result, nameofpatient=nameofpatient, model_counter=counter, total_counter=counter2)
        except ValueError as err:
            return render_template(
                'home.html',
                all_count=counter2,
                error_msg=str(err),
                form_data=request.form
            ), 400
        except Exception:
            return render_template('error.html'), 500

@app.route('/about')
def about():
    return render_template('disease.html')


@app.errorhandler(500)
def internal_error(error):

    return render_template('error.html')


@app.errorhandler(404)
def not_found(error):
    return "404 error",404

if __name__ == '__main__':
    app.run( debug = True)