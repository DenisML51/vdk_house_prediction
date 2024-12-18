from app import app
from flask import render_template
from models.model_metrics import linear_metrics, lasso_metrics, ridge_metrics, elastic_metrics, forest_metrics, \
    svr_metrics
from models.model_visual import linear_reg_visual, lasso_regression_visual, ridge_regression_visual, elastic_net_visual, \
    random_forest_visual, sv_regression_visual
import os
import pandas as pd
from flask import session
import pickle


@app.route('/metrics', methods=['GET'])
def metrics():
    if 'models_data' in session:
        print("Данные загружаются из сессии")
        models_data = pickle.loads(session['models_data'])
    else:
        print("Данные формируются впервые")
        df = pd.read_csv('C:\\Coding\\vdk_prices - Copy\\data_2.csv')

        models_data = []
        save_path = 'static/images'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        models = [
            {'name': 'Linear Reg', 'metrics': linear_metrics, 'visual': linear_reg_visual},
            {'name': 'Lasso', 'metrics': lasso_metrics, 'visual': lasso_regression_visual},
            {'name': 'Ridge', 'metrics': ridge_metrics, 'visual': ridge_regression_visual},
            {'name': 'Elastic', 'metrics': elastic_metrics, 'visual': elastic_net_visual},
            {'name': 'Random Forest', 'metrics': forest_metrics, 'visual': random_forest_visual},
            {'name': 'SVR', 'metrics': svr_metrics, 'visual': sv_regression_visual},
        ]

        for model in models:
            metrics_data = model['metrics'](df)

            metrics_data = metrics_data.round(3)

            model['visual'](df, save_path)

            graph_paths = [f'{model["name"].lower().replace(" ", "_")}_{i}.png' for i in range(1, 5)]

            models_data.append({
                'model_name': model['name'],
                'metrics_data': metrics_data.to_dict(orient='records')[0],
                'graph_paths': graph_paths
            })

        session['models_data'] = pickle.dumps(models_data)

    return render_template('metrics.html', models_data=models_data)



if __name__ == "__main__":
    app.run(debug=True)
