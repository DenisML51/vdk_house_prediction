from flask import render_template, request
import pandas as pd
import joblib
from app import app
from models.model_predict import predict_price_with_model


def format_price(price):
    return f"{price:,.0f}".replace(",", " ")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'get':
        model_name = request.form['model']

        user_data = {
            'rooms': float(request.form['rooms']),
            'area': float(request.form['area']),
            'kitchen_area': float(request.form['kitchen_area']),
            'Годпостройки': int(request.form['Годпостройки']),
            'Количествоэтажей': int(request.form['Количествоэтажей']),
            'Количествобалконов': int(request.form['Количествобалконов']),
            'Количествоквартир': int(request.form['Количествоквартир']),
            'Количествоподъездов': int(request.form['Количествоподъездов']),
            'number_of_owners': int(request.form['number_of_owners']),
            'number_of_elevators': int(request.form['number_of_elevators'])
        }

        one_hot_fields = [
            'renovation_Евро', 'renovation_Косметический', 'renovation_Отсутствует',
            'deal_type_Свободная продажа',
            'Материалстен_Железобетон', 'Материалстен_Кирпично-монолитный', 'Материалстен_Кирпичный',
            'Материалстен_Монолитный', 'Материалстен_Панельный', 'Материалстен_Смешанные',
            'Сериядома_1-513', 'Сериядома_121', 'Сериядома_121. многоквартирный дом',
            'Сериядома_14/14, индивидуальный', 'Сериядома_2.многоквартирный', 'Сериядома_25',
            'Сериядома_89152-к примогражданпроект', 'Сериядома_II-01', 'Сериядома_II-03',
            'Сериядома_II-32', 'Сериядома_II-66', 'Сериядома_v-кирпичный', 'Сериядома_ПП-83',
            'Сериядома_у-0101',
            'Типперекрытий_Железобетонный', 'Типперекрытий_Монолитный', 'Типперекрытий_Смешанный',
            'Типфундамента_Иной', 'Типфундамента_Ленточный', 'Типфундамента_Сборный',
            'Типфундамента_Свайный', 'Типфундамента_Сплошной',
            'Горячееводоснабжение_Поквартирный котел',
            'Теплоснабжение_Центральное',
            'energy_efficiency_class_C', 'energy_efficiency_class_D', 'energy_efficiency_class_E',
            'ventilation_Приточно-вытяжная'
        ]

        for field in one_hot_fields:
            user_data[field] = 0

        selected_categories = [
            request.form.get('renovation', ''),
            request.form.get('deal_type', ''),
            request.form.get('Материалстен', ''),
            request.form.get('Сериядома', ''),
            request.form.get('Типперекрытий', ''),
            request.form.get('Типфундамента', ''),
            request.form.get('Горячееводоснабжение', ''),
            request.form.get('Теплоснабжение', ''),
            request.form.get('energy_efficiency_class', ''),
            request.form.get('ventilation', '')
        ]

        for category in selected_categories:
            if category:
                user_data[category] = 1

        prediction = predict_price_with_model(user_data, model_name)
        formatted_prediction = format_price(prediction)
        return render_template('predict.html', prediction=formatted_prediction)

    return render_template('predict.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
