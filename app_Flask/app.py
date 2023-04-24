from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Загрузка сохраненных архитектур и весов моделей
with open('best_neural_network1_architecture.json', 'r') as json_file:
    best_neural_network1_architecture = json_file.read()

with open('best_neural_network2_architecture.json', 'r') as json_file:
    best_neural_network2_architecture = json_file.read()

# Создание моделей и загрузка весов
best_neural_network1 = model_from_json(best_neural_network1_architecture)
best_neural_network1.load_weights('best_neural_network1_weights.h5')

best_neural_network2 = model_from_json(best_neural_network2_architecture)
best_neural_network2.load_weights('best_neural_network2_weights.h5')

# Загрузка сохраненного масштабировщика
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Получаем значения параметров из формы
        density = float(request.form['density'])
        modulus = float(request.form['modulus'])
        hardener = float(request.form['hardener'])
        epoxy = float(request.form['epoxy'])
        flash_point = float(request.form['flash_point'])
        surface_density = float(request.form['surface_density'])
        resin_consumption = float(request.form['resin_consumption'])
        weave_angle = float(request.form['weave_angle'])
        weave_step = float(request.form['weave_step'])
        weave_density = float(request.form['weave_density'])
        tensile_modulus = float(request.form['tensile_modulus'])
        tensile_strength = float(request.form['tensile_strength'])

        # Вычисляем рекомендуемое соотношение матрица-наполнитель
        matrix_filler_ratios = np.linspace(0.1, 1.0, num=100).reshape(-1, 1)
        other_features = np.array(
            [1, density, modulus, hardener, epoxy, flash_point, surface_density, resin_consumption, weave_angle,
             weave_step, weave_density,])
        new_data = np.tile(other_features, (100, 1))
        new_data[:, 0] = matrix_filler_ratios[:, 0]

        # Масштабирование новых данных с использованием сохраненного масштабирования объекта
        new_data_scaled = scaler.transform(new_data)

        predicted_y1_values = best_neural_network1.predict(new_data_scaled)
        predicted_y2_values = best_neural_network2.predict(new_data_scaled)
        optimal_index = np.argmax(predicted_y1_values * predicted_y2_values)
        optimal_ratio = matrix_filler_ratios[optimal_index][0]

        recommended_ratio = f"{optimal_ratio:.2f}"

        # Возвращаем шаблон с результатами
        return render_template('results.html', recommended_ratio=recommended_ratio)
    else:
        # Возвращаем форму для ввода параметров
        return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True)
