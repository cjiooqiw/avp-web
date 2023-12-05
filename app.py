from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# 加载机器学习模型
model = joblib.load('trained_model.pkl')  # 替换为你的模型文件路径

@app.route('/')
def home():
    # 渲染主页
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            # 处理文件上传
            file = request.files['file']
            # 这里添加文件处理和模型预测的代码
            # 例如，读取文件内容，然后进行预测
            # file_content = file.read()
            # input_data = np.array([...]).reshape(-1, 1)
            # prediction = model.predict(input_data)[0]
            # result = 'Positive' if prediction == 1 else 'Negative'
        elif 'input_text' in request.form:
            # 处理直接输入的文本
            input_text = request.form['input_text']
            # 将输入转换为二维数组
            input_data = np.array([len(input_text)]).reshape(-1, 1)
            # 进行预测
            prediction = model.predict(input_data)[0]
            result = 'Positive' if prediction == 1 else 'Negative'

    return render_template('predict.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
