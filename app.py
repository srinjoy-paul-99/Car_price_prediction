from flask import Flask,render_template,request
import numpy as np
import pandas as pd

df_new = pd.read_csv('Car_price.csv')

app = Flask(__name__)

@app.route('/')

def hello():
    return render_template('index.html')

@app.route('/recommend',methods=['POST'])
def recommend():
    company = request.form.get('company')
    year = request.form.get('year')
    kms = request.form.get('kms')
    fuel = request.form.get('fuel')
    x = recommend_price(company,year,kms,fuel)
    return render_template('index.html',price=x)

def recommend_price(company,year,kms,fuel):
    m = df_new.columns
    d = {}
    for i in m:
        d[i] = 0
    if 'company_' + company in d:
        d['company_' + company] = 1
    if 'fuel_type_' + fuel in d:
        d['fuel_type_' + fuel] = 1
    d['year'] = year
    d['kms_driven'] = kms
    b = df_new.append(d,ignore_index=True).iloc[-1].to_frame().transpose()
    b = b.drop(columns=['Price'])
    X = df_new.drop(columns=['Price'])
    y = df_new['Price'].values
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=1)
    from sklearn.linear_model import LinearRegression
    reg1 = LinearRegression()
    reg1.fit(X_train,y_train)
    p = reg1.predict(b)
    return int(p)

if __name__ == '__main__':
    app.run(debug=True)

