from flask import Flask

app=Flask(__name__)



@app.route("/")
def hellow():
    return "Hello a!"




if __name__=="__main__":
    app.run(host='0.0.0.0',debug=False,port=8000)