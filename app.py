from flask import Flask, request, jsonify,render_template
import os



app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template('kioskdisplay.html')


#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=port)
    app.run(host='0.0.0.0', port=5000, debug=True)

