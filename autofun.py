from flask import Flask, render_template
from apscheduler.schedulers.background import BackgroundScheduler
import time

app = Flask(__name__)
scheduler = BackgroundScheduler()

@app.route('/data')
def get_data():
    print('exec')
    # Do something to get the data you want to display
    data = time.strftime('%H:%M:%S')
    return data

@app.route('/')
def index():
    # Render the index.html template with the data from the /data route
    return render_template('index.html', data=get_data())

def scheduled_job():
    # Call the /data route every one hour
    app.test_client().get('/')

if __name__ == '__main__':
    # Schedule the job to run every one hour
    scheduler.add_job(scheduled_job, 'interval', minutes=5)
    scheduler.start()
    app.run(host='0.0.0.0', port=8080, debug=True)
