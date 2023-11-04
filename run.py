import os

from flask import Flask, request, jsonify,render_template
from google.cloud import storage

import genqa
app = Flask(__name__)

def generate_response(prompt):
    try:
        # prompt = "Indian budget"
        response = genqa.askchat(prompt)
        print("Elango start to see real data")
        print(response)
        return response
    except Exception as e:
        return e


@app.route("/files")
def files():
    # Get the list of files from the specific bucket folder
    files = get_list_of_files('herfy-dev-documents', 'documents/google-research-pdfs/')

    # Generate the HTML list element
    html_list_element = generate_html_list_element(files)
    return render_template("files.html", filelist =html_list_element )

@app.route("/")
@app.route("/chat")
def chat():
    # Get the list of files from the specific bucket folder
    # files = get_list_of_files('herfy-dev-documents', 'documents/google-research-pdfs/')

    # Generate the HTML list element
    # html_list_element = generate_html_list_element(files)
    return render_template("chat.html")


@app.route("/chat/get")
def get_bot_response():
    user_text = request.args.get('msg')
    return str(generate_response(user_text))

def get_list_of_files(bucket_name, folder_name):
    """Gets a list of files from a specific Google Cloud Storage bucket folder.

    Args:
        bucket_name: The name of the Google Cloud Storage bucket.
        folder_name: The name of the Google Cloud Storage bucket folder.

    Returns:
        A list of file names.
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=folder_name)

    files = []
    for blob in blobs:
        files.append(blob.name)

    return files

def generate_html_list_element(files):
    """Generates an HTML list element from a list of file names.

    Args:
        files: A list of file names.

    Returns:
        An HTML list element.
    """

    html_list_element = []
    for file in files:
        # html_list_element = """<li><a href="https://storage.googleapis.com/{}/{}">{}</a></li>""".format('herfy-dev-documents', file, file)
    # html_list_element += """</ul>"""
        html_list_element.append(file)
    return html_list_element




ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']

    # Validate the file
    if file and allowed_file(file.filename):

        # Upload the file to Google Cloud Storage
        bucket = storage.Client().get_bucket('herfy-dev-documents')
        # blob = bucket.blob(file.filename)
        folder = 'documents/google-research-pdfs/'
        blob = bucket.blob(folder + file.filename)
        blob.upload_from_file(file)

        # Return a success message
        return render_template("chat.html",success= True ) 

    # Return an error message
    return render_template("chat.html",success = False) 

if __name__ == '__main__':
    app.run(host=os.getenv('IP', '0.0.0.0'), 
        port=int(os.getenv('PORT', 8080)))



# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))