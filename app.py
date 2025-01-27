from flask import Flask, render_template
from routes import recommend  # Import the recommend blueprint

app = Flask(__name__)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for the recommendation page
@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

# Route for the history page
@app.route('/history')
def history():
    return render_template('history.html')

# Route for the register page
@app.route('/register')
def register():
    return render_template('register.html')

# Route for the signin page
@app.route('/signin')
def signin():
    return render_template('signin.html')

# Route for the signout page
@app.route('/signout')
def signout():
    return render_template('signout.html')

# Register the '/recommend' route from routes.py
app.register_blueprint(recommend, url_prefix='/recommend')

if __name__ == "__main__":
    app.run(debug=True)