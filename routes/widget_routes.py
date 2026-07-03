from flask import Blueprint

widget_bp = Blueprint('widget', __name__)

@widget_bp.route('/widget.js')
def serve_widget():
    # Read the static JavaScript file from the project root
    with open('widget.js', 'r') as f:
        return f.read(), 200, {'Content-Type': 'application/javascript'}