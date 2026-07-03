from flask import Blueprint, send_from_directory

widget_bp = Blueprint('widget', __name__)

@widget_bp.route('/widget.js')
def serve_widget():
    # Serve the widget.js file from the static folder
    return send_from_directory('static', 'widget.js')