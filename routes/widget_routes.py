from flask import Blueprint, send_file

widget_bp = Blueprint('widget', __name__)

@widget_bp.route('/widget.js')
def serve_widget():
    return send_file('widget.js', mimetype='application/javascript')