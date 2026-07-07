from flask import Blueprint, send_file, request
import time

widget_bp = Blueprint('widget', __name__)

@widget_bp.route('/widget.js')
def serve_widget():
    # Serve with cache-control headers to prevent caching
    response = send_file('widget.js', mimetype='application/javascript')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Alternative: serve with a version parameter (use in your HTML)
@widget_bp.route('/widget.js')
def serve_widget_versioned():
    # If you want to force a new version, you can add a query param in your HTML
    # e.g., <script src="/widget.js?v=20250707"></script>
    return send_file('widget.js', mimetype='application/javascript')