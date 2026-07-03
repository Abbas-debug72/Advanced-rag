# routes/__init__.py
from .auth_routes import auth_bp
from .chat_routes import chat_bp
from .widget_routes import widget_bp
from .dashboard_routes import dashboard_bp

# Expose blueprints for easy import
__all__ = ['auth_bp', 'chat_bp', 'widget_bp', 'dashboard_bp']