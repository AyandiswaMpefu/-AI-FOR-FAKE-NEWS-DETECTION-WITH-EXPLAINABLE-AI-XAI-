from django.urls import path
from . import views

urlpatterns = [
    # existing route for the form-backed view
    path("", views.classify_view, name="classify"),
    # AJAX route for base analyze
    path("ajax/analyze/", views.classify_ajax, name="classify_ajax"),
    # Endpoint polled for explanation readiness
    path("ajax/explain-status/", views.explain_status, name="explain_status"),
]
