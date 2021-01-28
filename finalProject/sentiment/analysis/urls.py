from django.conf.urls import url
from . import views

urlpatterns = [
	url('/text', views.getText, name ='Text Submition'),
	url('/analysis', views.getAnlysis, name ='Text Analysis'),
]
