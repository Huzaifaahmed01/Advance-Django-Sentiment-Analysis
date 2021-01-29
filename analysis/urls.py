from django.conf.urls import url
from django.urls import path, include
from . import views

urlpatterns = [
	url('text/', views.getText, name ='Text Submition'),
	url('analysis/', views.getAnalysis, name ='Analysis Report'),
    path('feedback',views.feedback,name="feedback"),
	url('', views.home, name ='Sentiment Analysis')
]
