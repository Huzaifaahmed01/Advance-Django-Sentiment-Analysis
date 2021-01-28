from django.forms import ModelForm
from .models import sentimentText
from django import forms

class InputForm(ModelForm):
    class Meta:
        model = sentimentText
        fields = '__all__'
