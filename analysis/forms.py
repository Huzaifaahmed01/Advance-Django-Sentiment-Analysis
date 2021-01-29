from django.forms import ModelForm
from .models import *
from django import forms

class InputForm(ModelForm):
    class Meta:
        model = sentimentText
        fields = '__all__'
        widgets = {
            'submittedText': forms.Textarea(attrs={'cols':85})
        }
