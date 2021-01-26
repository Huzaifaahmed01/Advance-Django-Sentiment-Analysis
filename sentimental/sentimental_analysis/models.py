from django.db import models
from django.core import validators
from django import forms
# Create your models here.
def numeric(value):
    if value is isnumeric():
        raise forms.ValidationError("Enter a text")
class TextArea(models.Model):
    text = models.TextField(max_length=1000,validators=[numeric])
    

