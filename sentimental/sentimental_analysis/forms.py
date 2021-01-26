from django.forms import ModelForm
from .models import TextArea

class InputForm(ModelForm):
    class Meta:
        model = TextArea
        fields = "__all__"