from django.shortcuts import render
from django.http import HttpResponse
from .forms import InputForm
from django.core.exceptions import ValidationError
from django import forms
# Create your views here.
def view(request):

    form = InputForm()
    
    # if request.method=="POST":
    #     result=request.POST.get("input_text")
    #     print(result)
    #     result = result.replace(" ", "").replace("\n", "").replace("\t", "")
    #     if result=="" or result.isnumeric():
    #         display="block;"
    #         text="center"
    #         context={"display":display,"textalign":text}
    #         return render(request,"index.html",context)

    #     else:
    #         return render(request,"result.html",context={"result":result})
    # # return HttpResponse("Hello World")
    return render(request,"index.html",context={'form': form })    
   

def result(request):
    form = InputForm()
    data = [76,24]
    # t = json.dumps(data_list)

    if request.method == "POST" :
        result = request.POST.get("text")
        if result.isnumeric():
            error_msg="Enter a text"

            return render(request,'index.html',{'error':error_msg, 'form': form})
    return render(request,"result.html",context={'result': result }, {'data': data})    
