from django.shortcuts import render
from django.http import HttpResponse
from .forms import InputForm
from django.core.exceptions import ValidationError
from django import forms
import json
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

    # data_list = {
    #  "labels":  ['YES', 'NO', 'NEVER'] ,
    #  "data" : [70, 20, 10]
    # }
    # dataJSON  = json.dumps(data_list)

    # data_list2 = {
    #     'hello': 'World',
    #     'geeks': 'forgeeks',
    #     'ABC': 123,
    #     456: 'abc',
    #     14000605: 1,
    #     "labels":  ['YES', 'NO', 'NEVER'] ,
    #     "data" : [70, 20, 10]
    # }
    # dataJSON  = json.dumps(data_list2)
    data = [{'name': 'Positive',
                    'y': 80,
                    'sliced': True,
                    'selected': True
                }, {
                    'name': 'Negative',
                    'y': 20
                }]

    dataJSON = json.dumps(data)

    labels =  ['YES', 'NO', 'NEVER']
    data = [70, 20, 10]
    emotion="sadness"
    if request.method == "POST" :
        result = request.POST.get("text")
        if result.isnumeric():
            error_msg="Enter a text"


            return render(request,'index.html',{'error':error_msg, 'form': form})

    # return render(request,"result.html" ,context={'result': result, 'data': dataJSON})
    # return render(request,"result.html" ,context={'result': result, 'data': data, 'labels': labels})
    return render(request, 'result.html', {
        'labels': labels,
        'data': dataJSON,
        'result': result,'emotion':emotion
    })
