from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from django.views.decorators.csrf import csrf_exempt
from .net import person_detection

# Create your views here.

def index(request):
    return render(request, 'detection.html')

@csrf_exempt
def detection(request):
    try:
        file = request.FILES['upload']
        image = imread(file.temporary_file_path())
        assert image is not None
    except:
        return bad_request(request, 'bad_request.html')
    dets = person_detection(image)

    return JsonResponse({'roi':dets.tolist()})