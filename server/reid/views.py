from django.shortcuts import render, render_to_response, redirect
from django.http import HttpResponse
from django.core.urlresolvers import reverse
from django.views.defaults import bad_request
from .models import Person

from django.views.decorators.csrf import csrf_exempt
from server.settings import *

from skimage.io import imread, imsave
from skimage.transform import resize
from cv2 import imread, resize, imwrite

from .net import reid, get_feature

import numpy as np

import random
import string

def id_generator(size=3, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
# Create your views here.

def index(request):
    context = {'persons': Person.objects.all()}
    return render(request, 'reid.html', context)

def new_person(request):
    return render(request, 'new_person.html')

def identification(request):
    return render(request, 'identification.html')

# @csrf_exempt
# def new(request):
#     return render(request, 'reid/new.html')

# @csrf_exempt
# def delete(request):
#     Person.objects.create(name=request.POST['name'])
#     print request.FILES
#     context = {'persons': Person.objects.all()}
#     return redirect('../', **context)

""" Re-identify posted image """
@csrf_exempt
def req_to_database(request):
    pass

""" Add new person to database """
@csrf_exempt
def add_to_database(request):
    try:
        name = request.POST['name']
        file = request.FILES['upload']
        image = imread(file.temporary_file_path())
        assert image is not None
    except:
        return bad_request(request, 'bad_request.html')

    path, feature_path = savefile(image, name)
    create_person(name, path, feature_path)
    return redirect('/reid/')

def savefile(image, name):
    # InMemoryUploadedFile
    if image.shape == (256,128,3):
        pass
    else:
        image = resize(image, (128,256))
        pass
    filename = name+'_'+id_generator()
    # save image file for display
    relative_path = '/static/reid_person/'+filename+'.jpg'
    abs_path = BASE_DIR + relative_path
    imwrite(abs_path, image)
    # save feature to npy file
    feature = get_feature(image)
    feature_relative_path = '/static/reid_feature/'+filename+'.npy'
    feature_path = BASE_DIR + feature_relative_path
    np.save(feature_path, feature)

    return relative_path, feature_relative_path

def create_person(name, path, feature_path):
    # print path
    Person.objects.create(name=name, image_path=path, feature=feature_path)
    return

""" Delete from database, protected """
@csrf_exempt
def del_from_database(request):
    pass
