from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

index_str = \
"""
<h1>Cloud Terminal Object Tracking and Person Re-identification.</h1>
<a href="reid">reid</a>
<a href="detection">detection</a>
"""

def index(request):
    return render(request, 'index.html')