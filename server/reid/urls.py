from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^identification/', views.identification, name='re-identification'),
    url(r'^new_person/', views.new_person, name='new person'),
    # post request
    url(r'^add_to_database/', views.add_to_database),
    url(r'^req_to_database/', views.req_to_database),
]