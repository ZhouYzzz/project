from __future__ import unicode_literals

from django.db import models

# Create your models here.

class Person(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=20)
    # upload = models.FileField()
    image_path = models.CharField(max_length=40, blank=True)
    feature = models.CharField(max_length=40, blank=True)
    def __str__(self):
        return self.name
